"""PPL benchmark V3: fix de re-quantizacao + deferred + outlier splitting."""
from __future__ import annotations
import gc, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_cache import TurboQuantCache, TQCacheConfig
from tq_cache_v3 import TurboQuantCacheV3, TQCacheV3Config

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TEXT = open("data/wiki.test.raw", "r", encoding="utf-8").read()

def compute_ppl(model, tokenizer, text, prefill_len=256, decode_len=256,
                cache_factory=None, device="cuda"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    total = prefill_len + decode_len
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=total).to(device)
    ids = enc.input_ids
    actual = ids.shape[1]
    if actual < total:
        decode_len = actual - prefill_len
    cache = cache_factory() if cache_factory else None
    with torch.inference_mode():
        out = model(ids[:, :prefill_len], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nlls = []
        for i in range(decode_len - 1):
            tok = ids[:, prefill_len + i: prefill_len + i + 1]
            out = model(tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            target = ids[:, prefill_len + i + 1]
            nll = torch.nn.functional.cross_entropy(out.logits[:, -1, :], target)
            if not torch.isnan(nll):
                nlls.append(nll.item())
    if not nlls:
        return float("inf"), 0, {}, 0
    avg = sum(nlls) / len(nlls)
    ppl = torch.exp(torch.tensor(avg)).item()
    metrics = cache.get_metrics() if hasattr(cache, "get_metrics") else {}
    return ppl, avg, metrics, len(nlls)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print("PPL V3: no-requ + deferred + outlier split — Qwen2.5-0.5B")
    print("=" * 75)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=device)
    model.requires_grad_(False)
    for m in model.modules():
        m.training = False
    nl = model.config.num_hidden_layers
    hd = model.config.hidden_size // model.config.num_attention_heads
    pf, dc = 256, 256
    print(f"Layers={nl} head_dim={hd} prefill={pf} decode={dc}")

    cfgs = [
        ("A_baseline", None),
        ("B_v1_3bit_uni", lambda: TurboQuantCache(
            TQCacheConfig(k_bits=3, v_bits=3, residual_window=64,
                          protect_layers_init=2, protect_layers_final=2, v_only=True),
            n_layers=nl, head_dim=hd, device=device)),
        ("C_v1_4bit_uni", lambda: TurboQuantCache(
            TQCacheConfig(k_bits=4, v_bits=4, residual_window=64,
                          protect_layers_init=2, protect_layers_final=2, v_only=True),
            n_layers=nl, head_dim=hd, device=device)),
        ("D_v3_3.5_defer", lambda: TurboQuantCacheV3(
            TQCacheV3Config(k_bits_low=3, k_bits_high=4, v_bits_low=3, v_bits_high=4,
                            n_outlier_channels=16, residual_window=64,
                            protect_layers_init=2, protect_layers_final=2,
                            v_only=True, deferred=True),
            n_layers=nl, head_dim=hd, device=device)),
        ("E_v3_3.5_no_defer", lambda: TurboQuantCacheV3(
            TQCacheV3Config(k_bits_low=3, k_bits_high=4, v_bits_low=3, v_bits_high=4,
                            n_outlier_channels=16, residual_window=64,
                            protect_layers_init=2, protect_layers_final=2,
                            v_only=True, deferred=False),
            n_layers=nl, head_dim=hd, device=device)),
        ("F_v3_3.5_kv_defer", lambda: TurboQuantCacheV3(
            TQCacheV3Config(k_bits_low=3, k_bits_high=4, v_bits_low=3, v_bits_high=4,
                            n_outlier_channels=16, residual_window=128,
                            protect_layers_init=4, protect_layers_final=4,
                            v_only=False, deferred=True),
            n_layers=nl, head_dim=hd, device=device)),
    ]

    base_ppl = None
    print(f"\n{'config':<25} {'PPL':<10} {'delta':<10} {'comp':<7} {'q_calls':<8} {'time':<6}")
    print("-" * 75)
    for name, fac in cfgs:
        t0 = time.time()
        ppl, nll, met, nt = compute_ppl(model, tokenizer, TEXT, pf, dc, fac, device)
        elapsed = time.time() - t0
        comp = met.get("compression_ratio", 1.0)
        qc = met.get("total_quantize_calls", 0)
        if base_ppl is None:
            base_ppl = ppl
            d = "baseline"
        else:
            dp = ((ppl / base_ppl) - 1) * 100
            d = f"{'+' if dp >= 0 else ''}{dp:.2f}%"
        print(f"{name:<25} {ppl:<10.4f} {d:<10} {comp:<7.2f} {qc:<8} {elapsed:<6.1f}s")

if __name__ == "__main__":
    main()
