"""
Benchmark A/B/C: testa os 2 patches do paper.
  A = Baseline FP16
  B = V1 turbo (3-bit uniforme, sem deferred)
  C = V2 turbo (3.5-bit mixed + deferred quantization)

Roda em Qwen2.5-0.5B-Instruct para comparacao direta com resultados anteriores.
"""
from __future__ import annotations

import gc
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tq_cache import TurboQuantCache, TQCacheConfig
from tq_cache_v2 import TurboQuantCacheV2, TQCacheV2Config

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT = """You are a helpful AI assistant. Read the following passage carefully and answer the question.

The transistor was invented at Bell Labs in 1947 by John Bardeen, Walter Brattain, and William Shockley. This invention revolutionized electronics by replacing bulky vacuum tubes with smaller, more reliable semiconductor devices. The team received the Nobel Prize in Physics in 1956 for their discovery. Bardeen later won a second Nobel Prize in 1972 for his work on superconductivity, making him the only person to win two Nobel Prizes in Physics.

Intel's first microprocessor, the 4004, was released in 1971. It contained 2,300 transistors and ran at 740 kHz. By comparison, modern processors contain billions of transistors and operate at frequencies measured in gigahertz.

Question: Who invented the transistor, in what year, and what prize did they receive?"""


def reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run(model, tokenizer, prompt, max_new=48, cache_obj=None, device="cuda"):
    reset()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    with torch.inference_mode():
        t0 = time.time()
        out = model(input_ids, past_key_values=cache_obj, use_cache=True)
        pkv = out.past_key_values
        if device == "cuda":
            torch.cuda.synchronize()
        prefill_t = time.time() - t0

        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = [next_tok]
        logits_list = [out.logits[:, -1, :].clone()]

        t0 = time.time()
        for _ in range(max_new - 1):
            out = model(next_tok, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(next_tok)
            logits_list.append(out.logits[:, -1, :].clone())
        if device == "cuda":
            torch.cuda.synchronize()
        decode_t = time.time() - t0

    tokens = torch.cat(gen_ids, dim=1)
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    logits = torch.stack(logits_list, dim=1)
    tps = max_new / max(decode_t, 1e-6)

    metrics = {}
    if hasattr(pkv, "get_metrics"):
        metrics = pkv.get_metrics()

    return {
        "text": text,
        "ids": tokens[0].cpu().tolist(),
        "prefill_s": prefill_t,
        "decode_s": decode_t,
        "tok_per_s": tps,
        "vram_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0,
        "cache_metrics": metrics,
    }, logits


def compare(base, test, base_logits, test_logits):
    ids_b = torch.tensor(base["ids"])
    ids_t = torch.tensor(test["ids"])
    n = min(len(ids_b), len(ids_t))
    match = (ids_b[:n] == ids_t[:n]).float().mean().item()
    div = -1
    for i in range(n):
        if ids_b[i] != ids_t[i]:
            div = i
            break
    lb = base_logits[0, :n].float()
    lt = test_logits[0, :n].float()
    cos = F.cosine_similarity(lb, lt, dim=-1).mean().item()
    top5_b = lb.topk(5, dim=-1).indices
    top5_t = lt.topk(5, dim=-1).indices
    overlap = []
    for i in range(n):
        o = len(set(top5_b[i].tolist()) & set(top5_t[i].tolist())) / 5
        overlap.append(o)
    return {
        "token_match": match,
        "first_div": div,
        "logits_cos": cos,
        "top5_overlap": sum(overlap) / len(overlap),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("TurboQuant Patch Benchmark — V1 vs V2 (outlier split + deferred)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=device)
    model.requires_grad_(False)
    for m in model.modules():
        m.training = False

    n_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Layers: {n_layers}, head_dim: {head_dim}")
    prompt_len = len(tokenizer(PROMPT).input_ids)
    print(f"Prompt: {prompt_len} tokens, gen: 48 tokens")

    configs = {
        "A_baseline_fp16": None,
        "B_v1_turbo3_uniform": ("v1", TQCacheConfig(
            k_bits=3, v_bits=3, residual_window=64,
            protect_layers_init=2, protect_layers_final=2, v_only=True,
        )),
        "C_v1_turbo4_uniform": ("v1", TQCacheConfig(
            k_bits=4, v_bits=4, residual_window=64,
            protect_layers_init=2, protect_layers_final=2, v_only=True,
        )),
        "D_v2_mixed3.5_deferred": ("v2", TQCacheV2Config(
            k_bits_low=3, k_bits_high=4, v_bits_low=3, v_bits_high=4,
            n_outlier_channels=32, residual_window=64,
            protect_layers_init=2, protect_layers_final=2,
            v_only=True, deferred=True,
        )),
        "E_v2_mixed3.5_no_defer": ("v2", TQCacheV2Config(
            k_bits_low=3, k_bits_high=4, v_bits_low=3, v_bits_high=4,
            n_outlier_channels=32, residual_window=64,
            protect_layers_init=2, protect_layers_final=2,
            v_only=True, deferred=False,
        )),
        "F_v2_mixed3.5_both_kv": ("v2", TQCacheV2Config(
            k_bits_low=3, k_bits_high=4, v_bits_low=3, v_bits_high=4,
            n_outlier_channels=32, residual_window=128,
            protect_layers_init=4, protect_layers_final=4,
            v_only=False, deferred=True,
        )),
    }

    results = {}
    base_result = None
    base_logits = None

    for name, cfg_info in configs.items():
        print(f"\n{'='*70}")
        print(f"Config: {name}")
        print(f"{'='*70}")

        cache = None
        if cfg_info is not None:
            version, cfg = cfg_info
            if version == "v1":
                cache = TurboQuantCache(cfg, n_layers=n_layers, head_dim=head_dim, device=device)
            else:
                cache = TurboQuantCacheV2(cfg, n_layers=n_layers, head_dim=head_dim, device=device)

        result, logits = run(model, tokenizer, PROMPT, max_new=48, cache_obj=cache, device=device)
        print(f"  Prefill:  {result['prefill_s']:.3f}s")
        print(f"  Decode:   {result['decode_s']:.3f}s ({result['tok_per_s']:.1f} tok/s)")
        print(f"  VRAM:     {result['vram_mb']:.1f} MB")
        print(f"  Text:     {result['text'][:150]}...")
        if result["cache_metrics"]:
            cm = result["cache_metrics"]
            print(f"  Compress: {cm.get('compression_ratio', 1):.2f}x")
            if "avg_bits_v" in cm:
                print(f"  Avg bits: K={cm.get('avg_bits_k', 0):.1f} V={cm.get('avg_bits_v', 0):.1f}")
            if "prefill_tokens" in cm:
                print(f"  Deferred: prefill={cm.get('prefill_tokens', 0)} decode={cm.get('decode_tokens', 0)}")

        if base_result is None:
            base_result = result
            base_logits = logits
        else:
            cmp = compare(base_result, result, base_logits, logits)
            print(f"  vs BASE: match={cmp['token_match']:.2%} cos={cmp['logits_cos']:.4f} top5={cmp['top5_overlap']:.2%} div@{cmp['first_div']}")

        results[name] = result

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'config':<30} {'tok/s':<8} {'match':<8} {'cos':<8} {'top5':<8} {'comp':<6} {'correct?':<8}")
    print("-" * 70)
    for name, r in results.items():
        tok_s = r["tok_per_s"]
        comp = r["cache_metrics"].get("compression_ratio", 1.0)
        correct = "YES" if "Shockley" in r["text"] and "1947" in r["text"] else "NO"
        if name == "A_baseline_fp16":
            print(f"{name:<30} {tok_s:<8.1f} {'100%':<8} {'1.000':<8} {'100%':<8} {comp:<6.2f} {correct:<8}")
        else:
            cmp = compare(base_result, r, base_logits, results[name] if isinstance(results[name], torch.Tensor) else None)
            # Re-compute since we stored results not logits
            print(f"{name:<30} {tok_s:<8.1f} {'—':<8} {'—':<8} {'—':<8} {comp:<6.2f} {correct:<8}")

    with open("bench_patches_result.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in results.items()},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to bench_patches_result.json")


if __name__ == "__main__":
    main()
