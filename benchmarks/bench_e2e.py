"""
Benchmark E2E: compara modelo com Turbo ON vs OFF em metricas reais.

Metricas:
1. Perplexity no prompt de referencia
2. VRAM peak durante inferencia
3. Tokens/sec (decode)
4. Compression ratio real (bytes de KV cache)
5. Cosine similarity dos logits vs baseline
6. Token match rate (top-1) vs baseline
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tq_cache import TurboQuantCache, TQCacheConfig


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

LONG_PROMPT = """You are a helpful AI assistant. I'll give you a long passage about the history of computing, and then ask you a specific question about it. Read carefully.

The history of computing begins well before the development of modern digital computers. The earliest counting devices include the abacus, which originated in ancient Mesopotamia and was used across various civilizations. In the 17th century, mathematicians like Blaise Pascal and Gottfried Wilhelm Leibniz designed mechanical calculators. Pascal's Pascaline could add and subtract, while Leibniz's Stepped Reckoner could also multiply and divide.

Charles Babbage's work in the 19th century is often cited as the true beginning of modern computing. His Difference Engine was designed to compute polynomial functions, and his more ambitious Analytical Engine, though never completed in his lifetime, included concepts like conditional branching and loops that are fundamental to modern computers. Ada Lovelace, who worked with Babbage, is often credited as the first computer programmer for her notes on the Analytical Engine.

The 20th century saw rapid advances. Alan Turing's theoretical work in the 1930s laid the foundation for computer science. His concept of a universal machine that could simulate any other machine became the basis for modern general-purpose computers. During World War II, Turing helped break German Enigma codes at Bletchley Park using early electromechanical computing devices.

The first fully electronic general-purpose computer was ENIAC, completed in 1945 at the University of Pennsylvania. It weighed 30 tons and used about 18,000 vacuum tubes. The invention of the transistor at Bell Labs in 1947 by John Bardeen, Walter Brattain, and William Shockley revolutionized computing by replacing bulky vacuum tubes with smaller, more reliable components.

The 1960s and 1970s saw the development of integrated circuits and the first microprocessors. Intel's 4004, released in 1971, was the first commercially available microprocessor. This led to the personal computer revolution of the 1970s and 1980s, with companies like Apple, IBM, and Microsoft rising to prominence.

The internet, which has its origins in ARPANET (a US Department of Defense project from the late 1960s), transformed computing in the 1990s by connecting computers worldwide. Tim Berners-Lee's invention of the World Wide Web in 1989 at CERN made the internet accessible to ordinary users.

In recent decades, we've seen the rise of mobile computing, cloud services, artificial intelligence, and quantum computing research. Smartphones have become powerful pocket computers, and AI systems based on neural networks can now perform tasks that once seemed impossibly complex.

Question: According to the passage, who invented the transistor and in what year?"""


def get_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def compute_ppl(model, tokenizer, text: str, device: str = "cuda") -> float:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    with torch.inference_mode():
        outputs = model(input_ids, labels=input_ids)
    return torch.exp(outputs.loss).item()


def run_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    use_turbo: bool = False,
    tq_cfg: TQCacheConfig | None = None,
    device: str = "cuda",
) -> tuple[dict, torch.Tensor]:
    reset_vram()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    past_key_values = None
    if use_turbo:
        assert tq_cfg is not None
        n_layers = model.config.num_hidden_layers
        head_dim = getattr(model.config, "head_dim", None) or (
            model.config.hidden_size // model.config.num_attention_heads
        )
        past_key_values = TurboQuantCache(
            tq_cfg, n_layers=n_layers, head_dim=head_dim, device=device
        )

    with torch.inference_mode():
        t0 = time.time()
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        if device == "cuda":
            torch.cuda.synchronize()
        prefill_time = time.time() - t0

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]
        all_logits = [outputs.logits[:, -1, :].clone()]

        t0 = time.time()
        for _ in range(max_new_tokens - 1):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            all_logits.append(outputs.logits[:, -1, :].clone())
        if device == "cuda":
            torch.cuda.synchronize()
        decode_time = time.time() - t0

    gen_tokens = torch.cat(generated, dim=1)
    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    logits_stack = torch.stack(all_logits, dim=1)

    metrics = {
        "prompt_len": prompt_len,
        "gen_len": max_new_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "decode_tok_per_s": max_new_tokens / max(decode_time, 1e-6),
        "vram_peak_mb": get_vram_mb(),
        "generated_text": gen_text,
        "generated_ids": gen_tokens[0].cpu().tolist(),
    }
    if use_turbo and past_key_values is not None:
        metrics["turbo_metrics"] = past_key_values.get_metrics()

    return metrics, logits_stack


def compare_outputs(baseline: dict, turbo: dict, logits_baseline: torch.Tensor,
                    logits_turbo: torch.Tensor) -> dict:
    ids_b = torch.tensor(baseline["generated_ids"])
    ids_t = torch.tensor(turbo["generated_ids"])
    min_len = min(len(ids_b), len(ids_t))
    token_match = (ids_b[:min_len] == ids_t[:min_len]).float().mean().item()

    diff_idx = -1
    for i in range(min_len):
        if ids_b[i] != ids_t[i]:
            diff_idx = i
            break

    logits_b = logits_baseline[0]
    logits_t = logits_turbo[0]
    min_t = min(logits_b.shape[0], logits_t.shape[0])
    cos = F.cosine_similarity(
        logits_b[:min_t].float(), logits_t[:min_t].float(), dim=-1
    ).mean().item()
    top1_b = logits_b[:min_t].argmax(dim=-1)
    top1_t = logits_t[:min_t].argmax(dim=-1)
    top1_agree = (top1_b == top1_t).float().mean().item()
    top5_b = logits_b[:min_t].topk(5, dim=-1).indices
    top5_t = logits_t[:min_t].topk(5, dim=-1).indices
    top5_overlap = []
    for i in range(min_t):
        overlap = len(set(top5_b[i].tolist()) & set(top5_t[i].tolist())) / 5
        top5_overlap.append(overlap)
    top5_mean = sum(top5_overlap) / len(top5_overlap)

    return {
        "token_match_rate": token_match,
        "first_divergence_at": diff_idx,
        "logits_cosine_sim": cos,
        "top1_agreement": top1_agree,
        "top5_overlap_rate": top5_mean,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--k_bits", type=int, default=4)
    ap.add_argument("--v_bits", type=int, default=4)
    ap.add_argument("--residual_window", type=int, default=64)
    ap.add_argument("--protect_init", type=int, default=2)
    ap.add_argument("--protect_final", type=int, default=2)
    ap.add_argument("--output", default="bench_result.json")
    ap.add_argument("--v_only", action="store_true", help="Quantize only V (K stays FP16)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("TurboQuant E2E Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device
    )
    model.requires_grad_(False)
    for m in model.modules():
        m.training = False
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded: {n_params:.1f}M params")
    print(f"Layers: {model.config.num_hidden_layers}, "
          f"hidden: {model.config.hidden_size}, "
          f"heads: {model.config.num_attention_heads}")

    prompt = LONG_PROMPT
    n_prompt_tokens = len(tokenizer(prompt).input_ids)
    print(f"\nPrompt: {n_prompt_tokens} tokens")

    print("\n" + "=" * 60)
    print("BASELINE - FP16 KV cache")
    print("=" * 60)
    baseline_metrics, baseline_logits = run_generation(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        use_turbo=False,
        device=device,
    )
    print(f"Prefill:  {baseline_metrics['prefill_time_s']:.3f}s")
    print(f"Decode:   {baseline_metrics['decode_time_s']:.3f}s "
          f"({baseline_metrics['decode_tok_per_s']:.1f} tok/s)")
    print(f"VRAM peak: {baseline_metrics['vram_peak_mb']:.1f} MB")
    print(f"Generated: {baseline_metrics['generated_text'][:200]}...")

    ppl_baseline = compute_ppl(model, tokenizer, prompt, device=device)
    print(f"PPL (prompt): {ppl_baseline:.3f}")

    print("\n" + "=" * 60)
    print(f"TURBO - K{args.k_bits}/V{args.v_bits} "
          f"window={args.residual_window} protect=[{args.protect_init},{args.protect_final}]")
    print("=" * 60)
    tq_cfg = TQCacheConfig(
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        residual_window=args.residual_window,
        protect_layers_init=args.protect_init,
        protect_layers_final=args.protect_final,
        v_only=args.v_only,
    )
    turbo_metrics, turbo_logits = run_generation(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        use_turbo=True,
        tq_cfg=tq_cfg,
        device=device,
    )
    print(f"Prefill:  {turbo_metrics['prefill_time_s']:.3f}s")
    print(f"Decode:   {turbo_metrics['decode_time_s']:.3f}s "
          f"({turbo_metrics['decode_tok_per_s']:.1f} tok/s)")
    print(f"VRAM peak: {turbo_metrics['vram_peak_mb']:.1f} MB")
    print(f"Generated: {turbo_metrics['generated_text'][:200]}...")
    if "turbo_metrics" in turbo_metrics:
        tm = turbo_metrics["turbo_metrics"]
        print("\nCompression metrics:")
        print(f"  Total quantize calls: {tm['total_quantize_calls']}")
        print(f"  FP16 equivalent:   {tm['bytes_fp16_equiv'] / 1024:.1f} KB")
        print(f"  Compressed:        {tm['bytes_compressed'] / 1024:.1f} KB")
        print(f"  Saved:             {tm['bytes_saved'] / 1024:.1f} KB")
        print(f"  Ratio:             {tm['compression_ratio']:.2f}x")

    print("\n" + "=" * 60)
    print("A/B COMPARISON")
    print("=" * 60)
    comparison = compare_outputs(baseline_metrics, turbo_metrics, baseline_logits, turbo_logits)
    print(f"Token match rate:     {comparison['token_match_rate']:.2%}")
    print(f"First divergence:     token #{comparison['first_divergence_at']}")
    print(f"Logits cosine sim:    {comparison['logits_cosine_sim']:.4f}")
    print(f"Top-1 agreement:      {comparison['top1_agreement']:.2%}")
    print(f"Top-5 overlap rate:   {comparison['top5_overlap_rate']:.2%}")

    speed_delta = (turbo_metrics['decode_tok_per_s']
                   / max(baseline_metrics['decode_tok_per_s'], 1e-6) - 1) * 100
    vram_delta = (turbo_metrics['vram_peak_mb']
                  / max(baseline_metrics['vram_peak_mb'], 1e-6) - 1) * 100
    print(f"\nSpeed delta:  {speed_delta:+.1f}% (turbo vs baseline)")
    print(f"VRAM delta:   {vram_delta:+.1f}% (turbo vs baseline)")

    result = {
        "config": {
            "model": args.model,
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "residual_window": args.residual_window,
            "protect_init": args.protect_init,
            "protect_final": args.protect_final,
            "max_new_tokens": args.max_new_tokens,
            "prompt_tokens": n_prompt_tokens,
        },
        "baseline": {k: v for k, v in baseline_metrics.items() if k != "generated_ids"},
        "turbo": {k: v for k, v in turbo_metrics.items() if k != "generated_ids"},
        "comparison": comparison,
        "baseline_ppl": ppl_baseline,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
