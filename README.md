# TurboQuant Reproduction Study

Independent reproduction and analysis of Google Research's TurboQuant (ICLR 2026) — a KV cache compression algorithm for LLM inference claiming 6x memory reduction with zero quality loss.

**TL;DR**: We reproduced the paper's core claim (+0.32% PPL at 4.57x compression), but only after discovering three critical implementation details the paper doesn't emphasize. Without them, quality degrades by 2756%. This repo documents the entire journey.

---

## Key Findings

| What we tested | Result |
|---|---|
| Paper's algorithm (Lloyd-Max + rotation) validated against theoretical bounds? | **Yes** — matches within 1-14% across all bit-widths |
| "3-bit zero loss" out of the box? | **No** — uniform 3-bit destroys quality (PPL +1653% on Qwen2.5, +54% on Llama-3.1-8B) |
| Does it work on some models? | **Yes** — Gemma-2-2B: turbo3 = +2.8% PPL (near-zero loss confirmed) |
| Can we fix it to work universally? | **Yes** — with 3 patches: PPL +0.32% at 4.57x compression on Qwen2.5 |

### The Three Patches That Close the Gap

1. **Never re-quantize already-quantized tokens** — the single most important fix. Without it, compound error grows exponentially and outputs diverge to NaN within ~100 decode steps.

2. **Mixed-precision outlier channel splitting** — the paper's "3.5-bit" means 32 outlier channels at 4-bit + remaining at 3-bit. Not uniform 3-bit everywhere.

3. **Deferred quantization** — keep KV in FP16 during prefill, only quantize when decode begins. Avoids prefill-phase error accumulation.

### Before and After Patches (Qwen2.5-0.5B, V-only, Shakespeare PPL)

| Version | PPL | Delta | Compression |
|---------|-----|-------|-------------|
| Baseline FP16 | 40.54 | — | 1.00x |
| V1 3-bit uniform (naive) | 1157.83 | +2756% | 4.92x |
| **V3 3.5-bit + patches** | **40.67** | **+0.32%** | **4.57x** |

---

## Cross-Model Benchmark (C++/CUDA, TurboQuant fork)

Built [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) from source with CUDA 12.4 in Docker. Tested turbo3 PPL delta across 4 models:

| Model | turbo3 PPL delta | Verdict |
|-------|-----------------|---------|
| Gemma-2-2B | +2.8% | Confirms the paper |
| Llama-3.2-3B | +12.7% | Tolerable |
| Llama-3.1-8B (paper's model) | +54.2% | Broken |
| Qwen2.5-3B | +1653% | Catastrophic |

**The algorithm's effectiveness is strongly architecture-dependent.** Gemma's sliding window attention and uniform KV distribution make it the ideal case. Qwen2.5's extreme K/V norm asymmetry (~1400x ratio) makes it the worst case.

---

## Repository Structure

```
turboquant/
├── paper/
│   └── RESEARCH.md              # Full research report (start here)
├── src/
│   ├── tq_core.py               # V1: Lloyd-Max + rotation + quantize
│   ├── tq_core_v2.py            # V2: + outlier channel splitting
│   ├── tq_cache.py              # V1: HuggingFace DynamicCache wrapper
│   ├── tq_cache_v3.py           # V3: + no-requ + deferred (final)
│   └── test_synthetic.py        # Validation against paper bounds
├── benchmarks/
│   ├── bench_ppl_v3.py          # PPL benchmark with patches
│   ├── bench_e2e.py             # E2E generation benchmark
│   ├── bench_patches.py         # A/B/C patch comparison
│   ├── docker_build.sh          # Build TurboQuant C++/CUDA in Docker
│   └── turbo_full_sweep.sh      # Cross-model sweep script
├── results/
│   ├── python/                  # Python prototype results (JSON)
│   ├── cpp_cuda/                # llama.cpp mainline results (logs)
│   └── cross_model/             # 4-model TurboQuant comparison (logs)
│       ├── gemma-2-2b/
│       ├── llama-3.2-3b/
│       └── llama-3.1-8b/
├── knowledge-base/
│   ├── papers/                  # Paper summaries (5 papers)
│   ├── articles/                # 17 technical articles collected
│   ├── blog-posts/              # 3 blog posts analyzed
│   ├── code-examples/           # 6 implementation guides
│   └── discussions/             # Community discussions
└── docs/
    └── SETUP.md                 # Reproduction instructions
```

## Quick Start

### Validate the Algorithm

```bash
cd src/
pip install torch numpy
python test_synthetic.py
```

Expected: all 12 tests pass (4 bit-widths x 3 dimensions), MSE within paper bounds.

### Run PPL Benchmark (with patches)

```bash
pip install transformers accelerate
cd benchmarks/
python bench_ppl_v3.py
```

Downloads Qwen2.5-0.5B-Instruct (~1 GB) automatically. Runs ~1 min on CUDA GPU.

### Build and Test C++/CUDA TurboQuant

Requires: Docker Desktop with NVIDIA Container Runtime.

```bash
# Build
docker run --gpus all -v $(pwd):/work nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  bash /work/benchmarks/docker_build.sh

# Test (download GGUF model first)
docker run --gpus all -v $(pwd):/work nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  bash /work/benchmarks/turbo_full_sweep.sh "gemma-2-2b" "/work/models/model.gguf" 99
```

---

## Knowledge Base

We collected 33 documents about TurboQuant from public sources before writing any code:

- **5 papers**: TurboQuant (ICLR 2026), PolarQuant (AISTATS 2026), QJL (AAAI 2025), PolarQuant v2, RaBitQ (SIGMOD 2024)
- **17 articles**: deep-dives, implementation guides, controversy analysis, market impact
- **6 code-example guides**: PyTorch, Triton/vLLM, MLX/Apple, Rust, ecosystem overview
- **GitHub ecosystem**: 20+ repos cataloged, 4 cloned for reference

See [`knowledge-base/`](knowledge-base/) for the full collection.

---

## References

- **TurboQuant**: Zandieh et al., "Online Vector Quantization with Near-optimal Distortion Rate," ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant**: [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL**: Zandieh et al., AAAI 2025. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- **RaBitQ**: Gao & Long, SIGMOD 2024. [arXiv:2405.12497](https://arxiv.org/abs/2405.12497)
- **RotorQuant**: [github.com/scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)
- **llama.cpp TurboQuant fork**: [github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)

## Author

**Alexsandro Furtado** — independent researcher

## License

MIT
