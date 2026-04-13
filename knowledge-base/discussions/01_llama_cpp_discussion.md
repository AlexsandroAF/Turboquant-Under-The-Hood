# TurboQuant - Extreme KV Cache Quantization (llama.cpp Discussion #20969)

> **Fonte**: https://github.com/ggml-org/llama.cpp/discussions/20969
> **Plataforma**: GitHub / ggml-org/llama.cpp

## Overview

Community discussion on implementing TurboQuant in llama.cpp. Multiple independent implementations across different hardware platforms.

## Key Technical Findings

### Core Algorithm
- **Random rotation/WHT transform** to normalize input distributions
- **Lloyd-Max scalar quantization** for 3-4 bit encoding
- **Optional QJL correction** for unbiased inner product estimation

### Norm Disparity Issue
Key vectors have dramatically different magnitudes than Value vectors (ranging from 4x to 182x ratio depending on model). Asymmetric bit allocation works better than uniform quantization.

### Algorithm 2 Controversy
Multiple developers independently concluded that the paper's QJL residual correction (Algorithm 2) performs worse than MSE-only quantization in practice. WHT+QJL showed promise, but random rotation+QJL degraded performance.

## Implementations Across Hardware

### Metal (Apple Silicon)
TheTom's fork: 4.9x compression with manageable speed degradation after fixing dequantization bottlenecks through byte-read batching.

### CUDA
Madreag: ported Metal kernels to RTX 5090, 4.6x KV compression on 27B models with full Flash Attention support.

### Vulkan
Mixed K/V type support, recent AMD GPU fixes enabling functional implementations.

### CPU
Aaryan-Kapoor: negligible speed penalties (20.1 vs 19.3 tok/s prompt throughput), 4.4x compression.

## Compression Results

| Type | Compression | Quality |
|------|-------------|---------|
| TQ3 | 4.9x | ~1-2% PPL increase on larger models |
| TQ4 | 3.8x | Near-baseline quality |
| Memory | 536K token context achievable on consumer hardware (vs 109K FP16) |

## Practical Challenges

1. **Metal JIT compilation**: Custom headers require inlining to prevent silent CPU fallback
2. **CUDA 13.1 incompatibility**: MMQ kernels segfault on certain compiler versions
3. **Context-length scaling**: Initial decode degradation at long contexts, resolved through optimized dequantization
4. **Block size tuning**: Block=32 proved superior to block=128 for Flash Attention parallelism

## Quantitative Comparison (Qwen3.5-35B, 110K context)

| Config | Memory | Speed |
|--------|--------|-------|
| FP16 KV | 768 MiB | 38.0 tok/s |
| Q4_0 | 216 MiB (-72%) | 24.0 tok/s (-37%) |
| TurboQuant | Direct computation on quantized values | No per-token dequantization |

## Community Consensus

- Deterministic WHT outperforms random rotation significantly at 2-4 bits
- Mixed precision K/V handling necessary for models with high norm ratios
- Block-size=128 optimization improves compression from 4.57x to 5.12x
- No official llama.cpp mainline merge yet; multiple active forks
