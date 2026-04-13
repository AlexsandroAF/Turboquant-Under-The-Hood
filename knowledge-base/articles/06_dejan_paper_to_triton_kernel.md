# TurboQuant: From Paper to Triton Kernel in One Session

> **Fonte**: https://dejan.ai/blog/turboquant/
> **Plataforma**: DEJAN AI
> **Data**: 2026

## Overview

Documents implementing Google Research's TurboQuant for LLM KV caches, tested on Gemma 3 4B. Achieved significant memory savings through custom Triton kernel.

## The Algorithm

### Stage 1 — Random Rotation + Lloyd-Max Quantization
Applies orthogonal rotation to each KV vector, transforming coordinate distributions to a known Beta distribution. Enables precomputing optimal scalar quantizers without per-block normalization constants. No training required.

### Stage 2 — QJL Residual Correction
1-bit Quantized Johnson-Lindenstrauss transform applied to quantization residuals for inner-product optimization. Produces unbiased attention score estimates but requires custom kernels.

## Implementation: Three Layers

### Core Algorithm
Computes Lloyd-Max codebooks by running 300 iterations of optimization over the Beta distribution. Codebooks are cached for reuse.

### Python KV Cache Integration
Patches HuggingFace's `DynamicCache` to quantize tensors on update calls. Simulates accuracy impact without saving actual memory.

### Triton Fused Kernel
Computes attention directly from compressed uint8 key indices, avoiding fp16 materialization. Key optimization:

```
⟨q, R^T · centroids[idx]⟩ = ⟨R · q, centroids[idx]⟩
```

Pre-rotating queries once instead of decompressing keys.

## Key Results

### Algorithm Validation (d=256, synthetic data)

| Bits | Cosine Similarity | Compression | 
|------|-------------------|-------------|
| 2-bit | 0.940 | 15.5x |
| 3-bit | 0.983 | 10.4x |
| 4-bit | 0.995 | 7.9x |

### Triton Kernel Performance
- Q@K^T speedup: ~1.22x on RTX 4090
- Output numerically exact (cosine similarity: 1.000000)

### End-to-End on Gemma 3 4B IT
- 4-bit fused: 16.5 tok/s, 4 MB VRAM delta (vs 26 MB baseline)
- 2-bit fused: 17.7 tok/s (same as fp16), 7 MB VRAM delta
- Output quality: character-identical to baseline on tested prompts

## Critical Implementation Lessons

### Mistake 1: QJL Integration
Adding the QJL correction term back to reconstructed vectors degraded cosine similarity to 0.69. The correction only works within custom attention kernels using the two-part representation directly.

### Mistake 2: Model Loading
Gemma 3 4B requires `Gemma3ForConditionalGeneration` and `AutoProcessor`, not `AutoModelForCausalLM`.

### Mistake 3: Hadamard Transform Performance
Python-loop implementation of Fast Walsh-Hadamard created thousands of tiny CUDA kernel launches. Fix: precomputed orthogonal matrices via QR decomposition.

### Mistake 4: Cache Subclassing
Subclassing `DynamicCache` broke across transformers versions. Solution: patch the `update()` method on a stock instance.

### Mistake 5: Processor vs Tokenizer
`Gemma3Processor` lacks `encode()`. Token counting requires `processor.tokenizer.encode()`.

## Fused Kernel Architecture

Each instance:
1. Loads pre-rotated query slice
2. Loads uint8 key indices
3. Gathers centroid values via table lookup
4. Accumulates partial dot products

Reduces data movement from ~2 bytes (fp16) to ~1 byte (uint8) per element.

GQA handling: `kv_head = q_head // gqa_ratio`

## Future Optimization Paths

- Value cache compression (requires second Triton kernel)
- Structured rotation with fused Hadamard (reduces memory from O(d²) to O(d))
- Sub-byte packing (4 indices per byte)
- Flash Attention integration for further IO efficiency
