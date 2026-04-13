# TurboQuant MLX: KV-Cache Compression on Apple Silicon

> **Fonte**: https://github.com/sharpner/turboquant-mlx
> **Licença**: MIT

## Overview

Reproduction of Google's TurboQuant on Apple Silicon using MLX. Achieves up to 5.5x KV-Cache compression. Two paths: V2 (speed, hardware acceleration) and V3 (quality, Lloyd-Max codebooks).

## Key Results (Apple M4 Max)

| Config | Quality Impact | Compression |
|--------|----------------|-------------|
| 4-bit | Quality-neutral (13.02 vs 12.94 baseline PPL) | 3.6x |
| 3-bit | 5.3% quality loss with rotation+QJL | ~4.5x |
| 3.5-bit mixed | Near-lossless (0.3% degradation) | ~4x |
| 2.5-bit mixed | Aggressive | 5.5x |

### Notable Finding

V2 3-bit rot+QJL beats fp16 on Gemma 3 (D=256): -1.1% perplexity improvement. Rotation + QJL correction acts as regularizer at larger head dimensions.

## Architecture

### V2 Path (Speed-Optimized)
- Affine quantization via `mx.quantize`
- Hardware-accelerated matrix multiplication
- Optional random rotation and norm-baking
- QJL residual correction as refinement

### V3 Path (Quality-Optimized)
- Lloyd-Max optimal codebooks (1-4 bit)
- Channel outlier splitting for fractional bit rates
- Software dequantization with centroid lookup
- Paper-correct implementation

## Technical Innovations

### Pre-allocation Strategy
Pre-allocated buffers with stride-based updates (step=256) instead of per-token concatenation. Reduces allocations from O(T) to O(T/256).

### Norm-Baking
Bakes L2 norms into quantized scales and biases, eliminating two element-wise operations from the hot path.

### Outlier Channel Splitting
After rotation, channels become statistically equivalent. Fixed splits enable fractional bit rates:
"3.5-bit: 64 channels @ 4-bit + 64 @ 3-bit = 3.5 bits/dim"

### QJL Residual Correction
Quantization residuals → random matrix projection → stored as 1-bit signs → used to correct attention score estimation via JL projection.

## QJL Analysis

The paper's TurboQuant_prod scheme (b-1 bit MSE + 1-bit QJL) consistently degrades quality. Root cause: centroid resolution loss through softmax amplification. With 4 centroids (2-bit) instead of 8 (3-bit), quantization creates coarser score representation that softmax amplifies exponentially.

**QJL works as an additional correction layer but fails as a bit replacement strategy.**

## Usage

```python
import mlx_lm
from turboquant.cache_v2 import TurboQuantKVCacheV2
import turboquant.patch as tq_patch

tq_patch.apply()
model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

cache = [TurboQuantKVCacheV2(head_dim=128, bits=4, 
         use_rotation=True, use_normalization=True)
         for _ in range(len(model.layers))]
```

## Recommendations

| Use Case | Config | Performance |
|----------|--------|-------------|
| Maximum speed | V2 4-bit LEAN | ~105% of fp16 speed at 8K tokens |
| Best 4-bit quality | V2 4-bit rotated | -0.8% PPL |
| Best 3-bit (D=256) | V2 3-bit rot+QJL | -1.1% PPL on Gemma |
| Near-lossless | V3 3.5-bit mixed | +0.3% PPL |
| Aggressive compression | V3 2.5-bit mixed | 5.5x compression |

## Project Structure

```
turboquant/
├── cache_v2.py / cache_v3.py    # quantized KV storage
├── attention_v2.py / attention_v3.py  # SDPA dispatch
├── codebook.py                  # Lloyd-Max optimal centroids
├── codebook_ops.py              # MLX pack/unpack for 2/3/4-bit
├── qjl.py                      # JL sign-bit encoding
├── rotation.py                  # QR rotation + JL matrices
└── patch.py                     # mlx-lm integration
```
