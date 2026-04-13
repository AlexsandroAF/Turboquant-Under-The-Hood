# turboquant-pytorch — From-Scratch PyTorch Implementation

> **Fonte**: https://github.com/tonbistudio/turboquant-pytorch
> **Licença**: MIT

## Overview

From-scratch PyTorch implementation of TurboQuant (ICLR 2026) for LLM KV cache compression. Tested on Windows with NVIDIA GPUs.

## Key Finding

The paper's primary innovation (QJL residual correction) actually degrades performance for KV cache compression. V3 removes this component and introduces asymmetric bit allocation, achieving better results.

## Performance Results

### Generation Test (V3)

| Config | Quality | Compression |
|--------|---------|-------------|
| K6/V4 with 128-token residual window | Perfect output | ~2x |
| K8/V4 with 128-token residual window | Perfect output | ~1.6x |
| K4/V4 with 128-token residual window | Partial success | ~3x |
| K4/V2 without residual window | Failed generation | ~5x |

### Attention Score Accuracy (8K context)

| Method | Compression | Cosine Sim | Top-1 Match |
|--------|-------------|------------|-------------|
| V3 K4/V2 | 5.1x | 0.9996 | 94% |
| V3 K4/V2 + protected layers | 3.6x | 0.9997 | 99% |
| V2 with QJL (3-bit) | 5.0x | 0.9945 | 86% |

## Technical Architecture

### Core Algorithm
Random orthogonal matrix rotation + Lloyd-Max optimal scalar quantization. Vectors are normalized, rotated, quantized coordinate-wise, and stored with their original norm.

### Why QJL Fails
While mathematically unbiased for raw inner products, QJL introduces random variance that exponentially amplifies when attention scores pass through softmax. Six independent implementations confirmed MSE-only outperforms MSE+QJL.

### V3 Improvements
1. **MSE-only compression** — eliminates QJL entirely
2. **Asymmetric K/V bits** — more precision to keys than values
3. **Bit-packed storage** — real compression ratios (V2 was 38% larger than uncompressed!)
4. **Layer-adaptive precision** — protects sensitive first/last layers

## Installation

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Requirements: Python 3.10+, CUDA-capable NVIDIA GPU, Windows 11 or Linux

## Usage

```bash
# Generation test
python -m turboquant.generation_test

# V3 vs V2 comparison
python -m turboquant.validate_v3

# Synthetic tests
python -m turboquant.test_turboquant
```

## Project Structure

```
compressors_v3.py     # V3 implementation with bit-packed storage
turboquant.py         # Core quantizers: TurboQuantMSE, TurboQuantProd
lloyd_max.py          # Lloyd-Max optimal scalar quantizer solver
generation_test.py    # Text generation validation
validate_v3.py        # V3 vs V2 comparison
test_turboquant.py    # Synthetic algorithm tests
```

## Key Classes

- **MSECompressor**: Single-stage compressor with bit-packed storage (V3 foundation)
- **TurboQuantV3**: Orchestrator for asymmetric key/value compression with layer adaptation
- **TurboQuantMSE**: Original Stage 1 quantizer
- **TurboQuantProd**: Original two-stage with QJL (reference)

## MSE Distortion (d=128, 1000 random unit vectors)

| Bits | Measured MSE | Paper Bound | Efficiency |
|------|--------------|-------------|-----------|
| 1-bit | 0.362 | 0.680 | 0.53x |
| 2-bit | 0.116 | 0.170 | 0.68x |
| 3-bit | 0.034 | 0.043 | 0.81x |
| 4-bit | 0.009 | 0.011 | 0.87x |

## Critical Observation

**High attention score similarity (99.5%+) does NOT guarantee correct text generation.** Compression without a residual window (keeping recent tokens in FP16) produces garbage output despite excellent attention metrics.
