# turboquant (Triton + vLLM) — 0xSero

> **Fonte**: https://github.com/0xSero/turboquant
> **Licença**: MIT (implícita)

## Overview

KV cache compression for LLM inference with Triton kernels and vLLM integration. 3-bit keys, 2-bit values.

## Key Performance (RTX 5090, Qwen3.5-27B-AWQ)

- Prefill throughput: +5.7% to 1,907 tokens/second
- Decode throughput: +3.1% to 1.303 tokens/second
- Maximum token capacity: 2x (457k → 914k tokens)
- Memory freed: 30GB across GPUs

### 8x RTX 3090 MoE (Qwen3.5-35B-A3B)

- Consistent needle retrieval (4/5 needles) up to 131k context
- KV cache savings: 30.9% per GPU at 131k context
- Enables 1.45x context extension or 3 additional concurrent requests

## Technical Architecture

### Compression Engine
- Random orthogonal rotation spreads information across dimensions
- Lloyd-Max optimal scalar quantization: b-1 bits to rotated values
- QJL projection captures residual sign bits (1 bit per dimension)
- Group quantization handles values with per-group scaling

### Storage & Integration
- Modular KV capture hooks for attention layers
- Compressed storage with bit-packing (4 values/byte for 2-bit)
- vLLM adapter for seamless engine integration
- Triton kernels for efficient GPU computation

### Quality
- 3-bit key compression: near-lossless cosine similarity (1.0)
- 2-bit values: 0.94 cosine similarity
- 4-bit values: 0.997 similarity
- Unbiased estimator: E[estimated inner product] = true inner product

## Installation & Usage

```bash
pip install -e .

# Validate theoretical claims (CPU only)
python validate_paper.py

# Security audit of all claims
python audit_claims.py

# Modular architecture tests
python -m pytest test_modular.py -v

# Benchmark (requires 4x RTX 3090 + model)
CUDA_VISIBLE_DEVICES=0,1,4,6 python proof.py
```

## Directory Structure

```
turboquant/
├── codebook.py          # Lloyd-Max quantizer for Beta distributions
├── quantizer.py         # Primary compression algorithms
├── kv_cache.py          # Memory management with bit-packing
├── integration/vllm.py  # vLLM engine adapter
└── triton_kernels.py    # GPU-accelerated kernels
```

## Known Limitations

- Prefill phase still allocates full paged cache; TQ frees only after prefill
- Compression only on full-attention layers; linear-attention/Mamba uncompressed
- 2-bit value quantization: quality bottleneck (0.94 similarity); 4-bit recommended for production
- Hybrid decode path dequantizes complete history to float32 per step
- Mixed-architecture models (MoE + linear-attention) see reduced benefit

## Requirements

- vLLM 0.18.0, PyTorch 2.10, CUDA 12.8
- RTX 5090 (32GB) or 8x RTX 3090 (24GB each)
- Python 3.12
