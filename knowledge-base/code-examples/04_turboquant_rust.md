# TurboQuant Rust — AbdelStark

> **Fonte**: https://github.com/AbdelStark/turboquant
> **Licença**: MIT
> **Status**: Alpha (2026-03-25)

## Overview

Rust library for research-grade vector quantization of LLM KV caches. Three evaluation paths: synthetic, trace-based per-head safetensors, and real-model ONNX decoder inference.

## Installation

```toml
[dependencies]
turboquant = "0.1.1"
```

Requirements: Rust 1.87.0+. Optional GPU feature with Burn/WGPU.

## Core APIs

| Type | Purpose |
|------|---------|
| `TurboQuantMSE` | Reconstruction-focused vector quantization |
| `TurboQuantProd` | Inner-product-optimized quantization |
| `BatchQuantizedMSE/Prod` | Compressed batch storage |
| `QuantizedKVCache` | KV cache quantization utilities |
| `MultiHeadKVCache` | Multi-head cache management |
| `KvTrace` | Trace loading for per-head workloads |
| `RealModelRunner` | End-to-end ONNX decoder execution |

## Supported Models

Verified end-to-end:
- `distilgpt2`
- `HuggingFaceTB/SmolLM2-135M-Instruct`

## Usage Examples

### ONNX Export
```bash
python3 scripts/export_hf_decoder_onnx.py \
  --preset distilgpt2 \
  --output-dir artifacts/distilgpt2-onnx
```

### Synthetic Benchmark
```bash
cargo run --release --example benchmark -- --workload synthetic --quick
```

### Real-Model Comparison
```bash
cargo run --release --example benchmark -- \
  --workload real-model \
  --real-model-dir artifacts/distilgpt2-onnx \
  --prompt "Summarize the role of a KV cache in one sentence." \
  --real-eval-mode compare \
  --bits 4 \
  --value-bits 4 \
  --real-key-strategy prod \
  --top-k 5 \
  --max-new-tokens 16
```

### Full Evaluation Pipeline
```bash
python3 scripts/run_real_model_eval.py \
  --preset distilgpt2 \
  --bits 2 4 8 \
  --strategies prod mse
```

## Real-Model Metrics

- Next-token logit RMSE
- Top-k prediction agreement percentages
- Token match rates and divergence occurrence rates
- Reference-token cross-entropy and perplexity
- Operation latency and tokens-per-second
- Exact vs quantized KV memory comparison

## Important Note

The quantized pipeline quantizes cache during decode, then reconstructs float tensors before the next ONNX Runtime step. This is genuine end-to-end execution with quantized cache reuse, not custom quantized attention kernels within ONNX Runtime.
