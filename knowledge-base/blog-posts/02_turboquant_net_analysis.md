# TurboQuant: Independent Analysis (TurboQuant.net)

> **Fonte**: https://turboquant.net/
> **Plataforma**: TurboQuant.net — Independent Analysis

## Core Technology

### Two-Stage Architecture

1. **PolarQuant** – Random rotation plus polar coordinate transformation eliminating per-block normalization overhead
2. **QJL** – 1-bit residual correction layer using unbiased inner-product estimation

### PolarQuant — How It Works

Groups d-dimensional vectors into pairs to obtain radii and angles, applies recursive polar transforms on the radii, then quantizes only the angles. Produces a concentrated distribution suitable for standard scalar quantizers.

### Coordinate Distribution Formula
```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) × (1 - x²)^((d-3)/2)
where x ∈ [-1, 1]
```

### Key Advantages Over Traditional Methods

| Feature | Traditional PQ | TurboQuant |
|---------|---|---|
| Requires training | Yes | No |
| Full-precision constants | Many | None |
| Indexing time | Long | ~Zero |
| Quality loss | Visible | Minimal |

## Benchmark Results

### KV Cache Compression
- **LongBench Score:** 50.06 at 3.5-bit (matches full cache baseline)
- **Needle In A Haystack:** Perfect 100% from 4K to 104K tokens
- **Memory Reduction:** 6x+
- **Attention Speed:** 8x on H100 at 4-bit precision

### Hardware Impact (RTX 4090 Configuration)

| Model | Before | After TurboQuant |
|-------|--------|------------------|
| Llama-3.1 70B at 100K context | 7 GPUs | 6 GPUs (saves 1 card) |
| Qwen-2.5 32B INT4 | 2 GPUs | 1 GPU |

## Implementation Pseudocode

### Step 1: Lloyd-Max Centroids
```python
centroids = lloyd_max_quantizer(
    distribution="beta",
    bits=b
)
```

### Step 2: Random Rotation
```python
G = np.random.randn(d, d)
Pi, _ = np.linalg.qr(G)
```

### Step 3: Quantization Primitives
```python
def quant(x, Pi, centroids):
    y = Pi @ x
    idx = find_nearest(y, centroids)
    return idx

def dequant(idx, Pi, centroids):
    y = centroids[idx]
    x = Pi.T @ y
    return x
```

### Step 4: Attention Integration
Store K/V in compressed form and estimate inner products using QJL during attention computation.

## Expert Analysis

> "KV cache compression is approaching its limit...the next major change is unlikely to come from compression alone."

Most easy gains via quantization (2-3x) and outlier handling (3-4x) are already deployed. TurboQuant pushes toward 4-4.5x but faces implementation challenges: "These methods are less GPU-friendly, harder to keep low-latency, and more difficult to stabilize."

## Known Risks

- Random seed bias (paper argues high-dimensional effects are negligible)
- Rotation matrix overhead (pre-generate and reuse)
- Residual norm storage (one FP16 scalar remains)

## Applications Beyond LLMs

- **Vector Search:** Improves recall with near-zero indexing overhead for FAISS-scale systems
- **Multimodal:** Same compression principles apply to image/video embeddings
- **Theory Extensions:** Combining with outlier methods could enable practical 2-bit systems

## Timeline

| Period | Milestone |
|--------|-----------|
| Q2 2026 | Open-source code and framework integrations |
| Q4 2026 | Commercial products (cloud-first) |
| 2027 | Potential normalization as LLM quantization standard |

## References

- **Main Paper:** arXiv 2504.19874 (ICLR 2026)
- **PolarQuant:** arXiv 2502.02617 (AISTATS 2026)
- **QJL:** ACM DL 10.1609/aaai.v39i24.34773 (AAAI 2025)
