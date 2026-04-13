# TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

> **Fonte**: https://arxiv.org/abs/2504.19874
> **Status**: ICLR 2026 (Accepted)
> **Autores**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research)
> **Data**: April 28, 2025

## Abstract

TurboQuant addresses vector quantization—a fundamental problem in source coding theory. The method handles both mean-squared error (MSE) and inner product distortion, overcoming limitations of existing methods that fail to achieve optimal distortion rates.

## Technical Approach

The algorithm employs random vector rotation to induce concentrated Beta distributions on coordinates, then applies coordinate-wise optimal scalar quantizers.

### Two-Stage Method

**Stage 1 - MSE Quantization**: Randomly rotates input vectors, inducing a concentrated Beta distribution on coordinates, and leverages the near-independence property of distinct coordinates in high dimensions to apply optimal scalar quantizers per coordinate.

**Stage 2 - 1-bit QJL Transform**: A two-stage approach combines MSE quantization with a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform to address bias issues in inner product estimation.

## Theoretical Results

- Achieves near-optimal distortion rates within approximately 2.7× of information-theoretic lower bounds
- For KV cache quantization: achieves quality neutrality at 3.5 bits per channel
- For nearest neighbor search: outperforms existing product quantization methods while reducing indexing overhead

## Key Formulas

### Coordinate Distribution After Rotation
```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) × (1 - x²)^((d-3)/2)
where x ∈ [-1, 1]
```

## Experimental Results

| Benchmark | Result |
|-----------|--------|
| LongBench (Llama-3.1-8B) | TurboQuant at 3.5 bits scores 50.06 — matching FP16 baseline |
| Needle-in-a-Haystack | 0.997 retrieval accuracy at 4× compression |
| Quality Parity Threshold | Full accuracy match at ~3.5 bits/channel |
| Attention Speedup | Up to 8× on NVIDIA H100 in 4-bit mode |
| Vector Search (GloVe d=200) | Outperforms PQ and RaBitQ on recall@k |

## Citation

```bibtex
@article{turboquant2025,
  title={TurboQuant: Online Vector Quantization with Near-Optimal Distortion},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

## Component Papers

- **PolarQuant**: arXiv 2502.02617 (AISTATS 2026) — First stage; rotates vectors into polar coordinates
- **QJL**: AAAI 2025 (DOI: 10.1609/aaai.v39i24.34773) — Second stage; 1-bit transform correcting inner-product bias

## Paper Structure

The work spans 25 pages and is categorized under Machine Learning, AI, Databases, and Data Structures.
