# TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate — PAPER COMPLETO

> **Fonte**: https://arxiv.org/html/2504.19874v1
> **Status**: ICLR 2026
> **Autores**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research)

---

## Abstract

Vector quantization addresses Shannon's source coding problem by compressing high-dimensional vectors while minimizing distortion. TurboQuant is a data-oblivious algorithm achieving near-optimal distortion rates for both MSE and inner product preservation across all bit-widths and dimensions.

Key contributions:
- MSE-optimal quantizer using random rotation and Beta distribution analysis
- Two-stage inner product quantizer combining MSE optimization with QJL transform
- Information-theoretic lower bounds proving near-optimality (within ~2.7× constant factor)
- Experimental validation on KV cache compression and nearest neighbor search

---

## 1. Introduction

### 1.1 Problem Definition

The quantization map Q: ℝ^d → {0,1}^B transforms d-dimensional vectors into B-bit binary strings. For bit-width b ≥ 0 where B = b·d:

**MSE Distortion:**
```
D_mse := E_Q[||x - Q^(-1)(Q(x))||_2^2]
```

**Inner Product Error:**
```
D_prod := E_Q[|⟨y,x⟩ - ⟨y,Q^(-1)(Q(x))⟩|^2]
```

**Unbiasedness requirement:**
```
E_Q[⟨y, Q^(-1)(Q(x))⟩] = ⟨y,x⟩
```

### 1.2 Related Work

- **Shannon's source coding theory**: Established distortion-rate function
- **Zador**: Advanced with high-resolution methods
- **Gersho**: Introduced lattice vector quantization
- **Online vs Offline**: Online = instant, no data tuning (KV cache). Offline = heavy preprocessing
- **Product Quantization (PQ)**: Standard for NN search, k-means codebook

### 1.3 Contributions

#### MSE Optimized TurboQuant
Distortion bounds:
```
D_mse ≤ (√3·π/2)·(1/4^b)

b=1: ~0.36
b=2: ~0.117
b=3: ~0.03
b=4: ~0.009
```

#### Inner Product TurboQuant
Two-stage: MSE quantizer (b-1 bits) + QJL on residuals:
```
Unbiased: E[⟨y, Q_prod^(-1)(Q_prod(x))⟩] = ⟨y,x⟩
D_prod ≤ (√3·π²·||y||_2²/d)·(1/4^b)
```

#### Lower Bounds
```
MSE:    D_mse(Q) ≥ 1/4^b
Inner:  D_prod(Q) ≥ (1/d)·(1/4^b)
```

---

## 2. Preliminaries

### 2.1 Coordinate Distribution on Hypersphere

**Lemma 1:** For x ∈ S^(d-1) uniformly distributed on unit hypersphere, coordinate x_j follows Beta distribution:

```
f_X(x) := Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)
```

In high dimensions: f_X(·) → N(0, 1/d)

### 2.2 Shannon Lower Bound (SLB)

**Lemma 2:** For source x ∈ ℝ^d with entropy h(x):
```
D(p_X, B) ≥ (d/2πe)·2^((2/d)(h(x)-B))
```

**Lemma 3:** For x ∈ S^(d-1) uniformly distributed:
```
D(B) ≥ 2^(-2B/d)
```

### 2.3 QJL: 1-bit Inner Product Quantization

**Definition 1:** QJL map Q_qjl: ℝ^d → {-1,+1}^d:
```
Q_qjl(x) := sign(S·x)  where S ∈ ℝ^(d×d) ~ N(0,1) i.i.d.
Q_qjl^(-1)(z) := (√(π/2)/d)·S^T·z  for z ∈ {-1,+1}^d
```

**Lemma 4:** For x ∈ S^(d-1) and y ∈ ℝ^d:
- Unbiased: E[⟨y, Q_qjl^(-1)(Q_qjl(x))⟩] = ⟨y,x⟩
- Variance: Var(⟨y, Q_qjl^(-1)(Q_qjl(x))⟩) ≤ (π/2d)·||y||_2²

---

## 3. TurboQuant: High Performance Quantization

### 3.1 Algorithm 1: TurboQuant_mse

```
Input: dimension d, bit-width b

SETUP:
1. Generate random rotation Π ∈ ℝ^(d×d) via QR decomposition of random Gaussian matrix
2. Construct codebook by solving continuous k-means:
   C(f_X, b) := min_(c_1≤...≤c_2^b) Σ_(i=1)^(2^b) ∫ |x-c_i|² f_X(x) dx

QUANTIZE (Quant_mse):
1. y ← Π·x                                    // Rotate
2. idx_j ← argmin_(k∈[2^b]) |y_j - c_k|      // Nearest centroid per coord
3. Output: idx

DEQUANTIZE (DeQuant_mse):
1. ỹ_j ← c_(idx_j)                            // Look up centroids
2. x̃ ← Π^T·ỹ                                  // Rotate back
3. Output: x̃
```

**Theorem 1:** For bit-width b ≥ 1, x ∈ S^(d-1):
```
D_mse := E[||x - x̃||_2²] ≤ (√3·π/2)·(1/4^b)
```

**Proof:** MSE decomposes as D_mse = d·C(f_X, b). Each coord follows Beta distribution. For b > 4, Panter-Dite high-resolution formula:
```
C(f_X, b) ≤ (1/12)·(∫ f_X(x)^(1/3) dx)³·(1/4^b) = (√3·π/2d)·(1/4^b)
```

#### Entropy Encoding Optimization
Codebook indices have probabilities p_ℓ := ∫ f_X(x) dx over Voronoi cell ℓ. Optimal prefix coding achieves ~5% reduction for b=4 (entropy ≈ 3.8).

### 3.2 Algorithm 2: TurboQuant_prod

```
Input: dimension d, bit-width b

SETUP:
1. Instantiate TurboQuant_mse with bit-width b-1
2. Generate random projection S ∈ ℝ^(d×d) ~ N(0,1) i.i.d.

QUANTIZE (Quant_prod):
1. idx ← Quant_mse(x)           // MSE quantize with b-1 bits
2. r ← x - DeQuant_mse(idx)     // Compute residual
3. qjl ← sign(S·r)              // 1-bit QJL quantization of residual
4. Output: (idx, qjl, ||r||_2)

DEQUANTIZE (DeQuant_prod):
1. x̃_mse ← DeQuant_mse(idx)
2. x̃_qjl ← (√(π/2)/d)·||r||_2·S^T·qjl
3. Output: x̃_mse + x̃_qjl
```

**Theorem 2:** For b ≥ 1, x ∈ S^(d-1), any y ∈ ℝ^d:

- **Unbiasedness:** E[⟨y, x̃⟩] = ⟨y,x⟩
- **Distortion:**
```
D_prod ≤ (√3·π²·||y||_2²/d)·(1/4^b)

b=1: ~1.57/d
b=2: ~0.56/d
b=3: ~0.18/d
b=4: ~0.047/d
```

**Proof:**
```
E[⟨y,x̃⟩|x̃_mse] = ⟨y,x̃_mse⟩ + E[⟨y,x̃_qjl⟩|x̃_mse]
                  = ⟨y,x̃_mse⟩ + ⟨y,r⟩         // QJL unbiasedness
                  = ⟨y,x⟩

Conditional distortion using QJL variance (Lemma 4):
E[|⟨y,x⟩ - ⟨y,x̃⟩|²|x̃_mse] ≤ (π/2d)·||r||_2²·||y||_2²

Unconditional:
D_prod ≤ (π/2d)·||y||_2²·D_mse(b-1)
```

### 3.3 Lower Bounds (Theorem 3)

For any randomized Q: S^(d-1) → {0,1}^(b·d):
```
MSE:    D_mse(Q) ≥ 1/4^b
Inner:  D_prod(Q) ≥ (1/d)·(1/4^b)
```

**Proof:** Yao's minimax principle → convert worst-case randomized to average-case deterministic. Apply Shannon lower bound (Lemma 3) to uniform distribution on hypersphere.

**Optimality Gap:**
- MSE: √3·π/2 ≈ 2.7× optimal
- Inner product: √3·π² ≈ 17.3× optimal (divided by d factor)
- For b=1: only ~1.45× away from optimal

---

## 4. Experiments

### 4.1 Needle-In-A-Haystack (Llama-3.1-8B-Instruct, 4× compression)

| Method | Score |
|--------|-------|
| Full-Precision | 0.997 |
| **TurboQuant** | **0.997** |
| PolarQuant | 0.995 |
| KIVI | 0.981 |
| PyramidKV | 0.895 |
| SnapKV | 0.858 |

### 4.2 LongBench-E (Llama-3.1-8B-Instruct)

| Method | KV Size (bits) | Avg Score |
|--------|----------------|-----------|
| Full Cache | 16 | 50.06 |
| KIVI (2-bit) | 3 | 48.50 |
| KIVI (3-bit) | 5 | 50.16 |
| PolarQuant (3.9-bit) | 3.9 | 49.78 |
| **TurboQuant (2.5-bit)** | **2.5** | **49.44** |
| **TurboQuant (3.5-bit)** | **3.5** | **50.06** |

Mixed-precision strategy: outlier/non-outlier split:
- 2.5-bit: 32 outlier channels @ 3-bits + 96 regular @ 2-bits
- 3.5-bit: Different ratio, higher precision

### 4.3 Nearest Neighbor Search

**Quantization Time (seconds):**

| Method | d=200 | d=1536 | d=3072 |
|--------|-------|--------|--------|
| PQ | 37.04 | 239.75 | 494.42 |
| RabitQ | 597.25 | 2267.59 | 3957.19 |
| **TurboQuant** | **0.0007** | **0.0013** | **0.0021** |

TurboQuant: ~340,000× faster than PQ, ~1,885,000× faster than RabitQ (d=3072)

---

## Key Technical Innovations Summary

1. **Random Rotation**: Induces Beta distribution on coordinates, enabling coordinate-independent quantization
2. **Two-Stage Inner Product**: MSE-optimal + 1-bit residual = unbiased inner product
3. **Optimal Scalar Quantizers**: Pre-computed Lloyd-Max centroids for Beta distributions
4. **Information-Theoretic Grounding**: Shannon lower bounds + Yao's minimax principle
