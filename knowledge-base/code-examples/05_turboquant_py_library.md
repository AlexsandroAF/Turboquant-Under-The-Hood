# turboquant-py — Python Library (PyPI)

> **Fonte**: https://pypi.org/project/turboquant-py/
> **GitHub**: https://github.com/msilverblatt/turboquant-py
> **Licença**: Apache-2.0
> **Versão**: 0.1.0 (March 27, 2026)

---

## ESTA É A MELHOR REFERÊNCIA PARA REIMPLEMENTAÇÃO — API limpa e completa

## Instalação

```bash
pip install turboquant-py
pip install "turboquant-py[torch]"  # Com aceleração PyTorch/CUDA/MPS
```

## Quick Start

### TurboQuant MSE Mode

```python
import numpy as np
from turboquant import TurboQuant

vectors = np.random.randn(1000, 384)
tq = TurboQuant(dim=384, bit_width=2, mode="mse", seed=42)

compressed = tq.quantize(vectors)
reconstructed = tq.dequantize(compressed)

mse = float(np.mean((vectors - reconstructed) ** 2))
print(f"Reconstruction MSE: {mse:.6f}")

# Save e reload
compressed.save("my_index")
from turboquant import CompressedVectors
reloaded = CompressedVectors.load("my_index")
```

### TurboQuant Inner Product Mode

```python
import numpy as np
from turboquant import TurboQuant

db = np.random.randn(10000, 768)
query = np.random.randn(768)

tq = TurboQuant(dim=768, bit_width=3, seed=42)
compressed = tq.quantize(db)

scores = tq.inner_product(query, compressed)
top10 = np.argsort(scores)[::-1][:10]
```

### QJL 1-Bit Quantization

```python
import numpy as np
from turboquant import QJL

db = np.random.randn(10000, 1536)
query = np.random.randn(1536)

qjl = QJL(dim=1536, seed=42)
compressed = qjl.quantize(db)

scores = qjl.inner_product(query, compressed)
top10 = np.argsort(scores)[::-1][:10]
```

## API Completa

### Classe TurboQuant

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `dim` | int | required | Dimensionalidade do vetor |
| `bit_width` | int | required | Bits por coordenada (1-4) |
| `mode` | str | `"inner_product"` | `"mse"` ou `"inner_product"` |
| `seed` | int/None | None | Seed para matrizes de rotação/projeção |
| `outlier_channels` | int | 0 | Canais high-magnitude para precisão extra |
| `outlier_bit_width` | int/None | None | Bit-width para canais outlier |

**Métodos:**
- `quantize(vectors)` → `CompressedVectors`
- `dequantize(compressed)` → `(n, dim)` array
- `inner_product(query, compressed)` → `(n,)` scores
- `quantize_batched(vectors, batch_size, output_path, entropy_encode)` → stream to disk

### Classe QJL

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `dim` | int | required | Dimensionalidade |
| `projection_dim` | int/None | dim | Dimensão da projeção (≤ dim) |
| `seed` | int/None | None | Seed para matriz de projeção |

**Métodos:**
- `quantize(vectors)` → `CompressedVectors`
- `inner_product(query, compressed)` → `(n,)` scores

### Classe CompressedVectors

Container in-memory com: índices bit-packed, normas L2 por vetor, arrays auxiliares (QJL signs, residual norms).

- Slicing: `compressed[start:end]`
- Merge: `CompressedVectors.concatenate(parts)`
- Save/Load: `compressed.save("path")` / `CompressedVectors.load("path")`
- Entropy encoding: `compressed.save("path", entropy_encode=True)` (Huffman)

### Classe CompressedStore

Vector store on-disk com memory-mapped arrays + brute-force top-k search.

```python
store = CompressedStore.load("path/to/dir")
results = store.search(query, k=10)  # → list[tuple[int, float]]
```

### Funções Utilitárias

```python
# Gerar codebook Lloyd-Max para distribuição Beta
from turboquant import compute_codebook, get_codebook
centroids, boundaries = compute_codebook(dim=256, bit_width=4)

# Calcular economia teórica
from turboquant import compute_theoretical_savings
savings = compute_theoretical_savings(dim=256, bit_width=4)
# {'entropy': 3.742, 'avg_bits_huffman': 3.779, 'savings_pct': 5.5}
```

## Formato de Storage

| Arquivo | Conteúdo |
|---------|----------|
| `meta.json` | dim, bit-width, mode, seed, outlier config, encoding flags |
| `indices.npy` | Índices bit-packed (ou `indices.huffman`) |
| `norms.npy` | Normas L2 por vetor |
| `huffman_table.json` | Tabela Huffman (quando entropy-encoded) |
| `.npy` adicionais | QJL signs, residual norms, outlier indices |

**NOTA**: Matrizes de rotação e projeção são reconstruídas do seed → overhead mínimo.

## Entropy Encoding

| Bit-width | Shannon Entropy | Huffman Avg | Saving |
|-----------|-----------------|-------------|--------|
| 1 | 1.000 | 1.000 | 0.0% |
| 2 | 1.911 | 1.989 | 0.5% |
| 3 | 2.819 | 2.876 | 4.1% |
| 4 | 3.742 | 3.779 | 5.5% |

## Benchmarks (all-MiniLM-L6-v2, dim=384)

| Método | Bits | MSE | Recall@1 | Recall@10 |
|--------|------|-----|----------|-----------|
| NaiveUniform | 2 | 0.001079 | 0.675 | 0.721 |
| TurboQuant-mse | 2 | 0.000305 | 0.755 | 0.817 |
| TurboQuant-mse | 3 | 0.000090 | 0.895 | 0.878 |
| TurboQuant-mse | 4 | 0.000025 | 0.895 | 0.918 |

**TurboQuant-mse em 2 bits: 3.5x menor MSE que naive uniform.**

## Como Funciona Internamente

### MSE Mode
1. Rotação ortogonal aleatória → coordenadas seguem Beta distribution
2. Codebook Lloyd-Max pré-computado analiticamente para essa distribuição
3. Quantização escalar ótima por coordenada
4. Rotação inversa na dequantização

### Inner-Product Mode
1. Quantiza a (b-1) bits usando codebook MSE
2. Computa residual entre original e reconstrução MSE
3. Aplica QJL (projeção Gaussiana + extração de sinal) no residual
4. Armazena MSE indices + sign bits + norma residual
5. Na query: inner product = MSE dot product + QJL correction scaled por √(π/2)/d

### QJL Standalone
Projeção por matriz Gaussiana S, armazena sign(S·k) + norma. Estimativa:
```
⟨q,k⟩ ≈ √(π/2)/m · ||k|| · ⟨S·q, sign(S·k)⟩
```
Não-enviesado, 1 bit por coordenada projetada.
