# PolarQuant v2: Leveraging Polar Transformation for Key Cache Quantization and Decoding Acceleration

> **Fonte**: https://arxiv.org/abs/2502.00527
> **Autores**: Songhao Wu et al.
> **Data**: February 1, 2025
> **NOTA**: Este é um paper DIFERENTE do PolarQuant (2502.02617) usado no TurboQuant. Mesma ideia base mas abordagem distinta.

---

## Core Innovation

Outliers em key vectors tipicamente aparecem em apenas uma de duas dimensões que são rotacionadas juntas pelos RoPE (Rotary Position Embeddings). Em coordenadas polares, esses outliers formam "padrões circulares bem-estruturados" — ideais para quantização eficiente.

## Abordagem Técnica

### Quantização em Coordenadas Polares

Para cada sub-vetor 2D de keys pós-RoPE:

1. **Converte para coordenadas polares:**
```
r_t[j] = √(K̃_t[2j]² + K̃_t[2j+1]²)
θ_t[j] = atan2(K̃_t[2j+1], K̃_t[2j]) + π
```

2. **Quantização assimétrica:**
   - **Raio**: n-bit quantization (não-negativo, sem zero-point)
   - **Ângulo**: m-bit quantization sobre [0, 2π]

3. **Encoding**: Divide o plano 2D em 2^(n+m) regiões usando 2^n raios × 2^m ângulos

### Decodificação via Table Lookup (INOVAÇÃO CHAVE)

Em vez de dequantização cara na inferência:

- Pré-computa lookup table com d/2 × 2^m entradas
- Mapeia índices de ângulo quantizados para pares (cos, sin) × raios quantizados
- **Substitui multiplicação de matrizes por table lookups** para inner product query-key
- Alcança **1.27× speedup** em multiplicação query-key

Fórmula do lookup:
```
[K̃_t[2j], K̃_t[2j+1]] = [cos(π·Q(θ_t[j])/2^(m-1)) · (Q(r_t[j]) · s_j),
                           sin(π·Q(θ_t[j])/2^(m-1)) · (Q(r_t[j]) · s_j)]
```

## Vantagens

1. **Sem agrupamento de tokens** — Distribuições mais suaves eliminam overhead
2. **Menos parâmetros** — Só armazena scaling para raios; ângulos usam quantização implícita
3. **Compatível com pós-RoPE** — Evita overhead de dequantização de métodos pré-RoPE
4. **Menos zero-points** — Raio não-negativo elimina storage

## Resultados

### Performance (4-bit, actual 4.16 bits)
- Comparable a full-precision em LongBench (Llama-2-7B, Mistral-7B, Llama-3.1-8B)
- Competitivo em MMLU e GSM8K

### Eficiência (NVIDIA A800)
| Contexto | PolarQuant | FP16 | Speedup |
|----------|-----------|------|---------|
| 4K tokens | 50.04 μs | 60.98 μs | 1.22x |
| 128K tokens | 526.65 μs | 668.91 μs | 1.27x |

## Relevância para Nosso Compactador

A ideia de table lookup para substituir multiplicação matricial é extremamente valiosa. Combinada com a abordagem do TurboQuant principal, pode dar speedup significativo.
