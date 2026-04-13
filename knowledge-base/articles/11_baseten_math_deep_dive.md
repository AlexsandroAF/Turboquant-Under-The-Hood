# I Spent 31 Hours on the Math Behind TurboQuant So You Don't Have To

> **Fonte**: https://www.baseten.co/blog/i-spent-31-hours-on-the-math-behind-turboquant-so-you-dont-have-to/
> **Plataforma**: Baseten
> **Data**: 2026

---

## Key Problem: KV Cache Bottleneck

"For Llama-3.1-8B with 32 layers, 8 KV heads, head dimension 128, and 128K context: 128,000 × 32 × 8 × 128 × 2 × 2 bytes = **16 GB de KV cache sozinho**."

Standard quantization requer scale factors e zero-points per-block em full precision → overhead que destrói compressão.

## Fundamento Matemático

### Propriedade 1: Preservação Gaussiana
Multiplicar qualquer vetor por matriz aleatória com entradas normais produz Gaussiana multivariada:
```
S·x ~ N(0, ‖x‖²·Im)
```

### Propriedade 2: Concentração de Norma
Para vetores com coordenadas de distribuição normal padrão, a norma segue distribuição gamma generalizada concentrando fortemente em torno de √d em altas dimensões.

## Algoritmo PolarQuant

Converte vetores d-dimensionais em coordenadas polares recursivamente:

1. **Nível 1**: Pareia coordenadas (x₁, x₂) → (r, θ) usando atan2
2. **Níveis Recursivos**: Raios resultantes → (r₁, r₂) → (R, Θ)
3. **Continua**: Até um único raio final

### Distribuição dos Ângulos

No nível ℓ, ângulos seguem:
```
fψ(ℓ)(ψ) ~ sin^(2^(ℓ-1)-1)
```

Concentração maior em níveis mais altos → menos bits necessários.

## Detalhes de Implementação

**Rotação Aleatória**: Inicializar com Gram-Schmidt, aplicar para pré-condicionar.

**Construção de Codebook**: Para cada nível, buckets de quantização usando Lloyd's algorithm baseado em distribuições CONHECIDAS dos ângulos — não em dados de calibração.

**Mapeamento**: Nível 1 usa 4 bits (16 buckets); níveis subsequentes usam 2 bits (4 buckets) pois ângulos concentram fortemente em π/4.

## Resultado de Compressão (d=128)

```
64 ângulos × 4 bits (nível 1)  = 256 bits
32 ângulos × 2 bits (nível 2)  =  64 bits
16 ângulos × 2 bits (nível 3)  =  32 bits
 8 ângulos × 2 bits (nível 4)  =  16 bits
 8 raios restantes × FP16      = 128 bits
────────────────────────────────────────
Total: 496 bits vs 2,048 original = 4.13× compressão
```

## Vantagens sobre NVFP4

- Zero scale factors per-block (zero overhead)
- Buckets desenhados analiticamente (sem calibração)
- Usa distribuições conhecidas dos ângulos ao invés de espaçamento uniforme

Performance de kernel: ~75% da velocidade cuBLAS em sequências longas.
