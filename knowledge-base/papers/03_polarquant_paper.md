# PolarQuant: Quantizing KV Caches with Polar Transformation

> **Fonte**: https://arxiv.org/abs/2502.02617 / https://arxiv.org/html/2502.02617v1
> **Status**: AISTATS 2026
> **Autores**: Pesquisadores de KAIST, Google Research, Yale
> **Data**: February 4, 2025

---

## Overview

PolarQuant é o Stage 1 do TurboQuant. Introduz quantização de KV cache usando transformação em coordenadas polares combinada com pré-condicionamento aleatório.

## Core Insight

Transforma embeddings KV em coordenadas polares em vez de quantizar representações cartesianas diretamente. Após pré-condicionamento aleatório, os ângulos polares exibem distribuição limitada e altamente concentrada com forma analiticamente computável.

## Eliminação de Normalização

A distribuição previsível elimina a necessidade de normalização explícita — um passo requerido por métodos tradicionais que introduz overhead significativo de memória porque parâmetros de quantização (zero point e scale) precisam ser armazenados em full precision por cada bloco de dados.

## Algoritmo

### Pré-condicionamento Aleatório
Matriz de rotação ortogonal compartilhada aplicada aos vetores de embedding antes da quantização. Preserva inner products enquanto randomiza distribuições de coordenadas.

### Transformação Polar Recursiva
Converte vetores d-dimensionais em forma polar através de algoritmo recursivo de log₂(d) níveis:

1. **Nível 1**: Agrupa pares de coordenadas (x₁, x₂) → (r, θ) usando atan2
2. **Níveis Recursivos**: Raios resultantes → próximo nível (r₁, r₂) → (R, Θ)
3. **Continua**: Até restar um único raio final

### Distribuição dos Ângulos

Em cada nível ℓ, ângulos seguem distribuição concentrando em π/4:
```
fψ(ℓ)(ψ) segue distribuição sin^(2^(ℓ-1)-1)
```

Níveis mais altos → concentração mais forte → melhor eficiência de quantização.

## Configuração Prática (Llama-3.1-8B, d=128)

- **Nível 1**: 4 bits (16 buckets), range [0, 2π)
- **Níveis 2-4**: 2 bits (4 buckets) cada, range [0, π/2]
- **Raios**: Full precision (FP16)
- **Total**: 3.875 bits por coordenada

## Cálculo de Compressão (d=128)

```
64 ângulos nível 1 × 4 bits  = 256 bits
32 ângulos nível 2 × 2 bits  =  64 bits
16 ângulos nível 3 × 2 bits  =  32 bits
 8 ângulos nível 4 × 2 bits  =  16 bits
 8 raios restantes × 16 bits = 128 bits
─────────────────────────────────────────
Total: 496 bits vs 2048 bits original = 4.13× compressão
```

## Resultados

- **Compressão**: >4.2× em tarefas de longo contexto
- **Needle-In-A-Haystack**: PolarQuant 0.991 vs KIVI 0.984 vs Full 0.997
- **LongBench**: 48.37 (online) vs 46.70 (KIVI)

## Vantagens sobre Quantização Tradicional

| Feature | Quantização Tradicional | PolarQuant |
|---------|------------------------|------------|
| Normalização | Necessária (overhead) | Eliminada |
| Calibração | Necessária | Não necessária |
| Dependência de dados | Sim | Não (data-oblivious) |
| Codebook | Aprendido dos dados | Analiticamente computado |

## Importância para Nosso Compactador

PolarQuant é a BASE do TurboQuant. Entender a transformação polar recursiva e a distribuição Beta dos ângulos é **essencial** para reimplementação.
