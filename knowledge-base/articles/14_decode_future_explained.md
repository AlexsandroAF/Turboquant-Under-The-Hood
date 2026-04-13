# TurboQuant Explained: 3-Bit KV Cache at 6× Compression

> **Fonte**: https://decodethefuture.org/en/turboquant-vector-quantization-kv-cache/
> **Plataforma**: Decode The Future

---

## Overview

TurboQuant: algoritmo de quantização vetorial de dois estágios (ICLR 2026) que comprime KV caches de LLM para 3 bits por coordenada com zero perda de accuracy. Pelo menos 6× redução de memória e até 8× attention mais rápido em H100.

## Como Funciona

### Stage 1: PolarQuant (MSE-Optimal)
Rotação ortogonal aleatória → coordenadas seguem Beta distribution concentrada → Lloyd-Max quantization com codebooks pré-computados → elimina overhead de normalização.

### Stage 2: QJL (1-Bit Residual)
Stage 1 minimiza MSE mas introduz bias em dot products. QJL: transformada Johnson-Lindenstrauss nos resíduos → cada coordenada reduzida a sign bit → estimador de inner product não-enviesado.

**Total: (b-1) bits PolarQuant + 1 bit QJL = b bits total**

## Benchmarks

| Métrica | 3-bit TQ | 4-bit TQ | FP16 |
|---------|---------|---------|------|
| LongBench | Quality neutral | Quality neutral | 100% |
| Needle-In-A-Haystack | Perfeito | Perfeito | Perfeito |
| Redução KV Memory | ≥6× | ≥4× | 1× |
| Speedup Attention (H100) | — | Até 8× | 1× |
| Training Required | Nenhum | Nenhum | — |

Qwen2.5-3B em 8K tokens: 289 MB → 58 MB a 3-bit.

## Information-Theoretic Bounds

TurboQuant opera dentro de ~2.7× do limite teórico de Shannon — nenhum algoritmo pode fundamentalmente performar >2.7× melhor no mesmo bit budget.

## Comparação com Alternativas

| Método | Tipo | Min Bits | Training | Inner Product Não-enviesado |
|--------|------|----------|----------|---------------------------|
| TurboQuant | Data-oblivious VQ | 2.5-3 | Não | Sim |
| KIVI | Per-channel scalar | 2 | Não | Não |
| KVQuant | NUQ + dense-sparse | 3 | Calibração | Não |
| CommVQ | Additive VQ | 2 | EM training | Não |

## Limitações

A maioria dos LLM systems já capturaram 2-3× via int8/int4 e outlier-aware. TurboQuant push para ~4-4.5× efetivo, com ganhos restantes cada vez mais caros ao nível de kernel. Algoritmo se aproxima dos limites informação-teóricos.
