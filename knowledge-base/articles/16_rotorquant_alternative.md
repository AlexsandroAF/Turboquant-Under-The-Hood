# RotorQuant: A Faster Alternative to TurboQuant

> **Fonte**: https://github.com/scrya-com/rotorquant
> **Paper**: https://www.scrya.com/rotorquant.pdf
> **Licença**: MIT
> **NOTA**: RotorQuant SUPERA TurboQuant em todos os benchmarks

---

## Overview

RotorQuant usa rotações block-diagonal baseadas em álgebra de Clifford ao invés da Walsh-Hadamard Transform completa do TurboQuant. Resultado: melhor PPL, mais rápido, muito menos parâmetros.

## Abordagem Técnica

### Fundamento Matemático

Usa Clifford algebra Cl(3,0) — rotor sandwich product com 4 componentes multivetor não-zero. Em vez de transformar TODAS as 128 dimensões com dependências sequenciais (WHT), aplica rotações independentes 2D ou 4D por bloco.

**Insight chave**: Vetores de atenção reais em LLMs ocupam manifolds low-rank. Decorrelação completa via transforms full-rank é DESNECESSÁRIA. Rotações de bloco pequenas são suficientes.

### Três Variantes

| Variante | Operação | FMAs (d=128) | Parâmetros |
|----------|----------|--------------|------------|
| RotorQuant | Cl(3,0) rotor sandwich | 2,400 | 372 |
| IsoQuant | Rotação quaternion 4D | 512 | 128 |
| **PlanarQuant** | **Rotação Givens 2D** | **256** | **128** |

**Surpreendente**: Rotações de bloco MAIS SIMPLES dão MELHOR perplexidade que WHT global!

### Deferred Quantization (CRÍTICO)

K-cache aloca como FP16 durante prefill, quantiza APENAS quando decode começa:
- Elimina compounding de erro de roundtrip quantization
- Remove overhead de dequantização em flash attention durante prefill
- **3× melhor PPL** que esquemas roundtrip simétricos

## Benchmarks vs TurboQuant

### Llama 3.1 8B (3-bit simétrico K+V, RTX 5090)

| Config | Decode | Prefill | PPL | Compression |
|--------|--------|---------|-----|-------------|
| FP16 baseline | 140 tok/s | 6,156 tok/s | 6.63 | 1x |
| **IsoQuant 3-bit** | **118** | **3,397** | **6.91** | **10.3x** |
| **PlanarQuant 3-bit** | **119** | **3,822** | **7.05** | **10.3x** |
| TurboQuant 3-bit | 93 | 722 | 7.07 | 10.3x |

### vs TurboQuant (mesma compressão):
- **Melhor PPL**: 6.91 vs 7.07
- **28% mais rápido decode**: 119 vs 93 tok/s
- **5.3× mais rápido prefill**: 3,822 vs 722 tok/s
- **44× menos parâmetros**: 128 vs 16,384

### VRAM Savings (10.3x Compression)

| Context | FP16 KV | Compressed | Saved |
|---------|---------|------------|-------|
| 8K | 288 MB | 28 MB | 260 MB |
| 32K | 1,152 MB | 112 MB | 1.04 GB |
| 128K | 4,608 MB | 447 MB | 4.16 GB |

## Por Que Mais Rápido?

**TurboQuant/WHT**: log₂(d) estágios de Walsh-Hadamard em 128 dimensões → dependências sequenciais, inerentemente serial.

**RotorQuant**: Rotações 2D/4D independentes por bloco → sem dependências cross-element, totalmente paralelizável, cache-friendly.

## Código

### llama.cpp

```bash
# IsoQuant 3-bit (melhor qualidade por bit)
./build/bin/llama-server -m model.gguf -ngl 99 \
  --cache-type-k iso3 --cache-type-v iso3

# PlanarQuant K-only (zero perplexity loss)
./build/bin/llama-server -m model.gguf -ngl 99 \
  --cache-type-k planar3 --cache-type-v f16
```

### Python/Triton

```python
from turboquant import IsoQuantMSE, PlanarQuantMSE

iq = IsoQuantMSE(d=128, bits=4, mode='fast', device='cuda')
x_hat, indices = iq(x)

pq = PlanarQuantMSE(d=128, bits=3, device='cuda')
x_hat, indices = pq(x)
```

## Detalhe Crítico: Rotação Inversa para V-Cache

V-cache dequantização PRECISA da rotação INVERSA. WHT do TurboQuant tem propriedades self-canceling em somas ponderadas de atenção. PlanarQuant/IsoQuant requerem Givens inverso ou quaternion inverso explícito — fix que melhorou PPL de 15,369 para 7.05.

## Relevância para Nosso Compactador

**RotorQuant é potencialmente MELHOR que TurboQuant para nossa implementação.** Considerar:
1. PlanarQuant (rotação Givens 2D) como base — mais simples, mais rápido
2. Deferred quantization é essencial
3. 44× menos parâmetros = muito mais prático
4. Combinar com Lloyd-Max codebooks do TurboQuant
