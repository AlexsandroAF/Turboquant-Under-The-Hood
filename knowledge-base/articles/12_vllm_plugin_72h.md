# I Shipped Google's TurboQuant as a vLLM Plugin 72 Hours After the Paper

> **Fonte**: https://dev.to/albertocodes/i-shipped-googles-turboquant-as-a-vllm-plugin-72-hours-after-the-paper-heres-what-nobody-else-473g
> **Autor**: Alberto Nieto
> **GitHub**: https://github.com/Alberto-Codes/turboquant-vllm
> **PyPI**: turboquant-vllm

---

## Quick Start

```bash
pip install turboquant-vllm[vllm]
vllm serve allenai/Molmo2-8B --attention-backend CUSTOM
```

### HuggingFace Integration

```python
from transformers import DynamicCache
from turboquant_vllm import CompressedDynamicCache

cache = DynamicCache()
compressed = CompressedDynamicCache(cache, head_dim=128, bits=4)
```

## Inovação: Teste com Vision-Language

Testou em Molmo2 com vídeo gerando ~11,000 visual tokens — 10x mais pressão de memória que cenários texto-only. Gap significativo em validações anteriores.

## Resultados

| Métrica | Baseline | TQ4 Compressed |
|---------|----------|----------------|
| KV cache (RTX 4090, 11K tokens) | 1,639 MiB | 435 MiB (**3.76x**) |
| Qualidade output | Descrições detalhadas | **Near-identical** |
| Overhead decode | — | 1.78x |

Episódio de 23 minutos processado a 24 tokens/s em Molmo2-8B.

## Diferenciação Técnica

**Arquitetura Plugin**: Usa entry point system oficial do vLLM (sem forks/monkey-patching).

**Dequantização Incremental**: Otimização nova — descomprime apenas tokens novos por step, reduzindo overhead de 3.36x para 1.78x. **Não documentada no paper original.**

**Cross-Platform**: Triton kernels rodam em NVIDIA CUDA e AMD ROCm sem modificações.

## Issues Descobertos

1. **FP16 Precision Failure**: Erros compostos em 36 camadas causam corrupção; FP32 necessário
2. **QJL Ineficiência**: Abordagem 2-bit MSE do paper desperdiça precisão; 3-bit MSE produz resultados idênticos
3. **Multi-Layer Fusion Drift**: 0.023 cosine gap por camada compõe para erros semânticos; Flash Attention fusion necessário

## Validação

- 180+ unit tests em 9 arquivos (95%+ coverage)
- 16 experimentos GPU documentados
- Cross-platform: RTX 4090 e AMD Radeon 890M
- End-to-end PyPI installation em stock vLLM container
