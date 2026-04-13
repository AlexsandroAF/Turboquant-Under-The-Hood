# Google Dropped TurboQuant Two Weeks Ago. The Community Already Made It Usable

> **Fonte**: https://dev.to/alanwest/google-dropped-turboquant-two-weeks-ago-the-community-already-made-it-usable-3h0k
> **Plataforma**: DEV Community
> **Data**: ~April 7, 2026

---

## Overview

Google publicou o paper em March 25. Em 2 semanas: 5 implementações independentes, fork llama.cpp rodando modelo 104B em MacBook, e integração vLLM ativa.

Google lançou research paper mas **sem código acompanhante**. 7.7 milhões de views em social media.

## O Que a Comunidade Construiu

### tonbistudio/turboquant-pytorch
PyTorch reference, clareza > performance. Early reports de 5x compressão tinham bug inflando resultados. Após correção, 3-bit key degrada em certos cenários. **Melhor para estudo algorítmico.**

### TheTom/llama-cpp-turboquant
C/C++ fork de llama.cpp com CPU e CUDA kernels. **18/18 testes passando, MSE matching paper dentro de 1%.** Opção mais production-ready para NVIDIA GPU Linux.

### TheTom/turboquant_plus
llama.cpp Metal para Apple Silicon. Tipos turbo3 e turbo4. **Destaque: modelo 104B, 128K context em M5 Max MacBook** — PPL 4.024, 74GB peak memory.

```bash
./bin/llama-cli \
  -m /path/to/model.gguf \
  -ctk turbo3 \
  -ctv turbo3 \
  -c 131072 \
  -n 512 \
  -p "Your prompt here"
```

### 0xSero/turboquant
Triton kernels para vLLM. K 3-bit, V 2-bit. Menos maduro, mas desenhado para serving cloud multi-user.

### scos-lab/turboquant
Paper reproduction com insights sobre ambiguidades na pesquisa original. **Útil para entender o algoritmo.**

## Achievement: M5 Max

Modelo 104B com 128K context em MacBook. Sem compressão: demandaria servidores multi-GPU. Community: 4.6x compressão mantendo qualidade — "identical to f16 baseline at temperature 0."

## Matriz de Comparação

| Implementação | Lang | Target | Maturidade | Uso |
|---|---|---|---|---|
| tonbistudio/pytorch | Python | Research | Stable | Estudo |
| TheTom/llama-cpp | C/C++/CUDA | Linux/NVIDIA | Solid | GPU deploy |
| TheTom/turboquant_plus | C/C++/Metal | macOS/Apple | Solid | Mac local |
| 0xSero/turboquant | Triton | vLLM/Cloud | Early | Multi-user |
| scos-lab/turboquant | Python | Research | Stable | Paper reproduction |

## Recomendações

- **Apple Silicon 64GB+**: turboquant_plus
- **NVIDIA**: TheTom/llama-cpp (testes passando)
- **vLLM produção**: Aguardar mainline integration
- **Conservador**: Aguardar suporte oficial llama.cpp/vLLM
