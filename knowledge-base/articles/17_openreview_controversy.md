# TurboQuant OpenReview: Reviews e Controvérsia

> **Fonte**: https://openreview.net/forum?id=tO3ASKZlok
> **Substack**: https://amicusai.substack.com/p/turboquant-a-spicy-openreview-session
> **Status**: OpenReview bloqueou acesso direto (403), conteúdo coletado de fontes secundárias

---

## Informações dos Reviews

### Reviewer Independente sobre RaBitQ
Um reviewer do ICLR apontou independentemente que "RaBitQ and variants are similar to TurboQuant in that they all use random projection" e requisitou explicitamente uma discussão e comparação mais completa.

### Resposta dos Autores
Na versão final do ICLR, os autores do TurboQuant:
- NÃO adicionaram discussão real sobre RaBitQ
- MOVERAM a descrição já incompleta de RaBitQ para o apêndice (piorando)

## Controvérsia Pública

### RaBitQ Team (Jianyang Gao, ETH Zurich)
Via X/Twitter e OpenReview:
- TurboQuant misrepresenta RaBitQ em 3 dimensões (teoria, metodologia, experimentos)
- Issues foram comunicados aos autores ANTES da submissão
- Autores acknowledged mas não corrigiram
- Complaint formal submetido para ICLR General Chairs, PC Chairs, Code and Ethics Chairs

### Google Response
Via OpenReview: Resposta dos autores sugere que RaBitQ team NÃO havia levantado os pontos técnicos por canais acadêmicos — contradito por email evidence.

### Academic Support
Stanford NLP Group retweetou as concerns do RaBitQ team, amplificando a visibilidade.

## Pontos Técnicos da Disputa

1. **Random rotation + quantization**: Técnica compartilhada entre RaBitQ (SIGMOD 2024) e TurboQuant
2. **Bounds teóricos**: RaBitQ prova atingir bounds ótimos (Alon-Klartag); TurboQuant alega "suboptimal"
3. **Benchmarks**: TurboQuant comparou RaBitQ em Python single-threaded vs TurboQuant em A100 GPU

## Relevância

Este contexto é importante para:
1. Entender que a técnica core (rotação + quantização) NÃO é nova do TurboQuant
2. RaBitQ (SIGMOD 2024) é um antecessor importante a ser estudado
3. Avaliar claims de novidade vs contribuição incremental
