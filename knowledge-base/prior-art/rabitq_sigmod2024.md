# RaBitQ: Quantizing High-Dimensional Vectors with Theoretical Error Bounds

> **Fonte**: https://arxiv.org/abs/2405.12497
> **PDF**: ../pdfs/rabitq_sigmod2024.pdf
> **Status**: SIGMOD 2024
> **Autores**: Jianyang Gao, Cheng Long (Nanyang Technological University)
> **Extended version**: arXiv:2409.09913 (SIGMOD 2025)

---

## IMPORTÂNCIA: Prior art direto do TurboQuant. Mesma técnica core (rotação aleatória + quantização).

## Contribuições Principais

1. **Garantias teóricas**: Estimador de distância não-enviesado com bound de erro O(1/√D) — provado assintoticamente ótimo per Alon & Klartag (FOCS 2017)

2. **Abordagem técnica**:
   - Normaliza vetores de dados na hipersfera unitária
   - Constrói codebook via vetores bi-valorados rotacionados aleatoriamente (2^D vetores com coordenadas ±1/√D)
   - Representa códigos de quantização como D-bit strings (vs códigos mais longos do PQ)

3. **Eficiência**: 3× mais rápido em estimativa de distância single-vector via operações bitwise; performance batch comparável ao PQ baseado em SIMD

4. **Vantagem empírica**: Accuracy superior com ~50% menos código que PQ; performance estável em datasets onde PQ falha (ex: MSong)

## Relação com TurboQuant

### Similaridades (o que a controvérsia aponta)
- Ambos usam **rotação aleatória (Johnson-Lindenstrauss)** antes da quantização
- Ambos alcançam **bounds ótimos** de Alon-Klartag
- RaBitQ publicado **1 ano antes** (May 2024 vs April 2025)

### Diferenças
- RaBitQ foca em **nearest neighbor search**; TurboQuant foca em **KV cache compression**
- RaBitQ usa vetores bi-valorados (±1/√D); TurboQuant usa Lloyd-Max scalar quantization
- TurboQuant adiciona QJL como Stage 2

### Controvérsia
- Autor do TurboQuant (Daliri) contactou Gao em Jan 2025 pedindo ajuda para debugar implementação Python de RaBitQ
- TurboQuant mischaracterizou RaBitQ como "grid-based PQ" no paper
- TurboQuant comparou RaBitQ em single-threaded CPU vs A100 GPU
- Ver: articles/07_rabitq_controversy.md e articles/17_openreview_controversy.md

## Para Ler o Paper Completo

O PDF está em: `referencia/pdfs/rabitq_sigmod2024.pdf`
