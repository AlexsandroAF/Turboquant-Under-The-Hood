# TurboQuant, One Month Later: Implementations, Controversy, and What Actually Works

> **Fonte**: https://www.frr.dev/posts/turboquant-one-month-later-implementations-controversy-benchmarks/
> **Plataforma**: frr.dev
> **Data**: ~April 2026

---

## Promessas vs Realidade

### Google Prometeu
"6x compression of the KV cache, an 8x speedup in attention logits on H100 GPUs, with zero degradation in benchmarks."

### Realidade
PolarQuant (rotação + coordenadas polares) entrega 4-5x compressão prática — substancialmente menos que 6x mas suficiente.

## Implementações por Tier

### Tier 1 — Fully Functional
- **turboquant-mlx**: 4.7x compression em M4 Max (3-bit K, 3-bit V)
- **turboquant-pytorch**: 5x compression com K4/V2 assimétrico
- **turboquant (0xSero)**: Triton kernels com vLLM integration

### Tier 2 — Functional Sem Benchmarks Independentes
- SwiftLM (Swift, servidor OpenAI-compatible)
- TurboVec (CPU, vector search)
- turboquant_mlx (1-3 bit assimétrico)

### Tier 3 — Em Review
- llama.cpp discussion com implementação CPU
- vLLM pull request pendente

## Descoberta Crítica: QJL Falha

"QJL degrades performance in practice." O componente de correção de erro "introduces more errors than it fixes."

**~80% do valor do TurboQuant vem de PolarQuant sozinho.**

MSE-only supera MSE+QJL em métricas de token matching.

## Controvérsia RaBitQ

Alegações formais de ética:
1. **Similaridade omitida** — Ambos usam rotações JL; Google mischaracterizou RaBitQ como "grid-based PQ"
2. **Teoria misrepresentada** — TurboQuant alegou garantias RaBitQ "suboptimal" apesar de provas rigorosas
3. **Benchmarking enviesado** — RaBitQ testado com "Python degradado, single-threaded, multithreading desabilitado" vs A100 GPUs

Google Research NÃO respondeu publicamente.

## Paths Práticos de Implementação

**Apple Silicon (MLX):**
```
4.6x compression, 98% retenção de throughput, near-FP16 perplexity
```

**CUDA (PyTorch):**
```
K4/V2 (4-bit keys, 2-bit values) = ~5x compression
```

**Swift (Production Server):**
```
85 tokens/s com Gemma-4-26B em M5 Pro; 100K-token contexts
```

## Limitações Atuais

- Sem integração com Apple FoundationModels framework
- vLLM production integration pendente
- Benchmarks de vector search não confiáveis (comparação enviesada com RaBitQ)

## Assessment

PolarQuant sozinho entrega 4-5x compressão. QJL adiciona ruído ao invés de corrigir bias na prática. Verificação independente é essencial.
