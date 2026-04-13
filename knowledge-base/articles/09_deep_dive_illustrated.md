# An Illustrated Deep Dive into TurboQuant: PolarQuant, QJL, and KV Cache Compression

> **Fonte**: https://darshanfofadiya.com/research-papers/turboquant/
> **Tipo**: Deep-dive técnico completo com exemplos numéricos

---

## 1. O Problema de Memória do KV Cache

Para Llama-70B com 1M tokens:
- **Por camada**: 4.096 GB em BF16
- **80 camadas**: 327.68 GB total
- **Por GPU (32 GPUs)**: 10.24 GB por GPU
- **KV cache é 2.34x MAIOR que os pesos do modelo** (140 GB)

### Alvos de Compressão

| Bitwidth | Por Camada | Total | Por GPU |
|----------|-----------|-------|---------|
| BF16 (16-bit) | 4.096 GB | 327.68 GB | 10.24 GB |
| INT4 (4-bit) | 1.024 GB | 81.92 GB | 2.56 GB |
| TurboQuant (3.5-bit) | 0.896 GB | 71.68 GB | 2.24 GB |

## 2. Por Que Quantização Padrão Falha

### O Problema dos Outliers

Exemplo de vetor KV:
```
Dims:    [0.02, -0.05, 0.03, 87.4, 0.01, -0.04, 0.06, -42.1]
Range:   8,740x diferença (87.4 / 0.01)
Energia: Dims 3+7 = 100% da energia total
```

**Quantização INT4 naive** (signed, 16 levels: -8 to +7):
```
Scale = 87.4 / 7 = 12.486
Quantizado: [0, 0, 0, 7, 0, 0, 0, -3]
Reconstruído: [0, 0, 0, 87.4, 0, 0, 0, -37.5]
Resultado: 6 de 8 dimensões DESTRUÍDAS (100% erro)
```

### Overhead da Normalização Per-Dimension

```
Valores quantizados: 1,024 × 4 bits = 512 bytes
Fatores de escala:   1,024 × 2 bytes = 2,048 bytes
Total: 2,560 bytes vs 2,048 bytes sem compressão → COMPRESSÃO FALHA
```

## 3. PolarQuant: Rotação Aleatória

### Construindo Matriz Ortogonal (Gram-Schmidt)

```
Step 1: Normaliza primeira linha
  e₀ = g₀ / ||g₀||

Step k (k=1,2,...,d-1): 
  proj_j = ⟨g_k, e_j⟩  para todo j anterior
  v_k = g_k - Σ proj_j × e_j
  e_k = v_k / ||v_k||
```

**Resultado**: Matriz ortogonal Π onde ΠᵀΠ = I. Cada entrada tipicamente ±0.1 a ±0.7.

### O Que a Rotação Faz (Exemplo 8D)

**Antes da rotação**:
```
x = [0.02, -0.05, 0.03, 87.4, 0.01, -0.04, 0.06, -42.1]
Ratio de range: 8,740x
||x||² = 9,411.18 (81.2% no dim 3, 18.8% no dim 7)
```

**Depois da rotação** (y = Π @ x):
```
y = [-38.3, 41.0, 50.2, 22.1, -27.6, -23.6, -33.1, 28.9]
Ratio de range: 2.27x (50.2 / 22.1)
||y|| = 97.01 (norma preservada EXATAMENTE)
Cada coordenada carrega energia similar
```

### Por Que Funciona em d=128

```
E[yᵢ²] = 1/d

Para d=128:
  Std = 1/√128 = 0.0884
  99% range: ±0.228
  Max erro de quantização (3-bit): 0.029
```

**Insight chave**: Erro por coordenada shrinks como 1/√d, mas número de coordenadas cresce como d. Ao somar erros para attention scores:

```
Erro de attention score ≈ √d × (erro por coord)
                        = √d × (1/√d) / (# levels)
                        = 1 / (# levels)
                        ≈ constante, independente de d
```

**Dimensões mais altas MELHORAM a precisão da compressão.**

## 4. Pipeline Completo PolarQuant (d=128)

### Setup (Uma vez)
- Gerar Π ortogonal (128×128, 32 KB de armazenamento)
- Pré-computar centroids Lloyd-Max para distribuição Beta(d=128)

### Write Path (Token entra no cache)
```
1. Extrair magnitude: r = ||k||, k_unit = k / r
2. Rotacionar: y = Π @ k_unit              (16,384 operações)
3. Quantizar: 128 coords → nearest de 8 centroids
4. Armazenar: 400 bits por vetor (384 bits + 16-bit norm)
```

### Read Path (Computação de atenção)
```
1. Rotacionar query UMA VEZ: q_rot = Π @ q  (16,384 operações)
2. Para cada key no cache:
   score = q_rot · k_quantized_rotated
   (NÃO precisa de rotação inversa — keys ficam no espaço rotacionado)
```

### Cálculo de Storage
```
Por vetor:  128 coords × 3 bits + 16 bits norm = 400 bits = 50 bytes
Por token por camada (K+V): 8 heads × 2 × 50 = 800 bytes
80 camadas, 1M tokens: 64 GB
Compressão: 327.68 / 64 = 5.12x
```

## 5. QJL: Corrigindo o Bias

### O Problema: Ranking Flips

Com PolarQuant sozinho, quantização introduz bias determinístico que pode flipar rankings.

**Exemplo**: Vetores 2D, quantização 2-bit (4 levels):
```
Scores reais:     Q·K₁ = 0.17 (K₁ vence)
                  Q·K₂ = 0.10

Após quantização:
Estimados:        Q·K₁_hat = 0.05 (K₂ vence!)
                  Q·K₂_hat = 0.25

Swing de bias: 0.27 vs gap real: 0.07
Ranking INVERTIDO; erro = 3.9x a diferença real
```

### Solução QJL: Correção Não-enviesada

**Step 1: Computar residual**
```
e = K - K_hat (o erro que PolarQuant introduziu)
```

**Step 2: Projetar e armazenar sinais**
```
S = matriz de sinais aleatória (m × d, cada entrada ±1)
z = S @ e                        // projetar residual
Armazenar: sign(z) como m bits   // (m=48-64)
```

**Step 3: Estimar correção na inferência**
```
z_Q = S @ Q                      // projetar query full-precision
correction = (1/m) × Σ(z_Q[i] × sign(z[i]))
Final score = ⟨Q, K_hat⟩ + correction
```

### Por Que QJL Funciona
- **Não-enviesado**: E[correction] = ⟨Q, e⟩ exatamente
- **Erro shrinks como 1/√m**: Mais sign bits = melhor estimativa
- **Custo negligível**: m=48-64 bits vs 400 bits da quantização

### Memória com QJL
```
PolarQuant:     400 bits
QJL (m=48):      48 bits
Total:          448 bits por vetor
Por coordenada: 448 / 128 = 3.5 bits
```

## 6. Estimador Completo TurboQuant

```
Score real de atenção: ⟨Q, K⟩

Estimativa TurboQuant:
  ⟨Q, K_hat⟩ + (1/m) × Σ(projection_Q[i] × sign_K[i])
   ├─ Componente enviesado (PolarQuant)
   └─ Correção não-enviesada (QJL)

Em expectativa:
  E[TurboQuant] = ⟨Q, K_hat⟩ + ⟨Q, e⟩
               = ⟨Q, K⟩  (recupera score real perfeitamente)
```

## 7. Impacto na Infraestrutura

### Memória por GPU (Llama-70B, 1M tokens)

| Componente | FP16 | TurboQuant |
|-----------|------|-----------|
| Weights (FSDP) | 4.38 GB | 4.38 GB |
| KV cache | 10.24 GB | 2.24 GB |
| Activations | 0.50 GB | 0.50 GB |
| Overhead | 2.00 GB | 2.00 GB |
| **Total** | **17.12 GB** | **9.12 GB** |

### Capacidade de Batch
```
Requests concorrentes 1M-token:
  FP16:  6 requests
  TQ:    26 requests (4.3x melhoria)
```

### GPUs Necessárias
```
Mesmo workload (1M-token context):
  FP16:  6 GPUs
  TQ:    3 GPUs

128K context (mais realista):
  FP16 KV cache:  41.94 GB (apertado em 1 GPU)
  TQ KV cache:    9.18 GB  (fácil em 1 GPU)
```

## 8. Limites Teóricos

```
Shannon limit: lower bound informação-teórico

TurboQuant: MSE ≈ 2.7 × MSE_min

Interpretação:
  - 2.7x do ótimo teórico
  - ~1.4 bits adicionais poderiam ser salvos no limite de Shannon
  - Espaço para melhoria: apenas 3.5 → 2.1 bits
  - Ganhos fáceis já exauridos
```

## 9. Caveats Importantes

1. **Apenas modelos 8B testados** — escalação para 70B+ não provada
2. **8x speedup é apenas attention** — melhoria full inference 2-4x
3. **Sem código oficial** — implementações comunitárias existem
4. **Não deployed em produtos Google** — apenas pesquisa
5. **KV cache é um componente** — não elimina necessidade de FSDP/TP

## 10. Fórmulas Chave

| Conceito | Fórmula | Interpretação |
|---------|---------|-----------------|
| Matriz ortogonal | ΠᵀΠ = I | Rotação preserva norma e inner products |
| Variância por coord | E[yᵢ²] = 1/d | Maior d → clustering mais tight |
| Erro de attention | O(1/√d) × (1/levels) | Independente da dimensão |
| QJL não-enviesado | E[correction] = ⟨Q, e⟩ | Ruído cancela em média |
| Convergência do erro | σ_error ∝ 1/√m | Mais sign bits → melhor estimativa |
