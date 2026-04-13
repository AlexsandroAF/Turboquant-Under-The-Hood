# TurboQuant Animated: Watch Vector Quantization Happen

> **Fonte**: https://mesuvash.github.io/blog/2026/turboquant-interactive/
> **Autor**: Suvash Sedhain
> **Tipo**: Visualização interativa 2D/3D

---

## Passos do Algoritmo (Visualizados)

### Step 1: Normalize
Armazena norma (comprimento) separadamente. Divide vetor pela norma → projeção no círculo/esfera unitário(a). Separa magnitude de direção.

### Step 2: Apply Random Rotation
Multiplica TODOS os vetores pela MESMA matriz de rotação aleatória fixa (gerada uma vez de um seed). Produz coordenadas com distribuição conhecida e previsível. Compartilhada entre encoder e decoder sem custo de transmissão.

### Step 3: Quantize to Grid
Usando distribuição previsível da rotação, pré-computa posicionamento ótimo do grid. Cada coordenada arredondada para um de 2^b valores. Armazenada como índice. **ÚNICA operação com perda.**

### Step 4a: Undo Rotation
Reconstrói: lookup de valores do grid pelos índices, depois multiplica pela transposta da rotação.

### Step 4b: Rescale by Norm
Multiplica aproximação unitária pela norma armazenada. Erro de reconstrução vem SOMENTE do Step 3.

## Design do Grid

Grid é **NÃO-UNIFORME** — centroids ótimos para distribuição conhecida:

- **Em 2D** (círculo unitário): coordenadas seguem distribuição arcseno f(y) = 1/(π√(1−y²)), concentrando perto de ±1 → centroids clusteram nas bordas
- **Em alta dimensão** (d ≥ 128): coordenadas concentram fortemente perto de zero (near-Gaussian) → maior eficiência do grid

### Fórmulas dos Centroids

```
Centroids:  c_k = n(√(1−b_k²) − √(1−b_{k+1}²))/π
Boundaries: b_k = −cos(kπ/n)
```

## Exemplo de Compressão

Para vetor 128-dimensional em 3 bits:
```
128 coords × 3 bits = 384 bits
+ 1 float para norma = ~16 bits
Total: 50 bytes (de 256 bytes em FP16)
= 5× compressão
```

## Garantia Teórica

"O erro é provavelmente dentro de 2.72× do melhor que QUALQUER método poderia alcançar, mesmo um com conhecimento perfeito dos dados."

## Vantagem Chave

"Sem dados de treino necessários, sem codebook para aprender. O grid é fixo e universal."
