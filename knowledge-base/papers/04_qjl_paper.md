# QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead

> **Fonte**: https://arxiv.org/abs/2406.03482
> **Status**: AAAI 2025
> **Autores**: Amir Zandieh, Majid Daliri, Insu Han
> **GitHub**: https://github.com/amirzandieh/QJL (Apache-2.0)
> **Data**: June 5, 2024

---

## Overview

QJL é o Stage 2 do TurboQuant. Transformada Johnson-Lindenstrauss seguida de quantização em sign-bit para compressão de KV cache sem overhead de memória.

## Contribuição Principal

Estimador assimétrico para inner product: aplica QJL a um vetor e JL padrão (sem quantização) ao outro → estimador não-enviesado com distorção mínima.

## Definição Matemática

```
Q_qjl(x) := sign(S·x)     onde S ∈ ℝ^(d×d) ~ N(0,1) i.i.d.
Q_qjl^(-1)(z) := (√(π/2)/d)·S^T·z     para z ∈ {-1,+1}^d
```

### Propriedades (Lemma 4 do TurboQuant paper)

Para x ∈ S^(d-1) e y ∈ ℝ^d:
- **Não-enviesado**: E[⟨y, Q_qjl^(-1)(Q_qjl(x))⟩] = ⟨y,x⟩
- **Variância**: Var(⟨y, Q_qjl^(-1)(Q_qjl(x))⟩) ≤ (π/2d)·||y||_2²

## Resultados

Redução de >5× na memória do KV cache quantizando para 3 bits, sem comprometer accuracy, com runtime mais rápido.

## Código Oficial — Uso

```bash
# Instalação
git clone git@github.com:amirzandieh/QJL.git
cd QJL
pip install -r requirements.txt

# Build CUDA kernel
cd qjl_kernel
python setup.py build_ext --inplace
```

### Avaliação no LongBench

```bash
python longbench.py \
    --model_name "lmsys/longchat-7b-v1.5-32k" \
    --dtype "float16" \
    --key_quantization_bits 256 \
    --key_quantization_bits_initial_layers 512 \
    --initial_layers_count 15 \
    --outlier_count_general 8 \
    --outlier_count_initial_layers 8 \
    --value_quantization_bits 2 \
    --group_size 32 \
    --buffer_size 128 \
    --seed 42 \
    --dataset_name [dataset_name] \
    --n_data 150
```

## Parâmetros Importantes

- `key_quantization_bits`: Número de bits para projeção QJL das keys (256 = d×2)
- `key_quantization_bits_initial_layers`: Mais bits para camadas iniciais (512)
- `initial_layers_count`: Quantas camadas iniciais recebem mais bits (15)
- `outlier_count_general`: Canais outlier tratados separadamente (8)
- `value_quantization_bits`: Bits para quantização de values (2)
- `group_size`: Tamanho do grupo para group quantization (32)
- `buffer_size`: Tokens recentes mantidos em FP16 (128)

## Citação

```bibtex
@article{zandieh2024qjl,
  title={QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead},
  author={Zandieh, Amir and Daliri, Majid and Han, Insu},
  journal={arXiv preprint arXiv:2406.03482},
  year={2024}
}
```

## Relevância para Nosso Compactador

QJL é o componente de correção de bias. **PORÉM**: múltiplas implementações independentes mostraram que QJL degrada performance na prática quando usado com softmax attention. Para nosso compactador, considerar usar MSE-only (sem QJL) ou QJL apenas como correção adicional, nunca como substituição de bit.
