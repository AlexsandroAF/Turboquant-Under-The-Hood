# Mapa de Referência para Reimplementação

## PDFs dos Papers (5 arquivos)

| Paper | Arquivo | Tamanho |
|-------|---------|---------|
| TurboQuant (ICLR 2026) | `pdfs/turboquant_iclr2026.pdf` | 862 KB |
| PolarQuant v1 (AISTATS 2026) | `pdfs/polarquant_aistats2026.pdf` | 1.1 MB |
| PolarQuant v2 (table lookup) | `pdfs/polarquant_v2_2502.00527.pdf` | 1.2 MB |
| QJL (AAAI 2025) | `pdfs/qjl_aaai2025.pdf` | 4.9 MB |
| RaBitQ (SIGMOD 2024) — prior art | `pdfs/rabitq_sigmod2024.pdf` | 2.1 MB |

---

## Repositórios Clonados (4 repos)

### 1. turboquant-py — MELHOR REFERÊNCIA PARA REIMPLEMENTAÇÃO

**Caminho**: `repos/turboquant-py/`

| Arquivo | O que faz | Prioridade |
|---------|-----------|------------|
| `src/turboquant/turboquant.py` | **Classe principal TurboQuant** — quantize, dequantize, inner_product | **CRÍTICO** |
| `src/turboquant/codebook.py` | **Lloyd-Max solver para Beta distribution** — compute_codebook() | **CRÍTICO** |
| `src/turboquant/qjl.py` | **Classe QJL** — 1-bit quantize, inner_product | **IMPORTANTE** |
| `src/turboquant/_bitpack.py` | Bit-packing utilities — empacota indices em bytes | **IMPORTANTE** |
| `src/turboquant/storage.py` | CompressedVectors — save/load, slicing, concatenation | MÉDIO |
| `src/turboquant/_entropy.py` | Huffman encoding — compressão adicional | BAIXO |
| `src/turboquant/_accel.py` | PyTorch/CUDA acceleration | BAIXO |
| `src/turboquant/__init__.py` | API pública exports | REFERÊNCIA |
| `scripts/precompute_codebooks.py` | Gera codebooks offline | ÚTIL |
| `examples/synthetic_test.py` | Teste básico | REFERÊNCIA |
| `benchmarks/bench_all.py` | Benchmark completo | REFERÊNCIA |

### 2. turboquant-pytorch — IMPLEMENTAÇÃO FROM-SCRATCH

**Caminho**: `repos/turboquant-pytorch/`

| Arquivo | O que faz | Prioridade |
|---------|-----------|------------|
| `turboquant.py` | **Core: TurboQuantMSE + TurboQuantProd** | **CRÍTICO** |
| `lloyd_max.py` | **Lloyd-Max quantizer solver** | **CRÍTICO** |
| `compressors_v3.py` | **V3: MSE-only + asymmetric K/V + bit-packing** | **CRÍTICO** |
| `compressors.py` | V2: with QJL (referência) | IMPORTANTE |
| `test_turboquant.py` | Tests sintéticos — valida contra bounds do paper | IMPORTANTE |
| `validate_v3.py` | Compara V3 vs V2 | ÚTIL |
| `generation_test.py` | Teste de geração com modelo real | ÚTIL |

### 3. QJL — CÓDIGO OFICIAL DOS AUTORES DO PAPER

**Caminho**: `repos/QJL/`

| Arquivo | O que faz | Prioridade |
|---------|-----------|------------|
| `qjl_kernel/qjl_kernel.py` | **Python wrapper dos CUDA kernels** | **CRÍTICO** |
| `qjl_kernel/csrc/qjl_quant_kernel.cu` | **CUDA: quantização QJL** | **IMPORTANTE** |
| `qjl_kernel/csrc/qjl_score_kernel.cu` | **CUDA: score computation com QJL** | **IMPORTANTE** |
| `qjl_kernel/csrc/qjl_gqa_score_kernel.cu` | CUDA: GQA-aware score kernel | IMPORTANTE |
| `qjl_kernel/csrc/quantization.cu` | CUDA: quantização de values | REFERÊNCIA |
| `qjl_kernel/new_pack.py` | Bit packing utilities | REFERÊNCIA |
| `models/llama3_qjl.py` | **Integração com Llama-3** | IMPORTANTE |
| `models/llama3_utils_qjl.py` | Utilities de KV cache para Llama-3 | IMPORTANTE |
| `eval_long_bench.py` | Avaliação LongBench | REFERÊNCIA |

### 4. rotorquant — ALTERNATIVA SUPERIOR

**Caminho**: `repos/rotorquant/`

| Arquivo | O que faz | Prioridade |
|---------|-----------|------------|
| `turboquant/turboquant.py` | **Implementação TurboQuant de referência** | **CRÍTICO** |
| `turboquant/lloyd_max.py` | **Lloyd-Max solver** | **CRÍTICO** |
| `turboquant/rotorquant.py` | **RotorQuant: Clifford algebra rotation** | **IMPORTANTE** |
| `turboquant/planarquant.py` | **PlanarQuant: Givens 2D rotation (mais rápido)** | **IMPORTANTE** |
| `turboquant/isoquant.py` | **IsoQuant: Quaternion 4D rotation** | **IMPORTANTE** |
| `turboquant/clifford.py` | Álgebra de Clifford Cl(3,0) | IMPORTANTE |
| `turboquant/compressors.py` | Compressors para KV cache | IMPORTANTE |
| `turboquant/triton_kernels.py` | Triton GPU kernels | REFERÊNCIA |
| `turboquant/triton_planarquant.py` | Triton PlanarQuant | REFERÊNCIA |
| `turboquant/rabitq.py` | **Implementação RaBitQ para comparação** | ÚTIL |
| `turboquant/validate.py` | Validação contra paper | REFERÊNCIA |
| `tests/test_lloyd_max.py` | Tests do Lloyd-Max | REFERÊNCIA |

---

## Prior Art

| Arquivo | Descrição |
|---------|-----------|
| `prior-art/rabitq_sigmod2024.md` | Resumo do RaBitQ + contexto da controvérsia |

---

## Ordem de Leitura Recomendada para Reimplementação

### Fase 1: Entender o Core
1. `repos/turboquant-py/src/turboquant/codebook.py` — Lloyd-Max para Beta
2. `repos/turboquant-py/src/turboquant/turboquant.py` — Classe principal
3. `repos/turboquant-pytorch/turboquant.py` — Implementação alternativa
4. `repos/turboquant-pytorch/lloyd_max.py` — Lloyd-Max alternativo

### Fase 2: Entender V3 (o que funciona na prática)
5. `repos/turboquant-pytorch/compressors_v3.py` — MSE-only + asymmetric + bit-packing
6. `repos/rotorquant/turboquant/turboquant.py` — Referência RotorQuant
7. `repos/rotorquant/turboquant/lloyd_max.py` — Lloyd-Max RotorQuant

### Fase 3: Estudar Alternativas de Rotação
8. `repos/rotorquant/turboquant/planarquant.py` — Givens 2D (mais rápido)
9. `repos/rotorquant/turboquant/isoquant.py` — Quaternion 4D
10. `repos/rotorquant/turboquant/clifford.py` — Clifford algebra

### Fase 4: QJL (se necessário)
11. `repos/QJL/qjl_kernel/qjl_kernel.py` — Python wrapper
12. `repos/turboquant-py/src/turboquant/qjl.py` — Implementação Python pura
13. `repos/QJL/qjl_kernel/csrc/qjl_quant_kernel.cu` — CUDA kernel

### Fase 5: Integração com Modelos
14. `repos/QJL/models/llama3_qjl.py` — Como integrar com transformer
15. `repos/rotorquant/turboquant/compressors.py` — KV cache compression
