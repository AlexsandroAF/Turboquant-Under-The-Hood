# Knowledge Base — Reading Guide

We collected everything publicly available about TurboQuant before writing a single line of code. This knowledge base contains 33 documents from academic papers, blog posts, implementation guides, community discussions, and critical analyses.

Here's how to navigate it depending on what you're looking for.

---

## "I want to understand the algorithm" (start here)

Read these in order:

1. **[papers/02_turboquant_paper_FULL.md](papers/02_turboquant_paper_FULL.md)** — The complete paper with all algorithms, theorems, and proofs extracted from the arXiv HTML. This is the primary source. Algorithms 1 and 2 are the ones you need to implement.

2. **[articles/09_deep_dive_illustrated.md](articles/09_deep_dive_illustrated.md)** — The best walkthrough we found. Includes worked numerical examples showing exactly what happens when you rotate a vector with outliers, why naive quantization fails, and how QJL corrects bias. Start here if the paper feels dense.

3. **[articles/15_interactive_visualization.md](articles/15_interactive_visualization.md)** — Description of an interactive 2D/3D animation that visualizes each step. The actual visualization lives at [mesuvash.github.io](https://mesuvash.github.io/blog/2026/turboquant-interactive/). Worth visiting to build intuition.

4. **[articles/11_baseten_math_deep_dive.md](articles/11_baseten_math_deep_dive.md)** — 31 hours of math analysis covering the polar coordinate transformation in detail. Good if you want to understand *why* the Beta distribution emerges after rotation.

## "I want to implement it"

5. **[code-examples/05_turboquant_py_library.md](code-examples/05_turboquant_py_library.md)** — The cleanest API reference. Documents the `turboquant-py` PyPI package with full class signatures, storage format, and benchmarks. Best starting point for writing your own.

6. **[code-examples/01_turboquant_pytorch.md](code-examples/01_turboquant_pytorch.md)** — From-scratch PyTorch implementation. Documents the crucial V3 finding: QJL hurts in practice, MSE-only is better, and asymmetric K/V bits matter.

7. **[articles/06_dejan_paper_to_triton_kernel.md](articles/06_dejan_paper_to_triton_kernel.md)** — Step-by-step implementation of a Triton GPU kernel. Documents five mistakes the author made and how to fix them. Invaluable if you're writing custom kernels.

8. **[MAPA_REFERENCIA.md](MAPA_REFERENCIA.md)** — Maps every source file in the 4 reference repositories we cloned, ranked by priority for reimplementation.

## "I want to know what actually works in practice"

9. **[articles/10_one_month_later.md](articles/10_one_month_later.md)** — "One month later" retrospective covering which implementations delivered, what QJL's real-world status is, and the RaBitQ controversy.

10. **[articles/13_community_two_weeks.md](articles/13_community_two_weeks.md)** — Five community implementations analyzed with a comparison matrix (language, maturity, target hardware).

11. **[discussions/01_llama_cpp_discussion.md](discussions/01_llama_cpp_discussion.md)** — The llama.cpp community discussion (#20969). Contains hard-won practical findings: WHT beats random rotation, block-size tuning matters, norm disparity between K and V is real.

## "I want to understand the controversy"

12. **[articles/07_rabitq_controversy.md](articles/07_rabitq_controversy.md)** — The RaBitQ team's allegations against TurboQuant: methodological similarity omitted, theory misrepresented, unfair benchmarks.

13. **[articles/17_openreview_controversy.md](articles/17_openreview_controversy.md)** — The OpenReview discussion, ICLR reviewer comments, and Google's response.

## "I want alternatives to TurboQuant"

14. **[articles/16_rotorquant_alternative.md](articles/16_rotorquant_alternative.md)** — RotorQuant uses Clifford algebra block-diagonal rotations instead of full WHT. Claims better PPL, 28% faster decode, 44x fewer parameters. Includes PlanarQuant (Givens 2D) which is the fastest variant.

## Component papers

15. **[papers/03_polarquant_paper.md](papers/03_polarquant_paper.md)** — PolarQuant (Stage 1 of TurboQuant). Recursive polar coordinate transformation.

16. **[papers/04_qjl_paper.md](papers/04_qjl_paper.md)** — QJL (Stage 2). 1-bit Johnson-Lindenstrauss correction. Includes the official CUDA kernel code repository.

17. **[papers/05_polarquant_v2_table_lookup.md](papers/05_polarquant_v2_table_lookup.md)** — A different PolarQuant paper using table lookup for query-key inner product instead of matrix multiply.

18. **[prior-art/rabitq_sigmod2024.md](prior-art/rabitq_sigmod2024.md)** — RaBitQ summary and its relationship to TurboQuant.

---

## Full File List

### Papers (5)
| File | Paper | Venue |
|------|-------|-------|
| [01_turboquant_paper_arxiv.md](papers/01_turboquant_paper_arxiv.md) | TurboQuant (summary) | ICLR 2026 |
| [02_turboquant_paper_FULL.md](papers/02_turboquant_paper_FULL.md) | TurboQuant (complete) | ICLR 2026 |
| [03_polarquant_paper.md](papers/03_polarquant_paper.md) | PolarQuant | AISTATS 2026 |
| [04_qjl_paper.md](papers/04_qjl_paper.md) | QJL | AAAI 2025 |
| [05_polarquant_v2_table_lookup.md](papers/05_polarquant_v2_table_lookup.md) | PolarQuant v2 | 2025 |

### Articles (17)
| # | Source | Focus |
|---|--------|-------|
| 01 | DEV Community | Developer guide with code |
| 02 | Hackaday | Critical analysis vs NVFP4 |
| 03 | Analytics Vidhya | Didactic explanation |
| 04 | MindStudio | Market impact, stock drops |
| 05 | Medium | TurboQuant vs traditional quantization |
| 06 | DEJAN AI | Triton kernel implementation |
| 07 | DEV Community | RaBitQ controversy |
| 08 | TechCrunch | Mainstream "Pied Piper" coverage |
| 09 | darshanfofadiya.com | Illustrated deep dive with examples |
| 10 | frr.dev | One month later retrospective |
| 11 | Baseten | 31 hours of math analysis |
| 12 | DEV Community | vLLM plugin in 72 hours |
| 13 | DEV Community | Community implementations in 2 weeks |
| 14 | Decode The Future | Technical explainer |
| 15 | mesuvash.github.io | Interactive 2D/3D visualization |
| 16 | scrya-com/rotorquant | RotorQuant alternative |
| 17 | OpenReview/X | ICLR review controversy |

### Blog Posts (3)
| File | Source |
|------|--------|
| [01_google_research_blog.md](blog-posts/01_google_research_blog.md) | Official Google Research |
| [02_turboquant_net_analysis.md](blog-posts/02_turboquant_net_analysis.md) | Independent analysis with pseudocode |
| [03_turbo_quant_com_resources.md](blog-posts/03_turbo_quant_com_resources.md) | Resource compilation |

### Code Examples (6)
| File | Language | Platform |
|------|----------|----------|
| [01_turboquant_pytorch.md](code-examples/01_turboquant_pytorch.md) | Python/PyTorch | NVIDIA CUDA |
| [02_turboquant_triton_vllm.md](code-examples/02_turboquant_triton_vllm.md) | Python/Triton | vLLM production |
| [03_turboquant_mlx_apple.md](code-examples/03_turboquant_mlx_apple.md) | Python/MLX | Apple Silicon |
| [04_turboquant_rust.md](code-examples/04_turboquant_rust.md) | Rust | CPU + WGPU |
| [05_turboquant_py_library.md](code-examples/05_turboquant_py_library.md) | Python | Complete API reference |
| [06_github_ecosystem.md](code-examples/06_github_ecosystem.md) | Multi | 20+ repos cataloged |
