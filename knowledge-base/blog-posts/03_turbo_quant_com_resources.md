# TurboQuant Paper, arXiv & GitHub — Research Resources

> **Fonte**: https://turbo-quant.com/turboquant-paper
> **Plataforma**: turbo-quant.com

## Paper Details

- **Título**: TurboQuant: Online Vector Quantization with Near-Optimal Distortion
- **Autores**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research)
- **arXiv**: 2504.19874
- **Status**: ICLR 2026
- **Blog Google**: March 24, 2026

## Experimental Findings

| Benchmark | Result |
|-----------|--------|
| LongBench (Llama-3.1-8B) | 50.06 at 3.5-bit = FP16 baseline; KIVI at 3-bit: 48.50 |
| Needle-in-a-Haystack | 0.997 at 4× compression (FP16 match); SnapKV: 0.858 |
| Quality Parity | Full accuracy at ~3.5 bits/channel |
| Attention Speedup | Up to 8× on H100 in 4-bit |
| Vector Search (GloVe d=200) | Outperforms PQ and RaBitQ on recall@k |

## Component Papers

- **PolarQuant**: arXiv 2502.02617 (AISTATS 2026)
- **QJL**: AAAI 2025 (DOI: 10.1609/aaai.v39i24.34773)

## Community Discussions

- Hacker News (166+ comments): https://news.ycombinator.com/item?id=47513475
- vLLM forum: https://discuss.vllm.ai/t/turboquant-kv-cache-compression/2503
- Reddit r/LocalLLaMA: https://www.reddit.com/r/LocalLLaMA/search/?q=turboquant
- X/Twitter: https://x.com/search?q=turboquant&f=top

## Citation

```bibtex
@article{turboquant2025,
  title={TurboQuant: Online Vector Quantization with Near-Optimal Distortion},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

## FAQ

- **Training required?** No — data-oblivious and online
- **Which models?** Any transformer; benchmarked on Gemma, Mistral, Llama-3.1-8B
- **vs KIVI?** Better quality at 3.5 bits without calibration
- **Zero-loss?** At 3.5 bits on long-context benchmarks, yes; at 2.5 bits, small drops
- **Independently reproduced?** Not widely published as of April 2026
