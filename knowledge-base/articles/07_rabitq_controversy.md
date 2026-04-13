# TurboQuant and RaBitQ: What the Public Story Gets Wrong

> **Fonte**: https://dev.to/gaoj0017/turboquant-and-rabitq-what-the-public-story-gets-wrong-1i00
> **Autor**: Jianyang Gao (ETH Zurich, first author of RaBitQ)
> **Plataforma**: DEV Community

## Background: RaBitQ

RaBitQ emerged from Gao's PhD research at Nanyang Technological University:
- **Original paper** (arXiv:2405.12497, May 2024) → SIGMOD 2024
- **Extended version** (arXiv:2409.09913, September 2024) → SIGMOD 2025

The method applies random rotation (Johnson-Lindenstrauss transforms) before quantization and proves it achieves asymptotically optimal error bounds (Alon-Klartag, FOCS 2017).

## Three Core Problems with TurboQuant Paper

### Problem 1: Methodological Similarity Omitted

Both RaBitQ and TurboQuant employ random rotation before quantization—fundamental structural overlap. Yet TurboQuant never explicitly states this connection.

In January 2025, TurboQuant's second author Majid Daliri contacted Gao requesting help debugging his Python implementation of RaBitQ, demonstrating detailed technical knowledge. Despite this, subsequent submissions mischaracterized RaBitQ as "grid-based PQ."

### Problem 2: False Claims About Theory

TurboQuant asserts RaBitQ's guarantees are "suboptimal, likely due to loose analysis." However, Theorem 3.2 in the extended RaBitQ paper rigorously proves optimal bounds per Alon-Klartag. The work was invited to present at a FOCS-affiliated workshop.

Gao's team provided detailed explanations of why the interpretation was incorrect. Daliri acknowledged communicating this to all co-authors. The incorrect characterization persisted.

### Problem 3: Deliberately Unfair Experiments

TurboQuant tested RaBitQ using a degraded implementation and single-core CPU with multithreading disabled, while testing TurboQuant on NVIDIA A100 GPU. The reported speed gap was several orders of magnitude.

Daliri's email: "we were using a single-core CPU instance, and multiprocessing was indeed disabled...we weren't fully utilizing parallelism."

## Timeline

| Date | Event |
|------|-------|
| May 2024 | RaBitQ posted with open-source code |
| Sep 2024 | Extended RaBitQ version released |
| Jan 2025 | Daliri contacts Gao requesting debugging assistance |
| Apr 2025 | TurboQuant appears on arXiv |
| May 2025 | Gao explains theoretical errors; Daliri ceases communication |
| Nov 2025 | RaBitQ team discovers ICLR submission; contacts PC chairs |
| Jan 2026 | TurboQuant accepted to ICLR 2026 |
| Mar 2026 | Google promotes paper (tens of millions of views) |
| Current | Partial acknowledgment received; no full correction |

## Actions Taken

- Posted public comment on ICLR OpenReview
- Submitted formal complaints to ICLR General Chairs, PC Chairs, and Code and Ethics Chairs
- Plan to release detailed technical report on arXiv

## Importance for Our Knowledge Base

This controversy is critical context for understanding TurboQuant's relationship to prior work and evaluating the paper's claims about novelty and theoretical contributions.
