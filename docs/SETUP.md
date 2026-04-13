# Setup and Reproduction Guide

## Prerequisites

- Python 3.10+ with pip
- NVIDIA GPU with CUDA support (tested on RTX 3050 Laptop, 4 GB VRAM)
- Docker Desktop with NVIDIA Container Runtime (for C++/CUDA benchmarks)
- ~15 GB disk space (models + Docker images)

## Python Prototype

### Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate numpy
```

### Validate the algorithm

```bash
cd src/
python test_synthetic.py
```

This runs TurboQuant's Lloyd-Max codebook + random rotation against the paper's theoretical distortion bounds. No GPU required (runs on CPU too), takes ~10 seconds.

### Run PPL benchmark

```bash
cd benchmarks/
python bench_ppl_v3.py
```

Downloads Qwen2.5-0.5B-Instruct (~1 GB) on first run. Compares:
- A: Baseline FP16
- B: V1 3-bit uniform (broken, for reference)
- C: V1 4-bit uniform
- D: V3 3.5-bit with patches (outlier split + deferred + no-requ)
- E: V3 3.5-bit without deferred
- F: V3 3.5-bit K+V with deferred

Expected runtime: ~3-5 minutes on CUDA GPU.

## C++/CUDA TurboQuant (llama.cpp fork)

### Prerequisites

- Docker Desktop running
- NVIDIA GPU visible to Docker (`docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`)

### Download model

```bash
mkdir -p models/
curl -L -o models/qwen2.5-3b-instruct-q4_k_m.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
```

### Build from source

```bash
# Clone the TurboQuant fork
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
git fetch origin feature/turboquant-kv-cache
git checkout FETCH_HEAD

# Build in Docker container
docker run --rm --gpus all \
  -v "$(pwd)/..:/work" \
  nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  bash /work/benchmarks/docker_build.sh
```

Build time: ~60-90 minutes (198 CUDA files).

### Run benchmark

```bash
# Start persistent container
docker run -d --name tqbench --gpus all \
  -v "$(pwd)/..:/work" \
  nvidia/cuda:12.4.1-devel-ubuntu22.04 sleep 86400

# Run sweep
docker exec tqbench bash /work/benchmarks/turbo_full_sweep.sh \
  "qwen2.5-3b" "/work/models/qwen2.5-3b-instruct-q4_k_m.gguf" 99

# Cleanup
docker stop tqbench && docker rm tqbench
```

### Available KV cache types in the fork

| Type | Bits | Algorithm |
|------|------|-----------|
| f16 | 16 | Baseline (no compression) |
| q8_0 | 8 | Naive symmetric scalar |
| q4_0 | 4 | Naive symmetric scalar |
| turbo4 | 4 | TurboQuant (rotation + Lloyd-Max) |
| turbo3 | 3 | TurboQuant |
| turbo2 | 2 | TurboQuant |

## Troubleshooting

**"CUDA out of memory"**: Reduce `-ngl` (GPU layers) or use a smaller model. RTX 3050 fits models up to ~3B fully, 8B needs partial offload (`-ngl 18`).

**Docker GPU not visible**: Ensure Docker Desktop has "Use the WSL 2 based engine" enabled and NVIDIA Container Toolkit is installed. Test with `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.

**CUDA Toolkit install fails on Windows**: Don't bother. Use Docker instead (Section above). Windows CUDA installer has driver constraint issues that are hard to debug.

**NaN in PPL benchmarks**: You're re-quantizing already-quantized tokens. Use `tq_cache_v3.py` (V3), not `tq_cache.py` (V1). V3 tracks quantization boundaries per layer and never re-quantizes.
