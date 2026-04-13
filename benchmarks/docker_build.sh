#!/bin/bash
# Build llama-cpp-turboquant dentro do container CUDA 12.4 devel
# Mount: prototype/ = /work
set -e

cd /work/llama-cpp-turboquant

echo "=== Branch / commit ==="
git log --oneline -1 || true

echo "=== Installing build deps ==="
apt-get update -qq
apt-get install -y -qq cmake build-essential git ccache curl libcurl4-openssl-dev

echo "=== CMake configure ==="
mkdir -p build
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=ON \
    2>&1 | tail -30

echo "=== Building (this may take 20-40 min) ==="
cmake --build build \
    --config Release \
    -j$(nproc) \
    --target llama-bench llama-cli llama-perplexity llama-server 2>&1 | tail -50

echo "=== Build done. Binaries: ==="
ls -la build/bin/ 2>&1 | head -30
