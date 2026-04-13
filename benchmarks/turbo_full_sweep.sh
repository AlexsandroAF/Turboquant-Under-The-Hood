#!/bin/bash
# Sweep completo: bench + PPL para um modelo especifico
# Uso: ./turbo_full_sweep.sh <model_name> <model_path> <ngl>
#   ngl=99 para full GPU, valor menor para offload parcial
set -e

MODEL_NAME="$1"
MODEL_PATH="$2"
NGL="${3:-99}"

export LD_LIBRARY_PATH=/work/llama-cpp-turboquant/build/bin
BENCH=/work/llama-cpp-turboquant/build/bin/llama-bench
PPL=/work/llama-cpp-turboquant/build/bin/llama-perplexity
CORPUS=/work/data/wiki.test.raw
OUT="/work/results_turbo/$MODEL_NAME"
mkdir -p "$OUT"

CONFIGS=(
    "baseline_f16:f16:f16"
    "q8_0:q8_0:q8_0"
    "q4_0:q4_0:q4_0"
    "turbo4:turbo4:turbo4"
    "turbo3:turbo3:turbo3"
    "turbo2:turbo2:turbo2"
)

echo "==============================================="
echo "MODEL: $MODEL_NAME"
echo "PATH:  $MODEL_PATH"
echo "NGL:   $NGL"
echo "==============================================="

echo ""
echo "##### BENCH SWEEP #####"
for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r name k v <<< "$cfg"
    echo ""
    echo "=== $MODEL_NAME / bench $name (K=$k V=$v) ==="
    LOG="$OUT/bench_${name}.log"
    $BENCH -m "$MODEL_PATH" -p 512 -n 128 \
        -ctk "$k" -ctv "$v" -fa 1 -ngl "$NGL" -r 2 \
        2>&1 | tee "$LOG" | tail -4
done

echo ""
echo "##### PPL SWEEP #####"
for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r name k v <<< "$cfg"
    echo ""
    echo "=== $MODEL_NAME / ppl $name (K=$k V=$v) ==="
    LOG="$OUT/ppl_${name}.log"
    $PPL -m "$MODEL_PATH" -f "$CORPUS" \
        --ctx-size 512 --chunks 20 \
        -ctk "$k" -ctv "$v" \
        -fa 1 -ngl "$NGL" \
        2>&1 | tee "$LOG" | grep -E "Final estimate" | tail -1
done

echo ""
echo "##### SUMMARY: $MODEL_NAME #####"
for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r name k v <<< "$cfg"
    ppl=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$OUT/ppl_${name}.log" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    pp=$(grep 'pp512' "$OUT/bench_${name}.log" 2>/dev/null | awk -F'|' '{print $(NF-1)}' | awk '{print $1}' | head -1)
    tg=$(grep 'tg128' "$OUT/bench_${name}.log" 2>/dev/null | awk -F'|' '{print $(NF-1)}' | awk '{print $1}' | head -1)
    kv=$(grep 'CUDA0' "$OUT/ppl_${name}.log" 2>/dev/null | grep -oE '\+[ ]+[0-9]+[ ]+\+' | head -1 | grep -oE '[0-9]+')
    printf "%-15s pp=%8s tg=%6s PPL=%9s KV=%4s MiB\n" "$name" "$pp" "$tg" "$ppl" "$kv"
done
