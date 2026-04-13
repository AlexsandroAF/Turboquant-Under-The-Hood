#!/bin/bash
# PPL sweep para todos os KV cache types
set -e
export LD_LIBRARY_PATH=/work/llama-cpp-turboquant/build/bin
PPL=/work/llama-cpp-turboquant/build/bin/llama-perplexity
MODEL=/work/models/qwen2.5-3b-instruct-q4_k_m.gguf
CORPUS=/work/data/wiki.test.raw
OUT=/work/results_turbo
mkdir -p "$OUT"

CONFIGS=(
    "baseline_f16:f16:f16"
    "q8_0:q8_0:q8_0"
    "q4_0:q4_0:q4_0"
    "turbo2:turbo2:turbo2"
    "turbo3:turbo3:turbo3"
    "turbo4:turbo4:turbo4"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r name k v <<< "$cfg"
    echo ""
    echo "=== PPL $name (K=$k V=$v) ==="
    LOG="$OUT/ppl_${name}.log"
    $PPL -m "$MODEL" -f "$CORPUS" \
        --ctx-size 512 --chunks 20 \
        -ctk "$k" -ctv "$v" \
        -fa 1 -ngl 99 \
        2>&1 | tee "$LOG" | grep -E "Final estimate" | tail -1
done

echo ""
echo "=== ALL PPL RESULTS ==="
grep -h "Final estimate" "$OUT"/ppl_*.log 2>/dev/null
