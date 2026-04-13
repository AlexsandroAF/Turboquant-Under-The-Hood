#!/bin/bash
# Sweep definitivo: baseline vs TurboQuant real vs quantization tradicional
set -e
export LD_LIBRARY_PATH=/work/llama-cpp-turboquant/build/bin
BENCH=/work/llama-cpp-turboquant/build/bin/llama-bench
MODEL=/work/models/qwen2.5-3b-instruct-q4_k_m.gguf
OUT=/work/results_turbo
mkdir -p "$OUT"

CSV="$OUT/bench_turbo.csv"
echo "config,k_type,v_type,test,tok_per_s,std" > "$CSV"

# 7 configs principais
CONFIGS=(
    "baseline_f16:f16:f16"
    "q8_0:q8_0:q8_0"
    "q4_0:q4_0:q4_0"
    "turbo2:turbo2:turbo2"
    "turbo3:turbo3:turbo3"
    "turbo4:turbo4:turbo4"
    "turbo4_k_q8_v:turbo4:q8_0"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r name k v <<< "$cfg"
    echo ""
    echo "=== $name (K=$k V=$v) ==="
    LOG="$OUT/bench_${name}.log"
    $BENCH -m "$MODEL" -p 512 -n 128 -ctk "$k" -ctv "$v" -fa 1 -ngl 99 -r 2 -o md 2>&1 | tee "$LOG" | tail -6
    # Parse
    grep -E "pp512|tg128" "$LOG" 2>/dev/null | while read -r line; do
        test=$(echo "$line" | grep -oE "pp512|tg128")
        tok=$(echo "$line" | awk -F'|' '{print $(NF-1)}' | xargs)
        main=$(echo "$tok" | awk '{print $1}')
        std=$(echo "$tok" | awk '{print $3}')
        echo "$name,$k,$v,$test,$main,$std" >> "$CSV"
    done
done

echo ""
echo "=== SUMMARY CSV ==="
cat "$CSV"
