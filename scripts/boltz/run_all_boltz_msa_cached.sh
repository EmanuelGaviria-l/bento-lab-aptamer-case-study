#!/usr/bin/env bash
# Run Boltz-2 on inputs/boltz_msa/ (cached MSAs -- no ColabFold calls at predict time).
#
# Usage (Boltz venv active, from repo root):
#   nohup bash scripts/boltz/run_all_boltz_msa_cached.sh > docs/boltz_msa_batch_log.txt 2>&1 &

set -uo pipefail

PROJECT_ROOT="/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"
INPUT_DIR="$PROJECT_ROOT/inputs/boltz_msa"
OUTPUT_DIR="$PROJECT_ROOT/results/boltz_msa"

mkdir -p "$OUTPUT_DIR"

total=0
run=0
failed=0

for yaml_path in "$INPUT_DIR"/*.yaml; do
    [ -e "$yaml_path" ] || { echo "No YAML files in $INPUT_DIR"; exit 1; }

    job_name=$(basename "$yaml_path" .yaml)
    total=$((total + 1))

    existing=$(find "$OUTPUT_DIR" -path "*${job_name}*/affinity_${job_name}.json" 2>/dev/null | head -1)
    if [ -n "$existing" ]; then
        echo "[SKIP] $job_name (affinity output already exists)"
        continue
    fi

    echo "[RUN]  $job_name  ($(date '+%Y-%m-%d %H:%M:%S'))"
    start_ts=$(date +%s)

    boltz predict "$yaml_path" \
        --no_kernels \
        --out_dir "$OUTPUT_DIR"

    status=$?
    elapsed=$(( $(date +%s) - start_ts ))

    if [ $status -eq 0 ]; then
        echo "[DONE] $job_name in ${elapsed}s"
        run=$((run + 1))
    else
        echo "[FAIL] $job_name after ${elapsed}s (exit code $status)"
        failed=$((failed + 1))
    fi
done

echo ""
echo "===== Batch summary ====="
echo "Total inputs:     $total"
echo "Run this session: $run"
echo "Failed:           $failed"
