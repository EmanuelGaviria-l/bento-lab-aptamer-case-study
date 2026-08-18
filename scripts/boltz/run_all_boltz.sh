#!/usr/bin/env bash
# Run Boltz-2 (structure + affinity) over every YAML input in inputs/boltz/.
#
# Resumable: boltz predict itself skips inputs whose "processed" cache
# already exists (see "All inputs are already processed" behavior seen
# during testing), so simply re-running this script after an interrupted
# session is safe.
#
# Usage (from the repo root, with the Boltz-2 venv active):
#   nohup bash scripts/boltz/run_all_boltz.sh > docs/boltz_batch_log.txt 2>&1 &
#
# Check progress while it runs:
#   tail -f docs/boltz_batch_log.txt
#   find results/boltz -iname "affinity_*.json" | wc -l

set -uo pipefail  # NOTE: no -e -- one job failing must not kill the batch

PROJECT_ROOT="/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"
INPUT_DIR="$PROJECT_ROOT/inputs/boltz"
OUTPUT_DIR="$PROJECT_ROOT/results/boltz"

mkdir -p "$OUTPUT_DIR"

total=0
run=0
failed=0

for yaml_path in "$INPUT_DIR"/*.yaml; do
    job_name=$(basename "$yaml_path" .yaml)
    total=$((total + 1))

    # If the affinity output already exists for this job, skip it.
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
