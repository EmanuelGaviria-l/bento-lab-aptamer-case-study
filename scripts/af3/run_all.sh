#!/usr/bin/env bash
# Run AlphaFold 3 over every JSON input in inputs/af3/.
#
# Resumable: if a job's output directory already exists (i.e. it already
# completed, or at least started previously), it is skipped. Safe to
# re-run after an interrupted session -- just re-launch the same command.
#
# Usage (from the repo root, with the AF3 venv active):
#   nohup bash scripts/af3/run_all.sh > docs/af3_batch_log.txt 2>&1 &
#
# Check progress while it runs:
#   tail -f docs/af3_batch_log.txt
#   ls results/af3 | wc -l          # how many completed so far

set -uo pipefail  # NOTE: no -e -- one job failing must not kill the batch

PROJECT_ROOT="/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"
INPUT_DIR="$PROJECT_ROOT/inputs/af3"
OUTPUT_DIR="$PROJECT_ROOT/results/af3"
MODEL_DIR="$HOME/alphafold3"
AF3_SCRIPT="$HOME/alphafold3/run_alphafold.py"

mkdir -p "$OUTPUT_DIR"

total=0
skipped=0
run=0
failed=0

for json_path in "$INPUT_DIR"/*.json; do
    job_name=$(basename "$json_path" .json)
    total=$((total + 1))

    if [ -d "$OUTPUT_DIR/$job_name" ]; then
        echo "[SKIP] $job_name (output already exists)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[RUN]  $job_name  ($(date '+%Y-%m-%d %H:%M:%S'))"
    start_ts=$(date +%s)

    python3 "$AF3_SCRIPT" \
        --json_path="$json_path" \
        --model_dir="$MODEL_DIR" \
        --output_dir="$OUTPUT_DIR" \
        --norun_data_pipeline

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
echo "Already done:     $skipped"
echo "Run this session: $run"
echo "Failed:           $failed"
