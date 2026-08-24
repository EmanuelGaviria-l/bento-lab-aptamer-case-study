#!/usr/bin/env bash
# Run AlphaFold 3 on inputs/af3_msa_v2/ (v2 expansion).
# Skips jobs that already have a .cif in results/af3_msa (original 73)
# or results/af3_msa_v2.
#
# Usage (GPU node, not login):
#   nohup bash scripts/af3/run_all_msa_v2.sh > docs/af3_msa_v2_batch_log.txt 2>&1 &

set -uo pipefail

PROJECT_ROOT="/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"
INPUT_DIR="$PROJECT_ROOT/inputs/af3_msa_v2"
OUTPUT_DIR="$PROJECT_ROOT/results/af3_msa_v2"
ORIG_DIR="$PROJECT_ROOT/results/af3_msa"
AF3_ROOT="$HOME/alphafold3"
MODEL_DIR="$AF3_ROOT"
AF3_PYTHON="$AF3_ROOT/.venv/bin/python"
AF3_SCRIPT="$AF3_ROOT/run_alphafold.py"

if [ ! -x "$AF3_PYTHON" ]; then
    echo "AF3 python not found at $AF3_PYTHON"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/docs"

echo "Using python: $AF3_PYTHON"
echo "AF3 cwd:      $AF3_ROOT"
echo "Inputs:       $INPUT_DIR"
echo "Outputs:      $OUTPUT_DIR"
echo ""

total=0
skipped=0
run=0
failed=0

job_has_cif() {
    local job_out="$1"
    [ -d "$job_out" ] || return 1
    find "$job_out" -name '*.cif' 2>/dev/null | grep -q .
}

for json_path in "$INPUT_DIR"/*.json; do
    [ -e "$json_path" ] || { echo "No JSON files in $INPUT_DIR"; exit 1; }

    job_name=$(basename "$json_path" .json)
    job_out="$OUTPUT_DIR/$job_name"
    orig_out="$ORIG_DIR/$job_name"
    total=$((total + 1))

    if job_has_cif "$job_out" || job_has_cif "$orig_out"; then
        echo "[SKIP] $job_name (cif already exists)"
        skipped=$((skipped + 1))
        continue
    fi

    if [ -d "$job_out" ]; then
        echo "[CLEAN] $job_name (incomplete output, removing)"
        rm -rf "$job_out"
    fi

    echo "[RUN]  $job_name  ($(date '+%Y-%m-%d %H:%M:%S'))"
    start_ts=$(date +%s)

    (
        cd "$AF3_ROOT" || exit 1
        "$AF3_PYTHON" "$AF3_SCRIPT" \
            --json_path="$json_path" \
            --model_dir="$MODEL_DIR" \
            --output_dir="$OUTPUT_DIR" \
            --norun_data_pipeline
    )

    status=$?
    elapsed=$(( $(date +%s) - start_ts ))

    if [ $status -eq 0 ] && job_has_cif "$job_out"; then
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
