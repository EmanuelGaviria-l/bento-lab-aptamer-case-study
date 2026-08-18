#!/usr/bin/env bash
# Inject AF3 coords into Boltz pre_affinity npz, then score with the
# Boltz affinity module (no diffusion). Resumable: skips jobs that
# already have affinity_*.json.
#
# Usage (Boltz venv, GPU node, from repo root):
#   nohup bash scripts/task6/run_all_af3_affinity.sh > docs/af3_boltz_affinity_log.txt 2>&1 &

set -uo pipefail

PROJECT_ROOT="/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"
AF3_DIR="$PROJECT_ROOT/results/af3_msa"
OUT_PRED="$PROJECT_ROOT/results/af3_boltz_affinity/predictions"

cd "$PROJECT_ROOT"

echo "=== Inject AF3 coordinates ==="
python3 scripts/task6/inject_af3_coords.py

echo ""
echo "=== Run Boltz affinity module on AF3 poses ==="

total=0
skipped=0
run=0
failed=0

for cif in "$AF3_DIR"/*/*_model.cif; do
    [ -e "$cif" ] || continue
    job=$(basename "$(dirname "$cif")")
    total=$((total + 1))

    if [ -f "$OUT_PRED/$job/affinity_$job.json" ]; then
        echo "[SKIP] $job"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[RUN]  $job  ($(date '+%Y-%m-%d %H:%M:%S'))"
    start_ts=$(date +%s)
    if python3 scripts/task6/run_affinity_from_af3.py --job "$job"; then
        elapsed=$(( $(date +%s) - start_ts ))
        echo "[DONE] $job in ${elapsed}s"
        run=$((run + 1))
    else
        elapsed=$(( $(date +%s) - start_ts ))
        echo "[FAIL] $job after ${elapsed}s"
        failed=$((failed + 1))
    fi
done

echo ""
echo "===== Batch summary ====="
echo "Total:   $total"
echo "Skip:    $skipped"
echo "Run:     $run"
echo "Failed:  $failed"
