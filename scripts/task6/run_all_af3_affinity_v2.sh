#!/usr/bin/env bash
# Task 6 on the v2 set: AF3 coords -> Boltz affinity head (no diffusion).
# Skips the original 73 if their affinity JSON already exists in
# results/af3_boltz_affinity/. Writes new jobs to
# results/af3_boltz_affinity_v2/. Does not overwrite the first 73.
#
# Usage (Boltz venv, GPU node, from repo root):
#   nohup bash scripts/task6/run_all_af3_affinity_v2.sh > docs/af3_boltz_affinity_v2_log.txt 2>&1 &

set -uo pipefail

PROJECT_ROOT="/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"
CSV="$PROJECT_ROOT/data/aptamer_subset_v2.csv"
OUT_PRED="$PROJECT_ROOT/results/af3_boltz_affinity_v2/predictions"
ORIG_PRED="$PROJECT_ROOT/results/af3_boltz_affinity/predictions"

cd "$PROJECT_ROOT"

echo "=== Inject AF3 coordinates (v2; skip jobs that already have affinity JSON) ==="
python3 scripts/task6/inject_af3_coords.py --v2

echo ""
echo "=== Run Boltz affinity module on AF3 poses ==="

total=0
skipped=0
run=0
failed=0

while IFS= read -r job; do
    [ -n "$job" ] || continue
    total=$((total + 1))

    if [ -f "$OUT_PRED/$job/affinity_$job.json" ] || [ -f "$ORIG_PRED/$job/affinity_$job.json" ]; then
        echo "[SKIP] $job"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[RUN]  $job  ($(date '+%Y-%m-%d %H:%M:%S'))"
    start_ts=$(date +%s)
    if python3 scripts/task6/run_affinity_from_af3.py --v2 --job "$job"; then
        elapsed=$(( $(date +%s) - start_ts ))
        echo "[DONE] $job in ${elapsed}s"
        run=$((run + 1))
    else
        elapsed=$(( $(date +%s) - start_ts ))
        echo "[FAIL] $job after ${elapsed}s"
        failed=$((failed + 1))
    fi
done < <(python3 - <<'PY'
import csv, re
from pathlib import Path
csv_path = Path("/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study/data/aptamer_subset_v2.csv")

def safe(text: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", text.strip())
    return text.strip("_") or "aptamer"

with csv_path.open(newline="") as f:
    for row in csv.DictReader(f):
        print(f"{row['Serial Number']}_{safe(row['Name of Aptamer'])}")
PY
)

echo ""
echo "===== Batch summary ====="
echo "Total:   $total"
echo "Skip:    $skipped"
echo "Run:     $run"
echo "Failed:  $failed"
