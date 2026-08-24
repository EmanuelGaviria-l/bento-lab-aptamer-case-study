#!/usr/bin/env bash
# Boltz-2 v2 expansion: inputs/boltz_msa_v2 -> results/boltz_msa_v2
# Smallest proteins first. Skips jobs that already have affinity JSON in
# either the original results/boltz_msa/ (first 73) or this v2 output dir.
#
# Usage (Boltz venv active, from repo root, on a GPU node — not the login node):
#   nohup bash scripts/boltz/run_all_boltz_msa_v2.sh > docs/boltz_msa_v2_batch_log.txt 2>&1 &

set -uo pipefail

PROJECT_ROOT="/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"
INPUT_DIR="$PROJECT_ROOT/inputs/boltz_msa_v2"
OUTPUT_DIR="$PROJECT_ROOT/results/boltz_msa_v2"
ORIG_DIR="$PROJECT_ROOT/results/boltz_msa"

mkdir -p "$OUTPUT_DIR"

if ! ls "$INPUT_DIR"/*.yaml >/dev/null 2>&1; then
    echo "No YAML files in $INPUT_DIR"
    exit 1
fi

mapfile -t yaml_paths < <(
    python3 - "$INPUT_DIR" << 'PY'
import sys
from pathlib import Path
import yaml

input_dir = Path(sys.argv[1])

def protein_len(path: Path) -> int:
    data = yaml.safe_load(path.read_text())
    total = 0
    for item in data.get("sequences", []):
        protein = item.get("protein")
        if protein and "sequence" in protein:
            total += len(protein["sequence"])
    return total

paths = sorted(input_dir.glob("*.yaml"), key=lambda p: (protein_len(p), p.name))
for path in paths:
    print(path)
PY
)

total=0
run=0
failed=0
skipped=0

for yaml_path in "${yaml_paths[@]}"; do
    job_name=$(basename "$yaml_path" .yaml)
    total=$((total + 1))

    existing=$(find "$OUTPUT_DIR" "$ORIG_DIR" -path "*${job_name}*/affinity_${job_name}.json" 2>/dev/null | head -1)
    if [ -n "$existing" ]; then
        echo "[SKIP] $job_name (affinity output already exists)"
        skipped=$((skipped + 1))
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
echo "Skipped existing: $skipped"
echo "Run this session: $run"
echo "Failed:           $failed"
