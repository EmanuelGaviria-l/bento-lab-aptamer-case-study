#!/usr/bin/env python3
"""
Collect Boltz-2 affinity predictions for all 73 aptamers and build a
ranked results spreadsheet, joined with the experimental Kd values.

Reads:
  data/aptamer_subset.csv
  results/boltz/boltz_results_<job>/predictions/<job>/affinity_<job>.json
    (one per aptamer)

Writes:
  report/boltz_affinity_ranking.csv

Columns:
  serial, aptamer_name, target, nucleic_type, kd_nm,
  affinity_pred_value (log10 IC50 uM -- primary ranking column, ascending
      = predicted tightest binders first),
  affinity_probability_binary (secondary reference column: predicted
      probability the aptamer is a binder at all)

Run from the bento-lab-aptamer-case-study folder.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", text.strip())
    return text.strip("_") or "aptamer"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Boltz-2 affinity predictions.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Boltz results directory (default: results/boltz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: report/boltz_affinity_ranking.csv)",
    )
    parser.add_argument(
        "--layout",
        choices=("boltz", "flat"),
        default="boltz",
        help=(
            "boltz: results/boltz_results_<job>/predictions/<job>/affinity_<job>.json; "
            "flat: results/<job>/affinity_<job>.json (Task 6 AF3-wired output)"
        ),
    )
    args = parser.parse_args()

    project_root = Path("/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study")
    csv_path = project_root / "data" / "aptamer_subset.csv"
    results_dir = args.results_dir or (project_root / "results" / "boltz")
    output_path = args.output or (project_root / "report" / "boltz_affinity_ranking.csv")

    if not csv_path.exists():
        sys.exit(f"Missing file: {csv_path}")
    if not results_dir.exists():
        sys.exit(f"Missing results directory: {results_dir}")

    df = pd.read_csv(csv_path)

    rows = []
    missing = []

    for _, row in df.iterrows():
        serial = str(row["Serial Number"])
        aptamer_name = str(row["Name of Aptamer"])
        job_name = f"{serial}_{safe_filename(aptamer_name)}"

        if args.layout == "flat":
            affinity_json = results_dir / job_name / f"affinity_{job_name}.json"
        else:
            affinity_json = (
                results_dir
                / f"boltz_results_{job_name}"
                / "predictions"
                / job_name
                / f"affinity_{job_name}.json"
            )

        if not affinity_json.exists():
            missing.append(job_name)
            continue

        with affinity_json.open() as f:
            affinity_data = json.load(f)

        rows.append(
            {
                "serial": serial,
                "aptamer_name": aptamer_name,
                "target": row["Target"],
                "nucleic_type": row["Type of Nucleic Acid"],
                "kd_nm": row["Kd (nM)"],
                "affinity_pred_value": affinity_data["affinity_pred_value"],
                "affinity_probability_binary": affinity_data[
                    "affinity_probability_binary"
                ],
            }
        )

    if missing:
        print(f"WARNING: {len(missing)} job(s) missing affinity output:")
        for job_name in missing:
            print(f"  - {job_name}")

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("affinity_pred_value", ascending=True)
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nWrote {len(results_df)} ranked rows to {output_path}")
    print("\nTop 5 predicted tightest binders (lowest affinity_pred_value):")
    print(
        results_df[
            ["rank", "aptamer_name", "target", "affinity_pred_value", "kd_nm"]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
