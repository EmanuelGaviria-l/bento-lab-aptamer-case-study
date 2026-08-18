#!/usr/bin/env python3
"""
Collect AlphaFold 3 confidence scores for all 73 aptamer complexes.

Reads:
  data/aptamer_subset.csv
  results/af3_msa/<job>/  (ranking_scores.csv + summary_confidences.json)

Writes:
  report/af3_msa_scores.csv

ipTM is AF3's interface confidence (higher = AF3 is more sure the chains
contact). That is NOT an affinity prediction like Boltz's affinity_pred_value.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", text.strip())
    return text.strip("_") or "aptamer"


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def load_ranking(job_dir: Path) -> dict | None:
    job_name = job_dir.name
    ranking_csv = first_existing(
        [
            job_dir / f"{job_name}_ranking_scores.csv",
            job_dir / "ranking_scores.csv",
        ]
    )
    if ranking_csv is None:
        matches = [
            p
            for p in job_dir.glob("*ranking_scores.csv")
            if "sample" not in p.name
        ]
        ranking_csv = matches[0] if matches else None
    if ranking_csv is None or not ranking_csv.exists():
        return None

    rows = list(csv.DictReader(ranking_csv.open()))
    if not rows:
        return None

    def score(row: dict) -> float:
        try:
            return float(row.get("ranking_score") or row.get("score") or 0)
        except ValueError:
            return 0.0

    return max(rows, key=score)


def load_summary(job_dir: Path) -> dict:
    job_name = job_dir.name
    chosen = first_existing(
        [
            job_dir / f"{job_name}_summary_confidences.json",
            job_dir / "summary_confidences.json",
        ]
    )
    if chosen is None:
        top_level = [
            p
            for p in job_dir.glob("*summary_confidences.json")
            if p.parent == job_dir
        ]
        chosen = top_level[0] if top_level else None
    if chosen is None:
        return {}
    return json.loads(chosen.read_text())


def job_has_cif(job_dir: Path) -> bool:
    return any(job_dir.rglob("*.cif"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AF3 scores into a CSV.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="AF3 results directory (default: results/af3_msa)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: report/af3_msa_scores.csv)",
    )
    args = parser.parse_args()

    project_root = args.project_root
    csv_path = project_root / "data" / "aptamer_subset.csv"
    results_dir = args.results_dir or (project_root / "results" / "af3_msa")
    output_path = args.output or (project_root / "report" / "af3_msa_scores.csv")

    if not csv_path.exists():
        sys.exit(f"Missing file: {csv_path}")
    if not results_dir.exists():
        sys.exit(f"Missing results directory: {results_dir}")

    with csv_path.open(newline="") as f:
        aptamers = list(csv.DictReader(f))

    rows = []
    missing = []

    for aptamer in aptamers:
        serial = str(aptamer["Serial Number"])
        name = str(aptamer["Name of Aptamer"])
        job_name = f"{serial}_{safe_filename(name)}"
        job_dir = results_dir / job_name

        if not job_dir.is_dir() or not job_has_cif(job_dir):
            missing.append(job_name)
            continue

        ranking = load_ranking(job_dir) or {}
        summary = load_summary(job_dir)

        def pick(*keys: str):
            for key in keys:
                if ranking.get(key) not in (None, ""):
                    return ranking[key]
                if summary.get(key) not in (None, ""):
                    return summary[key]
            return ""

        rows.append(
            {
                "serial": serial,
                "aptamer_name": name,
                "target": aptamer["Target"],
                "nucleic_type": aptamer["Type of Nucleic Acid"],
                "kd_nm": aptamer["Kd (nM)"],
                "ranking_score": pick("ranking_score"),
                "ptm": pick("ptm"),
                "iptm": pick("iptm"),
                "has_cif": True,
                "job_dir": str(job_dir),
            }
        )

    if missing:
        print(f"WARNING: {len(missing)} job(s) missing a .cif:")
        for job_name in missing:
            print(f"  - {job_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "serial",
        "aptamer_name",
        "target",
        "nucleic_type",
        "kd_nm",
        "ranking_score",
        "ptm",
        "iptm",
        "has_cif",
        "job_dir",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    if rows:
        print("Example (first row):")
        for key, value in rows[0].items():
            if key == "job_dir":
                continue
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
