#!/usr/bin/env python3
"""
Task 7: Correlation analysis between predicted affinity ranking and
experimental Kd ranking.

Reads:
  report/boltz_affinity_ranking.csv

Writes:
  report/correlation_analysis.txt

Metric: Spearman rank correlation (scipy.stats.spearmanr).
Spearman is appropriate here because we care about whether the *ranking*
of aptamers by predicted affinity agrees with the ranking by experimental
Kd, not whether the absolute values match on the same numeric scale
(affinity_pred_value is log10(IC50 in uM); Kd is in nM -- not directly
comparable in magnitude, only in rank order).

Direction check: for both metrics, LOWER = stronger binding
(lower Kd = tighter binding; lower affinity_pred_value = lower predicted
IC50 = more potent). So a well-calibrated model should show a POSITIVE
Spearman correlation between affinity_pred_value and kd_nm (both move in
the same direction), not negative.

Reports both:
  - Overall correlation across all 73 aptamers / 7 targets pooled
    (as literally requested by the case study).
  - Per-target correlation, since Kd magnitudes differ substantially
    between targets (e.g. a 0.5 nM Kd vs a 27 nM Kd are not on the same
    scale), so pooling across targets can mask or distort the true
    within-target ranking agreement. Reported as a secondary, more
    rigorous view -- documented as such in the report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def run_analysis(
    df: pd.DataFrame,
    label: str,
    pred_col: str,
    higher_is_better: bool,
) -> list[str]:
    lines = []
    lines.append(f"{label} — predicted score vs experimental Kd")
    lines.append("=" * 75)
    lines.append("")
    lines.append(
        "Metric: Spearman rank correlation coefficient (scipy.stats.spearmanr)."
    )
    direction = "higher = tighter / more confident" if higher_is_better else "lower = tighter binding"
    lines.append(f"Predicted column: {pred_col} ({direction}).")
    lines.append("Experimental: kd_nm (lower = tighter binding).")
    if higher_is_better:
        lines.append(
            "Because the predicted score is higher-is-better, AGREEMENT with "
            "experiment is a NEGATIVE Spearman rho (high confidence with low Kd)."
        )
    else:
        lines.append(
            "Because both scores are lower-is-better, AGREEMENT is a POSITIVE "
            "Spearman rho."
        )
    lines.append("")

    pred = df[pred_col].astype(float)
    kd = df["kd_nm"].astype(float)
    rho, pval = spearmanr(pred, kd)
    lines.append(f"Overall (n={len(df)}, all 7 targets pooled):")
    lines.append(f"  Spearman rho = {rho:.4f}")
    lines.append(f"  p-value      = {pval:.4g}")
    lines.append("")

    # Mean of per-target rhos: avoids mixing targets with very different Kd scales.
    per_target = []
    lines.append("Per-target breakdown:")
    lines.append(f"{'Target':<55}{'n':>4}{'rho':>10}{'p-value':>12}")
    lines.append("-" * 81)
    for target, group in df.groupby("target"):
        if len(group) < 3:
            lines.append(f"{target:<55}{len(group):>4}{'(n<3, skipped)':>22}")
            continue
        if group["kd_nm"].nunique() < 2:
            lines.append(
                f"{target:<55}{len(group):>4}{'(Kd ties, skipped)':>22}"
            )
            continue
        t_rho, t_pval = spearmanr(
            group[pred_col].astype(float), group["kd_nm"].astype(float)
        )
        per_target.append(t_rho)
        short_target = (target[:52] + "...") if len(target) > 55 else target
        lines.append(
            f"{short_target:<55}{len(group):>4}{t_rho:>10.4f}{t_pval:>12.4g}"
        )

    if per_target:
        lines.append("")
        lines.append(
            f"Mean per-target rho (n_targets={len(per_target)}, "
            "HIV-Rev-style ties excluded):"
        )
        lines.append(f"  {sum(per_target) / len(per_target):.4f}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 7 Spearman correlation analysis.")
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=None,
        help="Path to boltz_affinity_ranking.csv (default: report/boltz_affinity_ranking.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output text file (default: report/correlation_analysis.txt)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Boltz-2 (no-MSA baseline)",
        help="Label printed in the report header",
    )
    parser.add_argument(
        "--pred-column",
        type=str,
        default="affinity_pred_value",
        help="Column to correlate with kd_nm",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Set this for scores where higher means tighter/more confident (ipTM, probability).",
    )
    args = parser.parse_args()

    project_root = Path("/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study")
    ranking_path = args.ranking_csv or (project_root / "report" / "boltz_affinity_ranking.csv")
    output_path = args.output or (project_root / "report" / "correlation_analysis.txt")

    df = pd.read_csv(ranking_path)
    if args.pred_column not in df.columns:
        sys.exit(
            f"Column {args.pred_column!r} not in {ranking_path}. "
            f"Available: {list(df.columns)}"
        )

    lines = run_analysis(
        df,
        args.label,
        pred_col=args.pred_column,
        higher_is_better=args.higher_is_better,
    )
    report_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text)

    print(report_text)
    print(f"\nWrote analysis to {output_path}")


if __name__ == "__main__":
    main()
