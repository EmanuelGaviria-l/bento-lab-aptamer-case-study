#!/usr/bin/env python3
"""
List every unique protein chain that needs an MSA (usually ~9, not 73).

Reads:
  data/targets.yaml
  data/uniprot_cache/*.fasta   (already downloaded by build_*_inputs.py)

Writes:
  data/msa_cache/manifest.yaml
  data/msa_cache/fasta/<msa_id>.fasta
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


def msa_id(uniprot: str, label: str) -> str:
    slug = re.sub(r"[^\w]+", "_", label.lower()).strip("_")
    return f"{uniprot}_{slug}"


def slice_chains(full_sequence: str, chains: list[dict]) -> list[tuple[str, str, str]]:
    pieces = []
    for chain in chains:
        start, end = chain["range"]
        label = chain.get("label", "chain")
        piece = full_sequence[start - 1 : end]
        if not piece:
            raise ValueError(
                f"Empty slice for {label} range {chain['range']} "
                f"(sequence length {len(full_sequence)})"
            )
        pieces.append((label, piece, msa_id("", label)))  # placeholder, fixed below
    return pieces


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MSA manifest for unique protein chains.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="targets YAML (default: data/targets.yaml).",
    )
    args = parser.parse_args()

    project_root = args.project_root
    targets_path = args.targets or (project_root / "data" / "targets.yaml")
    cache_dir = project_root / "data" / "uniprot_cache"
    msa_dir = project_root / "data" / "msa_cache"
    fasta_dir = msa_dir / "fasta"
    manifest_path = msa_dir / "manifest.yaml"

    if not targets_path.exists():
        sys.exit(f"Missing file: {targets_path}")

    target_info = yaml.safe_load(targets_path.read_text())["targets"]

    chains_out = []
    seen_sequences: dict[str, str] = {}

    for entry in target_info:
        uniprot = entry["uniprot"]
        target_name = entry["name"]
        fasta_file = cache_dir / f"{uniprot}.fasta"
        full_sequence = None
        needs_uniprot = any("sequence" not in chain for chain in entry["chains"])

        if needs_uniprot:
            if not fasta_file.exists():
                sys.exit(
                    f"Missing UniProt cache for {uniprot}. "
                    f"Run build_*_inputs.py once first to download sequences."
                )
            lines = [
                line.strip()
                for line in fasta_file.read_text().splitlines()
                if not line.startswith(">")
            ]
            full_sequence = "".join(lines)

        for chain in entry["chains"]:
            label = chain.get("label", "chain")
            if "sequence" in chain:
                sequence = "".join(str(chain["sequence"]).split())
            else:
                start, end = chain["range"]
                sequence = full_sequence[start - 1 : end]
            if not sequence:
                sys.exit(f"Empty chain sequence for {target_name} / {label}")
            chain_msa_id = msa_id(uniprot, label)
            a3m_path = msa_dir / f"{chain_msa_id}.a3m"

            if sequence in seen_sequences:
                print(
                    f"NOTE: {chain_msa_id} shares sequence with {seen_sequences[sequence]}"
                )
            else:
                seen_sequences[sequence] = chain_msa_id

            fasta_dir.mkdir(parents=True, exist_ok=True)
            chain_fasta = fasta_dir / f"{chain_msa_id}.fasta"
            chain_fasta.write_text(f">{chain_msa_id}\n{sequence}\n")

            status = "ready" if a3m_path.exists() else "pending"
            chains_out.append(
                {
                    "msa_id": chain_msa_id,
                    "target": target_name,
                    "uniprot": uniprot,
                    "label": label,
                    "sequence_length": len(sequence),
                    "fasta": str(chain_fasta.relative_to(project_root)),
                    "a3m": str(a3m_path.relative_to(project_root)),
                    "status": status,
                }
            )

            print(
                f"- {chain_msa_id}: {target_name} / {label} "
                f"({len(sequence)} aa) -> {a3m_path.name} [{status}]"
            )

    manifest = {"chains": chains_out}
    msa_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.dump(manifest, sort_keys=False))

    pending = sum(1 for c in chains_out if c["status"] == "pending")
    print(f"\nWrote manifest with {len(chains_out)} unique chains to {manifest_path}")
    print(f"{pending} still need MSAs fetched.")


if __name__ == "__main__":
    main()
