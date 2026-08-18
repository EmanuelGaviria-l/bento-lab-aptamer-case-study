#!/usr/bin/env python3
"""
Build Boltz-2 YAML input files (with affinity prediction) for each row in
aptamer_subset.csv.

Reads:
  data/aptamer_subset.csv
  data/targets.yaml     (same corrected mature-chain ranges used for AF3)

Writes:
  inputs/boltz/<serial>_<aptamer_name>.yaml

Requires a patched Boltz-2 install (see docs/setup_log.md) that accepts
dna/rna chains as the affinity "binder" -- unmodified Boltz-2 will reject
these with "Chain X is not a ligand!".

Run from the bento-lab-aptamer-case-study folder, with the Boltz-2 venv
active.
"""

from __future__ import annotations

import argparse
import re
import string
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import yaml


def clean_aptamer_sequence(raw: str, nucleic_type: str) -> str:
    """Remove 5'/3' labels and keep only valid DNA/RNA letters."""
    seq = raw.strip()
    seq = seq.replace("5′", "").replace("3′", "")
    seq = seq.replace("5'", "").replace("3'", "")
    seq = seq.replace(" ", "").upper()

    if nucleic_type == "ssDNA":
        seq = seq.replace("U", "T")
        allowed = set("ACGT")
    elif nucleic_type == "ssRNA":
        seq = seq.replace("T", "U")
        allowed = set("ACGU")
    else:
        raise ValueError(f"Unsupported nucleic type: {nucleic_type}")

    cleaned = "".join(ch for ch in seq if ch in allowed)
    if not cleaned:
        raise ValueError(f"Could not parse aptamer sequence: {raw!r}")
    return cleaned


def fetch_uniprot_sequence(uniprot_id: str, cache_dir: Path) -> str:
    """Download the FULL (precursor) protein sequence from UniProt (cached on disk)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{uniprot_id}.fasta"

    if cache_file.exists():
        fasta = cache_file.read_text()
    else:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        print(f"  Downloading {uniprot_id} from UniProt...")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                fasta = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not download {uniprot_id}. Check internet access on the cluster."
            ) from exc
        cache_file.write_text(fasta)

    lines = [line.strip() for line in fasta.splitlines() if not line.startswith(">")]
    sequence = "".join(lines)
    if not sequence:
        raise RuntimeError(f"No sequence found for {uniprot_id}")
    return sequence


def slice_chains(full_sequence: str, chains: list[dict]) -> list[str]:
    """Slice the full UniProt precursor into mature chain(s), 1-indexed inclusive."""
    pieces = []
    for chain in chains:
        start, end = chain["range"]
        piece = full_sequence[start - 1:end]
        if not piece:
            raise ValueError(
                f"Empty slice for range {chain['range']} "
                f"(sequence length {len(full_sequence)}, label={chain.get('label')})"
            )
        pieces.append(piece)
    return pieces


def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", text.strip())
    return text.strip("_") or "aptamer"


def msa_id(uniprot: str, label: str) -> str:
    slug = re.sub(r"[^\w]+", "_", label.lower()).strip("_")
    return f"{uniprot}_{slug}"


def build_boltz_yaml(
    protein_sequences: list[str],
    protein_msa_paths: list[str | None],
    aptamer_sequence: str,
    nucleic_type: str,
) -> dict:
    chain_ids = iter(string.ascii_uppercase)
    sequences = []

    for protein_seq, msa_path in zip(protein_sequences, protein_msa_paths):
        protein_entry = {
            "id": next(chain_ids),
            "sequence": protein_seq,
        }
        if msa_path:
            protein_entry["msa"] = msa_path
        else:
            protein_entry["msa"] = "empty"
        sequences.append({"protein": protein_entry})

    nucleic_key = "dna" if nucleic_type == "ssDNA" else "rna"
    binder_id = next(chain_ids)
    sequences.append(
        {
            nucleic_key: {
                "id": binder_id,
                "sequence": aptamer_sequence,
            }
        }
    )

    return {
        "version": 1,
        "sequences": sequences,
        "properties": [{"affinity": {"binder": binder_id}}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Boltz-2 affinity YAML inputs for aptamers.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"),
        help="Path to the case study repo on Andromeda",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only build this many inputs (0 = all). Useful for testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be built without writing files",
    )
    parser.add_argument(
        "--with-msa",
        action="store_true",
        help=(
            "DEPRECATED: lets Boltz hit ColabFold during each predict run. "
            "Prefer --msa-cache instead."
        ),
    )
    parser.add_argument(
        "--msa-cache",
        action="store_true",
        help=(
            "Use cached protein MSAs from data/msa_cache/*.a3m. "
            "Writes to inputs/boltz_msa/."
        ),
    )
    args = parser.parse_args()

    if args.with_msa and args.msa_cache:
        sys.exit("Use either --with-msa OR --msa-cache, not both.")

    project_root = args.project_root
    csv_path = project_root / "data" / "aptamer_subset.csv"
    targets_path = project_root / "data" / "targets.yaml"
    if args.msa_cache:
        output_dir = project_root / "inputs" / "boltz_msa"
    elif args.with_msa:
        output_dir = project_root / "inputs" / "boltz_msa_remote"
    else:
        output_dir = project_root / "inputs" / "boltz"
    msa_cache_dir = project_root / "data" / "msa_cache"
    cache_dir = project_root / "data" / "uniprot_cache"

    if not csv_path.exists():
        sys.exit(f"Missing file: {csv_path}")
    if not targets_path.exists():
        sys.exit(f"Missing file: {targets_path}")

    df = pd.read_csv(csv_path)
    target_info = yaml.safe_load(targets_path.read_text())["targets"]

    target_map = {entry["name"]: entry for entry in target_info}
    missing_targets = sorted(set(df["Target"]) - set(target_map))
    if missing_targets:
        sys.exit(f"These targets are missing from targets.yaml:\n{missing_targets}")

    print(f"Found {len(df)} aptamers across {df['Target'].nunique()} targets.")

    full_seq_cache: dict[str, str] = {}
    built = 0

    for _, row in df.iterrows():
        if args.limit and built >= args.limit:
            break

        target_name = row["Target"]
        target_entry = target_map[target_name]
        uniprot_id = target_entry["uniprot"]
        chains = target_entry["chains"]
        aptamer_name = str(row["Name of Aptamer"])
        serial = str(row["Serial Number"])
        nucleic_type = row["Type of Nucleic Acid"]

        if uniprot_id not in full_seq_cache:
            full_seq_cache[uniprot_id] = fetch_uniprot_sequence(uniprot_id, cache_dir)

        protein_sequences = slice_chains(full_seq_cache[uniprot_id], chains)
        aptamer_sequence = clean_aptamer_sequence(row["Aptamer Sequence"], nucleic_type)

        protein_msa_paths: list[str | None] = []
        if args.msa_cache:
            for chain in chains:
                label = chain.get("label", "chain")
                a3m_path = msa_cache_dir / f"{msa_id(uniprot_id, label)}.a3m"
                if not a3m_path.exists():
                    sys.exit(
                        f"Missing MSA cache file: {a3m_path}\n"
                        "Run scripts/msa/fetch_protein_msas.py first."
                    )
                protein_msa_paths.append(str(a3m_path.resolve()))
        elif args.with_msa:
            protein_msa_paths = [None] * len(protein_sequences)
        else:
            protein_msa_paths = [None] * len(protein_sequences)

        job_name = f"{serial}_{safe_filename(aptamer_name)}"
        output_path = output_dir / f"{job_name}.yaml"

        payload = build_boltz_yaml(
            protein_sequences=protein_sequences,
            protein_msa_paths=protein_msa_paths,
            aptamer_sequence=aptamer_sequence,
            nucleic_type=nucleic_type,
        )

        chain_lens = "+".join(str(len(s)) for s in protein_sequences)
        print(
            f"- {job_name}: target={uniprot_id} ({len(chains)} chain(s), "
            f"len={chain_lens}), aptamer_len={len(aptamer_sequence)}, type={nucleic_type}"
        )

        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

        built += 1

    if args.dry_run:
        print(f"\nDry run complete. Would build {built} Boltz-2 YAML input files.")
    else:
        print(f"\nDone. Wrote {built} files to {output_dir}")


if __name__ == "__main__":
    main()
