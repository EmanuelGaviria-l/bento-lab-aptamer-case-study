#!/usr/bin/env python3
"""
Build AlphaFold 3 JSON input files for each row in aptamer_subset.csv.

Reads:
  data/aptamer_subset.csv
  data/targets.yaml

Writes:
  inputs/af3/<serial>_<aptamer_name>.json

Run from the bento-lab-aptamer-case-study folder.

CHANGE LOG (dataset-correction pass):
  targets.yaml now specifies one or more residue ranges ("chains") per
  target, corresponding to the mature/processed form of the protein
  (signal peptide, propeptide, and cleaved fragments removed) rather
  than the full UniProt precursor sequence. See targets.yaml for the
  rationale behind each target's ranges. Targets with two chains
  (Thrombin: light+heavy; Abrin: A+B) are now written as two separate
  "protein" entries in the AF3 JSON, each with its own chain id.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
import urllib.error
import urllib.request
from pathlib import Path

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
    """
    Slice the full UniProt precursor sequence into one or more mature
    chain sequences using 1-indexed, inclusive [start, end] ranges from
    targets.yaml.
    """
    pieces = []
    for chain in chains:
        start, end = chain["range"]
        piece = full_sequence[start - 1:end]  # UniProt ranges are 1-indexed inclusive
        if not piece:
            raise ValueError(
                f"Empty slice for range {chain['range']} "
                f"(sequence length {len(full_sequence)}, label={chain.get('label')})"
            )
        pieces.append(piece)
    return pieces


def resolve_chain_sequences(
    target_entry: dict,
    full_seq_cache: dict[str, str],
    cache_dir: Path,
) -> list[str]:
    """Use explicit chain sequences when present; otherwise UniProt ranges."""
    pieces: list[str] = []
    full: str | None = None
    for chain in target_entry["chains"]:
        if "sequence" in chain:
            seq = "".join(str(chain["sequence"]).split())
            if not seq:
                raise ValueError(f"Empty sequence for {chain.get('label')}")
            pieces.append(seq)
            continue
        uniprot_id = target_entry["uniprot"]
        if full is None:
            if uniprot_id not in full_seq_cache:
                full_seq_cache[uniprot_id] = fetch_uniprot_sequence(uniprot_id, cache_dir)
            full = full_seq_cache[uniprot_id]
        start, end = chain["range"]
        piece = full[start - 1:end]
        if not piece:
            raise ValueError(
                f"Empty slice for range {chain['range']} "
                f"(sequence length {len(full)}, label={chain.get('label')})"
            )
        pieces.append(piece)
    return pieces


def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", text.strip())
    return text.strip("_") or "aptamer"


def msa_id(uniprot: str, label: str) -> str:
    slug = re.sub(r"[^\w]+", "_", label.lower()).strip("_")
    return f"{uniprot}_{slug}"


def build_af3_json(
    job_name: str,
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
            "templates": [],
        }
        if msa_path:
            # Inline the cached A3M. AF3's --norun_data_pipeline validator
            # requires unpairedMsa AND pairedMsa as strings; unpairedMsaPath
            # alone raises "missing paired MSA".
            a3m_text = Path(msa_path).read_text()
            if not a3m_text.lstrip().startswith(">"):
                raise ValueError(f"MSA at {msa_path} does not look like A3M")
            protein_entry["unpairedMsa"] = a3m_text
            protein_entry["pairedMsa"] = ""
        else:
            # Fallback: empty MSAs (baseline / no-MSA mode)
            protein_entry["unpairedMsa"] = ""
            protein_entry["pairedMsa"] = ""
        sequences.append({"protein": protein_entry})

    nucleic_key = "dna" if nucleic_type == "ssDNA" else "rna"
    nucleic_entry = {
        "id": next(chain_ids),
        "sequence": aptamer_sequence,
    }
    if nucleic_key == "rna":
        # validate_fold_input requires unpaired MSA for RNA chains too;
        # DNA chains are not checked, so no field needed there.
        nucleic_entry["unpairedMsa"] = ""
    sequences.append({nucleic_key: nucleic_entry})

    return {
        "name": job_name,
        "modelSeeds": [1],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AF3 JSON inputs for aptamers.")
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
        "--msa-cache",
        action="store_true",
        help=(
            "Use cached protein MSAs from data/msa_cache/*.a3m (inlined as unpairedMsa). "
            "Writes to inputs/af3_msa/ instead of inputs/af3/."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Aptamer CSV (default: data/aptamer_subset.csv).",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="targets YAML (default: data/targets.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: inputs/af3 or inputs/af3_msa).",
    )
    args = parser.parse_args()

    project_root = args.project_root
    csv_path = args.csv or (project_root / "data" / "aptamer_subset.csv")
    targets_path = args.targets or (project_root / "data" / "targets.yaml")
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = project_root / "inputs" / ("af3_msa" if args.msa_cache else "af3")
    msa_cache_dir = project_root / "data" / "msa_cache"
    cache_dir = project_root / "data" / "uniprot_cache"

    if not csv_path.exists():
        sys.exit(f"Missing file: {csv_path}")
    if not targets_path.exists():
        sys.exit(f"Missing file: {targets_path}")

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"No rows in {csv_path}")

    target_info = yaml.safe_load(targets_path.read_text())["targets"]

    target_map = {entry["name"]: entry for entry in target_info}
    csv_targets = {row["Target"] for row in rows}
    missing_targets = sorted(csv_targets - set(target_map))
    if missing_targets:
        sys.exit(f"These targets are missing from targets.yaml:\n{missing_targets}")

    print(f"Found {len(rows)} aptamers across {len(csv_targets)} targets.")

    full_seq_cache: dict[str, str] = {}
    built = 0

    for row in rows:
        if args.limit and built >= args.limit:
            break

        target_name = row["Target"]
        target_entry = target_map[target_name]
        uniprot_id = target_entry["uniprot"]
        chains = target_entry["chains"]
        aptamer_name = str(row["Name of Aptamer"])
        serial = str(row["Serial Number"])
        nucleic_type = row["Type of Nucleic Acid"]

        protein_sequences = resolve_chain_sequences(
            target_entry, full_seq_cache, cache_dir
        )

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
        else:
            protein_msa_paths = [None] * len(protein_sequences)

        aptamer_sequence = clean_aptamer_sequence(row["Aptamer Sequence"], nucleic_type)
        job_name = f"{serial}_{safe_filename(aptamer_name)}"
        output_path = output_dir / f"{job_name}.json"

        payload = build_af3_json(
            job_name=job_name,
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
            output_path.write_text(json.dumps(payload, indent=2))

        built += 1

    if args.dry_run:
        print(f"\nDry run complete. Would build {built} AF3 input files.")
    else:
        print(f"\nDone. Wrote {built} files to {output_dir}")


if __name__ == "__main__":
    main()
