#!/usr/bin/env python3
"""
Task 6 wiring, step 1: copy AF3 atom coordinates into Boltz's pre_affinity npz.

Boltz's affinity pass does not read a CIF. It loads:
  predictions/<job>/pre_affinity_<job>.npz
which is a StructureV2 table (atoms / residues / chains). We keep that
topology (so token order, masks, and the binder crop stay valid) and
overwrite coords from AF3's top-ranked model.cif, matched by
(chain, residue index, atom name).

Run from the case-study repo, Boltz venv active.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study")


def norm_atom(name: str) -> str:
    """AF3 and Boltz sometimes differ on * vs ' and padding."""
    name = str(name).strip().strip('"').strip("'")
    name = name.replace("*", "'")
    return name.upper()


def is_hydrogen(atom_name: str) -> bool:
    return norm_atom(atom_name).startswith("H")


def parse_cif_atoms(cif_path: Path) -> tuple[
    dict[tuple[str, int, str], np.ndarray],
    dict[tuple[str, int], str],
]:
    """Parse mmCIF ATOM/HETATM coords keyed by (chain, seq_id, atom_name)."""
    text = cif_path.read_text()
    # AF3 writes a standard _atom_site loop. Find the loop header + rows.
    match = re.search(r"loop_\s*((?:_atom_site\.\S+\s+)+)", text)
    if not match:
        raise ValueError(f"No _atom_site loop in {cif_path}")

    columns = re.findall(r"_atom_site\.(\S+)", match.group(1))
    body = text[match.end() :]
    # Stop at next loop or trailing '#'
    stop = re.search(r"\n(?:loop_|#)", body)
    if stop:
        body = body[: stop.start()]

    rows = []
    current: list[str] = []
    for token in _tokenize_cif(body):
        current.append(token)
        if len(current) == len(columns):
            rows.append(current)
            current = []

    col = {name: i for i, name in enumerate(columns)}
    needed = [
        "label_asym_id",
        "label_seq_id",
        "label_atom_id",
        "label_comp_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
    ]
    missing = [n for n in needed if n not in col]
    if missing:
        raise ValueError(f"{cif_path} missing atom_site columns: {missing}")

    coords: dict[tuple[str, int, str], np.ndarray] = {}
    resnames: dict[tuple[str, int], str] = {}
    for row in rows:
        seq_raw = row[col["label_seq_id"]]
        if seq_raw in {".", "?"}:
            continue
        chain = str(row[col["label_asym_id"]])
        atom = norm_atom(row[col["label_atom_id"]])
        seq = int(seq_raw)
        resname = str(row[col["label_comp_id"]]).strip().upper()
        xyz = np.array(
            [float(row[col["Cartn_x"]]), float(row[col["Cartn_y"]]), float(row[col["Cartn_z"]])],
            dtype=np.float32,
        )
        coords[(chain, seq, atom)] = xyz
        resnames[(chain, seq)] = resname
    return coords, resnames


def _tokenize_cif(block: str) -> list[str]:
    tokens = []
    buf = []
    in_quote = False
    quote_char = ""
    for ch in block:
        if in_quote:
            if ch == quote_char:
                tokens.append("".join(buf))
                buf = []
                in_quote = False
            else:
                buf.append(ch)
            continue
        if ch in {'"', "'"}:
            in_quote = True
            quote_char = ch
            continue
        if ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
            continue
        buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return [t for t in tokens if t and not t.startswith("#")]


def choose_res_offset(residues, chains, af3_resnames: dict[tuple[str, int], str]) -> int:
    """Boltz res_idx is often 0-based; AF3 label_seq_id is 1-based.

    Pick the offset that makes residue *names* line up. Matching only atom
    names like CA/N/C/O would look successful even with an off-by-one shift.
    """
    best_off = 0
    best_hits = -1
    for off in (0, 1, -1, 2):
        hits = 0
        total = 0
        for chain in chains:
            chain_name = str(chain["name"])
            res_start = int(chain["res_idx"])
            res_end = res_start + int(chain["res_num"])
            for res in residues[res_start:res_end]:
                total += 1
                key = (chain_name, int(res["res_idx"]) + off)
                if af3_resnames.get(key) == str(res["name"]).strip().upper():
                    hits += 1
        if hits > best_hits:
            best_hits = hits
            best_off = off
    print(
        f"  residue-name alignment: offset={best_off:+d} "
        f"({best_hits} residues agree with AF3)"
    )
    return best_off


def inject_one(job_name: str, project_root: Path, dry_run: bool) -> None:
    boltz_npz = (
        project_root
        / "results"
        / "boltz_msa"
        / f"boltz_results_{job_name}"
        / "predictions"
        / job_name
        / f"pre_affinity_{job_name}.npz"
    )
    af3_cif = project_root / "results" / "af3_msa" / job_name / f"{job_name}_model.cif"
    out_dir = (
        project_root
        / "results"
        / "af3_boltz_affinity"
        / "predictions"
        / job_name
    )
    out_npz = out_dir / f"pre_affinity_{job_name}.npz"

    if not boltz_npz.exists():
        sys.exit(f"Missing Boltz pre_affinity file: {boltz_npz}")
    if not af3_cif.exists():
        sys.exit(f"Missing AF3 CIF: {af3_cif}")

    data = dict(np.load(boltz_npz, allow_pickle=True))
    atoms = data["atoms"]
    residues = data["residues"]
    chains = data["chains"]

    print("  Boltz chains:")
    for chain in chains:
        names = chain.dtype.names
        printable = {n: chain[n] for n in names}
        print(f"    {printable}")

    af3, af3_resnames = parse_cif_atoms(af3_cif)
    af3_chains = sorted({k[0] for k in af3})
    print(f"  AF3 chains: {af3_chains}  ({len(af3)} atoms, {len(af3_resnames)} residues)")

    offset = choose_res_offset(residues, chains, af3_resnames)

    new_coords = atoms["coords"].copy()
    matched_heavy = 0
    missing_heavy = 0
    matched_h = 0
    missing_h = 0
    missing_examples = []

    for chain in chains:
        chain_name = str(chain["name"])
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        for res in residues[res_start:res_end]:
            res_idx = int(res["res_idx"]) + offset
            atom_start = int(res["atom_idx"])
            atom_end = atom_start + int(res["atom_num"])
            for atom_i in range(atom_start, atom_end):
                atom_name = norm_atom(str(atoms[atom_i]["name"]))
                key = (chain_name, res_idx, atom_name)
                hydro = is_hydrogen(atom_name)
                if key in af3:
                    new_coords[atom_i] = af3[key]
                    if hydro:
                        matched_h += 1
                    else:
                        matched_heavy += 1
                else:
                    if hydro:
                        missing_h += 1
                    else:
                        missing_heavy += 1
                        if len(missing_examples) < 8:
                            missing_examples.append(
                                (key, str(res["name"]).strip())
                            )

    heavy_total = matched_heavy + missing_heavy
    frac = matched_heavy / heavy_total if heavy_total else 0.0
    print(
        f"{job_name}: heavy atoms {matched_heavy}/{heavy_total} ({frac:.1%}); "
        f"hydrogens {matched_h}/{matched_h + missing_h} "
        f"(AF3 usually has no H, so unmatched H is OK)"
    )
    if missing_examples:
        print(f"  unmatched heavy examples (chain, res, atom, boltz_resname): {missing_examples}")
    if frac < 0.9:
        sys.exit(
            f"Heavy-atom match rate {frac:.1%} is too low to trust. "
            "Inspect chain names / residue numbering before continuing."
        )

    atoms = atoms.copy()
    atoms["coords"] = new_coords
    data["atoms"] = atoms
    # StructureV2 also stores a separate coords ensemble table.
    if "coords" in data:
        coords_tbl = data["coords"].copy()
        coords_tbl["coords"] = new_coords
        data["coords"] = coords_tbl

    if dry_run:
        print("  dry-run: not writing")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **data)
    print(f"  wrote {out_npz}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject AF3 coords into Boltz pre_affinity npz.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--job", type=str, default="", help="One job name, e.g. 10000008_5A")
    parser.add_argument("--limit", type=int, default=0, help="Only process this many jobs (0 = all).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    af3_root = args.project_root / "results" / "af3_msa"
    if args.job:
        jobs = [args.job]
    else:
        jobs = sorted(p.name for p in af3_root.iterdir() if p.is_dir() and (p / f"{p.name}_model.cif").exists())

    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Injecting AF3 coords for {len(jobs)} job(s)")
    for job in jobs:
        inject_one(job, args.project_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
