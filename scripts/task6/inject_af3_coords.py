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
import csv
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study")


def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", text.strip())
    return text.strip("_") or "aptamer"


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


def find_pre_affinity(project_root: Path, job_name: str) -> Path | None:
    for folder in ("boltz_msa_v2", "boltz_msa"):
        path = (
            project_root
            / "results"
            / folder
            / f"boltz_results_{job_name}"
            / "predictions"
            / job_name
            / f"pre_affinity_{job_name}.npz"
        )
        if path.is_file():
            return path
    return None


def find_af3_cif(project_root: Path, job_name: str) -> Path | None:
    for folder in ("af3_msa_v2", "af3_msa"):
        job_dir = project_root / "results" / folder / job_name
        named = job_dir / f"{job_name}_model.cif"
        if named.is_file():
            return named
        if job_dir.is_dir():
            matches = sorted(job_dir.glob("*model.cif"))
            if matches:
                return matches[0]
    return None


def job_names_from_csv(csv_path: Path) -> list[str]:
    jobs = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            serial = str(row["Serial Number"])
            name = str(row["Name of Aptamer"])
            jobs.append(f"{serial}_{safe_filename(name)}")
    return jobs


def affinity_json_exists(project_root: Path, job_name: str, output_root: Path) -> bool:
    candidates = [
        output_root / "predictions" / job_name / f"affinity_{job_name}.json",
        project_root
        / "results"
        / "af3_boltz_affinity"
        / "predictions"
        / job_name
        / f"affinity_{job_name}.json",
    ]
    return any(p.is_file() for p in candidates)


def inject_one(
    job_name: str,
    project_root: Path,
    dry_run: bool,
    output_root: Path,
) -> bool:
    boltz_npz = find_pre_affinity(project_root, job_name)
    af3_cif = find_af3_cif(project_root, job_name)
    out_dir = output_root / "predictions" / job_name
    out_npz = out_dir / f"pre_affinity_{job_name}.npz"

    if boltz_npz is None:
        print(f"SKIP {job_name}: missing Boltz pre_affinity npz")
        return False
    if af3_cif is None:
        print(f"SKIP {job_name}: missing AF3 CIF")
        return False

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
        print(
            f"FAIL {job_name}: heavy-atom match {frac:.1%} is too low to trust"
        )
        return False

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
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **data)
    print(f"  wrote {out_npz}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject AF3 coords into Boltz pre_affinity npz.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--job", type=str, default="", help="One job name, e.g. 10000008_5A")
    parser.add_argument("--limit", type=int, default=0, help="Only process this many jobs (0 = all).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--v2",
        action="store_true",
        help="193-aptamer set: search v2+original folders, write to af3_boltz_affinity_v2.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Parent of predictions/ (default: results/af3_boltz_affinity or _v2 with --v2).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Aptamer CSV used to list jobs (default v2 CSV when --v2).",
    )
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        name = "af3_boltz_affinity_v2" if args.v2 else "af3_boltz_affinity"
        output_root = args.project_root / "results" / name

    if args.job:
        jobs = [args.job]
    elif args.v2 or args.csv:
        csv_path = args.csv or (args.project_root / "data" / "aptamer_subset_v2.csv")
        if not csv_path.exists():
            sys.exit(f"Missing CSV: {csv_path}")
        jobs = job_names_from_csv(csv_path)
    else:
        af3_root = args.project_root / "results" / "af3_msa"
        jobs = sorted(
            p.name
            for p in af3_root.iterdir()
            if p.is_dir() and (p / f"{p.name}_model.cif").exists()
        )

    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Injecting AF3 coords for {len(jobs)} job(s) -> {output_root}")
    ok = 0
    skipped = 0
    failed = 0
    for job in jobs:
        if args.v2 and affinity_json_exists(args.project_root, job, output_root):
            print(f"[SKIP] {job} (affinity json already exists)")
            skipped += 1
            continue
        if inject_one(job, args.project_root, args.dry_run, output_root):
            ok += 1
        else:
            failed += 1
    print(f"Inject summary: wrote/ok={ok} skip={skipped} fail={failed}")


if __name__ == "__main__":
    main()
