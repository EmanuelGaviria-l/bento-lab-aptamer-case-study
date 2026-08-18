#!/usr/bin/env python3
"""
Download one MSA per unique protein chain from the ColabFold MMseqs2 server.

Run AFTER extract_unique_chains.py. Saves:
  data/msa_cache/<msa_id>.a3m

Uses retries + validation so a bad/corrupt download does not poison the cache.
Re-running is safe: already-downloaded .a3m files are skipped.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tarfile
import time
from pathlib import Path

import yaml


def load_run_mmseqs2():
    try:
        from colabfold.colabfold import run_mmseqs2  # type: ignore

        return run_mmseqs2
    except ImportError:
        try:
            from colabfold.batch import run_mmseqs2  # type: ignore

            return run_mmseqs2
        except ImportError as exc:
            raise RuntimeError(
                "ColabFold is not installed in this Python environment. "
                "Activate your Boltz venv and try: uv pip install colabfold"
            ) from exc


def looks_like_a3m(text: str) -> bool:
    """True if this is MSA content, not a file path.

    ColabFold's run_mmseqs2 often RETURNS the A3M text itself (starts with '>').
    Treating that string as a Path causes OSError 36 (file name too long).
    """
    stripped = text.lstrip()
    return stripped.startswith(">") and "\n" in text


def validate_a3m_text(text: str, min_lines: int = 2) -> None:
    if not text.lstrip().startswith(">"):
        raise ValueError("does not look like A3M (missing '>' header)")
    if text.count("\n") < min_lines:
        raise ValueError("looks too short to be a useful MSA")


def validate_a3m(path: Path, min_lines: int = 2) -> None:
    validate_a3m_text(path.read_text(), min_lines=min_lines)


def save_a3m_text(text: str, dest_a3m: Path) -> None:
    validate_a3m_text(text)
    dest_a3m.parent.mkdir(parents=True, exist_ok=True)
    dest_a3m.write_text(text)


def extract_a3m_from_result(result) -> str | None:
    """If ColabFold returned A3M text (str or list of str), return it."""
    if isinstance(result, str) and looks_like_a3m(result):
        return result
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        if isinstance(first, str) and looks_like_a3m(first):
            return first
    return None


def as_existing_path(value: str) -> Path | None:
    """Treat value as a filesystem path only if it looks like one."""
    if not value or looks_like_a3m(value) or "\n" in value or len(value) > 400:
        return None
    path = Path(value)
    try:
        if path.exists():
            return path
    except OSError:
        return None
    return None


def extract_a3m_from_tar(tar_path: Path, dest_a3m: Path) -> None:
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getnames()
        a3m_members = [m for m in members if m.endswith(".a3m")]
        if not a3m_members:
            raise tarfile.ReadError(f"No .a3m files inside {tar_path}")

        # Prefer uniref.a3m when present (standard ColabFold output name).
        preferred = [m for m in a3m_members if m.endswith("uniref.a3m")]
        chosen = preferred[0] if preferred else a3m_members[0]

        extracted = tar.extractfile(chosen)
        if extracted is None:
            raise tarfile.ReadError(f"Could not extract {chosen} from {tar_path}")

        dest_a3m.parent.mkdir(parents=True, exist_ok=True)
        with dest_a3m.open("wb") as out:
            shutil.copyfileobj(extracted, out)


def fetch_one_chain(
    fasta_path: Path,
    out_a3m: Path,
    run_mmseqs2,
    max_retries: int,
    retry_wait_sec: int,
) -> None:
    sequence = fasta_path.read_text().splitlines()[-1].strip()
    work_dir = out_a3m.parent / ".work" / out_a3m.stem
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(work_dir / "job")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        print(f"    attempt {attempt}/{max_retries} ...")
        try:
            # ColabFold API: often RETURNS the A3M text, not a path.
            # Some versions instead write a .tar.gz or .a3m next to prefix.
            result = run_mmseqs2(
                sequence,
                prefix,
                use_env=True,
                use_pairing=False,
                use_templates=False,
            )

            a3m_text = extract_a3m_from_result(result)
            if a3m_text is not None:
                n_hits = a3m_text.count("\n>")
                print(f"    ColabFold returned A3M text ({n_hits} extra sequences)")
                save_a3m_text(a3m_text, out_a3m)
            else:
                candidate: Path | None = None
                if isinstance(result, (list, tuple)) and result and isinstance(result[0], str):
                    candidate = as_existing_path(result[0])
                elif isinstance(result, str):
                    candidate = as_existing_path(result)

                tar_path = Path(f"{prefix}.tar.gz")
                found_a3m = sorted(work_dir.rglob("*.a3m"))

                if candidate is not None and candidate.suffix == ".a3m":
                    shutil.copy2(candidate, out_a3m)
                elif tar_path.exists():
                    extract_a3m_from_tar(tar_path, out_a3m)
                elif found_a3m:
                    shutil.copy2(found_a3m[0], out_a3m)
                else:
                    raise FileNotFoundError(
                        f"Could not locate MSA output for {fasta_path.name}. "
                        f"result type={type(result).__name__}, "
                        f"tar={tar_path.exists()}, a3m files in work dir={len(found_a3m)}"
                    )

            validate_a3m(out_a3m)
            print(f"    saved {out_a3m} ({out_a3m.stat().st_size} bytes)")
            return

        except (tarfile.ReadError, gzip.BadGzipFile, OSError, ValueError, FileNotFoundError) as exc:
            last_error = exc
            msg = str(exc)
            if len(msg) > 300:
                msg = msg[:300] + " ... [truncated]"
            print(f"    failed: {msg}")
            if attempt < max_retries:
                time.sleep(retry_wait_sec)
        finally:
            for pattern in (f"{prefix}.tar.gz", f"{prefix}.tar"):
                p = Path(pattern)
                if p.exists():
                    p.unlink()

    raise RuntimeError(
        f"All {max_retries} attempts failed for {fasta_path.name}"
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch cached protein MSAs via ColabFold.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only fetch this many pending chains (0 = all pending). Good for testing.",
    )
    parser.add_argument(
        "--msa-id",
        type=str,
        default="",
        help="Fetch only one chain by msa_id (e.g. P04618_rev_full_length_no_processing).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--retry-wait-sec",
        type=int,
        default=45,
    )
    args = parser.parse_args()

    project_root = args.project_root
    manifest_path = project_root / "data" / "msa_cache" / "manifest.yaml"

    if not manifest_path.exists():
        sys.exit(
            f"Missing {manifest_path}. Run scripts/msa/extract_unique_chains.py first."
        )

    manifest = yaml.safe_load(manifest_path.read_text())
    run_mmseqs2 = load_run_mmseqs2()

    pending = [c for c in manifest["chains"] if c["status"] != "ready"]
    if args.msa_id:
        pending = [c for c in manifest["chains"] if c["msa_id"] == args.msa_id]
        if not pending:
            sys.exit(f"No chain with msa_id={args.msa_id!r} in manifest.")

    fetched = 0
    for chain in manifest["chains"]:
        if chain["status"] == "ready" and not args.msa_id:
            print(f"[SKIP] {chain['msa_id']} (already cached)")
            continue

        if args.msa_id and chain["msa_id"] != args.msa_id:
            continue

        if args.limit and fetched >= args.limit:
            break

        if chain["status"] == "ready" and args.msa_id:
            print(f"[SKIP] {chain['msa_id']} (already cached)")
            continue

        fasta_path = project_root / chain["fasta"]
        out_a3m = project_root / chain["a3m"]

        print(
            f"[FETCH] {chain['msa_id']} "
            f"({chain['target']} / {chain['label']}, {chain['sequence_length']} aa)"
        )

        fetch_one_chain(
            fasta_path=fasta_path,
            out_a3m=out_a3m,
            run_mmseqs2=run_mmseqs2,
            max_retries=args.max_retries,
            retry_wait_sec=args.retry_wait_sec,
        )
        chain["status"] = "ready"
        fetched += 1

        manifest_path.write_text(yaml.dump(manifest, sort_keys=False))

    print(f"\nDone. Fetched {fetched} MSA(s) this run.")
    ready = sum(1 for c in manifest["chains"] if c["status"] == "ready")
    print(f"Cache status: {ready}/{len(manifest['chains'])} chains ready.")


if __name__ == "__main__":
    main()
