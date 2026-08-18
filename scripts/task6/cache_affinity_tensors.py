#!/usr/bin/env python3
"""
Dump AffinityModule inputs for each AF3-wired complex.

Fine-tuning trains only the affinity heads on these tensors so we do not
re-run the 40s trunk every epoch.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_affinity_from_af3 import (  # noqa: E402
    PROJECT_ROOT,
    find_cache,
    find_manifest,
    install_af3_coord_hook,
)


FEAT_KEYS = [
    "token_pad_mask",
    "mol_type",
    "affinity_token_mask",
    "token_to_rep_atom",
    "affinity_mw",
]


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        return [to_cpu(v) for v in value]
    return value


def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w.-]+", "_", text.strip())
    return text.strip("_") or "aptamer"


def load_labels(project: Path) -> dict[str, dict]:
    rows = {}
    with (project / "data" / "aptamer_subset.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            job = f"{row['Serial Number']}_{safe_filename(row['Name of Aptamer'])}"
            kd_nm = float(row["Kd (nM)"])
            rows[job] = {
                "target": row["Target"],
                "kd_nm": kd_nm,
                "log10_kd_um": float(torch.log10(torch.tensor(max(kd_nm, 1e-12) / 1000.0))),
            }
    return rows


def load_affinity_model():
    install_af3_coord_hook()
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False
    steering_args.contact_guidance_update = False
    model = Boltz2.load_from_checkpoint(
        find_cache() / "boltz2_aff.ckpt",
        strict=True,
        predict_args={
            "recycling_steps": 5,
            "sampling_steps": 1,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(steering_args),
        use_kernels=False,
    )
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--job", default="")
    args = parser.parse_args()
    project = args.project_root
    labels = load_labels(project)
    jobs = [args.job] if args.job else sorted(labels)
    cache_dir = project / "results" / "af3_boltz_affinity" / "tensor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = load_affinity_model().cuda()
    captured: dict = {}

    def hook_module(mod):
        orig = mod.forward

        def wrapped(s_inputs, z, x_pred, feats, **kwargs):
            job = captured.get("current_job")
            if job and job not in captured:
                captured[job] = {
                    "s_inputs": s_inputs.detach().cpu(),
                    "z": z.detach().cpu(),
                    "x_pred": x_pred.detach().cpu(),
                    "feats": {
                        k: to_cpu(feats[k])
                        for k in FEAT_KEYS
                        if k in feats
                    },
                }
            return orig(s_inputs, z, x_pred, feats, **kwargs)

        mod.forward = wrapped

    if getattr(model, "affinity_ensemble", False):
        hook_module(model.affinity_module1)
    else:
        hook_module(model.affinity_module)

    for job in jobs:
        out_pt = cache_dir / f"{job}.pt"
        if out_pt.exists():
            print(f"[SKIP] {job}")
            continue
        injected = (
            project
            / "results"
            / "af3_boltz_affinity"
            / "predictions"
            / job
            / f"pre_affinity_{job}.npz"
        )
        if not injected.exists():
            print(f"[MISS] {job} (no injected npz)")
            continue

        processed = project / "results" / "boltz_msa" / f"boltz_results_{job}" / "processed"
        manifest = Manifest.load(find_manifest(processed))
        if len(manifest.records) != 1:
            recs = [r for r in manifest.records if r.id == job]
            manifest = Manifest(records=recs or list(manifest.records[:1]))

        captured["current_job"] = job
        dm = Boltz2InferenceDataModule(
            manifest=manifest,
            target_dir=project / "results" / "af3_boltz_affinity" / "predictions",
            msa_dir=processed / "msa",
            mol_dir=find_cache() / "mols",
            num_workers=0,
            constraints_dir=(processed / "constraints")
            if (processed / "constraints").exists()
            else None,
            template_dir=(processed / "templates")
            if (processed / "templates").exists()
            else None,
            extra_mols_dir=(processed / "mols") if (processed / "mols").exists() else None,
            override_method="other",
            affinity=True,
        )
        batch = next(iter(dm.predict_dataloader()))
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            model.predict_step(batch, 0)

        if job not in captured:
            print(f"[FAIL] {job} (affinity hook did not fire)")
            continue
        payload = captured.pop(job)
        payload.update(labels[job])
        payload["job"] = job
        torch.save(payload, out_pt)
        print(f"[SAVE] {out_pt}")

    print(f"Cache dir: {cache_dir}")


if __name__ == "__main__":
    main()
