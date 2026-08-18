#!/usr/bin/env python3
"""
Task 6 wiring, step 2: run Boltz-2's affinity module on AF3 coordinates.

Does NOT run Boltz diffusion. structure_module.sample is replaced so
x_pred is the injected AF3 coords from pre_affinity_*.npz.

Requires:
  scripts/task6/inject_af3_coords.py  (already run, not dry-run)
  original processed/ folder from the Boltz MSA batch
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from pytorch_lightning import Trainer

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzAffinityWriter
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2


PROJECT_ROOT = Path("/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study")


def find_cache() -> Path:
    cache = Path.home() / ".boltz"
    if not cache.exists():
        sys.exit(f"Boltz cache not found at {cache}")
    return cache


def find_manifest(processed: Path) -> Path:
    for name in ("manifest.json", "manifest.yaml"):
        path = processed / name
        if path.exists():
            return path
    sys.exit(f"No manifest.json in {processed}")


def install_af3_coord_hook() -> None:
    """Use loaded structure coords instead of sampling a new Boltz pose."""
    original_forward = Boltz2.forward

    def forward_with_af3_coords(self, feats, *args, **kwargs):
        orig_sample = self.structure_module.sample

        def fake_sample(*_a, **_k):
            coords = feats["coords"]
            if coords.ndim == 4:
                xyz = coords[:, 0]
            elif coords.ndim == 3:
                xyz = coords
            else:
                raise RuntimeError(f"Unexpected coords shape {tuple(coords.shape)}")
            return {"sample_atom_coords": xyz}

        self.structure_module.sample = fake_sample
        try:
            return original_forward(self, feats, *args, **kwargs)
        finally:
            self.structure_module.sample = orig_sample

    Boltz2.forward = forward_with_af3_coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Score AF3 poses with Boltz affinity module.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--job", required=True, help="e.g. 10000008_5A")
    parser.add_argument("--no-kernels", action="store_true", default=True)
    args = parser.parse_args()

    job = args.job
    project = args.project_root
    injected = (
        project
        / "results"
        / "af3_boltz_affinity"
        / "predictions"
        / job
        / f"pre_affinity_{job}.npz"
    )
    if not injected.exists():
        sys.exit(
            f"Missing {injected}\nRun: python3 scripts/task6/inject_af3_coords.py --job {job}"
        )

    boltz_run = project / "results" / "boltz_msa" / f"boltz_results_{job}"
    processed = boltz_run / "processed"
    if not processed.exists():
        sys.exit(f"Missing processed dir: {processed}")

    cache = find_cache()
    aff_ckpt = cache / "boltz2_aff.ckpt"
    if not aff_ckpt.exists():
        sys.exit(f"Missing affinity checkpoint: {aff_ckpt}")

    manifest_path = find_manifest(processed)
    manifest = Manifest.load(manifest_path)
    filtered = [r for r in manifest.records if r.id == job or r.id == Path(job).name]
    if not filtered:
        print("Manifest records:", [r.id for r in manifest.records])
        if len(manifest.records) == 1:
            filtered = list(manifest.records)
        else:
            sys.exit(f"Job {job} not in manifest {manifest_path}")
    manifest = Manifest(records=filtered)
    print(f"Using manifest record id={manifest.records[0].id}")

    out_dir = project / "results" / "af3_boltz_affinity"
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    mol_dir = cache / "mols"
    msa_dir = processed / "msa"
    constraints_dir = processed / "constraints" if (processed / "constraints").exists() else None
    template_dir = processed / "templates" if (processed / "templates").exists() else None
    extra_mols_dir = processed / "mols" if (processed / "mols").exists() else None

    install_af3_coord_hook()

    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs()
    diffusion_params = Boltz2DiffusionParams()
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False
    steering_args.contact_guidance_update = False

    predict_args = {
        "recycling_steps": 5,
        "sampling_steps": 1,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    print(f"Loading {aff_ckpt}")
    model = Boltz2.load_from_checkpoint(
        aff_ckpt,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
        use_kernels=not args.no_kernels,
    )
    model.eval()

    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=pred_dir,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=constraints_dir,
        template_dir=template_dir,
        extra_mols_dir=extra_mols_dir,
        override_method="other",
        affinity=True,
    )

    pred_writer = BoltzAffinityWriter(
        data_dir=processed / "structures" if (processed / "structures").exists() else pred_dir,
        output_dir=pred_dir,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        callbacks=[pred_writer],
        enable_checkpointing=False,
    )
    trainer.predict(model, datamodule=data_module, return_predictions=False)

    aff_json = pred_dir / job / f"affinity_{job}.json"
    if aff_json.exists():
        print(json.dumps(json.loads(aff_json.read_text()), indent=2))
        print(f"Wrote {aff_json}")
    else:
        # Writer uses record.id which may differ slightly.
        found = list(pred_dir.glob("**/affinity_*.json"))
        print("Affinity JSON not at expected path. Found:", found)
        for path in found:
            print(path)
            print(path.read_text())


if __name__ == "__main__":
    main()
