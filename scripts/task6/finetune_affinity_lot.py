#!/usr/bin/env python3
"""
Leave-one-target-out fine-tune of Boltz-2 affinity heads on UTexas Kd.

Trunk/structure stay frozen. We only train AffinityModule weights on the
cached (s, z, x_pred) tensors from AF3-wired complexes.

Held-out target Spearman is the number that counts. Training-set rho will
look good and is mostly overfitting — say that in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from scipy.stats import spearmanr
from torch import nn

from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_affinity_from_af3 import PROJECT_ROOT, find_cache  # noqa: E402


def load_cache(cache_dir: Path) -> list[dict]:
    items = []
    for path in sorted(cache_dir.glob("*.pt")):
        items.append(torch.load(path, map_location="cpu", weights_only=False))
    if not items:
        sys.exit(f"No tensors in {cache_dir}. Run cache_affinity_tensors.py first.")
    return items


def affinity_params(model: Boltz2):
    params = []
    for name, module in model.named_modules():
        if "affinity" in name.lower() and list(module.parameters(recurse=False)):
            params.extend(p for p in module.parameters() if p.requires_grad)
    # Fallback: any parameter with affinity in the name.
    if not params:
        params = [p for n, p in model.named_parameters() if "affinity" in n]
    return params


def freeze_non_affinity(model: Boltz2, heads_only: bool = True) -> None:
    """Freeze trunk. By default only train the small output MLPs, not the
    AffinityModule Pairformer (518 tensors) — that is too slow/overfit for n=65.
    """
    for name, param in model.named_parameters():
        if heads_only:
            param.requires_grad = (
                "affinity_heads" in name
                or "to_affinity" in name
                or "affinity_out" in name
            )
        else:
            param.requires_grad = "affinity" in name


def predict_value(model: Boltz2, item: dict, device: torch.device) -> torch.Tensor:
    s_inputs = item["s_inputs"].to(device)
    z = item["z"].to(device)
    x_pred = item["x_pred"].to(device)
    feats = {}
    for k, v in item["feats"].items():
        if torch.is_tensor(v):
            feats[k] = v.to(device)
        elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
            feats[k] = [x.to(device) for x in v]
        else:
            feats[k] = v
    if getattr(model, "affinity_ensemble", False):
        o1 = model.affinity_module1(
            s_inputs=s_inputs, z=z, x_pred=x_pred, feats=feats, multiplicity=1, use_kernels=False
        )
        o2 = model.affinity_module2(
            s_inputs=s_inputs, z=z, x_pred=x_pred, feats=feats, multiplicity=1, use_kernels=False
        )
        return (o1["affinity_pred_value"] + o2["affinity_pred_value"]) / 2
    out = model.affinity_module(
        s_inputs=s_inputs, z=z, x_pred=x_pred, feats=feats, multiplicity=1, use_kernels=False
    )
    return out["affinity_pred_value"]


def load_fresh_model(heads_only: bool = True) -> Boltz2:
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
        msa_args=asdict(msa_args := MSAModuleArgs()),
        steering_args=asdict(steering_args),
        use_kernels=False,
    )
    freeze_non_affinity(model, heads_only=heads_only)
    return model


def spearman_or_none(preds, labels) -> tuple[float | None, float | None]:
    if len(preds) < 3 or len(set(labels)) < 2:
        return None, None
    rho, p = spearmanr(preds, labels)
    if rho != rho:  # nan
        return None, None
    return float(rho), float(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--full-module",
        action="store_true",
        help="Train the entire AffinityModule (slow, 518 tensors). Default is output heads only.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix for output files, e.g. heads → report/af3_boltz_affinity_finetune_lot_heads.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = args.project_root / "results" / "af3_boltz_affinity" / "tensor_cache"
    items = load_cache(cache_dir)
    by_target: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_target[item["target"]].append(item)

    targets = sorted(by_target)
    out_dir = args.project_root / "report"
    out_dir.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = out_dir / f"af3_boltz_affinity_finetune_lot{suffix}.csv"
    json_path = out_dir / f"af3_boltz_affinity_finetune_lot{suffix}.json"

    held_rows = []
    fold_lines = []
    done_targets = set()
    if json_path.exists():
        prev = json.loads(json_path.read_text())
        fold_lines = list(prev.get("folds", []))
        done_targets = {f["target"] for f in fold_lines}
        if csv_path.exists():
            import csv as csvmod

            with csv_path.open() as f:
                held_rows = list(csvmod.DictReader(f))
        print(f"Resuming; already finished: {sorted(done_targets)}", flush=True)

    print(f"Cached complexes: {len(items)} across {len(targets)} targets", flush=True)
    print(f"Mode: {'full AffinityModule' if args.full_module else 'output heads only'}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)

    import csv as csvmod

    for held in targets:
        if held in done_targets:
            print(f"\n=== SKIP (already done): {held} ===", flush=True)
            continue
        train_items = [it for t, group in by_target.items() if t != held for it in group]
        test_items = by_target[held]
        print(
            f"\n=== Hold out: {held}  (train {len(train_items)}, test {len(test_items)}) ===",
            flush=True,
        )

        model = load_fresh_model(heads_only=not args.full_module).to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        n_param = sum(p.numel() for p in params)
        print(f"  trainable tensors: {len(params)}  params: {n_param:,}", flush=True)
        if not params:
            sys.exit("No trainable affinity-head parameters found. Check module names.")
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(1, args.epochs + 1):
            total = 0.0
            for i, item in enumerate(train_items, start=1):
                opt.zero_grad(set_to_none=True)
                pred = predict_value(model, item, device).reshape(-1)
                y = torch.tensor([item["log10_kd_um"]], device=device, dtype=pred.dtype)
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()
                total += float(loss.detach())
                if i % 10 == 0:
                    print(
                        f"  epoch {epoch:02d}  step {i}/{len(train_items)}",
                        flush=True,
                    )
            print(
                f"  epoch {epoch:02d}  train MSE {total / len(train_items):.4f}",
                flush=True,
            )

        model.eval()
        preds, kds = [], []
        fold_rows = []
        with torch.no_grad():
            for item in test_items:
                pred = float(predict_value(model, item, device).reshape(-1)[0])
                preds.append(pred)
                kds.append(item["kd_nm"])
                fold_rows.append(
                    {
                        "job": item["job"],
                        "target": held,
                        "kd_nm": item["kd_nm"],
                        "log10_kd_um": item["log10_kd_um"],
                        "affinity_pred_ft": pred,
                    }
                )
        held_rows.extend(fold_rows)
        rho, pval = spearman_or_none(preds, kds)
        msg = (
            f"  held-out Spearman rho={rho} p={pval}"
            if rho is not None
            else "  held-out Spearman skipped (ties or n<3)"
        )
        print(msg, flush=True)
        fold_lines.append({"target": held, "n": len(test_items), "rho": rho, "p": pval})

        with csv_path.open("w", newline="") as f:
            writer = csvmod.DictWriter(
                f,
                fieldnames=["job", "target", "kd_nm", "log10_kd_um", "affinity_pred_ft"],
            )
            writer.writeheader()
            writer.writerows(held_rows)
        rhos = [f["rho"] for f in fold_lines if f["rho"] is not None]
        summary = {
            "folds": fold_lines,
            "mean_heldout_rho": (sum(rhos) / len(rhos)) if rhos else None,
            "epochs": args.epochs,
            "lr": args.lr,
            "full_module": args.full_module,
            "tag": args.tag,
        }
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"  saved checkpoint {json_path}", flush=True)

        del model
        torch.cuda.empty_cache()

    rhos = [f["rho"] for f in fold_lines if f["rho"] is not None]
    summary = {
        "folds": fold_lines,
        "mean_heldout_rho": (sum(rhos) / len(rhos)) if rhos else None,
        "epochs": args.epochs,
        "lr": args.lr,
        "full_module": args.full_module,
        "tag": args.tag,
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
