# Aptamer Design Lab — Technical Case Study

Case study submission for Professor Bento's Aptamer Design Lab: evaluating AlphaFold 3 and Boltz-2 for aptamer structure and binding-affinity prediction, using the [UTexas Aptamer Database](https://zenodo.org/records/8387047).

Cluster: `/projects/bentosprg6/gavirial/bento-lab-aptamer-case-study` (Andromeda). Do not run AF3 or Boltz-2 on the login node.

## What's here

- **`docs/`** — setup notes and troubleshooting logs (Singularity, CUDA, MSA server, batch runs)
- **`scripts/af3/`** — build AF3 JSON inputs, batch runners, score collector
- **`scripts/boltz/`** — build Boltz-2 YAML inputs (with and without MSA), batch runners, affinity ranking + Spearman
- **`scripts/msa/`** — unique protein chains and ColabFold MSA cache in `data/msa_cache/` (9 chains on the assigned set; 33 with the expansion)
- **`scripts/task6/`** — copy AF3 coordinates into Boltz-2's affinity head (zero-shot) and leave-one-target-out fine-tune
- **`data/`** — assigned set: `aptamer_subset.csv` + `targets.yaml` (73 aptamers / 7 proteins). Extra set: `aptamer_subset_v2.csv` + `targets_v2.yaml` (193 / 30). Selection rationale in `README_data_selection.md`
- **`inputs/`** — in git: `inputs/af3/`, `inputs/boltz/`, `inputs/boltz_msa/`, `inputs/boltz_msa_v2/`. Not in git: `inputs/af3_msa/` and `inputs/af3_msa_v2/` (inlined MSAs; regenerate with `--msa-cache`)
- **`results/`** — prediction dumps on Andromeda only (`.cif` / `.npz` / `.pt`; not tracked). Rankings live in `report/`
- **`report/`** — ranked affinity CSVs, Spearman tables, 1-page report

## Environment

Andromeda HPC (Boston College), Slurm. AF3 and Boltz-2 are installed **natively** via `uv`. `singularity exec` works; `singularity build --fakeroot` failed on this account (`libsubid`). See `docs/*_log.txt`.

Boltz-2 required a source patch (`schema.py`, `parse_polymer()`, and a required `mw` field) so DNA/RNA can be the affinity binder; upstream only validates small-molecule ligands. Predict jobs use `--no_kernels`.

AF3 MSA inputs inline the protein alignment as `unpairedMsa` (empty `pairedMsa`) so `--norun_data_pipeline` works. Use `$HOME/alphafold3/.venv/bin/python` from `~/alphafold3`.

```bash
source /home/gavirial/boltz/.venv-boltz/bin/activate   # Boltz-2
cd ~/alphafold3 && source .venv/bin/activate           # AF3
```

## Dataset

Filtered from 1,495 entries to **73 aptamers / 7 single-protein targets** (ssDNA/ssRNA, numeric Kd, ≥7 clones): Thrombin, VEGF, bFGF, HIV-1 Rev, IgE, ABF1, Abrin. Sequences use UniProt mature/processed chains where relevant (Thrombin and Abrin: two chains each; VEGF, bFGF: cleaved forms). Full rationale in `data/README_data_selection.md`.

All 12 HIV-1 Rev entries share a single reported Kd (27.5 nM, Xu & Ellington 1996) — rank correlation is undefined for this target.

Extra robustness set (does not replace the 73): **193 aptamers / 30 proteins**. Same nucleic-acid + numeric-Kd filters, ≥4 clones and ≥2 distinct Kds, UniProt-able proteins. Dropped Cytotoxin CNY (cylindrospermopsin is a small molecule). Construct notes are in `data/targets_v2.yaml`.

## Spearman sign convention

- Boltz `affinity_pred_value` (log10 IC50 µM): lower = tighter → **agreement is a positive ρ**
- Boltz binder probability and AF3 ipTM: higher = better → **agreement is a negative ρ**
- Prefer **mean per-target ρ**. Pooled n mixes proteins with different typical Kd. Skip a target when Kd or the model score is constant.

## What was done (Tasks 1–7 + bonus)

1. **Environment setup** — native AF3 + Boltz-2 install, workflow familiarization
2. **Dataset preparation** — filtering, target sequence correction, input generation
3. **AF3 structure predictions** — all 73 aptamer–target complexes (MSA-cached); extra 120 on the v2 set
4. **Boltz-2 affinity adaptation** — patched to accept nucleic-acid binders
5. **Boltz-2 affinity predictions** — all 73 complexes (without and with protein MSA), ranked by predicted affinity; extra 120 on v2
6. **Bonus: affinity module transplant** — wired AF3 coordinates into Boltz-2's affinity head (zero-shot), then leave-one-target-out fine-tune (full module and heads-only on 73; heads-only on 193)
7. **Correlation analysis** — Spearman between predicted scores and experimental Kd, pooled and per-target

## Key finding

No method showed a statistically significant, correctly directed correlation with experimental Kd once evaluated **per-target** rather than pooled. The one small-n exception on the assigned set is Abrin under AF3 interface confidence (ipTM), ρ = −0.79 (p = 0.02, n = 8). Cached protein MSA did not improve Boltz-2 ranking over single-sequence mode. Pooled binder-probability (ρ ≈ −0.26, p = 0.027) and pooled AF3 ipTM (ρ ≈ +0.25, wrong direction) look stronger than the within-target means; those pooled hits are between-protein artifacts.

In the bonus task, AF3-derived coordinates and Boltz-2's own predicted pose gave nearly identical affinity scores for the same aptamer — the affinity head relies more on sequence embeddings than 3D structure. Leave-one-target-out fine-tuning made cross-protein generalization *worse* than zero-shot (mean held-out ρ: +0.16 → −0.09 heads-only → −0.30 full module).

The 193-aptamer expansion washed the pooled artifacts out (Boltz pred_value pooled ρ ≈ 0; AF3 ipTM pooled ρ ≈ −0.10) and still did not recover experimental order. Heads-only LOT on 30 proteins: mean held-out ρ ≈ −0.08. Full tables and limitations are in `report/`.

## Reproducing

GPU jobs must run on a Slurm GPU node. Ranking already-computed `report/` CSVs is CPU-only.

```bash
# Protein MSAs once → data/msa_cache/
python3 scripts/msa/extract_unique_chains.py
python3 scripts/msa/fetch_protein_msas.py

# Assigned 73 inputs
python3 scripts/af3/build_af3_inputs.py
python3 scripts/af3/build_af3_inputs.py --msa-cache
python3 scripts/boltz/build_boltz_inputs.py
python3 scripts/boltz/build_boltz_inputs.py --msa-cache

# GPU — AF3 must use ~/alphafold3/.venv
bash scripts/af3/run_all.sh
bash scripts/af3/run_all_msa.sh
bash scripts/boltz/run_all_boltz.sh
bash scripts/boltz/run_all_boltz_msa_cached.sh   # not run_all_boltz_msa.sh

# Rank + Spearman (no-MSA, then MSA)
python3 scripts/boltz/rank_boltz_affinity.py
python3 scripts/boltz/correlation_analysis.py
python3 scripts/boltz/rank_boltz_affinity.py \
  --results-dir results/boltz_msa --output report/boltz_affinity_ranking_msa.csv
python3 scripts/boltz/correlation_analysis.py \
  --ranking-csv report/boltz_affinity_ranking_msa.csv \
  --output report/correlation_analysis_msa.txt \
  --label "Boltz-2 (cached protein MSAs)"
```

v2 extra set (193 / 30). Do not overwrite `aptamer_subset.csv`, `targets.yaml`, or the original 73 `report/` files.

```bash
python3 scripts/msa/extract_unique_chains.py --targets data/targets_v2.yaml
python3 scripts/msa/fetch_protein_msas.py

python3 scripts/boltz/build_boltz_inputs.py --msa-cache \
  --csv data/aptamer_subset_v2.csv --targets data/targets_v2.yaml \
  --output-dir inputs/boltz_msa_v2
python3 scripts/af3/build_af3_inputs.py --msa-cache \
  --csv data/aptamer_subset_v2.csv --targets data/targets_v2.yaml \
  --output-dir inputs/af3_msa_v2

bash scripts/boltz/run_all_boltz_msa_v2.sh
bash scripts/af3/run_all_msa_v2.sh
bash scripts/task6/run_all_af3_affinity_v2.sh

python3 scripts/task6/cache_affinity_tensors.py --v2
python3 scripts/task6/finetune_affinity_lot.py --v2
```

Rankers take `--csv data/aptamer_subset_v2.csv` and write `report/*_v2*`.
