# Aptamer Design Lab — Technical Case Study

Case study submission for Professor Bento's Aptamer Design Lab: evaluating AlphaFold 3 and Boltz-2 for aptamer structure and binding-affinity prediction, using the [UTexas Aptamer Database](https://zenodo.org/records/8387047).

## What's here

- **`docs/`** - setup notes and troubleshooting log (Singularity, failure, CUDA version conflicts, MSA server issues)
- **`scripts/af3/`** - build AF3 JSON inputs, batch runner
- **`scripts/boltz/`** - build Boltz-2 YAML inputs (with and without MSA), batch runner, affinity ranking + Spearman correlation analysis
- **`scripts/msa/`** - extract the 9 unique protein chains and cache ColabFold MSAs in `data/msa_cache/`
- **`scripts/task6/`** - copy AF3 coordinates into Boltz-2's affinity head (zero-shot) and leave-one-target-out fine-tune
- **`data/`** - `aptamer_subset.csv` (the 73-aptamer / 7-target working set), `targets.yaml` (UniProt IDs + corrected mature-chain residue ranges), selection rationale
- **`inputs/`** - generated AF3 and Boltz-2 input files (inputs/af3/_msa/ not tracked in git; regenerate via the scripts below)
- **`results/`** - prediction outputs (structures, confidences, affinity JSONs; raw `.cif` / `.npz` / `.pt` not tracked, ~2.8GB on Andromeda)
- **`report/`** - Ranked affinity CSVs, correlation analyses, final 1-page report

## Environment

Andromeda HPC (Boston College), Slurm scheduler. Both AF3 and Boltz-2 are installed **natively** via `uv` rather than in containers - Singularity was non-functional on this cluster (see `docs/*_log.txt`). Boltz-2 required a source patch (`schema.py`, `parse_polymer()`, and a required `mw` field) to support DNA/RNA chains as the affinity "binder"; upstream only validates for small-molecule ligands, though the underlying tokenizer/model architecture has no such restriction.
AF3 MSA inputs inline the protein alignment as `unpairedMsa` (with an empty `pairedMsa`) so `--norun_data_pipeline` works. Boltz-2 predict jobs use `--no_kernels`.

## Dataset

Filtered from 1,495 total entries to 73 aptamers (ssDNA/ssRNA, numeric Kd reported) across 7 single-protein targets with ≥7 clones each: Thrombin, VEGF, bFGF, HIV-1 Rev, IgE, ABF1, Abrin. Targets: sequences were corrected from UniProt precursor to mature/processed form where relevant (Thrombin, Abrin: two chains each; VEGF, bFGF: cleaved forms). Full rationale in `data/README_data_selection.md`. Note: all 12 HIV-1R Rev entries share a single reported Kd (27.5 nM, Xu & Ellington 1996) - rank correlation is undefined for this target.

## What was done (Tasks 1-7 + bonus)

1. **Environment setup** - native AF3 + Boltz-2 install, workflow familiarization
2. **Dataset preparation** - filtering, target sequence correction, input generation
3. **AF3 structure predictions** - all 73 aptamer-target complexes
4. **Boltx-2 affinity adaptation** - patched to accept nucleic-acid binders
5. **Boltz-2 affinity predictions** - all 73 complexes (both without and with real MSA), ranked by predicted affinity
6. **Bonus: affinity module transplant** - wired AF3's predicted coordinates into Boltz-2's affinity head (zero-shot), then fine-tuned with leave-one-target-out cross-validation (full module and heads-only variants)
7. **Correlation analysis** Spearman rank correlation between predicted affinity and experimental Kd, pooled and per-target, across all of the above

## Key finding

No method showed a statistically significant, correctly directed correlation with experimental Kd once evaluated per-target rather than pooled. The one exception was Abrin under AF3 interface confidence (ipTM), rho = -0.79 (p = 0.02, n = 8). Real MSA did not improve Boltz-2's pooled correlation over single-sequence mode. In the bonus task, AF3-derived coordinates and Boltz-2'sown predicted pose gave nearly identical affinity scores for the same aptamer - suggesting the affinity head relies more on sequence embeddings than 3D structure - and leave-one-target-out fine-tuning made cross-protein generalization *worse* than a zero-shot baseline (mean held-out rho: +0.16 -> -0.09 -> -0.30 as more of the module was fine-tuned), consistent with overfitting to the 6 training targets. Full analysis and limitations in `report/`.

## Reproducing

```bash
# Protein MSAs once (9 chains) → data/msa_cache/
python3 scripts/msa/extract_unique_chains.py
python3 scripts/msa/fetch_protein_msas.py

# Inputs
python3 scripts/af3/build_af3_inputs.py
python3 scripts/af3/build_af3_inputs.py --msa-cache
python3 scripts/boltz/build_boltz_inputs.py
python3 scripts/boltz/build_boltz_inputs.py --msa-cache

# GPU node — AF3 must use ~/alphafold3/.venv
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
