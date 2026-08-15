# Aptamer Design Lab — Technical Case Study

Case study submission for Professor Bento's Aptamer Design Lab, evaluating AlphaFold 3 and Boltz-2 for aptamer structure and affinity prediction.

## Structure

- `docs/` — setup notes and troubleshooting log
- `scripts/af3/` — AlphaFold 3 run scripts and modified inputs
- `scripts/boltz/` — Boltz-2 run scripts and modified inputs
- `data/` — dataset notes (raw data not tracked, see data/README.md)
- `results/` — prediction outputs and analysis (lightweight files only)
- `report/` — final 1-page report

## Environment

Andromeda HPC (BC), Slurm scheduler. AF3 and Boltz-2 installed natively via `uv` (Singularity was non-functional on this cluster — see `docs/setup_log.md` for details).
