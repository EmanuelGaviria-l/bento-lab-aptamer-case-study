# Dataset selection: UTexas Aptamer Database subset

Source: [UTexas Aptamer Database, Zenodo record 8387047](https://zenodo.org/records/8387047) (Sept 2023 version, 1,495 entries).

## Filters applied (see `prepare_dataset.py` for the exact logic)

| Step | Rows remaining |
|---|---|
| Start | 1,495 |
| ssDNA / ssRNA only (drop chemically modified aptamers AF3/Boltz-2 can't represent) | 1,334 |
| Kd (nM) reported | 977 |
| Restricted to 7 well-defined single-protein targets with >=7 replicates | 73 |

## Targets kept

Thrombin (Human), VEGF, bFGF, HIV-1 Rev, IgE (Human), ABF1 protein, Abrin toxin — chosen for having enough aptamer replicates per target (8-13) for a meaningful within-target Spearman correlation (Task 7), and for being single, well-characterized proteins rather than small molecules, whole cells, or huge multi-subunit complexes.

## Known limitation / next step

The dataset gives target *names* only, not sequences. Each target's protein sequence (e.g. via UniProt) still needs to be looked up before building AF3/Boltz-2 complex inputs.

## Excluded target types (with rationale)

- Small molecules (e.g. Hematoporphyrin IX, Chloramphenicol, 17-beta-Estradiol) — would require ligand (SMILES) modeling, a different setup than an aptamer-target structural complex.
- Whole cells / organisms (e.g. HepG2 cells, S. typhimurium) — no single target structure.
- E. coli RNA polymerase — large multi-subunit complex (~400 kDa), excluded for compute tractability given the project deadline.
