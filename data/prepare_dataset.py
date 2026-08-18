"""
Prepare a working subset of the UTexas Aptamer Database (Sept 2023 version)
for AlphaFold 3 / Boltz-2 structural and affinity prediction.

Selection rationale (documented for the case study report):

1. Nucleic acid type: restricted to pure ssDNA / ssRNA. AF3/Boltz-2 model
   canonical nucleotides; chemically modified aptamers (2'-fluoro-RNA,
   2'-amino-RNA, LNA, etc.) use non-standard residues the models cannot
   represent, so they are excluded.

2. Kd (nM): rows with no reported Kd are dropped -- affinity correlation
   (Task 7) requires a numeric experimental value to compare against.

3. Target type: only targets that are single, well-defined PROTEINS with
   a determinable amino-acid sequence are kept. AF3/Boltz-2 predict
   aptamer-target COMPLEXES, which requires a structural representation
   of the target. This excludes:
     - small molecules (e.g. Hematoporphyrin IX, Chloramphenicol,
       17-beta-Estradiol) -- these would need to be modeled as ligands
       (SMILES), not as the "target" chain in a nucleic-acid complex,
       which is a fundamentally different modeling setup than what
       Task 3/4 asks for (aptamer-target structural complexes).
     - whole cells / cell lines / organisms (e.g. HepG2 cells,
       S. typhimurium, cancer cell lines) -- there is no single defined
       molecular structure to fold against.
     - very large multi-subunit complexes (e.g. E. coli RNA polymerase,
       ~400 kDa, multiple subunits) -- excluded to keep compute
       tractable given the project deadline; noted as a limitation.

4. Target selection: kept targets with >=7 aptamer entries after the
   above filters, to have enough data points per target for a
   meaningful within-target Spearman correlation in Task 7. Prioritized
   single-chain, well-characterized proteins commonly used as aptamer
   benchmarks in the literature (thrombin, VEGF, bFGF, HIV-1 Rev, IgE),
   which also makes results easier to sanity-check against known
   structures/literature.

This is a deliberate, documented judgment call given the open-ended
nature of the task and the Aug 31 deadline -- not the only valid
approach.
"""

import pandas as pd

SOURCE = "utexas_aptamer_raw.xlsx"
OUTPUT = "aptamer_subset.csv"

TARGETS_KEPT = [
    "Thrombin, Human",
    "Vascular Endothelial Growth Factor (VEGF)",
    "Basic fibroblast growth factor (bFGF)",
    "Human immunodeficiency virus type 1 Rev (HIV-1 Rev)",
    "Immunoglobulin E (IgE), Human",
    "Autonomously replicating sequence\u2010binding factor 1 (ABF1) protein",
    "Abrin toxin, Abrus precatorius (A.precatorius)",
]

EXCLUDED_TARGET_TYPES = {
    "Hepatoma HepG2 cells, Human": "whole cell line, no single target structure",
    "Hematoporphyrin IX (HPIX)": "small molecule, not a protein/nucleic-acid target",
    "17 beta-Estradiol (17\u03b2-Estradiol) (E2\uff09": "small molecule",
    "Salmonella typhimurium (S. typhimurium)": "whole organism",
    "Chloramphenicol (Cam)": "small molecule",
    "Escherichia coli (E. coli) core bacterial RNA polymerase (RNAP)": (
        "large multi-subunit complex (~400 kDa); excluded for tractability, "
        "noted as a limitation"
    ),
}


def main():
    df = pd.read_excel(SOURCE)
    df.columns = [c.strip() for c in df.columns]

    n_start = len(df)

    # Step 1: pure ssDNA / ssRNA only
    df = df[df["Type of Nucleic Acid"].isin(["ssDNA", "ssRNA"])]
    n_after_type = len(df)

    # Step 2: must have a numeric Kd
    df = df[df["Kd (nM)"].notna()]
    n_after_kd = len(df)

    # Step 3: restrict to the curated protein-target list
    df = df[df["Target"].isin(TARGETS_KEPT)]
    n_after_target = len(df)

    # Keep only the columns relevant to structural/affinity modeling
    cols = [
        "Serial Number",
        "Name of Aptamer",
        "Target",
        "Type of Nucleic Acid",
        "Aptamer Sequence",
        "Sequence Length",
        "GC Content",
        "Kd (nM)",
        "Journal DOI",
        "Citation",
    ]
    df_out = df[cols].sort_values(["Target", "Kd (nM)"]).reset_index(drop=True)
    df_out.to_csv(OUTPUT, index=False)

    print(f"Starting rows:                 {n_start}")
    print(f"After ssDNA/ssRNA filter:       {n_after_type}")
    print(f"After Kd not-null filter:       {n_after_kd}")
    print(f"After protein-target selection: {n_after_target}")
    print(f"\nFinal dataset written to {OUTPUT}: {len(df_out)} aptamers "
          f"across {df_out['Target'].nunique()} targets\n")
    print("Aptamers per target:")
    print(df_out["Target"].value_counts())


if __name__ == "__main__":
    main()
