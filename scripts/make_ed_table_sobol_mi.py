#!/usr/bin/env python3
"""Generate ED Table 2: MI vs Sobol ST rank comparison.

Outputs:
  data/ed_table2_sobol_mi.csv   — full comparison table
  data/ed_table2_sobol_mi.tex   — LaTeX tabular (top 10 by MI rank)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).parent.parent / "data"
MI_JSON = DATA_DIR / "mi_results_k3.json"
SOBOL_JSON = DATA_DIR / "sobol_results.json"
OUT_CSV = DATA_DIR / "ed_table2_sobol_mi.csv"
OUT_TEX = DATA_DIR / "ed_table2_sobol_mi.tex"

METRICS = ["doub_time", "symmetry", "colony_growth"]
METRIC_LABELS = {"doub_time": "DT", "symmetry": "Sym", "colony_growth": "CG"}

# Parameters to flag in LaTeX output
AGREEMENT_PARAM = "CELL_VOLUME_MU"
ARTIFACT_PARAM = "COMPRESSION_TOLERANCE"


def ranks_from_scores(scores, higher_is_better=True):
    arr = np.array(scores)
    if higher_is_better:
        order = np.argsort(-arr)
    else:
        order = np.argsort(arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


def main():
    with open(MI_JSON) as f:
        mi_data = json.load(f)
    with open(SOBOL_JSON) as f:
        sobol_data = json.load(f)

    names = mi_data[METRICS[0]]["names"]

    rows = []
    for i, name in enumerate(names):
        row = {"Parameter": name}
        for metric in METRICS:
            label = METRIC_LABELS[metric]
            mi_score = mi_data[metric]["MI"][i]
            sobol_st = sobol_data[metric]["ST"][i]
            mi_rank = int(ranks_from_scores(mi_data[metric]["MI"])[i])
            sobol_rank = int(ranks_from_scores(sobol_data[metric]["ST"])[i])
            row[f"{label}_MI"] = round(mi_score, 4)
            row[f"{label}_MI_rank"] = mi_rank
            row[f"{label}_ST"] = round(sobol_st, 4)
            row[f"{label}_ST_rank"] = sobol_rank
            row[f"{label}_rank_diff"] = sobol_rank - mi_rank
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

    # Spearman ρ between MI and Sobol ST rankings per metric
    print("\nSpearman ρ (MI vs Sobol ST rankings):")
    for metric in METRICS:
        label = METRIC_LABELS[metric]
        mi_scores = mi_data[metric]["MI"]
        st_scores = sobol_data[metric]["ST"]
        rho, pval = spearmanr(mi_scores, st_scores)
        print(f"  {label}: ρ={rho:.3f}, p={pval:.3e}")

    # LaTeX: top 10 by average MI rank across metrics
    avg_mi = np.mean([mi_data[m]["MI"] for m in METRICS], axis=0)
    top10_idx = np.argsort(-avg_mi)[:10]

    col_header = " & ".join(
        [
            f"\\textbf{{{METRIC_LABELS[m]}-MI-Rk}} & \\textbf{{{METRIC_LABELS[m]}-ST-Rk}}"
            for m in METRICS
        ]
    )
    header = f"\\textbf{{Parameter}} & {col_header} \\\\\n\\hline"

    tex_rows = []
    for idx in top10_idx:
        name = names[idx]
        tex_name = name.replace("_", "\\_")
        if name == AGREEMENT_PARAM:
            tex_name = f"\\textbf{{{tex_name}}}$^\\dagger$"
        elif name == ARTIFACT_PARAM:
            tex_name = f"{tex_name}$^*$"
        vals = []
        for metric in METRICS:
            mi_rank = int(ranks_from_scores(mi_data[metric]["MI"])[idx])
            st_rank = int(ranks_from_scores(sobol_data[metric]["ST"])[idx])
            vals.extend([str(mi_rank), str(st_rank)])
        tex_rows.append(tex_name + " & " + " & ".join(vals) + " \\\\")

    tex = (
        "% ED Table 2 — MI vs Sobol ST rank comparison (top 10 by MI)\n"
        "% $\\dagger$ = consensus (high in both); $^*$ = posterior-width artifact\n"
        "\\begin{tabular}{l" + "rr" * len(METRICS) + "}\n"
        "\\hline\n" + header + "\n"
        + "\n".join(tex_rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )

    OUT_TEX.write_text(tex)
    print(f"\nSaved {OUT_TEX}")

    # Print COMPRESSION_TOLERANCE ranks for context
    ct_idx = names.index(ARTIFACT_PARAM)
    print(f"\n{ARTIFACT_PARAM} ranks:")
    for metric in METRICS:
        mi_rank = int(ranks_from_scores(mi_data[metric]["MI"])[ct_idx])
        st_rank = int(ranks_from_scores(sobol_data[metric]["ST"])[ct_idx])
        print(f"  {METRIC_LABELS[metric]}: MI rank={mi_rank}, Sobol ST rank={st_rank}")

    cv_idx = names.index(AGREEMENT_PARAM)
    print(f"\n{AGREEMENT_PARAM} ranks:")
    for metric in METRICS:
        mi_rank = int(ranks_from_scores(mi_data[metric]["MI"])[cv_idx])
        st_rank = int(ranks_from_scores(sobol_data[metric]["ST"])[cv_idx])
        print(f"  {METRIC_LABELS[metric]}: MI rank={mi_rank}, Sobol ST rank={st_rank}")


if __name__ == "__main__":
    main()
