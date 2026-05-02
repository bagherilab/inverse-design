#!/usr/bin/env python3
"""Generate ED Table 1: MI sensitivity to k-neighbor parameter.

Outputs:
  data/ed_table1_mi_sensitivity.csv   — full rank table
  data/ed_table1_mi_sensitivity.tex   — LaTeX tabular
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).parent.parent / "data"
MI_SENS = DATA_DIR / "mi_sensitivity_results.json"
OUT_CSV = DATA_DIR / "ed_table1_mi_sensitivity.csv"
OUT_TEX = DATA_DIR / "ed_table1_mi_sensitivity.tex"

METRICS = ["doub_time", "symmetry", "colony_growth"]
METRIC_LABELS = {"doub_time": "DT", "symmetry": "Sym", "colony_growth": "CG"}
KS = [3, 5, 10]


def ranks_from_mi(mi_scores):
    arr = np.array(mi_scores)
    order = np.argsort(-arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


def main():
    with open(MI_SENS) as f:
        data = json.load(f)

    names = data[METRICS[0]]["k3"]["names"]
    n = len(names)

    rows = []
    for i, name in enumerate(names):
        row = {"Parameter": name}
        for metric in METRICS:
            label = METRIC_LABELS[metric]
            for k in KS:
                key = f"k{k}"
                mi = data[metric][key]["MI"][i]
                rank = int(ranks_from_mi(data[metric][key]["MI"])[i])
                row[f"{label}_MI_k{k}"] = round(mi, 4)
                row[f"{label}_rank_k{k}"] = rank
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

    # Spearman ρ between k values per metric
    rho_lines = []
    for metric in METRICS:
        label = METRIC_LABELS[metric]
        mi3 = data[metric]["k3"]["MI"]
        mi5 = data[metric]["k5"]["MI"]
        mi10 = data[metric]["k10"]["MI"]
        r35, p35 = spearmanr(mi3, mi5)
        r310, p310 = spearmanr(mi3, mi10)
        r510, p510 = spearmanr(mi5, mi10)
        rho_lines.append(
            f"{label}: ρ(k3,k5)={r35:.3f} (p={p35:.3e}), "
            f"ρ(k3,k10)={r310:.3f} (p={p310:.3e}), "
            f"ρ(k5,k10)={r510:.3f} (p={p510:.3e})"
        )

    print("\nSpearman ρ across k values:")
    for line in rho_lines:
        print(" ", line)

    # LaTeX table: show rank at each k for top-10 by k=3 MI, averaged across metrics
    avg_mi_k3 = np.mean(
        [data[m]["k3"]["MI"] for m in METRICS], axis=0
    )
    top10_idx = np.argsort(-avg_mi_k3)[:10]

    col_header = " & ".join(
        [f"\\textbf{{{METRIC_LABELS[m]}-k{k}}}" for m in METRICS for k in KS]
    )
    header = f"\\textbf{{Parameter}} & {col_header} \\\\\n\\hline"

    tex_rows = []
    for idx in top10_idx:
        name = names[idx]
        vals = []
        for metric in METRICS:
            for k in KS:
                rank = int(ranks_from_mi(data[metric][f"k{k}"]["MI"])[idx])
                vals.append(str(rank))
        tex_rows.append(name.replace("_", "\\_") + " & " + " & ".join(vals) + " \\\\")

    rho_caption = "; ".join(
        f"{METRIC_LABELS[m]}: min ρ={min(spearmanr(data[m]['k3']['MI'], data[m]['k10']['MI'])[0], spearmanr(data[m]['k5']['MI'], data[m]['k10']['MI'])[0]):.2f}"
        for m in METRICS
    )

    tex = (
        "% ED Table 1 — MI sensitivity to k-neighbor\n"
        "% Caption note: " + rho_caption + "\n"
        "\\begin{tabular}{l" + "r" * (len(METRICS) * len(KS)) + "}\n"
        "\\hline\n" + header + "\n"
        + "\n".join(tex_rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )

    OUT_TEX.write_text(tex)
    print(f"\nSaved {OUT_TEX}")


if __name__ == "__main__":
    main()
