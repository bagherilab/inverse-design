#!/usr/bin/env python3
"""Render S4 from original SIR statistics and canonical pm50 error summaries.

The SIR source is s3://bagherilab-working/pohao/inverse_design/sir/with_history/
n3_t365_l30.zip (sha256 f2099b72c6216d3174dbe8ec8c7764c350d6889e1f412df4a40a9bae8a5a9e87).
No existing raster content is reused. Both panels show mean +/- SD error bars with the lower end at max(mean-SD, 0).
"""
from pathlib import Path
import argparse, hashlib, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "body/extended_data/figures/error_iteration/ed_figure1_generation_convergence.png"
SIR_METRICS = ["peak_I", "time_to_peak", "final_R", "area_I", "growth_rate"]
# Preserve the target constants used by the original figure script.
SIR_TARGETS = np.array([0.837117, 33.378653, 0.832338, 0.448831, 0.006877])
# Formal ABC_SMC_RF_N1024_pm50 chain, generations 0--9.
ARCADE_GENERATION_MEAN = np.array(
    [
        22.003204,
        19.623944,
        16.418118,
        14.674043,
        13.797424,
        14.096706,
        13.806979,
        13.806916,
        13.374288,
        13.467998,
    ]
)
ARCADE_GENERATION_SD = np.array(
    [
        25.369941,
        22.520575,
        21.435539,
        20.456437,
        18.910836,
        18.676605,
        18.735075,
        17.731450,
        17.891964,
        18.475863,
    ]
)

# Main pm50 chains at generation 4, one chain for each nominal N.
N_VALUES = np.array([128, 256, 512, 1024])
ARCADE_N_MEAN = np.array([18.494858, 15.399306, 15.042215, 13.797424])
ARCADE_N_SD = np.array([21.408296, 18.663722, 17.457525, 18.910836])


def summary(path):
    frame = pd.read_csv(path)[SIR_METRICS].replace([np.inf, -np.inf], np.nan)
    errors = np.abs(frame.mean().to_numpy() - SIR_TARGETS) / SIR_TARGETS * 100
    return float(np.mean(errors)), float(np.std(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sir-dir", type=Path, required=True, help="Extracted with_history/n3_t365_l30 directory")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--evidence", type=Path)
    args = ap.parse_args()
    generation_paths = [args.sir_dir / f"n512/iter_{g}/statistics.csv" for g in range(10)]
    sample_paths = [args.sir_dir / f"n{n}/iter_4/statistics.csv" for n in N_VALUES]
    gm, gs = np.array([summary(p) for p in generation_paths]).T
    nm, ns = np.array([summary(p) for p in sample_paths]).T
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6,
        "axes.labelsize": 7, "axes.linewidth": .5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.835))
    fig.subplots_adjust(left=.07, right=.99, bottom=.15, top=.94, hspace=.40, wspace=.10)
    limits = {}
    for col, (name, marker, means, sds, nmeans, nsds) in enumerate([
        ("SIR", "o", gm, gs, nm, ns),
        ("ARCADE", "s", ARCADE_GENERATION_MEAN, ARCADE_GENERATION_SD, ARCADE_N_MEAN, ARCADE_N_SD),
    ]):
        for row in range(2):
            ax = axes[row, col]
            # Both panels: mean +/- population SD, lower end truncated at zero.
            x, mu, sd = (np.arange(len(means)), means, sds) if row == 0 else (np.asarray(N_VALUES), nmeans, nsds)
            lower = np.maximum(mu-sd, 0); upper = mu+sd
            ax.errorbar(x, mu, yerr=[mu-lower, upper-mu], marker=marker, color="black", lw=1.8, ms=5, capsize=4, elinewidth=1)
            if row == 0:
                ax.set_xticks(x); ax.set_xlabel("Generation")
            else:
                ax.set_xscale("log", base=2); ax.set_xticks(N_VALUES)
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter()); ax.set_xlabel("Sample size (N)")
            # Include every plotted endpoint, with a data-derived 5% range margin.
            lo, hi = min(0., float(lower.min())), float(upper.max())
            margin = .05 * (hi-lo)
            ax.set_ylim(lo-margin, hi+margin)
            assert ax.get_ylim()[0] < lower.min() and ax.get_ylim()[1] > upper.max()
            limits[f"{name}_{row}"] = dict(lower=lower.tolist(), upper=upper.tolist(), ylim=list(ax.get_ylim()))
            if col == 0: ax.set_ylabel("Error (%)")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            ax.grid(axis="y", ls="--", alpha=.4); ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(width=.5, length=2.5)
    for row, letter in enumerate("AB"):
        pos=axes[row,0].get_position(); fig.text(pos.x0-.024, pos.y1+.022, letter, weight="bold", size=9)
    fig.legend(handles=[Line2D([], [], color="black", marker=m, label=n) for m,n in [("o","SIR"),("s","ARCADE")]], loc="lower right", frameon=False, fontsize=7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600); fig.savefig(args.output.with_suffix(".pdf")); plt.close(fig)
    if args.evidence:
        args.evidence.write_text(json.dumps(dict(sir_generation_mean=gm.tolist(), sir_generation_sd=gs.tolist(), sir_sample_mean=nm.tolist(), sir_sample_sd=ns.tolist(), limits=limits, input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in generation_paths+sample_paths}), indent=2)+"\n")

if __name__ == "__main__":
    main()
