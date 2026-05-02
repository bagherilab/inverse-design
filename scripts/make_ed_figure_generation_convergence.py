#!/usr/bin/env python3
"""Generate ED Figure 1: generation-wise total error convergence for SIR and ARCADE.

Output: results/figures/error_iteration/ed_figure1_generation_convergence.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inverse_design.plotting.style import (
    FONT_BASE,
    FONT_LABEL,
    FONT_PANEL,
    WIDTH_DOUBLE,
    apply_style,
    savefig_both,
)

OUT_DIR = Path(__file__).parent.parent / "results" / "figures" / "error_iteration"
OUT_PNG = OUT_DIR / "ed_figure1_generation_convergence.png"

SIR_DIR = Path("/home/pohaoc2/UW/bagherilab/SIR_OUTPUT/n3_t365_l30/n512")
SIR_METRICS = ["peak_I", "time_to_peak", "final_R", "area_I", "growth_rate"]
SIR_TARGETS = np.array([0.837117, 33.378653, 0.832338, 0.448831, 0.006877])
SIR_N_ITER = 10

ARCADE_DIR = Path(
    "/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_breast"
)
ARCADE_METRICS = ["doub_time", "doub_time_std", "symmetry", "symmetry_std", "colony_growth"]
ARCADE_TARGETS = np.array([45.5, 13.79, 0.806, 0.067, 18.3])
ARCADE_N_ITER = 10

_SIR_BASE = Path("/home/pohaoc2/UW/bagherilab/SIR_OUTPUT/n3_t365_l30")
SIR_N_DIRS = [_SIR_BASE / f"n{n}" for n in [128, 256, 512, 1024]]
SIR_N_VALUES = [128, 256, 512, 1024]

_ARCADE_BASE = Path("/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT")
ARCADE_N_DIRS = [
    _ARCADE_BASE / "ABC_SMC_RF_N128_combined_grid_breast",
    _ARCADE_BASE / "ABC_SMC_RF_N256_combined_grid_breast",
    _ARCADE_BASE / "ABC_SMC_RF_N512_combined_grid_breast",
    _ARCADE_BASE / "ABC_SMC_RF_N1024_combined_grid_breast",
]
ARCADE_N_VALUES = [128, 256, 512, 1024]


def pct_error(sim_val, target):
    if target == 0:
        return np.nan
    return abs(sim_val - target) / abs(target) * 100


def total_error_per_iter(folder, metric_names, targets, n_iters):
    means, stds = [], []
    for i in range(n_iters):
        csv_path = folder / f"iter_{i}" / "final_metrics.csv"
        df = pd.read_csv(csv_path)
        metric_errors = []
        for col, tgt in zip(metric_names, targets):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            metric_errors.append(pct_error(vals.mean(), tgt))
        means.append(float(np.nanmean(metric_errors)))
        stds.append(float(np.nanstd(metric_errors)))
    return means, stds


def error_vs_n(n_dirs, metric_names, targets, iter_num=4):
    means, stds = [], []
    for folder in n_dirs:
        csv_path = folder / f"iter_{iter_num}" / "final_metrics.csv"
        df = pd.read_csv(csv_path)
        metric_errors = []
        for col, tgt in zip(metric_names, targets):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            metric_errors.append(pct_error(vals.mean(), tgt))
        means.append(float(np.nanmean(metric_errors)))
        stds.append(float(np.nanstd(metric_errors)))
    return means, stds


def _row_panel_letters(fig, axes):
    """Place one bold panel letter per row (A = generation, B = sample size), figure coords."""
    for row, letter in enumerate("AB"):
        row_axes = axes[row, :]
        posns = [ax.get_position() for ax in row_axes]
        x0 = min(p.x0 for p in posns)
        y1 = max(p.y1 for p in posns)
        fig.text(
            x0 - 0.012,
            y1 + 0.018,
            letter,
            fontsize=FONT_PANEL,
            fontweight="bold",
            va="bottom",
            ha="right",
            clip_on=False,
        )


def _style_ax(ax, show_ylabel=True):
    ax.set_ylabel("Error (%)" if show_ylabel else "", fontsize=FONT_LABEL)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_BASE, width=0.5, length=2.5)


def main():
    apply_style()
    print("Computing SIR generation errors...")
    sir_means, sir_stds = total_error_per_iter(SIR_DIR, SIR_METRICS, SIR_TARGETS, SIR_N_ITER)
    print("Computing ARCADE generation errors...")
    arcade_means, arcade_stds = total_error_per_iter(
        ARCADE_DIR, ARCADE_METRICS, ARCADE_TARGETS, ARCADE_N_ITER
    )
    print("Computing SIR sample-size errors...")
    sir_n_means, sir_n_stds = error_vs_n(SIR_N_DIRS, SIR_METRICS, SIR_TARGETS)
    print("Computing ARCADE sample-size errors...")
    arcade_n_means, arcade_n_stds = error_vs_n(ARCADE_N_DIRS, ARCADE_METRICS, ARCADE_TARGETS)

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_DOUBLE, 3.60))

    # Row 0: generation convergence
    gen_configs = [
        ("SIR (N=512)", sir_means, sir_stds, "o", axes[0, 0], True),
        ("ARCADE (N=512)", arcade_means, arcade_stds, "s", axes[0, 1], False),
    ]
    for label, means, stds, marker, ax, show_ylabel in gen_configs:
        gens = list(range(len(means)))
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(gens, means_arr, marker=marker, color="black", linewidth=1.8, markersize=5,
                label=label)
        ax.fill_between(
            gens, np.maximum(means_arr - stds_arr, 0), means_arr + stds_arr,
            color="gray", alpha=0.2,
        )
        ax.set_xlabel("Generation", fontsize=FONT_LABEL)
        ax.set_xticks(gens)
        _style_ax(ax, show_ylabel=show_ylabel)

    # Row 1: error vs. sample size
    n_configs = [
        ("SIR", SIR_N_VALUES, sir_n_means, sir_n_stds, "o", axes[1, 0], True),
        ("ARCADE", ARCADE_N_VALUES, arcade_n_means, arcade_n_stds, "s", axes[1, 1], False),
    ]
    for label, n_vals, means, stds, marker, ax, show_ylabel in n_configs:
        ax.errorbar(
            n_vals, means, yerr=stds,
            marker=marker, color="black", linewidth=1.8, markersize=5,
            capsize=4, elinewidth=1, label=label,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_vals)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("Sample size (N)", fontsize=FONT_LABEL)
        _style_ax(ax, show_ylabel=show_ylabel)

    # Single figure-level legend bottom-right
    import matplotlib.lines as mlines
    sir_handle = mlines.Line2D([], [], color="black", marker="o", markersize=5, label="SIR")
    arcade_handle = mlines.Line2D([], [], color="black", marker="s", markersize=5, label="ARCADE")
    fig.legend(handles=[sir_handle, arcade_handle], loc="lower right",
               fontsize=FONT_LABEL, frameon=False, bbox_to_anchor=(1.0, -0.05))

    fig.tight_layout()
    _row_panel_letters(fig, axes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    savefig_both(fig, OUT_PNG)
    print(f"Saved {OUT_PNG}")

    # Print table for manuscript
    print("\n--- Generation error table ---")
    print(f"{'Gen':>4}  {'SIR mean':>10}  {'SIR std':>8}  {'ARCADE mean':>12}  {'ARCADE std':>10}")
    for i, (sm, ss, am, as_) in enumerate(zip(sir_means, sir_stds, arcade_means, arcade_stds)):
        print(f"{i:>4}  {sm:>9.2f}%  {ss:>7.2f}%  {am:>11.2f}%  {as_:>9.2f}%")


if __name__ == "__main__":
    main()
