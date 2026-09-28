#!/usr/bin/env python3
"""Generate 5-panel SIR summary figure."""

import pathlib
import random
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inverse_design.analyze.analyze_sir import (
    analyze_parameter_errors,
    load_histories_from_dir,
    plot_joint_distribution_4params_compact,
    plot_sensitivity_heatmap,
)
from inverse_design.models.sir.sir import SIR_ABM
from inverse_design.plotting.style import (  # noqa: E402
    FONT_BASE,
    FONT_LABEL,
    FONT_METRIC,
    FONT_PANEL,
    WIDTH_DOUBLE,
    apply_style,
    savefig_both,
)
from inverse_design.plotting.colormap import sir_compartment_color

# Unified typography for panels A-D from the shared figure style.
FIG_TICK = FONT_BASE + 2
FIG_AXIS_LABEL = FONT_LABEL + 2
FIG_METRIC = FONT_METRIC + 2
FIG_PANEL = FONT_PANEL + 2
PANEL_B_RIGHT_SHIFT = 0.025
PANEL_D_RIGHT_SHIFT = 0.03

HIST_DIR = ROOT / "results/SIR/with_history/n3_t365_l30/n512"
BASE_DIR = ROOT / "results/SIR/n3_t365_l30/n512"
TARGET = ROOT / "results/SIR/n3_t365_l30/target_values.csv"
TRUE_P = ROOT / "results/SIR/n3_t365_l30/true_params.csv"
OUT = ROOT / "results/figures/SIR/SIR.png"
# Display labels matching the manuscript notation for PI, PR, IIF, and ISF.
PARAM_LABELS = [r"$P_I$", r"$P_R$", r"$I_{IF}$", r"$I_{SF}$"]


def _panel_label(ax, label, x=-0.20):
    ax.text(
        x,
        1.16,
        label,
        transform=ax.transAxes,
        fontsize=FIG_PANEL,
        fontweight="bold",
        va="top",
        clip_on=False,
    )


def _run_reference_sir(seed, true_params):
    np.random.seed(seed)
    random.seed(seed)
    model = SIR_ABM(
        lattice_size=30,
        initial_infected_fraction=true_params["IIF"],
        initial_susceptible_fraction=true_params["ISF"],
        PI=true_params["PI"],
        PR=true_params["PR"],
        Pm=1.0,
    )
    model.run(max_time=365)
    return pd.DataFrame(model.history)


def main():
    apply_style()
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 4.11), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        5,
        height_ratios=[3, 2],
        width_ratios=[1, 1, 1.2, 1.0, 1.2],
        hspace=0.35,
        wspace=0.4,
        figure=fig,
    )

    true_params = pd.read_csv(TRUE_P).iloc[0]

    # Panel A
    gs_a = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[0, 0:2], hspace=0.20, wspace=0.18)
    axes_a = np.array(
        [[fig.add_subplot(gs_a[row, col]) for col in range(2)] for row in range(3)]
    )
    histories_by_generation = {
        "gen-0": load_histories_from_dir(HIST_DIR, iteration=0),
        "gen-4": load_histories_from_dir(HIST_DIR, iteration=4),
    }
    compartments = ["S", "I", "R"]
    reference_histories = [_run_reference_sir(seed, true_params) for seed in range(5)]

    for row, compartment in enumerate(compartments):
        for col, generation in enumerate(["gen-0", "gen-4"]):
            ax = axes_a[row, col]
            for history in histories_by_generation[generation]:
                ax.plot(
                    history["time"],
                    history[compartment],
                    color="#aaaaaa",
                    alpha=0.3,
                    linewidth=0.4,
                )

            if generation == "gen-4":
                # Reference parameter curves — colored dashed
                for ref_hist in reference_histories:
                    ax.plot(
                        ref_hist["time"],
                        ref_hist[compartment],
                        color=sir_compartment_color(compartment),
                        alpha=0.7,
                    linewidth=1.0,
                        linestyle="--",
                        zorder=15,
                    )

            if row == 0:
                ax.set_title(generation, fontsize=FIG_AXIS_LABEL, fontweight="normal")
            if col == 0:
                ax.set_ylabel(compartment, fontsize=FIG_AXIS_LABEL)
            else:
                ax.set_yticklabels([])
            ax.set_xlim(0, 400)
            ax.set_xticks([0, 200, 400])
            ax.set_ylim(0, 1)
            # Rows 0–1 (S, I): x tick marks only; row 2 (R): tick marks + labels + shared x label
            if row < 2:
                ax.tick_params(axis="x", bottom=True, labelbottom=False)
            else:
                ax.tick_params(axis="x", bottom=True, labelbottom=True)
                if col == 0:
                    ax.set_xlabel("Time (day)", fontsize=FIG_AXIS_LABEL)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="major", labelsize=FIG_TICK, width=0.5, length=2.5)

    axes_a[0, 1].legend(
        handles=[Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", label="Reference")],
        loc="upper right",
        frameon=False,
        fontsize=FIG_TICK,
        handlelength=1.2,
        labelspacing=0.2,
    )

    # A panel label — positioned relative to the outer A gridspec bounding box
    a_bbox = gs[0, 0:2].get_position(fig)
    fig.text(
        a_bbox.x0 - 0.012,
        a_bbox.y1 + 0.010,
        "A",
        fontsize=FIG_PANEL,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )

    # Panel B
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, 2:5], hspace=0.40, wspace=0.42)
    axes_b = np.array(
        [[fig.add_subplot(gs_b[row, col]) for col in range(3)] for row in range(2)]
    )
    prior = pd.read_csv(BASE_DIR / "iter_0/params.csv")
    posterior = pd.read_csv(BASE_DIR / "iter_4/params.csv")
    plot_joint_distribution_4params_compact(
        prior,
        posterior,
        true_params.to_numpy(),
        param_names=PARAM_LABELS,
        axes=axes_b,
        tick_labelsize=FIG_TICK,
        axis_labelsize=FIG_AXIS_LABEL,
        top_row_xlabel_pad=-1,
        bottom_row_xlabel_pad=-6,
        ylabel_pad=2,
    )
    for ax in axes_b.flat:
        pos = ax.get_position()
        ax.set_position([pos.x0 + PANEL_B_RIGHT_SHIFT, pos.y0, pos.width, pos.height])
    # Panel B legend in the band above the panel, level with the panel A column titles.
    b_left = axes_b[0, 0].get_position()
    b_right = axes_b[0, 2].get_position()
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markersize=4, markerfacecolor="#aaaaaa",
                   markeredgecolor="black", markeredgewidth=0.2, label="gen-0"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=4, markerfacecolor="#bb883b",
                   markeredgecolor="black", markeredgewidth=0.2, label="gen-4"),
            Line2D([0], [0], marker="*", linestyle="none", markersize=8, markerfacecolor="#bb883b",
                   markeredgecolor="black", markeredgewidth=0.9, label="Reference"),
        ],
        loc="lower center",
        bbox_to_anchor=((b_left.x0 + b_right.x1) / 2, b_left.y1 + 0.002),
        ncol=3,
        frameon=False,
        fontsize=FIG_TICK,
        handletextpad=0.2,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    # Panel C
    ax_c = fig.add_subplot(gs[1, 0:2])
    analyze_parameter_errors(
        BASE_DIR,
        n_particles=512,
        n_iterations=5,
        ax=ax_c,
        show_swarm=True,
        tick_labelsize=FIG_TICK,
        axis_labelsize=FIG_AXIS_LABEL,
        metric_labelsize=FIG_METRIC,
        x_tick_rotation=0,
        x_tick_ha="center",
    )
    _panel_label(ax_c, "C")

    # Panel D (sensitivity heatmap)
    ax_e = fig.add_subplot(gs[1, 2:5])
    params_df = pd.read_csv(BASE_DIR / "iter_0/params.csv")
    metrics_df = pd.read_csv(BASE_DIR / "iter_0/statistics.csv")
    plot_sensitivity_heatmap(
        params_df,
        metrics_df,
        ax=ax_e,
        annot_fontsize=FIG_AXIS_LABEL,
        tick_labelsize=FIG_TICK,
        metric_labelsize=FIG_METRIC,
        cbar_tick_labelsize=FIG_TICK,
        cbar_ylabel_fontsize=FIG_AXIS_LABEL,
        x_tick_rotation=0,
        x_tick_ha="center",
        cbar_ticks=[-1.0, 1.0],
        cbar_label_position="left",
        cbar_labelpad=10,
    )
    ax_e.set_yticklabels(PARAM_LABELS, rotation=0)
    heatmap_pos = ax_e.get_position()
    ax_e.set_position([
        heatmap_pos.x0 + PANEL_D_RIGHT_SHIFT,
        heatmap_pos.y0,
        heatmap_pos.width,
        heatmap_pos.height,
    ])
    cbar = ax_e.collections[0].colorbar
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([
        cbar_pos.x0 + PANEL_D_RIGHT_SHIFT + 0.01,
        cbar_pos.y0,
        cbar_pos.width,
        cbar_pos.height,
    ])
    cbar.ax.yaxis.labelpad = 2
    # Black border around heatmap
    for spine in ax_e.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.5)
    # Label close to the left edge of this wide panel
    _panel_label(ax_e, "D", x=-0.06)

    panel_top_y = a_bbox.y1 + 0.010
    b_bbox = gs[0, 2:5].get_position(fig)
    fig.text(
        b_bbox.x0 - 0.012,
        panel_top_y,
        "B",
        fontsize=FIG_PANEL,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    savefig_both(fig, OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
