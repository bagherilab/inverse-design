"""Parameter-comparison bar charts and swarm plots."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from inverse_design.plotting.theme import apply_publication_style
from inverse_design.vis.utils import (
    find_mode_bin,
    load_redundancy_analysis,
    create_parameter_order,
    peak_markers,
    peak_colors,
)

__all__ = [
    "find_mode_bin",
    "plot_best_sample_percentage_changes_split",
    "plot_multi_boxcharts",
]


def plot_best_sample_percentage_changes_split(
    all_simulation_metrics_list,
    all_simulation_params_list,
    target_metrics: dict,
    scenario_names,
    redundancy_files=None,
    save_path: str | None = None,
):
    """Two split bar-chart figures comparing simplified-model scenarios.

    Creates one figure for *individual* parameters and another for *grouped*
    (redundant) parameters, with an optional broken y-axis on the grouped figure.

    Returns ``(fig1, ax1, fig2, ax2)``.
    """
    apply_publication_style(font_size=12, axes_linewidth=1.0)
    _hatch = ["///", "...", "+++", "|||"]
    target_arr = np.array(list(target_metrics.values()))

    best_samples = []
    for i, (sim_metrics_list, _) in enumerate(zip(all_simulation_metrics_list, scenario_names)):
        last_iter = sim_metrics_list[-1]
        errors = [
            np.mean(np.abs((last_iter[j, :] - target_arr) / target_arr) * 100)
            for j in range(last_iter.shape[0])
        ]
        best_idx = int(np.argmin(errors))
        row = all_simulation_params_list[i].iloc[best_idx]
        all_names = list(row.index)
        drop = {"input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"}
        keep = [name for name in all_names if name not in drop]
        best_samples.append({name: row[name] for name in keep})

    individual_params: list = []
    grouped_params: list = []
    manual_rep_order: list = []
    scenarios_config: list = []

    if redundancy_files:
        try:
            individual_params, scenarios_config = load_redundancy_analysis(redundancy_files)
            manual_rep_order = [
                "GLUCOSE_CONCENTRATION",
                "LACTATE_RATE_SIGMA",
                "GLUCOSE_UPTAKE_RATE_SIGMA",
                "METABOLIC_PREFERENCE_MU",
            ]
            all_param_names, _, _, _ = create_parameter_order(
                individual_params, scenarios_config, manual_rep_order
            )
            available = list(best_samples[0].keys())
            individual_params = [p for p in individual_params if p in available]
            grouped_params = [
                p for p in all_param_names if p not in individual_params and p in available
            ]
        except Exception as exc:
            print(
                f"Warning: could not load redundancy analysis ({exc}). Falling back to alphabetical."
            )
            all_params = sorted(best_samples[0].keys())
            mid = len(all_params) // 2
            individual_params, grouped_params = all_params[:mid], all_params[mid:]
    else:
        all_params = sorted(best_samples[0].keys())
        mid = len(all_params) // 2
        individual_params, grouped_params = all_params[:mid], all_params[mid:]

    prior_values = best_samples[0]
    scenario_labels = list(scenario_names[1:])

    def _create_subplot(param_list, use_broken_axis: bool, fig_width: float):
        if not param_list:
            return None, None
        prior_means = np.array([prior_values[p] for p in param_list])
        param_pct_changes = []
        for scenario_name, best_sample in zip(scenario_names[1:], best_samples[1:]):
            available = [p for p in param_list if p in best_sample]
            indices = [param_list.index(p) for p in available]
            vals = np.array([best_sample[p] for p in available])
            pct = ((vals - prior_means[indices]) / prior_means[indices]) * 100
            full = np.zeros(len(param_list))
            for j, p in enumerate(available):
                if not (np.isnan(pct[j]) or np.isinf(pct[j])):
                    full[param_list.index(p)] = pct[j]
            param_pct_changes.append(full)

        n_params = len(param_list)
        n_scenarios = len(scenario_labels)
        bar_width = 0.8 / n_scenarios
        x_pos = np.arange(n_params)
        all_vals = np.concatenate(param_pct_changes) if param_pct_changes else np.array([0.0])
        valid = all_vals[~np.isnan(all_vals) & ~np.isinf(all_vals)]

        if use_broken_axis:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(fig_width, 5), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1}
            )
            axes = [ax1, ax2]
            ax1.set_ylim(150, float(np.max(valid)) + 50 if len(valid) > 0 else 700)
            ax2.set_ylim(float(np.min(valid)) - 10 if len(valid) > 0 else -100, 75)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))
            axes = [ax]
            if len(valid) > 0:
                ax.set_ylim(float(np.min(valid)) - 10, float(np.max(valid)) + 50)

        for i, (pct_changes, sname) in enumerate(zip(param_pct_changes, scenario_labels)):
            xp = x_pos + (i - n_scenarios / 2 + 0.5) * bar_width
            bar_kw = dict(
                alpha=0.7,
                color="white",
                edgecolor="black",
                linewidth=1.0,
                hatch=_hatch[i % len(_hatch)],
                label=sname,
            )
            if use_broken_axis:
                ax1.bar(xp, pct_changes, bar_width, **bar_kw)
                ax2.bar(xp, pct_changes, bar_width, **{**bar_kw, "label": "_nolegend_"})
            else:
                ax.bar(xp, pct_changes, bar_width, **bar_kw)

        for axis in axes:
            if use_broken_axis:
                axis.spines["top"].set_visible(axis is not ax1)
            else:
                axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_linewidth(1.0)
            axis.spines["bottom"].set_linewidth(1.0)
            axis.set_axisbelow(True)
            axis.tick_params(axis="both", which="major", labelsize=12, width=1.5)

        def _apply_x_labels(target_ax):
            target_ax.set_xticks(x_pos)
            target_ax.set_xticklabels(param_list, rotation=45, ha="right", fontsize=12)
            if redundancy_files and scenarios_config and manual_rep_order:
                try:
                    group_colors = ["#f6b192", "#d25849", "#a51429", "#67001f"]
                    rep_to_color = {r: group_colors[k] for k, r in enumerate(manual_rep_order)}
                    for lbl in target_ax.get_xticklabels():
                        txt = lbl.get_text()
                        for rep, redundants in scenarios_config[-1]["redundant_params"].items():
                            if txt in redundants:
                                lbl.set_color(rep_to_color.get(rep, "black"))
                            if txt == rep:
                                lbl.set_fontweight("bold")
                                lbl.set_color(rep_to_color.get(rep, "black"))
                except Exception:
                    pass

        if use_broken_axis:
            ax1.set_xticks([])
            _apply_x_labels(ax2)
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=2)
            d = 0.01
            for axis, y0, y1 in [
                (ax1, 0, 2 * d),
                (ax1, -2 * d, 0),
                (ax2, 1 - 2 * d, 1),
                (ax2, 1, 1 + 2 * d),
            ]:
                axis.plot(
                    (-d, +d),
                    (y0, y1),
                    transform=axis.transAxes,
                    color="k",
                    clip_on=False,
                    linewidth=2,
                )
            ax1.legend(loc="upper right", fontsize=12, frameon=True, fancybox=True, shadow=True)
            for axis in [ax1, ax2]:
                for lbl in axis.get_yticklabels():
                    lbl.set_fontweight("bold")
            plt.subplots_adjust(hspace=0.05)
            return fig, (ax1, ax2)
        else:
            _apply_x_labels(ax)
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=2)
            for lbl in ax.get_yticklabels():
                lbl.set_fontweight("bold")
            plt.tight_layout()
            return fig, ax

    fig1, ax1 = _create_subplot(individual_params, use_broken_axis=False, fig_width=6.0)
    fig2, ax2 = _create_subplot(grouped_params, use_broken_axis=True, fig_width=10.0)

    if save_path:
        for fig, suffix in [(fig1, "_individual"), (fig2, "_grouped")]:
            if fig is not None:
                path = (
                    save_path.replace(".png", f"{suffix}.png")
                    if ".png" in save_path
                    else f"{save_path}{suffix}.png"
                )
                fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    else:
        for fig in [fig1, fig2]:
            if fig is not None:
                fig.show()

    return fig1, ax1, fig2, ax2


def plot_multi_boxcharts(
    target_df: pd.DataFrame,
    csv_files,
    case_names,
    target_metrics=("doub_time", "symmetry", "colony_growth"),
    figsize=(8, 3),
    case_colors=None,
    font_size: int = 14,
    metric_titles: dict[str, str] | None = None,
    error_stat: str = "std",
    marker_label_y: float = -0.075,
    save_path: str | None = None,
):
    """Bar charts with error bars and swarm plots for multiple CSV scenarios.

    The first case comes from *target_df* (pre-calculated means/stds).  Subsequent
    cases are loaded from *csv_files*.  Shape markers identify each case on the
    x-axis instead of long text labels.
    """
    target_metrics = list(target_metrics)
    colors = {
        "symmetry": "#486b45",
        "doub_time": "#bb883b",
        "act_ratio": "#af1b0a",
        "colony_growth": "#545aab",
        "symmetry_std": "#679A63",
        "doub_time_std": "#ddc39d",
        "colony_growth_std": "#a9acd5",
        "act_ratio_std": "#d78d85",
    }
    if error_stat not in {"std", "iqr"}:
        raise ValueError("error_stat must be 'std' or 'iqr'")

    all_centers, all_yerr, all_raw_data = [], [], []
    case_centers = [target_df[m].iloc[0] for m in target_metrics]
    case_yerr = []
    for metric in target_metrics:
        q1_col = f"{metric}_q1"
        q3_col = f"{metric}_q3"
        std_col = f"{metric}_std"
        center = float(target_df[metric].iloc[0])
        if error_stat == "iqr" and q1_col in target_df and q3_col in target_df:
            case_yerr.append(
                [
                    max(0.0, center - float(target_df[q1_col].iloc[0])),
                    max(0.0, float(target_df[q3_col].iloc[0]) - center),
                ]
            )
        elif std_col in target_df:
            std_val = float(target_df[std_col].iloc[0])
            case_yerr.append([std_val, std_val])
        else:
            case_yerr.append([0.0, 0.0])
    all_centers.append(case_centers)
    all_yerr.append(case_yerr)
    all_raw_data.append(None)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)[target_metrics]
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        centers, yerr = [], []
        for metric in target_metrics:
            values = df[metric].dropna()
            if error_stat == "iqr":
                q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
                centers.append(float(median))
                yerr.append([float(median - q1), float(q3 - median)])
            else:
                mean = values.mean()
                std = values.std()
                centers.append(float(mean))
                yerr.append([float(std), float(std)])
        all_centers.append(centers)
        all_yerr.append(yerr)
        all_raw_data.append([df[m].values for m in target_metrics])
    all_centers = np.array(all_centers, dtype=float)
    all_yerr = np.array(all_yerr, dtype=float)

    fig, axes = plt.subplots(1, len(target_metrics), figsize=figsize, sharey=False)
    if len(target_metrics) == 1:
        axes = [axes]
    marker_to_size = {"o": 100, "s": 95, "^": 105, "D": 75, "*": 160, "p": 100}

    for i, metric in enumerate(target_metrics):
        ax = axes[i]
        x_positions = np.arange(len(case_names))
        for j, (x_pos, center_val, error_vals) in enumerate(
            zip(x_positions, all_centers[:, i], all_yerr[:, i])
        ):
            if j == 0:
                bar_color = colors.get(metric, "#cccccc")
            elif case_colors is not None and (j - 1) < len(case_colors):
                bar_color = case_colors[j - 1]
            else:
                bar_color = "white"
            lower_err, upper_err = float(error_vals[0]), float(error_vals[1])
            ax.bar(
                x_pos,
                float(center_val),
                yerr=[[lower_err], [upper_err]],
                capsize=5,
                alpha=0.7 if j == 0 else 1.0,
                color=bar_color,
                edgecolor="black",
                linewidth=1.5,
            )
        for j, case_raw in enumerate(all_raw_data):
            if case_raw is not None:
                ax.scatter(
                    np.random.normal(j, 0.02, len(case_raw[i])),
                    case_raw[i],
                    color="black",
                    alpha=0.6,
                    s=20,
                    zorder=3,
                )
        x_labels = ["EXP"] + [""] * (len(case_names) - 1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=0, ha="center")
        y_min, y_max = ax.get_ylim()
        for j, x_pos in enumerate(x_positions):
            if j > 0 and (j - 1) < len(peak_markers) and (j - 1) < len(peak_colors):
                mk = peak_markers[j - 1]
                s = marker_to_size[mk]
                kw = dict(
                    marker=mk,
                    color=peak_colors[j - 1],
                    s=s,
                    clip_on=False,
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=10,
                )
                if j < 3:
                    kw["facecolor"] = "none"
                ax.scatter(
                    x_pos,
                    marker_label_y,
                    transform=ax.get_xaxis_transform(),
                    **kw,
                )
        ax.set_ylim(y_min if metric == "colony_growth" else 0, y_max)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="both", which="major", labelsize=font_size, width=1.0)
        if metric_titles is not None:
            ax.set_title(metric_titles.get(metric, metric), fontsize=font_size + 1, pad=6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes
