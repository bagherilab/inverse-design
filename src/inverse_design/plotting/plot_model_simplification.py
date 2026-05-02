"""Panel helpers for the ARCADE model simplification figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks
from inverse_design.plotting.plot_dendrogram import colorize_ward_clusters_threshold


CLUSTER_COLORS = [
    "mediumorchid",
    "darkorange",
    "lightcoral",
    "skyblue",
    "gold",
    "lightgreen",
    "plum",
    "crimson",
]

DROP_COLS = {"input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"}
ACRONYMS = {"ATP"}


def short_name(param: str) -> str:
    return param.replace("_MU", "").replace("_SIGMA", "").replace("_", " ").title()


def sentence_case_name(param: str) -> str:
    stem = param.removesuffix("_MU").removesuffix("_SIGMA")
    words = stem.split("_")
    formatted_words = []
    for idx, word in enumerate(words):
        if word in ACRONYMS:
            formatted_words.append(word)
        elif idx == 0:
            formatted_words.append(word.capitalize())
        else:
            formatted_words.append(word.lower())
    return " ".join(formatted_words)


def load_lr_data(lr_json: Path) -> List[dict]:
    data = json.loads(Path(lr_json).read_text())
    models = data["prediction_models"]["models"]
    return sorted(
        models,
        key=lambda model: model["prediction_quality"]["correlation_r"],
        reverse=True,
    )


def load_dend_data(dend_json: Path) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    data = json.loads(Path(dend_json).read_text())
    return (
        data["representative_parameters"],
        data["redundant_params"],
        data["individual_params"],
    )


def load_posterior(posterior_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(posterior_csv)
    drop = [col for col in df.columns if col in DROP_COLS]
    return df.drop(columns=drop).select_dtypes(include=[np.number])


def load_peak_posterior(posterior_csv: Path, peak: int, n_samples: int = 50) -> pd.DataFrame:
    posterior = load_posterior(posterior_csv)
    pca_result, peak_positions, *_ = perform_pca_and_find_peaks(posterior, n_components=2)
    peak_idx = peak - 1
    if peak_idx < 0 or peak_idx >= len(peak_positions):
        raise ValueError(f"Peak {peak} is unavailable; found {len(peak_positions)} peaks")

    distances = np.sqrt(np.sum((pca_result[:, :2] - peak_positions[peak_idx]) ** 2, axis=1))
    closest_indices = np.argsort(distances)[:n_samples]
    return posterior.iloc[closest_indices].copy()


def _signed_r(model: dict) -> float:
    r_val = model["prediction_quality"]["correlation_r"]
    slope = model["linear_model"]["slope"]
    return -abs(r_val) if slope < 0 else abs(r_val)


def _format_scientific(value: float) -> str:
    if value == 0:
        return "0"
    mantissa, exponent = f"{abs(value):.2e}".split("e")
    exponent_value = int(exponent)
    if exponent_value == 0:
        return mantissa.rstrip("0").rstrip(".")
    return rf"{mantissa.rstrip('0').rstrip('.')} \times 10^{{{exponent_value}}}"


def _format_equation_lines(model: dict) -> str:
    slope = model["linear_model"]["slope"]
    intercept = model["linear_model"]["intercept"]
    slope_sign = "-" if slope < 0 else ""
    sign = "+" if intercept >= 0 else "-"
    return "\n".join(
        [
            rf"$y = {slope_sign}{_format_scientific(slope)}x$",
            rf"${sign} {_format_scientific(intercept)}$",
        ]
    )


def draw_scatter_panels(
    axes,
    models: List[dict],
    posterior: pd.DataFrame,
    font_size: int = 10,
) -> None:
    for i, (ax, model) in enumerate(zip(axes, models[:3])):
        pred_col = model["predictor_parameter"]["name"]
        tgt_col = model["target_parameter"]["name"]
        slope = model["linear_model"]["slope"]
        intercept = model["linear_model"]["intercept"]
        r_val = _signed_r(model)

        x = posterior[pred_col].values
        y = posterior[tgt_col].values

        ax.scatter(
            x,
            y,
            s=15,
            edgecolors="black",
            facecolors="none",
            alpha=0.5,
            linewidths=0.8,
        )

        x_sorted = np.sort(x)
        ax.plot(x_sorted, slope * x_sorted + intercept, "k--", linewidth=1.0, alpha=0.85)

        ax.text(
            0.05,
            0.95,
            f"R = {r_val:.2f}\n{_format_equation_lines(model)}",
            transform=ax.transAxes,
            fontsize=max(font_size - 2, 1),
            va="top",
            ha="left",
            linespacing=1.25,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "boxstyle": "square,pad=0.12",
            },
            zorder=10,
        )

        ax.set_xlabel(sentence_case_name(pred_col), fontsize=font_size)
        ax.set_ylabel(sentence_case_name(tgt_col), fontsize=font_size)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="major", labelsize=max(font_size - 1, 1), width=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        ax.xaxis.get_offset_text().set_visible(False)
        ax.set_box_aspect(1)


def draw_equation_panel(ax: Axes, models: List[dict]) -> None:
    ax.axis("off")
    fontsize = 4.0 if len(models) > 8 else 7

    def fmt(value: float) -> str:
        return f"{value:.3g}"

    lines = ["Linear predictions (R >= 0.8)", ""]
    for model in models:
        pred = short_name(model["predictor_parameter"]["name"])
        tgt = short_name(model["target_parameter"]["name"])
        slope = model["linear_model"]["slope"]
        intercept = model["linear_model"]["intercept"]
        sign = "+" if intercept >= 0 else "-"
        lines.append(f"{tgt} =")
        lines.append(f"  {fmt(slope)}*{pred} {sign} {fmt(abs(intercept))}")

    ax.text(
        0.05,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=fontsize,
        family="monospace",
        va="top",
        ha="left",
        linespacing=1.05,
        clip_on=True,
    )


def draw_dendrogram_panel(
    ax: Axes,
    posterior: pd.DataFrame,
    ward_threshold: float = 1.0,
    return_details: bool = False,
    font_size: int = 10,
):
    param_names = np.array(posterior.columns.tolist())
    corr_matrix = np.corrcoef(posterior.values, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    _, cluster_labels, clusters, rep_params, redundant_params, _ = colorize_ward_clusters_threshold(
        data=corr_matrix,
        param_names=param_names,
        ward_threshold=ward_threshold,
        ax=ax,
        verbose=False,
    )

    ax.set_xlim(0, 2)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("Ward Distance", fontsize=font_size, fontweight="normal")
    ax.tick_params(axis="x", which="major", labelsize=max(font_size - 1, 1), width=0.8)
    ax.tick_params(axis="y", which="major", labelsize=max(font_size - 2, 1), width=0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    for label in ax.get_xticklabels():
        label.set_fontweight("normal")

    individual_params = [
        str(param_names[i])
        for i, cluster_label in enumerate(cluster_labels)
        if cluster_label not in clusters
    ]
    rep_label_names = {short_name(param).capitalize().strip() for param in rep_params}
    individual_label_names = {short_name(param).capitalize().strip() for param in individual_params}
    ytick_texts = []
    ytick_is_perturbable = []
    for label in ax.get_yticklabels():
        text = label.get_text().strip()
        is_rep = text in rep_label_names
        is_individual = text in individual_label_names
        is_perturbable = is_rep or is_individual
        ytick_texts.append(f"* {text}" if is_rep else text)
        ytick_is_perturbable.append(is_perturbable)
    ax.set_yticks(ax.get_yticks(), labels=ytick_texts)

    label_colors = {}
    for i, rep in enumerate(rep_params):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        for param in [rep, *redundant_params.get(rep, [])]:
            label_colors[short_name(param).capitalize().strip()] = color
    for param in individual_params:
        label_colors[short_name(param).capitalize().strip()] = "0.45"

    for label, is_perturbable in zip(ax.get_yticklabels(), ytick_is_perturbable):
        label_text = label.get_text().removeprefix("* ").strip()
        color = label_colors.get(label_text, "black")
        label.set_color(color)
        label.set_fontweight("bold" if is_perturbable else "normal")

    if return_details:
        return rep_params, redundant_params, individual_params
    return rep_params


def draw_status_panel(
    ax: Axes,
    rep_params: List[str],
    redundant_params: Dict[str, List[str]],
    individual_params: List[str],
    font_size: int = 10,
) -> None:
    ax.axis("off")
    total_lines = (
        3
        + len(rep_params)
        + len(individual_params)
        + sum(len(params) for params in redundant_params.values())
    )
    fontsize = font_size if total_lines <= 18 else max(font_size - 2, 1)
    line_h = min(0.055, 0.88 / max(total_lines, 1))
    y = 0.97

    def put(text, color="black", weight="normal", alpha=1.0, indent=0):
        nonlocal y
        ax.text(
            0.04 + indent * 0.06,
            y,
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            alpha=alpha,
            va="top",
            clip_on=True,
        )
        y -= line_h

    put("Perturb freely", weight="bold")
    for i, rep in enumerate(rep_params):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        put(short_name(rep), color=color, weight="bold", indent=1)
    for param in individual_params:
        put(short_name(param), color="gray", weight="bold", indent=1)

    y -= line_h * 0.5
    put("Fixed at default", weight="bold")
    for i, rep in enumerate(rep_params):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        for redundant in redundant_params.get(rep, []):
            put(short_name(redundant), color=color, alpha=0.55, indent=1)
