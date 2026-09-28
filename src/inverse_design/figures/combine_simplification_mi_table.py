#!/usr/bin/env python3
"""Simplification MI table: cluster parameter ranking across simplification levels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypeAlias

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg", force=False)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inverse_design.io.cluster_profiles import load_cluster_centroids  # noqa: E402
from inverse_design.vis.utils import (  # noqa: E402
    peak_colors,
    peak_marker_markersize,
    peak_markers,
)
from inverse_design.plotting.style import (  # noqa: E402
    DPI,
    FONT_LABEL,
    FONT_PANEL,
    WIDTH_DOUBLE,
    apply_style,
    savefig_both,
)

ARCADE_OUTPUT_DIR = REPO_ROOT.parent / "ARCADE_OUTPUT"
DEFAULT_N1024_DIR = ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2"


def _gather_lr_json_paths_under_linear_block(block_dir: Path) -> tuple[list[Path], list[Path]]:
    """Collect ``prediction_summary.json`` and ``lr_predictions*.json`` under one *linear* run dir."""
    prediction_summary: list[Path] = []
    lr_predictions: list[Path] = []

    def _collect_from_dir(d: Path) -> None:
        for path in d.glob("prediction_summary.json"):
            if path.is_file():
                prediction_summary.append(path)
        for path in d.glob("lr_predictions*.json"):
            if path.is_file():
                lr_predictions.append(path)

    _collect_from_dir(block_dir)
    for sub in sorted(block_dir.iterdir()):
        if sub.is_dir():
            _collect_from_dir(sub)
            for sub2 in sub.iterdir():
                if sub2.is_dir():
                    _collect_from_dir(sub2)
    return prediction_summary, lr_predictions


def discover_lr_summary_json_for_r_threshold(
    threshold: str,
    arcade_root: Path | None = None,
) -> Path | None:
    """LR chain JSON for one correlation threshold (matches ``*_linear_<threshold>_`` run name).

    Lower *r* typically yields more prediction edges → more derived params → fewer MI cells;
    higher *r* → fewer edges → more roots → more MI cells. Each r-series table column uses its
    own file when present.
    """
    root = arcade_root or ARCADE_OUTPUT_DIR
    if not root.is_dir():
        return None
    token = f"_linear_{threshold}_"
    prediction_summary: list[Path] = []
    lr_predictions: list[Path] = []
    for block_dir in sorted(root.glob("*_linear_*")):
        if not block_dir.is_dir() or token not in block_dir.name:
            continue
        ps, lp = _gather_lr_json_paths_under_linear_block(block_dir)
        prediction_summary.extend(ps)
        lr_predictions.extend(lp)
    if prediction_summary:
        return min(prediction_summary)
    if lr_predictions:
        return min(lr_predictions)
    return None


def discover_default_lr_summary_json(arcade_root: Path | None = None) -> Path | None:
    """Pick one LR chain JSON under ``<arcade_root>/*_linear_*`` (any threshold).

    Prefers ``prediction_summary.json``, else any ``lr_predictions*.json``.
    """
    root = arcade_root or ARCADE_OUTPUT_DIR
    if not root.is_dir():
        return None
    prediction_summary: list[Path] = []
    lr_predictions: list[Path] = []
    for block_dir in sorted(root.glob("*_linear_*")):
        if not block_dir.is_dir():
            continue
        ps, lp = _gather_lr_json_paths_under_linear_block(block_dir)
        prediction_summary.extend(ps)
        lr_predictions.extend(lp)
    if prediction_summary:
        return min(prediction_summary)
    if lr_predictions:
        return min(lr_predictions)
    return None


DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "out" / "pca_cluster_arcade_inputs_breast_only_mean_2" / "summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "simplification"
DEFAULT_MI_SUMMARY_JSON = REPO_ROOT / "data" / "summary_MI.json"

FIG_BG = "#ffffff"
CARD_BG = "#ffffff"
# transAxes x for panel letters A and B (matches ``combine_simplification_metrics.PANEL_AB_LETTER_X``).
PANEL_AB_LETTER_X = -0.1
CARD_EDGE = "#bdbdbd"
TEXT = "#222222"
MUTED = "#666666"

_CLUSTER_COLOR_OFFSET = 2
MI_NULL_THRESHOLD = 0.05
MI_RANDOM_STATE = 42

SIMP_METRICS = ["doub_time", "symmetry", "colony_growth"]
SIMP_METRIC_SHORT = ["DT", "Sym", "CG"]
ALL_LEVELS = [
    ("orig", None, None),
    ("r09", "linear", "0.9"),
    ("r08", "linear", "0.8"),
    ("r07", "linear", "0.7"),
    ("t10", "dendrogram", "1.0"),
    ("t125", "dendrogram", "1.25"),
    ("t15", "dendrogram", "1.5"),
]
LEVEL_LABELS = {
    "orig": "Original",
    "r09": "0.9",
    "r08": "0.8",
    "r07": "0.7",
    "t10": "1.0",
    "t125": "1.25",
    "t15": "1.5",
}
R_SERIES_KEYS = ["r09", "r08", "r07"]
T_SERIES_KEYS = ["t10", "t125", "t15"]


def _r_series_derived_param_names(lr_summary_path: Path | None) -> frozenset[str]:
    """Parameters predicted from roots in the LR chain (children); MI omitted in r-series columns."""
    if lr_summary_path is None or not lr_summary_path.is_file():
        return frozenset()
    try:
        with lr_summary_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return frozenset()
    names: set[str] = set()
    for model in data.get("prediction_models", {}).get("models", []):
        tgt = model.get("target_parameter") or {}
        name = tgt.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    derived = (
        data.get("parameter_categorization", {}).get("derived_parameters", {}).get("parameters", [])
    )
    for entry in derived:
        name = entry.get("parameter_name")
        if isinstance(name, str) and name:
            names.add(name)
    return frozenset(names)


_META_COLS = {
    "input_folder",
    "X_SPACING",
    "Y_SPACING",
    "DISTANCE_TO_CENTER",
}
# Shared y for peak marker and "Peak k" in header (transAxes).
HEADER_PEAK_ROW_Y = 0.5

CLUSTER_BIO = {
    1: "Nutrient-seeking",
    2: "Hypoxia-resistant",
    3: "Necrosis-dominated",
    4: "Volume-dominated",
}
CLUSTER_N = {1: 465, 2: 195, 3: 259, 4: 105}

MetricCell: TypeAlias = dict[str, float | int] | None
ParamCell: TypeAlias = dict[str, MetricCell] | None
LevelMI: TypeAlias = dict[str, ParamCell]
ClusterMI: TypeAlias = dict[str, LevelMI]


def _simp_run_dir(series: str, threshold: str, peak_k: int) -> Path | None:
    run_name = f"ABC_SMC_RF_N512_combined_grid_{series}_{threshold}_p{peak_k}_mean_only"
    run_dir = ARCADE_OUTPUT_DIR / run_name / run_name / "iter_4"
    return run_dir if run_dir.exists() else None


def _numeric_param_cols(param_df: pd.DataFrame) -> list[str]:
    candidates = [col for col in param_df.columns if col not in _META_COLS]
    numeric = param_df[candidates].select_dtypes(include=[np.number])
    return [col for col in numeric.columns if numeric[col].std(skipna=True) > 0]


def _compute_mi(
    param_df: pd.DataFrame, metrics_df: pd.DataFrame
) -> tuple[LevelMI, dict[str, float]]:
    """Compute per-level redesign-normalized MI and Spearman direction.

    For each metric, raw MI values are min-max normalized across parameters
    within the current peak/level. Every metric is retained in the returned
    cells, including explicit zeros, so downstream averaging across DT/Sym/CG is
    equivalent to sum(values) / 3.
    """
    join_col = "input_folder"
    if join_col not in param_df.columns or join_col not in metrics_df.columns:
        raise ValueError("Both parameter and metric dataframes must include input_folder")

    metrics_clean = metrics_df[[join_col] + SIMP_METRICS].copy()
    metrics_clean[SIMP_METRICS] = metrics_clean[SIMP_METRICS].replace([np.inf, -np.inf], np.nan)
    metrics_clean = metrics_clean.dropna(subset=SIMP_METRICS, how="all")
    df = param_df.merge(metrics_clean, on=join_col)
    param_cols = _numeric_param_cols(param_df)
    param_cols = [col for col in param_cols if col in df.columns]
    if not param_cols or df.empty:
        return {}, {}

    raw_mi: dict[str, dict[str, float]] = {param: {} for param in param_cols}
    directions: dict[str, dict[str, int]] = {param: {} for param in param_cols}
    median_fill = df[param_cols].median(numeric_only=True)

    for metric in SIMP_METRICS:
        y_series = df[metric].replace([np.inf, -np.inf], np.nan)
        valid = y_series.dropna().index
        if len(valid) < 3:
            for param in param_cols:
                raw_mi[param][metric] = 0.0
                directions[param][metric] = 0
            continue

        x_values = df.loc[valid, param_cols].fillna(median_fill)
        y_values = y_series.loc[valid].to_numpy()
        mi_values = mutual_info_regression(x_values, y_values, random_state=MI_RANDOM_STATE)
        for idx, param in enumerate(param_cols):
            mi_value = float(mi_values[idx])
            raw_mi[param][metric] = mi_value if np.isfinite(mi_value) else 0.0
            corr, _ = spearmanr(x_values[param].to_numpy(), y_values)
            directions[param][metric] = int(np.sign(corr)) if np.isfinite(corr) else 0

    for metric in SIMP_METRICS:
        vals = {param: raw_mi[param][metric] for param in param_cols}
        min_value = min(vals.values())
        max_value = max(vals.values())
        denom = max_value - min_value
        for param in param_cols:
            raw_mi[param][metric] = (vals[param] - min_value) / denom if denom > 0 else 0.0

    result: LevelMI = {}
    rank_strength: dict[str, float] = {}
    for param in param_cols:
        rank_strength[param] = float(
            sum(raw_mi[param][m] for m in SIMP_METRICS if np.isfinite(raw_mi[param][m]))
        )

    for param in param_cols:
        cell: dict[str, MetricCell] = {}
        for metric in SIMP_METRICS:
            value = raw_mi[param][metric]
            cell[metric] = {"mi": value, "dir": directions[param][metric]}
        result[param] = cell
    return result, rank_strength


def _assign_to_clusters(
    param_df: pd.DataFrame, centroids: dict[int, tuple[float, float]]
) -> np.ndarray:
    """Assign each parameter row to the nearest summary centroid in PCA space."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    param_cols = _numeric_param_cols(param_df)
    values = param_df[param_cols].fillna(param_df[param_cols].median(numeric_only=True))
    coords = PCA(n_components=2, random_state=MI_RANDOM_STATE).fit_transform(
        StandardScaler().fit_transform(values)
    )
    centroid_keys = sorted(centroids)
    centroid_arr = np.array([centroids[key] for key in centroid_keys])
    distances = np.linalg.norm(coords[:, None, :] - centroid_arr[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    return np.array([centroid_keys[idx] for idx in nearest])


def _load_orig_mi(
    n1024_dir: Path, summary_json: Path, peak_k: int
) -> tuple[LevelMI, dict[str, float]]:
    """Compute MI for the original posterior filtered to one PCA peak."""
    param_df = pd.read_csv(n1024_dir / "iter_4" / "all_param_df.csv")
    metrics_df = pd.read_csv(n1024_dir / "iter_4" / "final_metrics_seed.csv")
    centroids = load_cluster_centroids(summary_json)
    assignments = _assign_to_clusters(param_df, centroids)
    selected_params = param_df.loc[assignments == peak_k].reset_index(drop=True)
    selected_folders = set(selected_params["input_folder"])
    selected_metrics = metrics_df[metrics_df["input_folder"].isin(selected_folders)]
    return _compute_mi(selected_params, selected_metrics.reset_index(drop=True))


def _gather_cluster_mi(
    peak_k: int, n1024_dir: Path, summary_json: Path
) -> tuple[ClusterMI, dict[str, float]]:
    """Return MI results for Orig plus all simplification levels for one peak."""
    orig_level, orig_rank_strength = _load_orig_mi(n1024_dir, summary_json, peak_k)
    result: ClusterMI = {"orig": orig_level}
    for level_key, series, threshold in ALL_LEVELS[1:]:
        if series is None or threshold is None:
            continue
        run_dir = _simp_run_dir(series, threshold, peak_k)
        if run_dir is None:
            result[level_key] = {}
            continue
        param_df = pd.read_csv(run_dir / "all_param_df.csv")
        metrics_df = pd.read_csv(run_dir / "final_metrics_seed.csv")
        mi_level, _ = _compute_mi(param_df, metrics_df)
        result[level_key] = mi_level
    return result, orig_rank_strength


def _cell_score(cell: ParamCell) -> float:
    if cell is None:
        return 0.0
    values = [
        float(value["mi"])
        for value in cell.values()
        if value is not None and np.isfinite(float(value["mi"]))
    ]
    return float(sum(values))


def _stable_params(cluster_mi: ClusterMI, series_keys: list[str]) -> set[str]:
    """Return params whose rank changes by no more than one across a series."""
    ranks_per_level: list[dict[str, int]] = []
    for level_key in series_keys:
        level_data = cluster_mi.get(level_key, {})
        scored = [
            (param, _cell_score(cell)) for param, cell in level_data.items() if cell is not None
        ]
        scored.sort(key=lambda item: -item[1])
        ranks_per_level.append({param: rank for rank, (param, _) in enumerate(scored)})

    all_params = set(param for ranks in ranks_per_level for param in ranks)
    stable: set[str] = set()
    for param in all_params:
        fallback = len(all_params) + 5
        param_ranks = [ranks.get(param, fallback) for ranks in ranks_per_level]
        if max(param_ranks) - min(param_ranks) <= 1:
            stable.add(param)
    return stable


def _all_params(cluster_mi: ClusterMI) -> list[str]:
    """All available params, preserving the source dataframe order."""
    params: list[str] = []
    seen: set[str] = set()
    for level_key, _, _ in ALL_LEVELS:
        for param in cluster_mi.get(level_key, {}):
            if param not in seen:
                params.append(param)
                seen.add(param)
    return params


def _params_sorted_by_orig_mi(cluster_mi: ClusterMI) -> list[str]:
    """All params for this cluster, rows ordered by displayed Orig mi_mean (desc).

    Sorts by the same value shown in the Orig column: average of above-threshold
    normalized MI across metrics. Params with no visible Orig signal sort last.
    """
    params = _all_params(cluster_mi)
    orig_data = cluster_mi.get("orig", {})

    def sort_key(param: str) -> tuple[int, float, str]:
        cell = orig_data.get(param) if orig_data else None
        if cell is None:
            return (1, 0.0, param)
        mi_values = [
            float(cell[m]["mi"])
            for m in SIMP_METRICS
            if cell.get(m) is not None and np.isfinite(float(cell[m]["mi"]))
        ]
        if not mi_values:
            return (1, 0.0, param)
        return (0, -float(np.mean(mi_values)), param)

    return sorted(params, key=sort_key)


def _mi_bg_color(value: float) -> str:
    """Map normalized MI to discrete low / medium / high fill colors."""
    if not np.isfinite(value):
        value = 0.0
    value = float(np.clip(value, 0.0, 1.0))
    if value > 0.7:
        return "#ef8a62"
    if value >= 0.3:
        return "#999999"
    return "#67a9cf"


def _mi_text_color(_direction: int) -> str:
    return "#1A1A1A"


def _mi_direction_arrow(direction: int) -> str:
    if direction > 0:
        return "↑"
    if direction < 0:
        return "↓"
    return ""


def _format_mi_direction(value: float, direction: int) -> str:
    arrow = _mi_direction_arrow(direction)
    label = "<0.01" if 0 < value < 0.005 else f"{value:.2f}"
    return label + arrow


def _param_label(param: str) -> str:
    return param.replace("_MU", "").replace("_SIGMA", " sigma").replace("_", " ").title()


def _build_r_series_hide_mi_by_level(arcade_root: Path | None) -> dict[str, frozenset[str]]:
    """Per r column: derived param names from the LR summary for that correlation threshold."""
    out: dict[str, frozenset[str]] = {}
    for level_key, _series, threshold in ALL_LEVELS:
        if level_key not in R_SERIES_KEYS or threshold is None:
            continue
        path = discover_lr_summary_json_for_r_threshold(threshold, arcade_root)
        out[level_key] = _r_series_derived_param_names(path)
    return out


def _draw_mi_cluster_block(
    fig,
    subplot_spec,
    peak_k: int,
    cluster_mi: ClusterMI,
    params: list[str],
    font_size: int,
    panel_label: str | None = None,
    *,
    section_levels: list[tuple[str, str | None, str | None]] | None = None,
    section_title: str | None = None,
    show_peak_header: bool = True,
    show_param_labels: bool = True,
    content_x1_override: float | None = None,
    label_w_override: float | None = None,
    r_series_hide_mi_by_level: dict[str, frozenset[str]] | None = None,
):
    """Render one cluster block with all parameters and 7 MI level columns."""
    color_idx = (peak_k - 1 + _CLUSTER_COLOR_OFFSET) % len(peak_colors)
    cluster_color = peak_colors[color_idx]
    cluster_marker = peak_markers[color_idx]

    card_ax = fig.add_subplot(subplot_spec)
    card_ax.set_facecolor(CARD_BG)
    card_ax.set_xticks([])
    card_ax.set_yticks([])
    card_ax.set_zorder(0)
    for spine in card_ax.spines.values():
        spine.set_visible(False)

    outer = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=subplot_spec, height_ratios=[0.05, 1.0], hspace=0.08
    )

    header_ax = fig.add_subplot(outer[0])
    header_ax.set_zorder(2)
    header_ax.patch.set_alpha(0)
    header_ax.axis("off")
    # Match metric bar charts: ``plot(..., markersize=...)`` in points (not scatter area).
    # Panel letter + marker + title: same layout convention as metrics ``_plot_section`` (left, y=0.5).
    marker_x = 0.018
    title_x = 0.062
    if panel_label:
        header_ax.text(
            PANEL_AB_LETTER_X,
            0.5,
            panel_label,
            ha="left",
            va="center",
            fontsize=FONT_PANEL,
            fontweight="bold",
            color="#000000",
            transform=header_ax.transAxes,
            clip_on=False,
        )
        marker_x = 0.038
        title_x = 0.088
    if show_peak_header:
        _ms = peak_marker_markersize(cluster_marker)
        header_ax.plot(
            marker_x,
            HEADER_PEAK_ROW_Y,
            marker=cluster_marker,
            markersize=_ms,
            markerfacecolor=cluster_color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            linestyle="none",
            transform=header_ax.transAxes,
            clip_on=False,
            zorder=3,
        )
        _n_samples = CLUSTER_N.get(peak_k)
        _peak_title = (
            f"Peak {peak_k} (n = {_n_samples})" if _n_samples is not None else f"Peak {peak_k}"
        )
        header_ax.text(
            title_x,
            HEADER_PEAK_ROW_Y,
            _peak_title,
            ha="left",
            va="center",
            fontsize=font_size,
            fontweight="normal",
            color="#000000",
            transform=header_ax.transAxes,
        )

    ax = fig.add_subplot(outer[1])
    ax.set_zorder(2)
    ax.patch.set_alpha(0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    levels = list(section_levels or ALL_LEVELS)
    n_rows = max(len(params), 1)
    n_levels = len(levels)
    # Label gutter for parameter names (clip_off so long names are not cut off).
    content_x0 = 0.04
    content_x1 = 0.96 if content_x1_override is None else content_x1_override
    label_w = 0.24 if label_w_override is None else label_w_override
    cell_w = (content_x1 - label_w) / n_levels
    header_h = 0.115
    row_h = (1.0 - header_h) / n_rows
    col_xs = [label_w + idx * cell_w for idx in range(n_levels)]
    table_x0 = col_xs[0] if col_xs else label_w
    table_width = cell_w * n_levels
    table_y1 = 1.0 - header_h

    ax.add_patch(
        mpatches.Rectangle(
            (table_x0, 0.0),
            table_width,
            table_y1,
            fill=False,
            edgecolor="#000000",
            lw=0.8,
            zorder=30,
        )
    )

    group_bands: list[tuple[str, int, int]] = []
    r_indices = [idx for idx, (level_key, _, _) in enumerate(levels) if level_key in R_SERIES_KEYS]
    t_indices = [idx for idx, (level_key, _, _) in enumerate(levels) if level_key in T_SERIES_KEYS]
    if section_title and r_indices and not t_indices:
        group_bands.append((section_title, r_indices[0], r_indices[-1]))
    elif section_title and t_indices and not r_indices:
        group_bands.append((section_title, t_indices[0], t_indices[-1]))
    else:
        if r_indices:
            group_bands.append(("Pairwise", r_indices[0], r_indices[-1]))
        if t_indices:
            group_bands.append(("Hierarchy", t_indices[0], t_indices[-1]))

    for label, start_idx, end_idx in group_bands:
        x0 = col_xs[start_idx]
        x1 = col_xs[end_idx] + cell_w
        ax.add_patch(
            mpatches.Rectangle(
                (x0, 1.0 - header_h * 0.48),
                x1 - x0,
                header_h * 0.38,
                facecolor="#eeeeee",
                edgecolor="#c5c5c5",
                lw=0.4,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            1.0 - header_h * 0.29,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="normal",
            color="#000000",
            clip_on=False,
        )

    for col_idx, (level_key, _, _) in enumerate(levels):
        ax.text(
            col_xs[col_idx] + cell_w * 0.5,
            1.0 - header_h * 0.78,
            LEVEL_LABELS[level_key],
            ha="center",
            va="center",
            fontsize=font_size,
            color="#000000",
            clip_on=False,
        )
    for row_idx, param in enumerate(params):
        row_top = 1.0 - header_h - row_idx * row_h
        row_bottom = row_top - row_h
        if show_param_labels:
            ax.text(
                label_w - 0.006,
                row_bottom + row_h * 0.5,
                _param_label(param),
                ha="right",
                va="center",
                fontsize=font_size,
                color="#000000",
                clip_on=False,
            )

        for col_idx, (level_key, _, _) in enumerate(levels):
            x = col_xs[col_idx]
            level_data = cluster_mi.get(level_key, {})
            _hide_for_level = (
                (r_series_hide_mi_by_level or {}).get(level_key, frozenset())
                if level_key in R_SERIES_KEYS
                else frozenset()
            )
            if level_key in R_SERIES_KEYS and param in _hide_for_level:
                ax.add_patch(
                    mpatches.Rectangle(
                        (x, row_bottom),
                        cell_w,
                        row_h,
                        facecolor="#ebebeb",
                        edgecolor="#d0d0d0",
                        lw=0.3,
                    )
                )
                ax.text(
                    x + cell_w * 0.5,
                    row_bottom + row_h * 0.5,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color=MUTED,
                    fontweight="normal",
                    clip_on=False,
                )
                continue
            cell = level_data.get(param)
            # Missing level, missing param row, or empty metric dict → "—" (clip_on=False avoids blank cells).
            if level_data == {} or not cell:
                ax.add_patch(
                    mpatches.Rectangle(
                        (x, row_bottom),
                        cell_w,
                        row_h,
                        facecolor="#f2f2f2",
                        edgecolor="#d0d0d0",
                        lw=0.3,
                    )
                )
                ax.text(
                    x + cell_w * 0.5,
                    row_bottom + row_h * 0.5,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color=MUTED,
                    fontweight="normal",
                    clip_on=False,
                )
                continue

            mi_values = [
                float(cell[m]["mi"])
                for m in SIMP_METRICS
                if cell.get(m) is not None and np.isfinite(float(cell[m]["mi"]))
            ]
            dir_values = [cell[m]["dir"] for m in SIMP_METRICS if cell.get(m) is not None]
            mi_mean = float(np.mean(mi_values)) if mi_values else 0.0
            if not np.isfinite(mi_mean):
                mi_mean = 0.0
            direction = int(np.sign(sum(dir_values))) if dir_values else 0
            ax.add_patch(
                mpatches.Rectangle(
                    (x, row_bottom),
                    cell_w,
                    row_h,
                    facecolor=_mi_bg_color(mi_mean),
                    edgecolor="#d0d0d0",
                    lw=0.3,
                )
            )
            ax.text(
                x + cell_w * 0.5,
                row_bottom + row_h * 0.5,
                _format_mi_direction(mi_mean, direction),
                ha="center",
                va="center",
                fontsize=font_size,
                color=_mi_text_color(direction),
                fontweight="normal",
                clip_on=False,
            )

    if r_indices and t_indices:
        series_divider_x = col_xs[t_indices[0]]
        ax.plot(
            [series_divider_x, series_divider_x],
            [0.0, 1.0],
            color="#000000",
            lw=1.4,
            zorder=20,
        )

    return header_ax


def _plot_mi_grid(
    fig,
    subplot_spec,
    args: argparse.Namespace,
    *,
    panel_label: str | None = None,
    section_levels: list[tuple[str, str | None, str | None]] | None = None,
    section_title: str | None = None,
    show_param_labels: bool = True,
    content_x1_override: float | None = None,
    label_w_override: float | None = None,
):
    peaks_to_plot = tuple(getattr(args, "peaks_to_plot", (1, 2, 3, 4)))
    show_peak_header = getattr(args, "show_peak_header", True)
    arcade_root = getattr(args, "lr_arcade_root", None)
    r_series_hide_mi_by_level = _build_r_series_hide_mi_by_level(arcade_root)
    grid = gridspec.GridSpecFromSubplotSpec(
        1, len(peaks_to_plot), subplot_spec=subplot_spec, hspace=0.0, wspace=0.04
    )
    cluster1_header_ax = None
    first_peak = peaks_to_plot[0]
    for peak_idx, peak_k in enumerate(peaks_to_plot):
        data, _ = _gather_cluster_mi(peak_k, args.n1024_dir, args.summary_json)
        params = _params_sorted_by_orig_mi(data)
        ha = _draw_mi_cluster_block(
            fig,
            grid[0, peak_idx],
            peak_k,
            data,
            params,
            args.font_size,
            panel_label=panel_label if peak_k == first_peak else None,
            section_levels=section_levels,
            section_title=section_title,
            show_peak_header=show_peak_header,
            show_param_labels=show_param_labels,
            content_x1_override=content_x1_override,
            label_w_override=label_w_override,
            r_series_hide_mi_by_level=r_series_hide_mi_by_level,
        )
        if peak_k == first_peak:
            cluster1_header_ax = ha
    return cluster1_header_ax


def plot_mi_table_panel(
    fig,
    subplot_spec,
    args: argparse.Namespace,
    *,
    panel_label: str | None = None,
    section_levels: list[tuple[str, str | None, str | None]] | None = None,
    section_title: str | None = None,
    show_param_labels: bool = True,
    content_x1_override: float | None = None,
    label_w_override: float | None = None,
):
    """Entry point for the combined wrapper. Returns cluster 1 header axes (or None)."""
    return _plot_mi_grid(
        fig,
        subplot_spec,
        args,
        panel_label=panel_label,
        section_levels=section_levels,
        section_title=section_title,
        show_param_labels=show_param_labels,
        content_x1_override=content_x1_override,
        label_w_override=label_w_override,
    )


def _serialize_param_cell_for_json(
    pcell: ParamCell,
) -> dict[str, dict[str, float | int] | None] | None:
    if pcell is None:
        return None
    out: dict[str, dict[str, float | int] | None] = {}
    for m in SIMP_METRICS:
        mc = pcell.get(m)
        if mc is None:
            out[m] = None
        else:
            out[m] = {"mi": float(mc["mi"]), "dir": int(mc["dir"])}
    return out


def export_mi_summary_to_json(out_path: Path, args: argparse.Namespace) -> None:
    """Compute MI for all four PCA peaks and all simplification levels; write JSON summary.

    Same logic as panel A: ``_gather_cluster_mi`` / ``_compute_mi`` (sklearn MI + Spearman sign),
    including full per-parameter cells (not display-masked for r-series derived params).
    """
    lr_root = getattr(args, "lr_arcade_root", None)
    summary: dict = {
        "version": 1,
        "description": (
            "Per-parameter mutual information vs doubling time, symmetry, and colony growth for "
            "each PCA-defined posterior peak (1-4) and each simplification level. "
            "Original: N1024 iter_4 particles assigned to peak via 2D PCA + nearest centroid in "
            "summary_json. Other columns: N512 ABC_SMC_RF_N512_combined_grid_{linear|dendrogram}_"
            "{threshold}_p{peak}_mean_only/iter_4. "
            "MI from sklearn.feature_selection.mutual_info_regression (random_state=42), "
            "min-max normalized per metric within each level to [0,1]. "
            "dir is sign(Spearman rho) between parameter and that metric. "
            "Zero-valued entries are retained so cell means equal sum(DT, Sym, CG) / 3."
        ),
        "simp_metrics": list(SIMP_METRICS),
        "level_keys": [k for k, _, _ in ALL_LEVELS],
        "level_display": {k: LEVEL_LABELS[k] for k, _, _ in ALL_LEVELS},
        "mi_null_threshold": MI_NULL_THRESHOLD,
        "mutual_info_random_state": MI_RANDOM_STATE,
        "paths": {
            "n1024_dir": str(args.n1024_dir),
            "summary_json": str(args.summary_json),
            "lr_arcade_root": str(lr_root) if lr_root is not None else None,
        },
        "peaks": {},
    }

    for peak_k in range(1, 5):
        cluster_mi, orig_rank_strength = _gather_cluster_mi(
            peak_k, args.n1024_dir, args.summary_json
        )
        params_sorted = _params_sorted_by_orig_mi(cluster_mi)
        levels_out: dict[str, dict] = {}
        for level_key, _, _ in ALL_LEVELS:
            level_data = cluster_mi.get(level_key)
            if not level_data:
                levels_out[level_key] = {}
                continue
            levels_out[level_key] = {
                param: _serialize_param_cell_for_json(level_data.get(param))
                for param in sorted(level_data.keys())
            }
        summary["peaks"][str(peak_k)] = {
            "cluster_n": CLUSTER_N[peak_k],
            "cluster_label": CLUSTER_BIO[peak_k],
            "params_sorted_by_orig_mi": params_sorted,
            "orig_rank_strength": {
                p: float(orig_rank_strength[p])
                for p in sorted(orig_rank_strength.keys())
                if np.isfinite(orig_rank_strength[p])
            },
            "levels": levels_out,
        }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {out_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplification MI ranking table.")
    parser.add_argument("--n1024-dir", type=Path, default=DEFAULT_N1024_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--lr-arcade-root",
        type=Path,
        default=None,
        help=(
            "Directory to search for linear simplification runs (default: "
            f"{ARCADE_OUTPUT_DIR}). Each r column (0.9 / 0.8 / 0.7) loads LR JSON from "
            "``*_linear_<r>_*/**/(prediction_summary.json|lr_predictions*.json)`` so derived "
            "parameter counts match that correlation threshold."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--font-size", type=int, default=FONT_LABEL)
    parser.add_argument(
        "--export-mi-json",
        type=Path,
        default=DEFAULT_MI_SUMMARY_JSON,
        help=f"Write MI summary for all peaks/levels (default: {DEFAULT_MI_SUMMARY_JSON}).",
    )
    parser.add_argument(
        "--no-export-mi-json",
        action="store_true",
        help="Skip writing summary_MI JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.10, 2.44), dpi=args.dpi, facecolor=FIG_BG)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.975, bottom=0.035)
    plot_mi_table_panel(fig, fig.add_gridspec(1, 1)[0], args)
    save_path = args.output_dir / "simplified_mi_table.png"
    savefig_both(fig, save_path, facecolor=FIG_BG, pad_inches=0)
    plt.close(fig)
    print(f"Saved {save_path}")
    if not args.no_export_mi_json:
        export_mi_summary_to_json(args.export_mi_json, args)


if __name__ == "__main__":
    main()
