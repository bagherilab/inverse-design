#!/usr/bin/env python3
"""Weighted-density and extinction figures for reviewer R2.12.

Expected campaign layout is the preserved ``sir_r212`` Hyak output:

    published/n512/iter_4/{params.csv,statistics.csv}
    published/target_values.csv
    replicates/replicate_{1,2}/iter_4/{params.csv,weights.csv}
    extinction/{extinction_summary.csv,histories/<group>/draw_*.csv}

The published forest weights are replayed deterministically with the recorded
target.  Each fitted cloud uses its own prior-range-normalized PCA basis.  The
three fits still use the same fixed bandwidth factor (the 2-D Scott factor for
nominal n=512), grid resolution, maximum filter, and relative-density threshold.
This prevents changes in ESS from silently changing the amount of smoothing
between runs without treating coordinates from independent PCA bases as shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import maximum_filter
from scipy.stats import gaussian_kde

from reviewer_figure_style import (  # noqa: E402
    OKABE_ITO,
    PEAK_COLORS,
    PLOS_DOUBLE_WIDTH,
    apply_plos_style,
    assert_text_inside_figure,
    four_spines,
    in_axis_label,
    panel_label,
    save_figure,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "analysis_outputs/reviewer_closeout"
DEFAULT_FIG = REPO / "body/updated_figures/figures/reviewer_closeout"
PARAMS = ["PI", "PR", "IIF", "ISF"]
RANGES = {"PI": (0.001, 0.5), "PR": (0.0001, 0.1), "IIF": (0.01, 0.5), "ISF": (0.49, 0.99)}
GROUPS = ["published", "replicate_1", "replicate_2"]
DISPLAY_LABELS = {
    "published": "Replicate 1",
    "replicate_1": "Replicate 2",
    "replicate_2": "Replicate 3",
    "reference": "Reference",
}
COLORS = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["green"]]
BW = 512 ** (-1 / 6)
GRID = 200
WINDOW = 5
THRESHOLD = 0.05

# Scatter-area compensation copied from ``../inverse_design``
# (src/inverse_design/figures/combine_fit_exp_figure.py).  Matplotlib marker
# paths do not have equal visible areas at a common ``s`` value, so preserve the
# figure-specific area ratios rather than squaring the Line2D marker sizes.
PEAK_MARKER_AREA_PT2 = {
    "o": 30,
    "s": 27,
    "^": 37,
    "D": 30,
    "v": 37,
    "p": 42,
}
# No star: Fig 2B reserves the star for the known generating parameters.
DENSITY_PEAK_MARKERS = ["^", "D", "v", "p", "o", "s"]
DENSITY_PEAK_COLORS = [
    *PEAK_COLORS,
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
]


def inverse_design_source() -> Path:
    """Locate the companion inverse_design source tree on local or Hyak hosts."""
    candidates = [
        REPO.parent / "inverse_design" / "src",
        Path(os.environ.get("DED_INVERSE_DESIGN", "/home/pohaoc2/UW/bagherilab/inverse_design")) / "src",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the companion inverse_design/src tree")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_weights(path: Path) -> np.ndarray:
    weights = pd.read_csv(path)["weight"].to_numpy(float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(f"Invalid weights: {path}")
    return weights / weights.sum()


def published_weights(campaign: Path) -> np.ndarray:
    sys.path.insert(0, str(inverse_design_source()))
    from inverse_design.rf.drf import DRF

    run = campaign / "published/n512/iter_4"
    parameters = pd.read_csv(run / "params.csv").to_numpy(float)
    statistics = pd.read_csv(run / "statistics.csv").to_numpy(float)
    target = pd.read_csv(campaign / "published/target_values.csv").to_numpy(float).ravel()
    model = DRF(n_trees=50, min_samples_leaf=5, n_try=None, random_state=202608080, criterion="CART")
    model.fit(parameters, statistics)
    weights = np.asarray(model.predict_weights(target), dtype=float)
    return weights / weights.sum()


def load_clouds(campaign: Path) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    paths = {
        "published": campaign / "published/n512/iter_4",
        "replicate_1": campaign / "replicates/replicate_1/iter_4",
        "replicate_2": campaign / "replicates/replicate_2/iter_4",
    }
    clouds = {name: pd.read_csv(path / "params.csv")[PARAMS] for name, path in paths.items()}
    weights = {
        "published": published_weights(campaign),
        "replicate_1": normalized_weights(paths["replicate_1"] / "weights.csv"),
        "replicate_2": normalized_weights(paths["replicate_2"] / "weights.csv"),
    }
    for name in GROUPS:
        if len(clouds[name]) != len(weights[name]):
            raise ValueError(f"Particle/weight mismatch in {name}")
    return clouds, weights


def independent_bases(
    clouds: dict[str, pd.DataFrame],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Project each fitted cloud into its own prior-range-normalized PCA basis."""
    coordinates: dict[str, np.ndarray] = {}
    explained: dict[str, np.ndarray] = {}
    for name in GROUPS:
        values = clouds[name][PARAMS].to_numpy(float).copy()
        for column, parameter in enumerate(PARAMS):
            low, high = RANGES[parameter]
            values[:, column] = (values[:, column] - low) / (high - low)
        centered = values - values.mean(axis=0)
        _, singular, components = np.linalg.svd(centered, full_matrices=False)
        components = components[:2].copy()
        for component in components:
            pivot = np.argmax(np.abs(component))
            if component[pivot] < 0:
                component *= -1
        coordinates[name] = centered @ components.T
        explained[name] = singular[:2] ** 2 / np.sum(singular**2)
    return coordinates, explained


def shared_grid(coordinates: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pooled = np.vstack([coordinates[name] for name in GROUPS])
    padding = 0.04 * np.ptp(pooled, axis=0)
    x = np.linspace(pooled[:, 0].min() - padding[0], pooled[:, 0].max() + padding[0], GRID)
    y = np.linspace(pooled[:, 1].min() - padding[1], pooled[:, 1].max() + padding[1], GRID)
    xx, yy = np.meshgrid(x, y)
    return x, y, xx, yy


def independent_grids(
    coordinates: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    grids = {}
    for name in GROUPS:
        values = coordinates[name]
        padding = 0.04 * np.ptp(values, axis=0)
        x = np.linspace(values[:, 0].min() - padding[0], values[:, 0].max() + padding[0], GRID)
        y = np.linspace(values[:, 1].min() - padding[1], values[:, 1].max() + padding[1], GRID)
        xx, yy = np.meshgrid(x, y)
        grids[name] = (x, y, xx, yy)
    return grids


def density_modes(
    coordinates: np.ndarray,
    weights: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bandwidth: float = BW,
    threshold: float = THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, xx, yy = grid
    kde = gaussian_kde(coordinates.T, weights=weights, bw_method=bandwidth)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(GRID, GRID)
    hits = (density == maximum_filter(density, size=(WINDOW, WINDOW))) & (density > threshold * density.max())
    yi, xi = np.nonzero(hits)
    centroids = np.column_stack([x[xi], y[yi]])
    peak_density = kde(centroids.T)
    order = np.argsort(-peak_density)
    return density, centroids[order], peak_density[order]


def detector_sensitivity(
    coordinates: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    grids: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Shared detector-setting sweep on each fitted cloud's coordinate grid."""
    rows = []
    for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
        for threshold in (0.05, 0.10, 0.25, 0.50):
            for name in GROUPS:
                _, centroids, peak_density = density_modes(
                    coordinates[name],
                    weights[name],
                    grids[name],
                    bandwidth=BW * multiplier,
                    threshold=threshold,
                )
                rows.append(
                    {
                        "bandwidth_multiplier": multiplier,
                        "bandwidth_factor": BW * multiplier,
                        "relative_density_threshold": threshold,
                        "group": name,
                        "mode_count": len(centroids),
                        "second_to_first_peak_density_ratio": (
                            float(peak_density[1] / peak_density[0])
                            if len(peak_density) >= 2
                            else 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def peak_table(
    centroids: dict[str, np.ndarray],
    peak_densities: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    x_name: str,
    y_name: str,
) -> pd.DataFrame:
    rows = []
    for name in GROUPS:
        for mode, (centroid, peak_density) in enumerate(
            zip(centroids[name], peak_densities[name]), 1
        ):
            rows.append(
                {
                    "group": name,
                    "mode": mode,
                    x_name: centroid[0],
                    y_name: centroid[1],
                    "kde_density": peak_density,
                    "ess": 1 / np.square(weights[name]).sum(),
                }
            )
    return pd.DataFrame(rows)


def nearest_peak(coordinates: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.argmin(
        np.sum((coordinates[:, None, :] - centroids[None, :, :]) ** 2, axis=2),
        axis=1,
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    four_spines(axis)


def plot_density_particles(
    axis: plt.Axes,
    coordinates: np.ndarray,
    weights: np.ndarray,
    centroids: np.ndarray,
) -> None:
    """ARCADE-like particles and marked KDE maxima on a white background."""
    if len(centroids) > len(DENSITY_PEAK_MARKERS):
        raise ValueError(
            f"Detected {len(centroids)} maxima but only "
            f"{len(DENSITY_PEAK_MARKERS)} distinct peak markers are configured"
        )
    sizes = 6 + 36 * np.sqrt(weights / weights.max())
    draw_order = np.argsort(sizes)[::-1]
    axis.scatter(
        coordinates[draw_order, 0],
        coordinates[draw_order, 1],
        s=sizes[draw_order],
        color="#7A7A7A",
        alpha=0.18,
        linewidths=0,
        rasterized=True,
    )
    for mode, centroid in enumerate(centroids):
        marker = DENSITY_PEAK_MARKERS[mode % len(DENSITY_PEAK_MARKERS)]
        axis.scatter(
            *centroid,
            marker=marker,
            s=PEAK_MARKER_AREA_PT2[marker],
            facecolor=DENSITY_PEAK_COLORS[mode % len(DENSITY_PEAK_COLORS)],
            edgecolor="black",
            linewidth=0.65,
            zorder=5,
        )
    style_axis(axis)


def load_extinction(campaign: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = campaign / "extinction"
    records = []
    for group in ["published", "replicate_1", "replicate_2", "reference"]:
        for path in sorted((root / "histories" / group).glob("draw_*.csv")):
            frame = pd.read_csv(path)
            time_column = "time" if "time" in frame.columns else "t"
            infected_column = "infected" if "infected" in frame.columns else "I"
            for time, infected in zip(frame[time_column], frame[infected_column]):
                records.append({"group": group, "draw": int(path.stem.split("_")[-1]), "time_days": float(time), "infected": float(infected)})
    source = pd.DataFrame(records)
    if source.empty:
        raise FileNotFoundError(f"No extinction histories under {root}")
    common_times = np.arange(0, 5000 + 25, 25, dtype=float)
    interpolated = []
    for (group, draw), frame in source.groupby(["group", "draw"]):
        frame = frame.sort_values("time_days")
        infected = np.interp(common_times, frame.time_days, frame.infected, right=0.0)
        for time, value in zip(common_times, infected):
            interpolated.append({"group": group, "draw": draw, "time_days": time, "infected": value})
    interpolated_frame = pd.DataFrame(interpolated)
    summary = (
        interpolated_frame.groupby(["group", "time_days"])["infected"]
        .agg(q25=lambda x: x.quantile(0.25), median="median", q75=lambda x: x.quantile(0.75), n="size")
        .reset_index()
    )
    return source, summary


def add_particle_legend(axis: plt.Axes, anchor_y: float) -> None:
    """Weight-size legend inside ``axis``; must not cover the replicate tag."""
    particle_handle = Line2D([], [], marker="o", linestyle="none", color="#7A7A7A", alpha=0.45, markersize=4, label=r"Particle (area $\propto$ weight)")
    legend = axis.legend(
        handles=[particle_handle],
        loc="upper right",
        bbox_to_anchor=(0.99, anchor_y),
        borderaxespad=0,
        frameon=False,
        fontsize=5.8,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.35,
    )
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    axis_box = axis.get_window_extent(renderer)
    if not axis_box.contains(legend_box.x0, legend_box.y0) or not axis_box.contains(
        legend_box.x1, legend_box.y1
    ):
        raise RuntimeError("Particle legend extends outside its panel")
    tag_box = axis.texts[0].get_bbox_patch().get_window_extent(renderer)
    if legend_box.overlaps(tag_box):
        raise RuntimeError("Particle legend overlaps the replicate tag")


def plot_pca(
    coordinates: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
    explained: dict[str, np.ndarray],
    grids: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    output: Path,
    *,
    split_formats: bool = False,
) -> None:
    apply_plos_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(PLOS_DOUBLE_WIDTH, 2.05),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.08},
    )
    for column, name in enumerate(GROUPS):
        axis = axes[column]
        plot_density_particles(axis, coordinates[name], weights[name], centroids[name])
        x, y, _, _ = grids[name]
        axis.set_xlim(x[0], x[-1])
        axis.set_ylim(y[0], y[-1])
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        in_axis_label(axis, DISPLAY_LABELS[name])
        axis.set_xlabel(f"PC1 ({100 * explained[name][0]:.1f}%)")
        axis.set_ylabel(f"PC2 ({100 * explained[name][1]:.1f}%)")
        panel_label(axis, chr(ord("A") + column), x=-0.18, y=1.02)
    add_particle_legend(axes[1], anchor_y=0.86)
    # ``save_figure`` uses Matplotlib's 0.1-inch tight-bbox padding (15 px at
    # the 150-dpi layout renderer); independent axes can place an edge tick in
    # that padding even though it remains fully inside the exported image.
    assert_text_inside_figure(fig, tolerance_px=15.0)
    save_figure(fig, output, split_formats=split_formats)
    plt.close(fig)


def _draw_parallel_break_marks(left: plt.Axes, right: plt.Axes, length_pt: float = 4.5) -> None:
    """Draw physically parallel 45-degree break marks on unequal-width axes."""
    left.figure.canvas.draw()
    half_px = length_pt * left.figure.dpi / 72.0
    slopes = []
    for axis, x_center in ((left, 1.0), (right, 0.0)):
        dx = half_px / axis.bbox.width
        dy = half_px / axis.bbox.height
        kwargs = dict(color="black", clip_on=False, linewidth=0.7)
        for y_center in (0.0, 1.0):
            axis.plot(
                (x_center - dx, x_center + dx),
                (y_center - dy, y_center + dy),
                transform=axis.transAxes,
                **kwargs,
            )
        slopes.append((2 * dy * axis.bbox.height) / (2 * dx * axis.bbox.width))
    if not np.allclose(slopes, 1.0, atol=1e-12):
        raise RuntimeError(f"Broken-axis marks are not parallel: display slopes={slopes}")


def plot_extinction(extinction: pd.DataFrame, output: Path, *, split_formats: bool = False) -> None:
    apply_plos_style()
    fig = plt.figure(figsize=(PLOS_DOUBLE_WIDTH, 2.05), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.05)
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1], sharey=left)

    extinction_colors = {"published": COLORS[0], "replicate_1": COLORS[1], "replicate_2": COLORS[2], "reference": "#777777"}
    for group in ["published", "replicate_1", "replicate_2", "reference"]:
        sub = extinction[extinction.group.eq(group)].sort_values("time_days")
        values_x = sub.time_days.to_numpy(float)
        for axis in (left, right):
            axis.plot(
                values_x,
                sub["median"].to_numpy(float),
                color=extinction_colors[group],
                linestyle="--" if group == "reference" else "-",
                label=DISPLAY_LABELS[group],
            )
            axis.fill_between(
                values_x,
                sub.q25.to_numpy(float),
                sub.q75.to_numpy(float),
                color=extinction_colors[group],
                alpha=0.12,
                linewidth=0,
            )
    left.axvline(365, color="black", linestyle=(0, (3, 2)), linewidth=0.8)
    left.text(
        357,
        0.97,
        "Day 365",
        transform=left.get_xaxis_transform(),
        ha="right",
        va="top",
        color="black",
    )
    left.set_xlim(0, 1000)
    right.set_xlim(1000, extinction.time_days.max())
    right.set_xticks([1000, 3000, 5000])
    left.set_ylim(-0.04, 0.86)
    left.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    left.set_ylabel("Infected fraction")
    left.set_xlabel("")
    right.set_xlabel("")
    for axis in (left, right):
        style_axis(axis)
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.tick_params(labelleft=False, left=False)
    _draw_parallel_break_marks(left, right)
    handles, labels = left.get_legend_handles_labels()
    legend = right.legend(handles, labels, loc="upper right", ncol=1, frameon=False, handlelength=1.6)
    shared_xlabel = fig.supxlabel("Simulation time (days)", y=-0.055)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlabel_box = shared_xlabel.get_window_extent(renderer)
    tick_boxes = [
        tick.get_window_extent(renderer)
        for axis in (left, right)
        for tick in axis.get_xticklabels()
        if tick.get_visible() and tick.get_text()
    ]
    if any(xlabel_box.overlaps(box) for box in tick_boxes):
        raise RuntimeError("Shared x-axis label overlaps an x tick label")
    legend_box = legend.get_window_extent(renderer)
    right_box = right.get_window_extent(renderer)
    if not right_box.contains(legend_box.x0, legend_box.y0) or not right_box.contains(
        legend_box.x1, legend_box.y1
    ):
        raise RuntimeError("Extinction legend extends outside the right axes")
    assert_text_inside_figure(fig)
    save_figure(fig, output, split_formats=split_formats)
    plt.close(fig)


def plot_pi_isf(
    coordinates: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
    output: Path,
    *,
    split_formats: bool = False,
) -> None:
    """Directly evaluate the PI--ISF plane used for the manuscript claim."""
    apply_plos_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(PLOS_DOUBLE_WIDTH, 2.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.08},
    )
    for column, name in enumerate(GROUPS):
        plot_density_particles(
            axes[column],
            coordinates[name],
            weights[name],
            centroids[name],
        )
        in_axis_label(axes[column], DISPLAY_LABELS[name])
        axes[column].set_xlabel(r"$P_I$")
        axes[column].set_xlim(0.10, 0.60)
        axes[column].set_ylim(0.45, 1.00)
        if column == 0:
            axes[column].set_ylabel(r"$I_{SF}$")
        panel_label(axes[column], chr(ord("A") + column), x=-0.16, y=1.02)
    # Second row, below the in-axis replicate tag.
    add_particle_legend(axes[0], anchor_y=0.84)
    assert_text_inside_figure(fig)
    save_figure(fig, output, split_formats=split_formats)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIG)
    parser.add_argument(
        "--render-extinction-only",
        action="store_true",
        help="Render the split extinction figure from the tracked summary CSV.",
    )
    parser.add_argument(
        "--reuse-sensitivity",
        action="store_true",
        help="Reuse only the unchanged direct-plane sensitivity CSV; fit-specific PCA sensitivity is always recomputed.",
    )
    parser.add_argument(
        "--render-density-only",
        action="store_true",
        help="Regenerate only the two density figures and their density outputs.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    if args.render_extinction_only:
        extinction_summary = pd.read_csv(args.out / "sir_extinction_timecourse_summary.csv")
        plot_extinction(extinction_summary, args.figure_dir / "sir_extinction", split_formats=True)
        return
    if args.campaign is None:
        parser.error("--campaign is required unless --render-extinction-only is used")
    campaign = args.campaign.resolve()
    clouds, weights = load_clouds(campaign)
    coordinates, explained = independent_bases(clouds)
    grids = independent_grids(coordinates)
    densities, centroids, peak_densities = {}, {}, {}
    for name in GROUPS:
        densities[name], centroids[name], peak_densities[name] = density_modes(
            coordinates[name], weights[name], grids[name]
        )
    peak_summary = peak_table(centroids, peak_densities, weights, "pc1", "pc2")
    peak_summary["pc1_variance_fraction"] = peak_summary["group"].map(
        {name: explained[name][0] for name in GROUPS}
    )
    peak_summary["pc2_variance_fraction"] = peak_summary["group"].map(
        {name: explained[name][1] for name in GROUPS}
    )
    sensitivity = detector_sensitivity(coordinates, weights, grids)

    pi_isf = {name: clouds[name][["PI", "ISF"]].to_numpy(float) for name in GROUPS}
    pi_isf_grid = shared_grid(pi_isf)
    pi_isf_grids = {name: pi_isf_grid for name in GROUPS}
    pi_isf_densities, pi_isf_centroids, pi_isf_peak_densities = {}, {}, {}
    for name in GROUPS:
        (
            pi_isf_densities[name],
            pi_isf_centroids[name],
            pi_isf_peak_densities[name],
        ) = density_modes(pi_isf[name], weights[name], pi_isf_grid)
    pi_isf_peak_summary = peak_table(
        pi_isf_centroids, pi_isf_peak_densities, weights, "PI", "ISF"
    )
    pi_isf_sensitivity_path = args.out / "sir_pi_isf_detector_sensitivity.csv"
    if args.reuse_sensitivity and pi_isf_sensitivity_path.is_file():
        pi_isf_sensitivity = pd.read_csv(pi_isf_sensitivity_path)
    else:
        pi_isf_sensitivity = detector_sensitivity(pi_isf, weights, pi_isf_grids)
    peak_summary.to_csv(args.out / "sir_common_kde_peaks.csv", index=False)
    sensitivity.to_csv(args.out / "sir_common_kde_detector_sensitivity.csv", index=False)
    pi_isf_peak_summary.to_csv(args.out / "sir_pi_isf_kde_peaks.csv", index=False)
    pi_isf_sensitivity.to_csv(args.out / "sir_pi_isf_detector_sensitivity.csv", index=False)
    plot_pca(
        coordinates,
        weights,
        centroids,
        explained,
        grids,
        args.figure_dir / "sir_common_kde_pca",
        split_formats=True,
    )
    plot_pi_isf(
        pi_isf,
        weights,
        pi_isf_centroids,
        args.figure_dir / "sir_pi_isf_weighted_kde",
        split_formats=True,
    )
    if not args.render_density_only:
        _, extinction_summary = load_extinction(campaign)
        extinction_summary.to_csv(args.out / "sir_extinction_timecourse_summary.csv", index=False)
        plot_extinction(extinction_summary, args.figure_dir / "sir_extinction", split_formats=True)
    source_files = (
        sorted(campaign.glob("published/n512/iter_4/*.csv"))
        + sorted(campaign.glob("published/*.csv"))
        + sorted(campaign.glob("replicates/replicate_*/iter_4/*.csv"))
        + sorted(campaign.glob("replicates/replicate_*/complete.json"))
        + sorted(campaign.glob("extinction/*.csv"))
        + sorted(campaign.glob("extinction/*.json"))
        + sorted(campaign.glob("extinction/histories/*/*.csv"))
    )
    current_source_records = [
        {"path": str(path), "sha256": sha256(path)} for path in source_files
    ]
    provenance_path = args.out / "sir_common_kde_provenance.json"
    if args.render_density_only and provenance_path.is_file():
        prior_source_records = json.loads(provenance_path.read_text()).get("source_files", [])
        prior_hashes = {record["sha256"] for record in prior_source_records}
        missing_hashes = [
            record["sha256"]
            for record in current_source_records
            if record["sha256"] not in prior_hashes
        ]
        if missing_hashes:
            raise RuntimeError(
                "Density-only source hashes do not match the preserved provenance: "
                + ", ".join(missing_hashes)
            )
        source_records = prior_source_records
    else:
        source_records = current_source_records
    provenance = {
        "campaign": str(campaign),
        "source_files": source_records,
        "basis": "separate prior-range-normalized PCA fitted independently to each 512-particle cloud",
        "explained_variance_fraction": {name: explained[name].tolist() for name in GROUPS},
        "kde": {"weighted": True, "particles_per_fit": 512, "ess_definition": "1/sum(normalized_weight^2)", "detector_definition": "same prespecified default SIR settings applied on each fit-specific PCA grid", "common_bandwidth_factor": BW, "bandwidth_definition": "nominal-n 2-D Scott factor, 512^(-1/6)", "grid_resolution_per_fit": GRID, "maximum_filter": WINDOW, "relative_density_threshold": THRESHOLD},
        "direct_manuscript_plane": "native PI--ISF coordinates; same weights, bandwidth factor, grid size, maximum filter, and threshold family as the PCA comparison",
        "mode_matching": "not performed because coordinates from independent PCA bases are not directly comparable",
        "detector_sensitivity": {
            "bandwidth_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
            "relative_density_thresholds": [0.05, 0.10, 0.25, 0.50],
            "pca_result": "the exact detected-peak count changes with the shared detector settings applied in each fit-specific PCA basis",
            "pi_isf_result": "the exact detected-peak count changes with the common detector setting",
        },
        "claim_boundary": "the four-parameter PCA maxima are supplementary to the direct PI--ISF analysis. Applying the same prespecified default SIR detector configuration to every 512-particle fit identifies multiple separated high-density regions whose fitted particles reproduce similar target metrics, supporting practical equifinality. The detector sweep shows that the exact numerical peak count depends on smoothing and detection settings. Extended trajectories diagnose the absorbing endpoint and do not change the 365-day calibration target or the fitted clouds.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(peak_summary.to_string(index=False))
    print("\nPI--ISF peaks")
    print(pi_isf_peak_summary.to_string(index=False))


if __name__ == "__main__":
    main()
