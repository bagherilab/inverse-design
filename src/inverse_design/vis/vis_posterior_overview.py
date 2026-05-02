"""Posterior / prior overview plots."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Polygon

from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks
from inverse_design.analyze.source_metrics import find_spacing_from_density
from inverse_design.plotting.theme import apply_publication_style
from inverse_design.vis.utils import find_mode_bin, peak_markers, peak_colors

__all__ = [
    "create_corner_plot",
    "export_pca_cluster_arcade_inputs",
    "extract_pca_cluster_parameters",
    "plot_marginal_distributions",
    "plot_pca_analysis",
    "plot_correlation_matrix",
    "plot_correlation_matrix_for_peaks",
    "plot_trace_analysis",
    "plot_summary_comparison",
    "plot_posterior_pair_density",
    "plot_pca_with_peaks",
    "write_arcade_input_from_params",
]


DEFAULT_ARCADE_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "sample_inputs" / "sample_combined_v3.xml"
)
DEFAULT_METADATA_COLUMNS = ("batch", "instance", "seed", "score", "distance")
DEFAULT_SOURCE_SIDE_LENGTH = 30 / np.sqrt(3)
NORMAL_VALUE_RE = re.compile(
    r"NORMAL\(\s*MU\s*=\s*([^,\s)]+)\s*,\s*SIGMA\s*=\s*([^\s)]+)\s*\)",
    flags=re.IGNORECASE,
)

ARCADE_PARAM_CONFIGS = {
    # Format: precision, clamp_to_unit_interval, XML subpath.
    "CELL_VOLUME": (1, False, None),
    "APOPTOSIS_AGE": (1, False, None),
    "NECROTIC_FRACTION": (6, True, None),
    "ACCURACY": (6, True, None),
    "AFFINITY": (6, True, None),
    "COMPRESSION_TOLERANCE": (6, False, None),
    "SYNTHESIS_DURATION": (3, False, "proliferation"),
    "BASAL_ENERGY": (6, False, "metabolism"),
    "PROLIFERATION_ENERGY": (12, False, "metabolism"),
    "MIGRATION_ENERGY": (12, False, "metabolism"),
    "METABOLIC_PREFERENCE": (6, False, "metabolism"),
    "CONVERSION_FRACTION": (6, False, "metabolism"),
    "RATIO_GLUCOSE_PYRUVATE": (6, False, "metabolism"),
    "LACTATE_RATE": (6, False, "metabolism"),
    "AUTOPHAGY_RATE": (12, False, "metabolism"),
    "GLUCOSE_UPTAKE_RATE": (6, False, "metabolism"),
    "ATP_PRODUCTION_RATE": (6, False, "metabolism"),
    "MIGRATORY_THRESHOLD": (6, False, "signaling"),
}

SOURCE_LAYER_CONFIGS = {
    "GLUCOSE_CONCENTRATION": "GLUCOSE",
    "OXYGEN_CONCENTRATION": "OXYGEN",
}


def _prepare_numeric_posterior_data(
    posterior_data: pd.DataFrame,
    drop_columns: tuple[str, ...] = DEFAULT_METADATA_COLUMNS,
) -> pd.DataFrame:
    """Return numeric posterior parameter columns with common metadata removed."""
    cleaned = posterior_data.drop(
        columns=[c for c in drop_columns if c in posterior_data], errors="ignore"
    )
    cleaned = cleaned.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    cleaned = cleaned.dropna(axis=0, how="any")
    if cleaned.empty:
        raise ValueError("No numeric posterior parameter rows remain after cleaning")
    return cleaned


def _project_to_pca(pca: PCA, scaler: StandardScaler, point: np.ndarray) -> np.ndarray:
    return pca.transform(scaler.transform(point.reshape(1, -1)))[0]


def _invert_pca_peak(
    peak_position: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
    n_components: int,
) -> np.ndarray:
    pc_vector = np.zeros(n_components)
    pc_vector[:2] = peak_position[:2]
    scaled_values = pca.inverse_transform(pc_vector.reshape(1, -1))
    return scaler.inverse_transform(scaled_values)[0]


def extract_pca_cluster_parameters(
    posterior_data: pd.DataFrame,
    n_components: int = 2,
    threshold_ratio: float = 0.1,
    neighborhood_size: int = 5,
    n_bins: int = 30,
    random_state: int = 0,
    drop_columns: tuple[str, ...] = DEFAULT_METADATA_COLUMNS,
) -> pd.DataFrame:
    """Extract mean, mode and inverse-PCA peak representatives from posterior data.

    The cluster rows use the peak coordinates found in the PC1/PC2 density map,
    padded with zero for any omitted higher principal components, and then
    inverted back through the PCA/scaler into the original parameter domain.
    """
    if n_components < 2:
        raise ValueError("n_components must be at least 2 for PC1/PC2 peak detection")

    posterior_params = _prepare_numeric_posterior_data(posterior_data, drop_columns)
    param_names = posterior_params.columns.tolist()
    data = posterior_params[param_names].values

    pca_result, peak_positions, _, pca, _, _, _, _ = perform_pca_and_find_peaks(
        data,
        n_components=n_components,
        threshold_ratio=threshold_ratio,
        neighborhood_size=neighborhood_size,
        random_state=random_state,
    )
    scaler = StandardScaler()
    scaler.fit(data)

    mean_point = np.array(posterior_params[param_names].mean(axis=0))
    mode_bins = find_mode_bin(posterior_params[param_names], n_bins=n_bins)
    mode_point = np.array([mode_bins[name]["mode_bin_center"] for name in param_names])

    rows = []
    for label, kind, point in [
        ("posterior_mean", "mean", mean_point),
        ("posterior_mode", "mode", mode_point),
    ]:
        pc = _project_to_pca(pca, scaler, point)
        record = {
            "label": label,
            "kind": kind,
            "cluster": np.nan,
            "cluster_size": len(posterior_params),
            "cluster_fraction": 1.0,
            "pc1": pc[0],
            "pc2": pc[1],
        }
        record.update(dict(zip(param_names, point)))
        rows.append(record)

    assignments = None
    if len(peak_positions) > 0:
        assignments = np.array(
            [
                np.argmin([np.linalg.norm(pt - peak) for peak in peak_positions])
                for pt in pca_result[:, :2]
            ]
        )

    for idx, peak in enumerate(peak_positions):
        params = _invert_pca_peak(peak, pca, scaler, n_components)
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak) ** 2, axis=1))
        nearest_index = int(np.argmin(distances))
        cluster_size = int(np.sum(assignments == idx)) if assignments is not None else 0
        record = {
            "label": f"cluster_{idx + 1:02d}",
            "kind": "pca_peak_inverse",
            "cluster": idx + 1,
            "cluster_size": cluster_size,
            "cluster_fraction": cluster_size / len(posterior_params),
            "pc1": float(peak[0]),
            "pc2": float(peak[1]),
            "nearest_sample_index": int(posterior_params.index[nearest_index]),
            "nearest_sample_pc_distance": float(distances[nearest_index]),
        }
        record.update(dict(zip(param_names, params)))
        rows.append(record)

    return pd.DataFrame(rows)


def _arcade_param_path(base_name: str, subfolder: str | None) -> str:
    return f"{subfolder}/{base_name}" if subfolder else base_name


def _finite_param(params: dict, param_name: str) -> bool:
    value = params.get(param_name)
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _format_arcade_value(
    value: float,
    precision: int,
    is_bounded: bool,
) -> str:
    value = float(value)
    if is_bounded:
        value = min(1.0, max(0.0, value))
    else:
        value = max(0.0, value)
    return f"{value:.{precision}f}"


def _normal_parts(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    match = NORMAL_VALUE_RE.search(value)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _update_population_xml(root: ET.Element, params: dict) -> dict:
    cancerous_pop = root.find(".//population[@id='cancerous']") or root.find(
        ".//population[@id='cancer']"
    )
    if cancerous_pop is None:
        raise ValueError("Could not find population with id='cancerous' or id='cancer'")

    written = {}
    for xml_param in cancerous_pop.findall("population.parameter"):
        param_id = xml_param.get("id")
        for base_param, (precision, is_bounded, subfolder) in ARCADE_PARAM_CONFIGS.items():
            if param_id != _arcade_param_path(base_param, subfolder):
                continue

            mu_key = f"{base_param}_MU"
            sigma_key = f"{base_param}_SIGMA"
            existing_mu, existing_sigma = _normal_parts(xml_param.get("value"))

            if _finite_param(params, base_param):
                formatted = _format_arcade_value(params[base_param], precision, is_bounded)
                xml_param.set("value", formatted)
                written[base_param] = float(formatted)
            elif _finite_param(params, mu_key) or _finite_param(params, sigma_key):
                mu = (
                    _format_arcade_value(params[mu_key], precision, is_bounded)
                    if _finite_param(params, mu_key)
                    else existing_mu
                )
                sigma = (
                    _format_arcade_value(params[sigma_key], precision, False)
                    if _finite_param(params, sigma_key)
                    else existing_sigma
                )
                if mu is None or sigma is None:
                    continue
                xml_param.set("value", f"NORMAL(MU={mu},SIGMA={sigma})")
                written[mu_key] = float(mu)
                written[sigma_key] = float(sigma)
    return written


def _template_radius_bound(root: ET.Element) -> float:
    series = root.find(".//series")
    if series is None:
        return 12.0
    radius = float(series.get("radius", 10))
    margin = float(series.get("margin", 0))
    return radius + margin


def _format_spacing_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None or not np.isfinite(float(value)):
        return ""
    return f"*:{int(round(float(value)))}"


def _update_source_xml(
    root: ET.Element,
    params: dict,
    side_length: float,
) -> dict:
    written = {}
    sites_component = root.find(".//component[@id='SITES']")
    if sites_component is not None:
        x_spacing = params.get("X_SPACING")
        y_spacing = params.get("Y_SPACING")
        if not (_finite_param(params, "X_SPACING") and _finite_param(params, "Y_SPACING")):
            if _finite_param(params, "CAPILLARY_DENSITY"):
                spacing = find_spacing_from_density(
                    float(params["CAPILLARY_DENSITY"]),
                    _template_radius_bound(root),
                    side_length,
                    MICRON_to_MM=1e-3,
                )
                if spacing is not None:
                    x_spacing, y_spacing = spacing

        if x_spacing is not None:
            x_value = _format_spacing_value(x_spacing)
            x_param = sites_component.find("component.parameter[@id='X_SPACING']")
            if x_param is not None:
                x_param.set("value", x_value)
            written["X_SPACING"] = x_value
        if y_spacing is not None:
            y_value = _format_spacing_value(y_spacing)
            y_param = sites_component.find("component.parameter[@id='Y_SPACING']")
            if y_param is not None:
                y_param.set("value", y_value)
            written["Y_SPACING"] = y_value
        if _finite_param(params, "CAPILLARY_DENSITY"):
            written["CAPILLARY_DENSITY"] = float(params["CAPILLARY_DENSITY"])

    for param_name, layer_id in SOURCE_LAYER_CONFIGS.items():
        if not _finite_param(params, param_name):
            continue
        layer = root.find(f".//layer[@id='{layer_id}']")
        if layer is None:
            continue
        value = f"{float(params[param_name]):.6g}"
        for xml_param in layer.findall("layer.parameter"):
            if (
                xml_param.get("operation") == "generator"
                or xml_param.get("id") == "INITIAL_CONCENTRATION"
            ):
                xml_param.set("value", value)
        written[param_name] = float(params[param_name])

    return written


def write_arcade_input_from_params(
    params: dict,
    output_path: str | Path,
    template_path: str | Path = DEFAULT_ARCADE_TEMPLATE,
    side_length: float = DEFAULT_SOURCE_SIDE_LENGTH,
) -> dict:
    """Write one ARCADE XML input from a parameter dictionary."""
    tree = ET.parse(template_path)
    root = tree.getroot()
    written = {}
    written.update(_update_population_xml(root, params))
    written.update(_update_source_xml(root, params, side_length))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return written


def _summary_lines(summary_df: pd.DataFrame) -> list[str]:
    lines = ["label, kind, PC1, PC2, Vol, Tolerance"]
    for _, row in summary_df.iterrows():
        vol = row.get("CELL_VOLUME_MU", np.nan)
        tolerance = row.get("COMPRESSION_TOLERANCE", np.nan)
        lines.append(
            f"{row['label']}: {row['kind']}, PC=({row['pc1']:.6g}, {row['pc2']:.6g}), "
            f"Vol={vol:.6g}, Tolerance={tolerance:.6g}"
        )
    return lines


def export_pca_cluster_arcade_inputs(
    posterior_data: pd.DataFrame,
    output_dir: str | Path,
    template_path: str | Path = DEFAULT_ARCADE_TEMPLATE,
    n_components: int = 2,
    threshold_ratio: float = 0.1,
    neighborhood_size: int = 5,
    n_bins: int = 30,
    random_state: int = 0,
    drop_columns: tuple[str, ...] = DEFAULT_METADATA_COLUMNS,
    side_length: float = DEFAULT_SOURCE_SIDE_LENGTH,
    include_mean_mode: bool = True,
) -> pd.DataFrame:
    """Save inverse-PCA cluster representatives and ARCADE XML inputs.

    Outputs are written under ``output_dir``:
    ``summary.csv`` and ``summary.json`` contain PC coordinates and parameter
    values; ``cluster_summary.txt`` gives a quick Vol/Tolerance readout; and
    ``inputs/input_*.xml`` contains the generated ARCADE input files.
    """
    output_dir = Path(output_dir)
    inputs_dir = output_dir / "inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    summary_df = extract_pca_cluster_parameters(
        posterior_data,
        n_components=n_components,
        threshold_ratio=threshold_ratio,
        neighborhood_size=neighborhood_size,
        n_bins=n_bins,
        random_state=random_state,
        drop_columns=drop_columns,
    )
    if not include_mean_mode:
        summary_df = summary_df[summary_df["kind"] == "pca_peak_inverse"].reset_index(drop=True)

    parameter_log = []
    for file_index, (_, row) in enumerate(summary_df.iterrows(), start=1):
        file_name = f"input_{file_index}.xml"
        params = {
            column: row[column] for column in summary_df.columns if column not in {"label", "kind"}
        }
        written = write_arcade_input_from_params(
            params=params,
            output_path=inputs_dir / file_name,
            template_path=template_path,
            side_length=side_length,
        )
        log_row = {
            "file_name": file_name,
            "label": row["label"],
            "kind": row["kind"],
            "pc1": row["pc1"],
            "pc2": row["pc2"],
        }
        log_row.update(written)
        parameter_log.append(log_row)

    summary_df.to_csv(output_dir / "summary.csv", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(json.loads(summary_df.to_json(orient="records")), fh, indent=2)
    pd.DataFrame(parameter_log).to_csv(output_dir / "parameter_log.csv", index=False)
    with open(output_dir / "cluster_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(_summary_lines(summary_df)) + "\n")

    return summary_df


def create_corner_plot(prior_data: pd.DataFrame, posterior_data: pd.DataFrame):
    """Corner plot comparing prior and posterior marginals and scatter."""
    param_names = prior_data.columns.tolist()
    fig, axes = plt.subplots(
        len(param_names), len(param_names), figsize=(12, 12), tight_layout=True
    )
    for i, param1 in enumerate(param_names):
        for j, param2 in enumerate(param_names):
            ax = axes[i, j]
            if i == j:
                ax.hist(
                    prior_data[param1],
                    bins=30,
                    alpha=0.5,
                    color="blue",
                    density=True,
                    label="Prior",
                )
                ax.hist(
                    posterior_data[param1],
                    bins=30,
                    alpha=0.5,
                    color="orange",
                    density=True,
                    label="Posterior",
                )
                ax.set_ylabel("Density")
                ax.legend()
            elif i > j:
                ax.scatter(
                    prior_data[param2],
                    prior_data[param1],
                    alpha=0.3,
                    s=1,
                    color="blue",
                    label="Prior",
                )
                ax.scatter(
                    posterior_data[param2],
                    posterior_data[param1],
                    alpha=0.3,
                    s=1,
                    color="orange",
                    label="Posterior",
                )
                ax.set_xlabel(param2)
                ax.set_ylabel(param1)
            else:
                ax.axis("off")
    plt.suptitle("Corner Plot: Prior vs Posterior")
    return fig


def plot_marginal_distributions(posterior_data: pd.DataFrame, best_sample_idx: int):
    """Marginal histograms annotated with mean, mode and PCA-peak positions."""
    param_names = posterior_data.columns.tolist()
    data = posterior_data[param_names].values
    mean_point = np.array(posterior_data[param_names].mean(axis=0))
    mode_bins = find_mode_bin(posterior_data[param_names], n_bins=30)
    mode_point = np.array([info["mode_bin_center"] for info in mode_bins.values()])
    pca_result, peak_positions, _, pca, _, _, _, _ = perform_pca_and_find_peaks(data, 2)

    peak_param_values = []
    for peak in peak_positions:
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak) ** 2, axis=1))
        peak_param_values.append(data[np.argmin(distances)])

    n_cols = 4
    n_rows = int(np.ceil(len(param_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 2 * n_rows))
    axes = axes.flatten()
    marker_to_size = {
        "o": 22 * 1.21,
        "s": 20 * 1.21,
        "^": 32 * 1.21,
        "D": 20 * 1.21,
        "*": 55 * 1.21,
        "p": 35 * 1.21,
    }

    for i, param in enumerate(param_names):
        ax = axes[i]
        ax.hist(
            posterior_data[param],
            bins=30,
            alpha=0.7,
            color="#A0522D",
            density=True,
            edgecolor="black",
            linewidth=1.5,
            label="Posterior",
        )
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.5)
        for idx, pt in enumerate([mean_point[i], mode_point[i]]):
            ax.scatter(
                pt,
                y_max * 1.1,
                marker=peak_markers[idx],
                s=marker_to_size[peak_markers[idx]] * 4,
                color=peak_colors[idx],
                edgecolor="black",
                linewidth=1.5,
                facecolor="none",
                zorder=10,
            )
        for j, peak_values in enumerate(peak_param_values):
            k = j + 2
            ax.scatter(
                peak_values[i],
                y_max * 1.1,
                marker=peak_markers[k % len(peak_markers)],
                s=marker_to_size[peak_markers[k % len(peak_markers)]] * 4,
                color=peak_colors[k % len(peak_colors)],
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.set_title(param.replace("_MU", "").replace("_", " ").title(), fontsize=12, pad=10)
        if i % n_cols == 0:
            ax.set_ylabel("Density", fontsize=12)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=14, width=1.5)
        ax.ticklabel_format(axis="x", style="scientific", scilimits=(-2, 2))
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2))
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        if param == "CAPILLARY_DENSITY":
            ax.set_xscale("log")
    for i in range(len(param_names), len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig


def plot_pca_analysis(prior_data: pd.DataFrame, posterior_data: pd.DataFrame):
    """PCA projection scatter + component loadings."""
    param_names = prior_data.columns.tolist()
    scaler = StandardScaler()
    combined_data = np.vstack([prior_data.values, posterior_data.values])
    scaler.fit(combined_data)
    prior_scaled = scaler.transform(prior_data.values)
    posterior_scaled = scaler.transform(posterior_data.values)
    pca = PCA(n_components=2)
    pca.fit(combined_data)
    prior_pca = pca.transform(prior_scaled)
    posterior_pca = pca.transform(posterior_scaled)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(prior_pca[:, 0], prior_pca[:, 1], alpha=0.5, color="blue", s=1, label="Prior")
    ax1.scatter(
        posterior_pca[:, 0], posterior_pca[:, 1], alpha=0.5, color="orange", s=1, label="Posterior"
    )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax1.legend()
    ax1.set_title("PCA Projection")
    loadings = pca.components_.T
    ax2.bar(range(len(param_names)), loadings[:, 0], alpha=0.7, label="PC1")
    ax2.bar(range(len(param_names)), loadings[:, 1], alpha=0.7, label="PC2")
    ax2.set_xticks(range(len(param_names)))
    ax2.set_xticklabels(param_names, rotation=45, ha="right")
    ax2.set_ylabel("Loading")
    ax2.legend()
    ax2.set_title("PCA Component Loadings")
    plt.tight_layout()
    return fig


def plot_correlation_matrix(posterior_data: pd.DataFrame):
    """Lower-triangle correlation heatmap with colorbar in upper-triangle space."""
    apply_publication_style(font_size=10, axes_linewidth=1.0)
    posterior_corr = posterior_data.corr()
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    mask = np.triu(np.ones_like(posterior_corr, dtype=bool), k=1)
    heatmap = sns.heatmap(
        posterior_corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
        cbar=False,
        linewidths=0,
        vmin=-1,
        vmax=1,
    )
    for i in range(len(posterior_corr.columns)):
        for j in range(i + 1):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="gray", linewidth=0.1))
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontweight="bold")
    cbar_ax = inset_axes(
        ax,
        width="40%",
        height="4%",
        loc="upper right",
        bbox_to_anchor=(0.1, 0.05, 0.8, 0.8),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=10, width=1.5)
    cbar.set_label("Correlation", fontsize=12, fontweight="bold", labelpad=20)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight("bold")
    plt.tight_layout()
    return fig


def plot_correlation_matrix_for_peaks(posterior_data: pd.DataFrame, n_components: int = 2):
    """Correlation heatmaps split by PCA peak membership."""
    param_names = posterior_data.columns.tolist()
    data = posterior_data[param_names].values
    _, peak_positions, _, _, _, _, _, peak_points = perform_pca_and_find_peaks(data, n_components)
    n_peaks = len(peak_positions)
    fig, axes = plt.subplots(1, n_peaks, figsize=(8 * n_peaks, 8))
    for i, peak_point in enumerate(peak_points):
        peak_df = pd.DataFrame(peak_point, columns=param_names)
        corr_matrix = peak_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_matrix = corr_matrix.mask(mask)
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            center=0,
            ax=axes[i],
            fmt=".2f",
            cbar=(i == len(peak_points) - 1),
        )
        axes[i].set_title(f"Correlation Matrix for Peak {i + 1}")
    plt.tight_layout()
    return fig


def plot_trace_analysis(prior_data: pd.DataFrame, posterior_data: pd.DataFrame):
    """Running-mean trace plots for parameter convergence."""
    param_names = prior_data.columns.tolist()
    n_cols = 4
    n_rows = int(np.ceil(len(param_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    for i, param in enumerate(param_names):
        ax = axes[i]
        indices = np.random.permutation(len(prior_data))
        prior_trace = prior_data[param].values[indices]
        posterior_trace = posterior_data[param].values[indices]
        x = np.arange(len(prior_trace))
        ax.plot(
            x[::50],
            (np.cumsum(prior_trace) / (x + 1))[::50],
            color="blue",
            alpha=0.7,
            label="Prior",
        )
        ax.plot(
            x[::50],
            (np.cumsum(posterior_trace) / (x + 1))[::50],
            color="orange",
            alpha=0.7,
            label="Posterior",
        )
        ax.set_title(f"{param} - Running Mean")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")
        ax.legend()
    for i in range(len(param_names), len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.suptitle("Running Means: Prior vs Posterior", y=1.02)
    return fig


def plot_summary_comparison(prior_data: pd.DataFrame, posterior_data: pd.DataFrame):
    """Four-panel summary statistics comparison (mean, std, delta-mean, uncertainty reduction)."""
    param_names = prior_data.columns.tolist()
    stats_df = pd.DataFrame(
        {
            "Parameter": param_names,
            "Prior Mean": prior_data.mean().values,
            "Prior Std": prior_data.std().values,
            "Posterior Mean": posterior_data.mean().values,
            "Posterior Std": posterior_data.std().values,
        }
    )
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(param_names))
    width = 0.35
    for ax, prior_col, post_col, ylabel, title in [
        (ax1, "Prior Mean", "Posterior Mean", "Mean", "Parameter Means"),
        (ax2, "Prior Std", "Posterior Std", "Standard Deviation", "Parameter Standard Deviations"),
    ]:
        ax.bar(x - width / 2, stats_df[prior_col], width, label="Prior", alpha=0.7)
        ax.bar(x + width / 2, stats_df[post_col], width, label="Posterior", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    mean_diff = stats_df["Posterior Mean"] - stats_df["Prior Mean"]
    ax3.bar(x, mean_diff, color="green", alpha=0.7)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(param_names, rotation=45, ha="right")
    ax3.set_ylabel("Mean Difference")
    ax3.set_title("Change in Mean (Posterior - Prior)")
    unc_reduction = stats_df["Prior Std"] / stats_df["Posterior Std"]
    ax4.bar(x, unc_reduction, color="purple", alpha=0.7)
    ax4.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(param_names, rotation=45, ha="right")
    ax4.set_ylabel("Uncertainty Reduction Ratio")
    ax4.set_title("Uncertainty Reduction (Prior Std / Posterior Std)")
    plt.tight_layout()
    return fig


def plot_posterior_pair_density(posterior_data: pd.DataFrame):
    """Pair-density plot using KDE contours on the lower triangle."""
    param_names = posterior_data.columns.tolist()
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    for i, param1 in enumerate(param_names):
        for j, param2 in enumerate(param_names):
            ax = axes[i, j]
            if i == j:
                kde = gaussian_kde(posterior_data[param1])
                x_range = np.linspace(
                    posterior_data[param1].min(), posterior_data[param1].max(), 100
                )
                ax.plot(x_range, kde(x_range), color="blue")
                ax.fill_between(x_range, kde(x_range), alpha=0.3, color="blue")
                ax.set_ylabel(param1)
            elif i > j:
                x = np.linspace(posterior_data[param2].min(), posterior_data[param2].max(), 100)
                y = np.linspace(posterior_data[param1].min(), posterior_data[param1].max(), 100)
                X, Y = np.meshgrid(x, y)
                kernel = gaussian_kde(np.vstack([posterior_data[param2], posterior_data[param1]]))
                Z = np.reshape(kernel(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)
                contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.6)
                ax.set_xlabel(param2)
                ax.set_ylabel(param1)
                if i == n_params - 1 and j == n_params - 2:
                    plt.colorbar(contour, ax=ax, label="Density")
            else:
                ax.axis("off")
    plt.suptitle("Posterior Parameter Pair Densities", y=1.02)
    plt.tight_layout()
    return fig


def plot_pca_with_peaks(
    posterior_data: pd.DataFrame,
    n_components: int = 2,
    fig=None,
    ax=None,
    font_size: int = 12,
    cluster_fill_alpha: float = 0.12,
    point_size: float = 6,
    save_path: str | None = None,
):
    """PCA scatter coloured by peak cluster with mean/mode/peak markers overlaid."""
    param_names = posterior_data.columns.tolist()
    data = posterior_data[param_names].values
    mean_point = np.array(posterior_data[param_names].mean(axis=0))
    mode_bins = find_mode_bin(posterior_data[param_names], n_bins=30)
    mode_point = np.array([info["mode_bin_center"] for info in mode_bins.values()])
    pca_result, peak_positions, _, pca, _, _, _, _ = perform_pca_and_find_peaks(data, n_components)

    scaler = StandardScaler()
    scaler.fit(data)
    mean_pca = pca.transform(scaler.transform(mean_point.reshape(1, -1)))[0]
    mode_pca = pca.transform(scaler.transform(mode_point.reshape(1, -1)))[0]

    apply_publication_style(font_size=font_size, axes_linewidth=1.0)
    owns_figure = fig is None or ax is None
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(3.5 * 0.8, 2.8 * 0.8))

    marker_to_size = {"o": 30, "s": 27, "^": 37, "D": 30, "*": 60, "p": 42}

    if len(peak_positions) > 0:
        assignments = np.array(
            [
                np.argmin([np.linalg.norm(pt - pk) for pk in peak_positions])
                for pt in pca_result[:, :2]
            ]
        )
        for i in range(len(peak_positions)):
            mask = assignments == i
            pts_2d = pca_result[mask, :2]
            if not np.any(mask):
                continue
            cluster_color = peak_colors[(i + 2) % len(peak_colors)]
            if cluster_fill_alpha > 0 and pts_2d.shape[0] >= 3:
                try:
                    hull = ConvexHull(pts_2d)
                    hull_verts = pts_2d[hull.vertices]
                    ax.add_patch(
                        Polygon(
                            hull_verts,
                            closed=True,
                            facecolor=cluster_color,
                            alpha=cluster_fill_alpha,
                            edgecolor=cluster_color,
                            linewidth=0.8,
                            zorder=1,
                        )
                    )
                except Exception:
                    pass
            ax.scatter(
                pts_2d[:, 0],
                pts_2d[:, 1],
                s=point_size,
                alpha=0.65,
                color=cluster_color,
                edgecolors="none",
                zorder=2,
                label=f"Cluster {i + 1}",
            )
    else:
        ax.scatter(
            pca_result[:, 0], pca_result[:, 1], c="#A0522D", alpha=0.75, s=30, edgecolors="none"
        )

    for idx, pt in enumerate([mean_pca, mode_pca]):
        marker = peak_markers[idx % len(peak_markers)]
        ax.scatter(
            pt[0],
            pt[1],
            color=peak_colors[idx % len(peak_colors)],
            s=marker_to_size[marker],
            alpha=1.0,
            marker=marker,
            edgecolors="black",
            facecolor="none",
            linewidths=1.0,
            zorder=8,
        )

    for i, peak in enumerate(peak_positions):
        k = i + 2
        marker = peak_markers[k % len(peak_markers)]
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak) ** 2, axis=1))
        pt_pca = pca_result[np.argmin(distances)]
        ax.scatter(
            pt_pca[0],
            pt_pca[1],
            color=peak_colors[k % len(peak_colors)],
            s=marker_to_size[marker],
            alpha=1.0,
            marker=marker,
            edgecolors="black",
            linewidths=1.0,
            zorder=8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=font_size)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=font_size)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=font_size - 1, width=1.0)
    if owns_figure:
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
