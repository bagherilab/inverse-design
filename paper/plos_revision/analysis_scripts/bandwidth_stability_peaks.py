"""Redraw the KDE covariance-factor sweep for the canonical Fig 3 cloud.

This uses the frozen unweighted Fig 3 convention: standardized 19-parameter
PCA, Scott covariance factor, a 100x100 grid, a 5x5 maximum filter, and a 5%
relative-density threshold.

The projection and peak identities are checked against the frozen assignment
artifact rather than a historical weighted partition.

Alongside the figure it writes the dense covariance-factor sweep matching the S4
sensitivity grid (0.5, 0.75, 1, 1.25, 1.5, 2, 2.5x) so the text can state the
range over which the detected-maximum count is stable.

Detector diagnostics consume the PC1/PC2 coordinates in the frozen assignment
artifact directly. The parameter CSV is used only to validate row alignment.

Run: ``python analysis_scripts/bandwidth_stability_peaks.py``.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from hashlib import sha256

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.stats import gaussian_kde

# Canonical detector settings frozen for Fig 3.
GRID = 100
WINDOW = 5
RELATIVE_DENSITY_THRESHOLD = 0.05

# The 19 perturbed parameters, in the order used by weighted_kde.py.
PARAMS = [
    "CELL_VOLUME_MU", "NECROTIC_FRACTION", "ACCURACY", "COMPRESSION_TOLERANCE",
    "SYNTHESIS_DURATION_MU", "BASAL_ENERGY_MU", "PROLIFERATION_ENERGY_MU",
    "MIGRATION_ENERGY_MU", "METABOLIC_PREFERENCE_MU", "CONVERSION_FRACTION_MU",
    "RATIO_GLUCOSE_PYRUVATE_MU", "LACTATE_RATE_MU", "AUTOPHAGY_RATE_MU",
    "GLUCOSE_UPTAKE_RATE_MU", "ATP_PRODUCTION_RATE_MU", "MIGRATORY_THRESHOLD_MU",
    "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION", "CAPILLARY_DENSITY",
]

# Frozen Fig 3 peak centres and memberships, for matching.
PAPER = np.array([
    [+1.5420984882, -2.0413592090],
    [-2.5753182829, -0.5309657005],
    [-1.1286583363, +2.1661655645],
    [+2.0985061600, +2.1661655645],
])
PAPER_N = [310, 315, 162, 237]
MATCH_TOLERANCE = 1.0

# Panel multiples of the Scott factor, in the order the figure shows
# them (two rows of three).
PANEL_MULTIPLES = [0.5, 0.8, 1.0, 1.1, 1.2, 2.0]
# Dense sweep reported in the S4 sensitivity grid.
SWEEP_MULTIPLES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]

# Marker convention of the 2026-06-04 figure: P1 green triangle, P2 purple
# diamond, P3 orange star, P4 yellow pentagon, gray cross for unmatched maxima.
PEAK_COLORS = ["#2ca02c", "#9467bd", "#ff7f0e", "#f7e11e"]
PEAK_MARKERS = ["^", "D", "*", "p"]
PEAK_SIZES = [190, 190, 320, 220]
UNMATCHED_COLOR = "#7f7f7f"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PARAMS = os.path.join(
    REPO_ROOT,
    "analysis_outputs/pm50_reduced_campaign/ARCADE_OUTPUT/"
    "ABC_SMC_RF_N1024_pm50/iter_4/all_param_df.csv",
)
DEFAULT_ASSIGNMENTS = os.path.join(
    REPO_ROOT, "analysis_outputs/pm50_fig3/canonical_peak_assignments.csv"
)


def pca2(X):
    """sklearn's StandardScaler + PCA(n_components=2), including svd_flip."""
    Z = (X - X.mean(0)) / X.std(0)
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    U, S, Vt = U[:, :2], S[:2], Vt[:2]
    for i in range(2):
        j = int(np.argmax(np.abs(Vt[i])))
        if Vt[i, j] < 0:
            Vt[i] *= -1
            U[:, i] *= -1
    explained = (S ** 2) / ((len(Z) - 1) * Z.var(0, ddof=1).sum())
    return U * S, explained


def project(params_csv):
    """Unweighted PCA basis over the 19 perturbed parameters."""
    frame = pd.read_csv(params_csv)
    missing = [name for name in PARAMS if name not in frame.columns]
    if missing:
        raise SystemExit("missing parameter columns: %s" % ", ".join(missing))
    return pca2(frame[PARAMS].to_numpy(float))


def scott_factor(n_particles, n_dim=2):
    return float(n_particles ** (-1.0 / (n_dim + 4)))


def detect_peaks(pc, factor, weights=None):
    """Local density maxima under the default detector settings."""
    kde = gaussian_kde(pc.T, bw_method=factor, weights=weights)
    # Match perform_pca_and_find_peaks exactly: asymmetric fixed padding is
    # part of the frozen Fig 3 detector convention.
    xs = np.linspace(pc[:, 0].min() - 2, pc[:, 0].max() + 2, GRID)
    ys = np.linspace(pc[:, 1].min() - 1, pc[:, 1].max() + 1, GRID)
    XX, YY = np.meshgrid(xs, ys)
    density = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(GRID, GRID)
    hits = (density == maximum_filter(density, size=(WINDOW, WINDOW))) & (
        density > density.max() * RELATIVE_DENSITY_THRESHOLD)
    yi, xi = np.nonzero(hits)
    centres = np.column_stack([xs[xi], ys[yi]])
    labels = np.argmin(((pc[:, None, :] - centres[None, :, :]) ** 2).sum(-1), axis=1)
    return centres, labels


def match_to_canonical(centres, tol=MATCH_TOLERANCE):
    """Nearest detected centre to each frozen Fig 3 peak, if within tolerance."""
    out = []
    for target in PAPER:
        if len(centres) == 0:
            out.append(None)
            continue
        d = np.linalg.norm(centres - target, axis=1)
        j = int(np.argmin(d))
        out.append((j, float(d[j])) if d[j] <= tol else None)
    return out


def sweep(pc, factor_default, multiples):
    """Detected maxima and canonical-peak recovery at each covariance factor."""
    rows = []
    for multiple in multiples:
        factor = multiple * factor_default
        centres, labels = detect_peaks(pc, factor)
        matches = match_to_canonical(centres)
        row = {
            "multiple": multiple,
            "covariance_factor": round(factor, 6),
            "mode_count": len(centres),
            "canonical_peaks_recovered": sum(m is not None for m in matches),
        }
        for k, match in enumerate(matches):
            row["P%d_members" % (k + 1)] = int((labels == match[0]).sum()) if match else 0
            row["P%d_canonical_members" % (k + 1)] = PAPER_N[k]
        rows.append(row)
    return pd.DataFrame(rows)


def draw(pc, factor_default, output_png):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0))
    for axis, multiple in zip(axes.ravel(), PANEL_MULTIPLES):
        factor = multiple * factor_default
        centres, labels = detect_peaks(pc, factor)
        matched = {m[0]: k for k, m in enumerate(match_to_canonical(centres)) if m is not None}

        for j in range(len(centres)):
            peak = matched.get(j)
            axis.scatter(
                pc[labels == j, 0], pc[labels == j, 1], s=9, linewidths=0, zorder=2,
                color=PEAK_COLORS[peak] if peak is not None else UNMATCHED_COLOR)
        for j, centre in enumerate(centres):
            peak = matched.get(j)
            axis.scatter(
                centre[0], centre[1], zorder=3, edgecolor="black", linewidths=1.1,
                marker=PEAK_MARKERS[peak] if peak is not None else "X",
                color=PEAK_COLORS[peak] if peak is not None else UNMATCHED_COLOR,
                s=PEAK_SIZES[peak] if peak is not None else 190)

        # Same notation as the caption and Methods: KDE covariance factor h.
        title = "$h$ = %.3f (%s)" % (factor, "Scott" if multiple == 1.0 else "%g$\\times$ Scott" % multiple)
        axis.set_title(title, fontsize=17)
        axis.set_xlabel("PC1", fontsize=15)
        axis.set_ylabel("PC2", fontsize=15)
        axis.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    fig.savefig(os.path.splitext(output_png)[0] + ".pdf")
    plt.close(fig)


def file_digest(path):
    with open(path, "rb") as handle:
        return sha256(handle.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=os.environ.get("DED_PUBLISHED_PARAMS", DEFAULT_PARAMS))
    parser.add_argument("--assignments", default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--figure", default="body/extended_data/figures/bandwidth_stability_peaks.png")
    parser.add_argument("--sweep", default="analysis_outputs/reviewer_closeout/bandwidth_stability_sweep.csv")
    parser.add_argument("--provenance", default="analysis_outputs/reviewer_closeout/bandwidth_stability_provenance.json")
    args = parser.parse_args()

    frozen = pd.read_csv(args.assignments)
    params = pd.read_csv(args.params)
    if frozen.input_folder.tolist() != params.input_folder.astype(str).tolist():
        raise RuntimeError("frozen assignment rows are not aligned to parameter rows")
    pc = frozen[["pc1", "pc2"]].to_numpy(float)
    # Explained variance is already frozen in the Fig 3 provenance; it is not
    # recomputed here because downstream detector diagnostics consume only PC coordinates.
    explained = np.array([np.nan, np.nan])
    counts = frozen.groupby("peak").size().sort_index().tolist()
    if counts != PAPER_N:
        raise RuntimeError(f"unexpected frozen P1--P4 counts: {counts}")
    factor_default = scott_factor(len(pc))
    print("particles %d | Scott covariance factor %.4f" % (len(pc), factor_default))
    print("using frozen canonical PC1/PC2 coordinates")

    panels = sweep(pc, factor_default, PANEL_MULTIPLES)
    dense = sweep(pc, factor_default, SWEEP_MULTIPLES)
    print("\npanel factors")
    print(panels.to_string(index=False))
    print("\ndense sweep")
    print(dense.to_string(index=False))

    stable = dense.loc[dense.canonical_peaks_recovered == 4, "multiple"]
    if len(stable):
        print("\nall four canonical peaks recovered at %.2f--%.2fx Scott"
              % (stable.min(), stable.max()))
    else:
        print("\nno covariance factor in the dense sweep recovers all four canonical peaks")

    os.makedirs(os.path.dirname(args.sweep), exist_ok=True)
    dense.to_csv(args.sweep, index=False)
    draw(pc, factor_default, args.figure)

    with open(args.provenance, "w") as handle:
        json.dump(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "assignment_convention": "frozen Fig 3 unweighted PCA/KDE",
                "params_csv": os.path.abspath(args.params),
                "params_sha256": file_digest(args.params),
                "assignments_csv": os.path.abspath(args.assignments),
                "assignments_sha256": file_digest(args.assignments),
                "n_particles": int(len(pc)),
                "scott_covariance_factor": factor_default,
                "detector": {
                    "grid": GRID,
                    "maximum_filter": WINDOW,
                    "relative_density_threshold": RELATIVE_DENSITY_THRESHOLD,
                    "weights": "none (unweighted canonical Fig 3 convention)",
                },
                "panel_multiples": PANEL_MULTIPLES,
                "sweep_multiples": SWEEP_MULTIPLES,
                "canonical_peak_centres": [list(map(float, p)) for p in PAPER],
                "canonical_peak_members": PAPER_N,
                "sweep_csv": os.path.abspath(args.sweep),
                "sweep_sha256": file_digest(args.sweep),
                "figure_png": os.path.abspath(args.figure),
                "figure_png_sha256": file_digest(args.figure),
                "figure_pdf": os.path.abspath(os.path.splitext(args.figure)[0] + ".pdf"),
                "figure_pdf_sha256": file_digest(os.path.splitext(args.figure)[0] + ".pdf"),
                "script": os.path.abspath(__file__),
                "script_sha256": file_digest(__file__),
                "command": "python analysis_scripts/bandwidth_stability_peaks.py",
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print("\nwrote %s, %s, %s" % (args.figure, args.sweep, args.provenance))


if __name__ == "__main__":
    main()
