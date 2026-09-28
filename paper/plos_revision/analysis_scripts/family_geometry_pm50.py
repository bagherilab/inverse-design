"""Is N=1024 justified by posterior geometry, and is the fourth peak seed-stable?

MAE stops separating the four particle counts long before generation 9
(sweep_report.csv: every N sits at 11-14% by g9, and the between-seed spread at
N=128 is wider than the gap between N=128 and N=1024). So the case for N=1024
cannot be made on MAE. This script measures what MAE cannot see: where the
posterior sits, how wide it is, and how many maxima it has -- and how much all
three move when only the seed changes.

Method. One basis, fitted once, unweighted, on the N=1024 main-family generation
9 cloud, and every other cloud is projected into it. Re-fitting PCA per cloud
gives a rotated basis with arbitrary component signs, so coordinates from two
fits are not comparable; this is the same constraint recorded in
compare_prioronly.pca_basis and peaks_vs_leaf.

Three quantities per cloud, all under the run's own DRF weights:

  centroid    weighted mean in PC1/PC2. Between-seed scatter at fixed N is the
              null: any N-dependence smaller than that is not evidence.
  trace cov   weighted total variance in the 2-D basis -- posterior width, the
              quantity a larger particle count is supposed to pin down.
  maxima      weighted-KDE peak count, bandwidth pinned at the published 0.3150
              for the reason in peaks_vs_leaf: scipy derives Scott's factor from
              neff, so a higher-ESS weighting would narrow the bandwidth and
              manufacture peaks on its own.

What this cannot say: three seeds bound the between-draw scatter, they do not
estimate it precisely. A centroid shift inside the seed scatter is unresolved,
not absent.

Usage:
    family_geometry_pm50.py [--gen 9] [--csv OUT.csv]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.stats import gaussian_kde

ROOT = os.environ.get("DED_HPC_ROOT", "/gscratch/cheme/chiu")
OUT = f"{ROOT}/ARCADE_OUTPUT"
CFG = f"{ROOT}/pm50/parameter_config.py"
PUBLISHED_BW = 0.3150

FAMILY = {"": "main", "_rep": "rep1", "_rep2": "rep2"}
SEED = {"main": 42, "rep1": 20260806, "rep2": 20260807}
NS = (128, 256, 512, 1024)
REFERENCE = ("1024", "")  # the basis is fitted here

# The 19 reported parameters, in the column order peaks_vs_leaf uses. The order
# matters: PC signs are fixed from the largest-magnitude loading.
PARAMS = ["CELL_VOLUME_MU", "NECROTIC_FRACTION", "ACCURACY", "COMPRESSION_TOLERANCE",
          "SYNTHESIS_DURATION_MU", "BASAL_ENERGY_MU", "PROLIFERATION_ENERGY_MU",
          "MIGRATION_ENERGY_MU", "METABOLIC_PREFERENCE_MU", "CONVERSION_FRACTION_MU",
          "RATIO_GLUCOSE_PYRUVATE_MU", "LACTATE_RATE_MU", "AUTOPHAGY_RATE_MU",
          "GLUCOSE_UPTAKE_RATE_MU", "ATP_PRODUCTION_RATE_MU", "MIGRATORY_THRESHOLD_MU",
          "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION", "CAPILLARY_DENSITY"]


def fixed_basis(X):
    """Scaler + 2-component PCA fitted on one cloud, as a projector for any cloud."""
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    Vt = Vt[:2].copy()
    for i in range(2):
        j = np.argmax(np.abs(Vt[i]))
        if Vt[i, j] < 0:
            Vt[i] *= -1
    centre = Z.mean(0)
    var = (S[:2] ** 2) / (S ** 2).sum()

    def project(Y):
        return ((Y - mu) / sd - centre) @ Vt.T

    return project, var


def detect(pc, weights, bw=PUBLISHED_BW, grid=200, window=5, thresh=0.05):
    kde = gaussian_kde(pc.T, bw_method=bw, weights=weights)
    xs = np.linspace(pc[:, 0].min(), pc[:, 0].max(), grid)
    ys = np.linspace(pc[:, 1].min(), pc[:, 1].max(), grid)
    XX, YY = np.meshgrid(xs, ys)
    dens = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(grid, grid)
    hit = (dens == maximum_filter(dens, size=(window, window))) & (dens > dens.max() * thresh)
    yi, xi = np.nonzero(hit)
    return np.column_stack([xs[xi], ys[yi]])


def load(run, g):
    par = pd.read_csv(f"{run}/iter_{g}/all_param_df.csv")
    idx = par["input_folder"].astype(str).str.extract(r"(\d+)").iloc[:, 0].astype(int)
    wdf = pd.read_csv(f"{run}/iter_{g}/weights.csv").set_index("input_index")["weight"]
    w = idx.map(wdf).fillna(0.0).to_numpy(float)
    return par[PARAMS].to_numpy(float), w / w.sum()


def main():
    argv = sys.argv[1:]
    gen = 9
    if "--gen" in argv:
        gen = int(argv[argv.index("--gen") + 1])
    csv_out = argv[argv.index("--csv") + 1] if "--csv" in argv else None

    ref = f"{OUT}/ABC_SMC_RF_N{REFERENCE[0]}_pm50{REFERENCE[1]}"
    Xr, _ = load(ref, gen)
    project, var = fixed_basis(Xr)
    print(f"basis: {os.path.basename(ref)} iter_{gen}, unweighted, "
          f"PC1+PC2 = {100 * var.sum():.1f}% of variance ({100 * var[0]:.1f} / {100 * var[1]:.1f})")
    print(f"every cloud below is projected into that basis; bandwidth pinned at {PUBLISHED_BW}\n")

    hdr = (f"{'N':>6s} {'family':>7s} {'seed':>9s} {'ESS':>7s} {'PC1':>8s} {'PC2':>8s} "
           f"{'trace cov':>10s} {'maxima':>7s}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for N in NS:
        for suf, fam in FAMILY.items():
            run = f"{OUT}/ABC_SMC_RF_N{N}_pm50{suf}"
            if not os.path.exists(f"{run}/iter_{gen}/weights.csv"):
                continue
            X, w = load(run, gen)
            pc = project(X)
            c = w @ pc
            d = pc - c
            trace = float((w[:, None] * d ** 2).sum())
            ess = float(1.0 / np.sum(w ** 2))
            cent = detect(pc, w)
            print(f"{N:>6d} {fam:>7s} {SEED[fam]:>9d} {ess:>7.1f} "
                  f"{c[0]:>8.3f} {c[1]:>8.3f} {trace:>10.3f} {len(cent):>7d}")
            rows.append(dict(N=N, family=fam, seed=SEED[fam], gen=gen, ess=ess,
                             pc1=c[0], pc2=c[1], trace_cov=trace, maxima=len(cent)))
        print()

    df = pd.DataFrame(rows)
    print("=" * 72)
    print("between-seed scatter at fixed N (the null), against the N-to-N change")
    print("=" * 72)
    print(f"{'N':>6s} {'centroid sd':>12s} {'trace cov mean':>15s} {'trace cov sd':>13s} {'maxima':>10s}")
    for N in sorted(df.N.unique()):
        s = df[df.N == N]
        spread = float(np.hypot(s.pc1.std(ddof=1), s.pc2.std(ddof=1)))
        print(f"{N:>6d} {spread:>12.3f} {s.trace_cov.mean():>15.3f} "
              f"{s.trace_cov.std(ddof=1):>13.3f} {sorted(s.maxima.tolist())!s:>10s}")

    ref_row = df[(df.N == 1024)]
    print(f"\ncentroid distance from the N=1024 same-family cloud:")
    for fam in ("main", "rep1", "rep2"):
        base = ref_row[ref_row.family == fam]
        if base.empty:
            continue
        b = base.iloc[0]
        line = [f"  {fam:>5s}:"]
        for N in NS[:-1]:
            r = df[(df.N == N) & (df.family == fam)]
            if r.empty:
                continue
            r = r.iloc[0]
            line.append(f"N={N} {np.hypot(r.pc1 - b.pc1, r.pc2 - b.pc2):.3f}")
        print("  ".join(line))

    if csv_out:
        df.to_csv(csv_out, index=False)
        print(f"\nwrote {csv_out}: {len(df)} clouds")


if __name__ == "__main__":
    main()
