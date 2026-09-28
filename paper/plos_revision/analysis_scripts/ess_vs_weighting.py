"""Is the ESS ceiling set by the particle count or by the forest geometry?

Measured 2026-08-07: ESS at generation 4 is 59.7 / 75.8 / 70.9 / 105.0 / 97.5 for
N = 128 / 256 / 512 / 1024 / 5000. From N=1024 to N=5000 the particle count rises
4.9x and the ESS *falls*. Whatever bounds the effective sample size, it is not the
number of particles -- so the 200,000 simulations that N=5000 cost bought no
additional effective sample at all.

`sa_ess.sbatch` already named the suspect: "the weighted support is bounded by
n_trees x leaf size rather than by the particle count". A DRF leaf holds at least
`min_samples_leaf` particles and there is no depth limit, so leaves stay small
however many particles exist, and only the particles sharing a leaf with the target
get weight. This script tests that directly by refitting the weighting on the
*same* particles with different forest geometry and watching the ESS move.

The prediction being tested is specific: ESS should track `n_trees x min_samples_leaf`
and be flat in N. If instead ESS is flat in the forest grid too, the ceiling is
something else and the diagnosis is wrong.

Nothing is re-simulated. Every cell reuses the stored particles and metrics; only
the weighting is recomputed, which is why this costs nothing and needs no cluster.

**These weights are a refit, not the run's own.** Generation g+1's particles were
drawn using the original weights, so a refit at generation g does not propagate --
it answers "what ESS would this weighting have given on this cloud", not "what would
the chain have done". That is the question here.

Usage:
    ess_vs_weighting.py [--gen 4] [--trees 50,150,250] [--leaf 5,15,25] [--csv OUT]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.environ.get("DED_INVERSE_DESIGN", "/home/pohaoc2/UW/bagherilab/inverse_design"), "src"))
from inverse_design.rf.drf import DRF  # noqa: E402

ROOT = os.environ.get(
    "DED_PM50_ROOT",
    "/tmp/claude-1000/-home-pohaoc2-UW-bagherilab-2025-InverseDesign-PHC-Copy-/"
    "4903db57-e4a5-4e59-b9ea-d1b9bae1b1a3/scratchpad/pm50",
)

# All five entries of targets.json. The DRF is fitted on every one of them, not on
# the three the MI analysis uses: `SIMP_METRICS` in combine_simplification_mi_table
# is doub_time/symmetry/colony_growth, but the weighting also sees doub_time_std and
# symmetry_std. Fitting on three understated ESS by 2.4x at N=1024 (43.5 against a
# stored 105.0); on five it recovers 91.8, inside forest-to-forest variation.
METRICS = ["doub_time", "symmetry", "colony_growth", "doub_time_std", "symmetry_std"]
PARAMS = ["CELL_VOLUME_MU", "NECROTIC_FRACTION", "ACCURACY", "COMPRESSION_TOLERANCE",
          "SYNTHESIS_DURATION_MU", "BASAL_ENERGY_MU", "PROLIFERATION_ENERGY_MU",
          "MIGRATION_ENERGY_MU", "METABOLIC_PREFERENCE_MU", "CONVERSION_FRACTION_MU",
          "RATIO_GLUCOSE_PYRUVATE_MU", "LACTATE_RATE_MU", "AUTOPHAGY_RATE_MU",
          "GLUCOSE_UPTAKE_RATE_MU", "ATP_PRODUCTION_RATE_MU", "MIGRATORY_THRESHOLD_MU",
          "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION", "CAPILLARY_DENSITY"]

PUBLISHED = (50, 5)     # the published forest: n_trees=50, min_samples_leaf=5

# The forest is randomised, so a refit cannot reproduce the chain's stored weights bit
# for bit -- the same point sir_ess_replay.py makes. Several forests are therefore fitted
# per cell and the spread reported; the validation gate is that the stored ESS falls
# inside that spread, not that it equals any single refit.
SEEDS = [0, 1, 2, 3, 4]


def load(run, gen):
    d = f"{ROOT}/{run}/iter_{gen}"
    par = pd.read_csv(f"{d}/all_param_df.csv")
    met = pd.read_csv(f"{d}/final_metrics.csv")
    if len(met) != len(par) or "input_folder" in met.columns:
        key = "input_folder" if "input_folder" in met.columns else met.columns[0]
        met = met.set_index(met[key].astype(str)).reindex(
            par["input_folder"].astype(str)).reset_index(drop=True)
    tgt = json.load(open(f"{ROOT}/{run}/targets.json"))
    return par, met, np.array([tgt[m] for m in METRICS], float)


def stored_ess(run, gen):
    """The chain's own ESS, for comparison against every refit."""
    d = f"{ROOT}/{run}/iter_{gen}"
    par = pd.read_csv(f"{d}/all_param_df.csv")
    idx = par["input_folder"].astype(str).str.extract(r"(\d+)").iloc[:, 0].astype(int)
    w = idx.map(pd.read_csv(f"{d}/weights.csv").set_index("input_index")["weight"])
    w = w.fillna(0.0).to_numpy(float)
    w = w / w.sum()
    return 1.0 / np.sum(w ** 2), int((w > 0).sum())


def refit_ess(par, met, tgt, n_trees, leaf):
    """Refit the DRF weighting on one cloud.

    The StandardScaler is not optional. `abc_smc_rf_arcade.py:446` fits one per
    generation on that generation's statistics and transforms the target with it, so
    the forest sees standardised statistics. Skipping it lets `doub_time` (~45) and
    `colony_growth` (~18) dominate every split while `symmetry` (~0.8) is effectively
    ignored, which understated ESS by 2.4x at N=1024 in testing.
    """
    from sklearn.preprocessing import StandardScaler

    ok = met[METRICS].notna().all(1).to_numpy()
    X = par.loc[ok, PARAMS].to_numpy(float)
    S = met.loc[ok, METRICS].to_numpy(float)
    scaler = StandardScaler().fit(S)
    S = scaler.transform(S)
    tgt = scaler.transform(np.asarray(tgt, float).reshape(1, -1)).ravel()

    ess, sup = [], []
    for seed in SEEDS:
        model = DRF(n_trees=n_trees, min_samples_leaf=leaf, random_state=seed)
        model.fit(X, S)
        w = np.asarray(model.predict_weights(tgt), float)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        ess.append(1.0 / np.sum(w ** 2))
        sup.append(int((w > 0).sum()))
    if not ess:
        return float("nan"), float("nan"), 0
    return float(np.mean(ess)), float(np.std(ess)), int(np.mean(sup))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=4)
    ap.add_argument("--trees", default="50,150,250")
    ap.add_argument("--leaf", default="5,15,25")
    ap.add_argument("--runs", default="ABC_SMC_RF_N128_pm50,ABC_SMC_RF_N256_pm50,"
                                      "ABC_SMC_RF_N512_pm50,ABC_SMC_RF_N1024_pm50,"
                                      "ABC_SMC_RF_N5000_pm50")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    trees = [int(t) for t in a.trees.split(",")]
    leaves = [int(x) for x in a.leaf.split(",")]

    rows = []
    for run in a.runs.split(","):
        try:
            par, met, tgt = load(run, a.gen)
        except FileNotFoundError:
            print(f"!! missing {run} iter_{a.gen}")
            continue
        se, ss = stored_ess(run, a.gen)
        n = len(par)
        print(f"\n{'=' * 66}\n{run}  iter_{a.gen}  n={n}"
              f"   stored ESS={se:.1f} ({100 * se / n:.2f}% of N), support={ss}\n{'=' * 66}")
        print(f"{'trees':>6s} {'leaf':>5s} {'t*leaf':>7s} {'ESS (mean+-sd)':>18s} "
              f"{'ESS/N':>7s} {'support':>8s}")
        for nt, lf in itertools.product(trees, leaves):
            e, sd, s = refit_ess(par, met, tgt, nt, lf)
            tag = ""
            if (nt, lf) == PUBLISHED:
                inside = abs(se - e) <= 2 * sd if sd > 0 else False
                tag = f"  <- published, stored {se:.1f} {'INSIDE' if inside else 'OUTSIDE'} +-2sd"
            print(f"{nt:6d} {lf:5d} {nt * lf:7d} {e:10.1f} +- {sd:-4.1f} "
                  f"{100 * e / n:6.2f}% {s:8d}{tag}")
            rows.append(dict(run=run, N=n, gen=a.gen, n_trees=nt, leaf=lf,
                             trees_x_leaf=nt * lf, ess=round(e, 2), ess_sd=round(sd, 2),
                             ess_frac=round(e / n, 5), support=s,
                             stored_ess=round(se, 2)))

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n{'=' * 66}\nESS averaged over N, by forest geometry\n{'=' * 66}")
        piv = df.pivot_table(index="n_trees", columns="leaf", values="ess", aggfunc="mean")
        print(piv.round(1).to_string())
        print("\nESS averaged over forest geometry, by N")
        print(df.groupby("N")["ess"].mean().round(1).to_string())
        print("\nspread: max/min ESS across the forest grid vs across N")
        print(f"  forest grid (mean over N): {piv.values.max() / piv.values.min():.2f}x")
        byn = df.groupby("N")["ess"].mean()
        print(f"  particle count (mean over grid): {byn.max() / byn.min():.2f}x")
    if a.csv and rows:
        df.to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
