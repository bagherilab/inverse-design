"""Step 15 of Dinh et al. Algorithm 6, replayed on the corrected run of record.

`analysis_scripts/smc_weight_replay.py` answered this question on the published
N1024 run, whose ESS column feeds
`body/extended_data/ed_table_reweighting_ess.tex`. That cloud carries the
seed-pairing defect and no Step 8, so every ESS in the table is wrong. This
script is the same computation against `ABC_SMC_RF_N1024_pm50`, which enforces
the prior at the +/-50% bounds and has the corrected pairing.

Three differences from the published replay, all forced by the run:

  * T is 10, not 5, and the schedule is not the published formula. The pm50
    chains were run with an explicit `DED_KERNEL_SCHEDULE`, logged per
    generation as
        0.08 0.06 0.04 0.02 0.018 0.016 0.014 0.012 0.01
    which reproduces 0.1*(1 - t/5) over its original four steps and then tapers
    linearly rather than collapsing to the 0.01 floor. Replaying with the
    published formula would use a kernel the sampler never used.

  * the `ratio+pi` variant is dropped. Step 8 is enforced here -- sweep_report
    finds zero out-of-box particles in any generation of any chain -- so pi is
    constant over the support and `ratio+pi` is `ratio` by construction. It is
    still asserted, not assumed: the script fails loudly if any particle is out
    of the box.

  * generations run 0-9, so the regenerated table has ten rows where the
    published one had five.

What this cannot say, unchanged from the published replay: generation t+1's
particles were drawn using the uncorrected weights of generation t, so this is a
re-weighting of one particle cloud, not a re-run. It bounds how far the missing
importance ratio moved the posterior given these particles.

Usage:
    smc_weight_replay_pm50.py [run_dir] [--tex OUT.tex] [--csv OUT.csv]

Set DED_PM50_CONFIG to the run's parameter_config.py when it is not at
$DED_HPC_ROOT/pm50/parameter_config.py (a copy is archived as
`analysis_outputs/reviewer_closeout/pm50_parameter_config.py`).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("DED_HPC_ROOT", "/gscratch/cheme/chiu")
CFG = os.environ.get("DED_PM50_CONFIG", f"{ROOT}/pm50/parameter_config.py")
DEFAULT_RUN = f"{ROOT}/ARCADE_OUTPUT/ABC_SMC_RF_N1024_pm50"

# Read off the DED_KERNEL_SCHEDULE lines in logs/prep_corrected_*.err. Keyed by
# the iteration that was *produced* with it, which is how the driver logs it.
SCHEDULE = {1: 0.08, 2: 0.06, 3: 0.04, 4: 0.02,
            5: 0.018, 6: 0.016, 7: 0.014, 8: 0.012, 9: 0.01}

_ns = {}
exec(open(CFG).read(), _ns)  # noqa: S102
RANGES = dict(_ns["PARAM_RANGES_MU_ONLY"])
RANGES.update(_ns["SOURCE_PARAM_RANGES"])

# The dimensions the Gaussian kernel perturbs: every free MU parameter plus the
# two continuous source parameters. X/Y_SPACING move under a discrete kernel and
# CAPILLARY_DENSITY is a deterministic function of the spacing pair, so neither
# is an independent Gaussian dimension.
GAUSS = [p for p, (lo, hi) in _ns["PARAM_RANGES_MU_ONLY"].items()
         if lo != hi and not p.endswith("_SIGMA")]
GAUSS += ["GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]

# The box Step 8 actually enforces: every dimension the sampler draws. X/Y_SPACING
# are discrete and bounded by construction. CAPILLARY_DENSITY and
# DISTANCE_TO_CENTER are excluded because they are not drawn -- density is a
# deterministic function of the spacing pair, and its nominal (25, 6571) floor is
# not a bound the sampler can violate: the lowest reachable pair evaluates to
# 24.84, 0.6% under, in every generation including the Sobol draw. Including it
# would report 13 "violations" in a prior draw that cannot violate the prior.
BOX = [p for p, (lo, hi) in RANGES.items()
       if lo != hi and p not in ("X_SPACING", "Y_SPACING",
                                 "CAPILLARY_DENSITY", "DISTANCE_TO_CENTER")]


def load(run, g):
    par = pd.read_csv(f"{run}/iter_{g}/all_param_df.csv")
    idx = par["input_folder"].astype(str).str.extract(r"(\d+)").iloc[:, 0].astype(int)
    wdf = pd.read_csv(f"{run}/iter_{g}/weights.csv").set_index("input_index")["weight"]
    w = idx.map(wdf).fillna(0.0).to_numpy(float)
    return par, w / w.sum()


def spacing(par):
    return np.column_stack([
        par["X_SPACING"].astype(str).str.split(":").str[-1].astype(float),
        par["Y_SPACING"].astype(str).str.split(":").str[-1].astype(float)])


def log_denominator(cur, prev, wprev, sf, with_spacing=True):
    """log sum_k w_{t-1}^(k) K_t(theta_t^(i) | theta_{t-1}^(k)) per current particle."""
    A = cur[GAUSS].to_numpy(float)
    B = prev[GAUSS].to_numpy(float)
    logk = np.zeros((len(A), len(B)))
    for j, p in enumerate(GAUSS):
        lo, hi = RANGES[p]
        sd = (hi - lo) * sf
        d = (A[:, j][:, None] - B[None, :, j]) / sd
        logk -= 0.5 * d ** 2 + np.log(sd * np.sqrt(2 * np.pi))
    if with_spacing:
        sa, sb = spacing(cur), spacing(prev)
        ok = ((np.abs(sa[:, 0][:, None] - sb[None, :, 0]) <= 4)
              & (np.abs(sa[:, 1][:, None] - sb[None, :, 1]) <= 4))
        logk += np.where(ok, -np.log(81.0), -np.inf)
    m = logk.max(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.sum(wprev[None, :] * np.exp(logk - m), axis=1)) + m[:, 0]


def norm_from_log(logw):
    finite = logw[np.isfinite(logw)]
    if len(finite) == 0:
        return np.zeros(len(logw))
    w = np.where(np.isfinite(logw), np.exp(logw - finite.max()), 0.0)
    return w / w.sum()


def ess(w):
    s = np.sum(w ** 2)
    return float(1.0 / s) if s > 0 else 0.0


def out_of_box(par):
    bad = np.zeros(len(par), bool)
    for p in BOX:
        if p not in par.columns:
            continue
        lo, hi = RANGES[p]
        v = pd.to_numeric(par[p], errors="coerce").to_numpy(float)
        bad |= (v < lo) | (v > hi)
    return int(bad.sum())


def main():
    argv = sys.argv[1:]
    tex = None
    if "--tex" in argv:
        i = argv.index("--tex")
        tex = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    csv_out = None
    if "--csv" in argv:
        i = argv.index("--csv")
        csv_out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    run = (argv[0] if argv else DEFAULT_RUN).rstrip("/")

    gens = sorted(int(d.split("_")[1]) for d in os.listdir(run)
                  if d.startswith("iter_") and os.path.exists(f"{run}/{d}/weights.csv"))
    print(f"run: {run}")
    print(f"generations: {gens[0]}-{gens[-1]}   Gaussian kernel dimensions: {len(GAUSS)}")
    print("Generation 0 is a Sobol draw from the prior, so the proposal IS the prior")
    print("and Algorithm 6 sets w* = w: no correction applies and no ratio is shown.\n")

    data = {g: load(run, g) for g in gens}

    viol = {g: out_of_box(data[g][0]) for g in gens}
    if any(viol.values()):
        print(f"REFUSING: Step 8 is not enforced in this run: {viol}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Step 8 check: 0 out-of-box particles in all {len(gens)} generations, so pi is")
    print("constant over the support and the ratio+pi variant is the ratio variant.\n")

    hdr = f"{'gen':>5s} {'n':>6s} {'support':>8s} {'ESS forest':>11s} {'ESS reweighted':>15s} {'ratio':>8s} {'kernel sd':>10s}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for g in gens:
        par, w_pub = data[g]
        e_pub = ess(w_pub)
        if g == 0:
            print(f"{g:>5d} {len(par):>6d} {int((w_pub > 0).sum()):>8d} "
                  f"{e_pub:>11.1f} {'---':>15s} {'---':>8s} {'---':>10s}")
            rows.append((g, e_pub, None, None))
            continue
        prev, w_prev = data[g - 1]
        sf = SCHEDULE[g]
        ld = log_denominator(par, prev, w_prev, sf)
        with np.errstate(divide="ignore"):
            w_ratio = norm_from_log(np.log(w_pub) - ld)
        e_r = ess(w_ratio)
        print(f"{g:>5d} {len(par):>6d} {int((w_pub > 0).sum()):>8d} "
              f"{e_pub:>11.1f} {e_r:>15.1f} {e_pub / e_r:>7.0f}x {100 * sf:>9.1f}%")
        rows.append((g, e_pub, e_r, e_pub / e_r))

    if csv_out:
        with open(csv_out, "w") as fh:
            fh.write("generation,ess_forest,ess_reweighted,ratio,kernel_sd_fraction\n")
            for g, e_pub, e_r, rat in rows:
                cells = [f"{e_pub:.6f}", "", "", ""] if e_r is None else [
                    f"{e_pub:.6f}", f"{e_r:.6f}", f"{rat:.6f}", f"{SCHEDULE[g]:g}"]
                fh.write(",".join([str(g)] + cells) + "\n")
        print(f"wrote {csv_out}")

    if tex:
        with open(tex, "w") as fh:
            fh.write("% ED Table - effect of the ABC-SMC-(D)RF importance-reweighting step on ESS\n")
            fh.write(f"% Source: {os.path.basename(run)}, the corrected run of record\n")
            fh.write("% (prior enforced at +/-50% of the model defaults, corrected seed pairing).\n")
            fh.write("% Reweighting replayed on the same particles with the run's own kernel\n")
            fh.write("% schedule (DED_KERNEL_SCHEDULE), not the published formula.\n")
            fh.write("% Generation 0 is a Sobol draw from the prior, so the proposal IS the prior\n")
            fh.write("% and no correction applies.\n")
            fh.write("\\begin{tabular}{lrrr}\n\\hline\n")
            fh.write("\\textbf{Generation} & \\textbf{ESS, forest weights} & "
                     "\\textbf{ESS, with reweighting} & \\textbf{Ratio} \\\\\n\\hline\n")
            for g, e_pub, e_r, rat in rows:
                if e_r is None:
                    fh.write(f"{g} & {e_pub:.1f} & --- & --- \\\\\n")
                else:
                    fh.write(f"{g} & {e_pub:.1f} & {e_r:.1f} & ${rat:.0f}\\times$ \\\\\n")
            fh.write("\\hline\n\\end{tabular}\n")
        print(f"\nwrote {tex}")


if __name__ == "__main__":
    main()
