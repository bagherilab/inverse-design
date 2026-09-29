"""Per-generation MAE, ESS and prior-box violations for every finished chain.

The MAE definition is lifted verbatim from `convergence.py` (which in turn came
from `analysis_scripts/conv.py`, the script that produced the N=512 numbers in
RESPONSES_0803.md): drop each target column's IQR outliers, take the median,
express the gap to the target as a percentage of |target|, then average over the
five fitted targets. Nothing here is a new metric -- if this script disagrees
with convergence.py on a generation both have seen, this script is wrong.

Usage:
    sweep_report.py            # every run, every generation present
    sweep_report.py <run_dir> <gen>   # one cell, for dry-running
"""
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = "/gscratch/cheme/chiu/ARCADE_OUTPUT"
CFG = "/gscratch/cheme/chiu/pm50/parameter_config.py"

FAMILY = {"": "main", "_rep": "rep1", "_rep2": "rep2"}
SEED = {"main": 42, "rep1": 20260806, "rep2": 20260807}

_ns = {}
exec(open(CFG).read(), _ns)  # noqa: S102
RANGES = _ns["PARAM_RANGES_MU_ONLY"]


def iqrf(s):
    q1, q3 = s.quantile([0.25, 0.75])
    i = q3 - q1
    return s[(s >= q1 - 1.5 * i) & (s <= q3 + 1.5 * i)]


def mae(fm, tar):
    d = fm.replace([np.inf, -np.inf], np.nan)
    return float(
        np.mean([abs((iqrf(d[m].dropna()).median() - tar[m]) / abs(tar[m]) * 100) for m in tar])
    )


def outside_box(par):
    """Particles with at least one parameter outside the configured prior box."""
    cols = [p for p in RANGES if p in par.columns and RANGES[p][0] != RANGES[p][1]]
    if not cols:
        return None, 0
    v = par[cols].apply(pd.to_numeric, errors="coerce")
    bad = pd.DataFrame(
        {c: (v[c] < RANGES[c][0]) | (v[c] > RANGES[c][1]) for c in cols}
    )
    return int(bad.any(axis=1).sum()), len(cols)


def cell(run_dir, g):
    d = f"{run_dir}/iter_{g}"
    tar = json.load(open(f"{run_dir}/targets.json"))
    fm = pd.read_csv(f"{d}/final_metrics.csv")
    par = pd.read_csv(f"{d}/all_param_df.csv")
    out, ncols = outside_box(par)
    ess = None
    if os.path.exists(f"{d}/ess.json"):
        ess = json.load(open(f"{d}/ess.json"))
    return {
        "rows": len(fm),
        "mae": mae(fm, tar),
        "outside": out,
        "cols_checked": ncols,
        "ess": None if ess is None else ess["ess"],
        "ess_fraction": None if ess is None else ess["ess_fraction"],
        "n_particles": None if ess is None else ess["n_particles"],
        "max_weight": None if ess is None else ess["max_weight"],
    }


if len(sys.argv) == 3:
    r = cell(sys.argv[1].rstrip("/"), int(sys.argv[2]))
    for k, v in r.items():
        print(f"{k:>14s}  {v}")
    raise SystemExit

rows = []
for N in (128, 256, 512, 1024):
    for suf, fam in FAMILY.items():
        run = f"{OUT}/ABC_SMC_RF_N{N}_pm50{suf}"
        if not os.path.isdir(run):
            continue
        for g in range(10):
            if not os.path.exists(f"{run}/iter_{g}/final_metrics.csv"):
                continue
            try:
                r = cell(run, g)
            except Exception as e:  # a generation mid-write
                print(f"skip N={N} {fam} g{g}: {e}", file=sys.stderr)
                continue
            r.update(N=N, family=fam, seed=SEED[fam], gen=g, run=os.path.basename(run))
            rows.append(r)

df = pd.DataFrame(rows)[
    ["N", "family", "seed", "gen", "run", "rows", "n_particles", "ess",
     "ess_fraction", "max_weight", "mae", "outside", "cols_checked"]
]
dst = "/gscratch/cheme/chiu/sweep_report.csv"
df.to_csv(dst, index=False)
print(f"wrote {dst}: {len(df)} generation-rows")
print()
print("MAE by generation, one column per family (N rows x gen)")
for N in sorted(df.N.unique()):
    sub = df[df.N == N]
    print(f"\n  N={N}")
    piv = sub.pivot_table(index="gen", columns="family", values="mae")
    print(piv.to_string(float_format=lambda x: f"{x:7.2f}"))
print()
tot = int(df.outside.fillna(0).sum())
print(f"prior-box violations across every generation of every run: {tot}")
