"""Per-region target metric errors for the published SIR generation 4 fit.

Assigns every generation 4 particle to its nearest detected PI--ISF density
maximum (standardized coordinates, unweighted assignment) and reports the
region-median target metric error.  Supports the scoped equifinality statement
in Results and the S4 Fig caption.

Inputs
    SIR_OUTPUT/n3_t365_l30/n512/iter_4/all_param_df.csv
    SIR_OUTPUT/n3_t365_l30/n512/iter_4/final_metrics.csv
    SIR_OUTPUT/n3_t365_l30/target_values.csv
    analysis_outputs/reviewer_closeout/sir_pi_isf_kde_peaks.csv  (published rows)

Output
    analysis_outputs/reviewer_closeout/sir_mode_target_errors.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CAMPAIGN = Path.home() / "UW/bagherilab/SIR_OUTPUT/n3_t365_l30"
PEAKS = Path("analysis_outputs/reviewer_closeout/sir_pi_isf_kde_peaks.csv")
OUT = Path("analysis_outputs/reviewer_closeout/sir_mode_target_errors.csv")


def main(campaign: Path, peaks: Path, out: Path) -> None:
    fit = campaign / "n512/iter_4"
    params = pd.read_csv(fit / "params.csv")
    metrics = pd.read_csv(fit / "statistics.csv")
    target = pd.read_csv(campaign / "target_values.csv").iloc[0]
    centroids = pd.read_csv(peaks)
    centroids = centroids[centroids.group == "published"][["PI", "ISF"]].to_numpy(float)

    coords = params[["PI", "ISF"]].to_numpy(float)
    mu, sd = coords.mean(0), coords.std(0)
    z, c = (coords - mu) / sd, (centroids - mu) / sd
    label = ((z[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)

    columns = list(metrics.columns)
    rows = []
    for k in range(len(centroids)):
        region = metrics[label == k]
        median = region.median()
        error = 100 * np.abs(median - target[columns]) / np.abs(target[columns])
        rows.append(
            {"mode": k + 1, "n": len(region), "mae_pct": error.mean()}
            | {f"median_{c}": median[c] for c in columns}
            | {f"err_pct_{c}": error[c] for c in columns}
        )
    table = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table[["mode", "n", "mae_pct"] + [f"err_pct_{c}" for c in columns]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    ap.add_argument("--peaks", type=Path, default=PEAKS)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    main(a.campaign, a.peaks, a.out)
