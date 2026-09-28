"""The bias axis of the forest-hyperparameter grid.

ESS on its own is not a quantity to maximise: in ABC-RF the forest replaces the
tolerance-based accept/reject, so `min_samples_leaf` plays the role of the ABC
tolerance and a larger leaf buys ESS by admitting particles further from the
observation. This reports what that costs, in the paper's own units.

For each (n_trees, min_samples_leaf) and each generation we compute, over the
target summary statistics:

  weighted MAE%   = mean_j sum_i w_i |s_ij - t_j| / |t_j|
  prior MAE%      = mean_j mean_i |s_ij - t_j| / |t_j|      (weights ignored)
  predictive err% = mean_j |sum_i w_i s_ij - t_j| / |t_j|

The prior MAE is identical across the grid -- the simulations are the same, only
the weights differ -- so it is the line the weighted MAE approaches as the leaf
grows. Distance below it is what the weighting is buying.
"""

import csv
import glob
import json
import os
import re

import numpy as np


def load_generation(run_dir, gen):
    """Return (weights, statistics, targets) aligned on particle index."""
    gen_dir = os.path.join(run_dir, "iter_%d" % gen)
    wf = os.path.join(gen_dir, "weights.csv")
    mf = os.path.join(gen_dir, "final_metrics.csv")
    tf = os.path.join(run_dir, "targets.json")
    if not all(os.path.exists(p) for p in (wf, mf, tf)):
        return None

    targets = json.load(open(tf))
    with open(mf) as fh:
        metrics = {r["input_folder"]: r for r in csv.DictReader(fh)}
    with open(wf) as fh:
        weight_rows = list(csv.DictReader(fh))

    weights, stats = [], []
    for row in weight_rows:
        key = "input_%s" % row["input_index"]
        if key not in metrics:
            continue
        try:
            values = [float(metrics[key][name]) for name in targets]
        except (KeyError, ValueError):
            continue
        weights.append(float(row["weight"]))
        stats.append(values)

    if not weights:
        return None
    return (
        np.array(weights),
        np.array(stats),
        np.array([targets[name] for name in targets]),
    )


def errors(weights, stats, targets):
    """Weighted MAE%, prior MAE% and predictive-mean error%, averaged over stats."""
    total = weights.sum()
    if total <= 0:
        return None
    w = weights / total
    rel = np.abs(stats - targets) / np.abs(targets)
    weighted_mae = float(np.mean(w @ rel))
    prior_mae = float(np.mean(np.mean(rel, axis=0)))
    predictive = float(np.mean(np.abs(w @ stats - targets) / np.abs(targets)))
    return weighted_mae, prior_mae, predictive


def main():
    combos = {}
    for run_dir in sorted(glob.glob(os.path.join(os.environ.get("DED_HPC_ROOT", "/gscratch/cheme/chiu"), "sa_ess/runs/*"))):
        m = re.search(r"_t(\d+)_l(\d+)$", run_dir)
        if not m:
            continue
        trees, leaf = int(m.group(1)), int(m.group(2))
        for gen in range(5):
            loaded = load_generation(run_dir, gen)
            if loaded is None:
                continue
            measured = errors(*loaded)
            if measured is None:
                continue
            ess_path = os.path.join(run_dir, "iter_%d" % gen, "ess.json")
            ess = json.load(open(ess_path))["ess"]
            combos.setdefault((trees, leaf), []).append((ess,) + measured)

    if not combos:
        print("no complete grids yet")
        return

    hdr = "%8s %5s %5s %8s %12s %11s %12s" % (
        "n_trees", "leaf", "gens", "ESS", "weighted MAE", "prior MAE", "pred. err",
    )
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(combos):
        vals = np.array(combos[key])
        ess, weighted, prior, predictive = vals.mean(axis=0)
        print(
            "%8d %5d %5d %8.1f %11.2f%% %10.2f%% %11.2f%%"
            % (key[0], key[1], len(vals), ess, 100 * weighted, 100 * prior,
               100 * predictive)
        )


if __name__ == "__main__":
    main()
