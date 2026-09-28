"""Summarise the forest-hyperparameter grid: does ESS follow n_trees x leaf size?"""

import csv
import glob
import json
import os
import re

rows = []
for d in sorted(glob.glob(os.path.join(os.environ.get("DED_HPC_ROOT", "/gscratch/cheme/chiu"), "sa_ess/runs/*"))):
    m = re.search(r"_t(\d+)_l(\d+)$", d)
    if not m:
        continue
    trees, leaf = int(m.group(1)), int(m.group(2))
    for g in range(5):
        wf = os.path.join(d, "iter_%d" % g, "weights.csv")
        ef = os.path.join(d, "iter_%d" % g, "ess.json")
        if not (os.path.exists(wf) and os.path.exists(ef)):
            continue
        w = [float(r["weight"]) for r in csv.DictReader(open(wf))]
        e = json.load(open(ef))
        rows.append((trees, leaf, g, len(w), sum(1 for x in w if x > 0), e["ess"]))

if not rows:
    print("no complete grids yet")
    raise SystemExit

# per-combination summary across the five generations, which are replicates here
combos = {}
for trees, leaf, g, n, sup, ess in rows:
    combos.setdefault((trees, leaf), []).append((n, sup, ess))

hdr = "%8s %5s %5s %7s %9s %9s %9s" % (
    "n_trees", "leaf", "gens", "cap", "support", "ESS", "ESS/n",
)
print(hdr)
print("-" * len(hdr))
for (trees, leaf) in sorted(combos):
    vals = combos[(trees, leaf)]
    n = sum(v[0] for v in vals) / len(vals)
    sup = sum(v[1] for v in vals) / len(vals)
    ess = sum(v[2] for v in vals) / len(vals)
    print(
        "%8d %5d %5d %7d %9.0f %9.1f %8.1f%%"
        % (trees, leaf, len(vals), trees * leaf, sup, ess, 100 * ess / n)
    )
