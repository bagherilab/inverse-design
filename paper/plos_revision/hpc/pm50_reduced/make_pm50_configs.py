#!/usr/bin/env python3
"""Write the 24 reduced-configuration configs for the pm50 re-run.

Each config re-derives its own constraints from the bounded N1024_pm50 parent
(the driver does the PCA, peak detection and correlation work itself), so the
only per-chain fields are the method, its threshold and the peak.

Generations 0-3 use the five-seed template; prepare_pm50.sbatch switches to the
ten-seed template at generation 4.
"""

import json
from pathlib import Path

ROOT = Path("/gscratch/cheme/chiu")
CONFIGS = ROOT / "bench_pm50_reduced/configs"

BASE = {
    "sobol_power": 9,
    "n_group": 1,
    "n_min_sample": 1,
    "n_cpu": 16,
    "radius": 10,
    "margin": 2,
    "hex_size": 30,
    "simplify_model": True,
    "n_iterations": 5,
    "rf_type": "DRF",
    "n_trees": 50,
    "min_samples_leaf": 5,
    "random_state": 42,
    "criterion": "CART",
    "subsample_ratio": 0.5,
    "mean_only": True,
    "template_path": "sample_inputs/sample_combined_v3_5seed.xml",
    "base_dir": str(ROOT / "ARCADE_OUTPUT/ABC_SMC_RF_N1024_pm50"),
    "base_s3_dir": None,
    "base_input_dir": str(ROOT / "ARCADE_INPUT"),
    "base_output_dir": str(ROOT / "ARCADE_OUTPUT"),
    "n_particles": 512,
    "compression_tolerance_range": [2.175, 6.525],
}

ARMS = [("linear", t) for t in ("0.9", "0.8", "0.7")] + [
    ("dendrogram", t) for t in ("1.0", "1.25", "1.5")
]

written = []
for method, threshold in ARMS:
    for peak in (1, 2, 3, 4):
        tag = f"pm50_{method}_{threshold}_p{peak}"
        cfg = dict(BASE)
        cfg["simplify_method"] = method
        cfg["correlation_threshold"] = float(threshold)
        cfg["peak_name"] = f"Peak {peak}"
        cfg["run_name"] = f"ABC_SMC_RF_N512_pm50_{method}_{threshold}_p{peak}_mean_only"
        path = CONFIGS / f"{tag}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        written.append(tag)

print(f"wrote {len(written)} configs to {CONFIGS}")
print(" ".join(written))
