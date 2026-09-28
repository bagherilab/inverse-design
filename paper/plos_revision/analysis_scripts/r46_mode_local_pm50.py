"""Build the pm50 single-mode four-method R4.6 campaign.

The published P1--P4 partition is withdrawn, so the comparison is rebuilt on the
corrected pm50 cloud and on one mode of it. Every selector sees the same data:
the complete membership of the largest density maximum of
``ABC_SMC_RF_N5000_pm50`` generation 4, unweighted, which is the only way to
separate the selection method from the selection scope. The frozen global table
could not do that, because Sobol and LASSO had only ever been run at whole-cloud
scope while Ward had only been run peak-locally.

Four arms at one budget:

* ``ward``   Ward at distance threshold 1.0 on the local correlation matrix,
             keeping the member nearest its cluster centroid in
             correlation-profile space. k is an output of the threshold.
* ``hybrid`` the same clusters, keeping the member with the largest mean of
             normalized Sobol total-order and absolute LASSO importance.
* ``sobol``  the k highest Sobol total-order scores.
* ``lasso``  the k highest absolute LASSO coefficients.

Because Ward fixes k, all four arms retain the same number of parameters and the
only variable is which ones. Importance averages the three selection metrics,
matching the frozen global campaign; ``surrogate_oob_r2`` is recorded next to it
because that average weights three metrics whose surrogate skill differs by more
than threefold.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import maximum_filter
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

CT_RANGE = [2.175, 6.525]
BW = 0.3150
WARD_THRESHOLD = 1.0
RNG_SEED = 20260810
SOBOL_BASE = 512
TREES = 300
METRICS = ("doub_time", "symmetry", "colony_growth")
SOURCE_RUN = "ABC_SMC_RF_N5000_pm50"
SOURCE_GENERATION = 4
# The driver re-detects modes with its own unweighted settings and indexes
# peak_positions by peak_name. Verified 2026-08-10: that detection also finds
# six modes on this cloud and its index 1 -- "Peak 2" -- has centroid
# (-2.654, -1.358) against (-2.902, -1.364) for the weighted mode selected here,
# so the label points at the same mode. The preset branch never reads the
# resulting peak_data; the name only has to be a valid index.
DRIVER_PEAK_NAME = "Peak 2"
METHODS = ("ward", "hybrid", "sobol", "lasso")
RECALIBRATION_SEEDS = (20260808, 20260809, 20260810)
HELD_OUT_SEEDS = tuple(range(93001, 93011))
SMOKE_ARM = "ml_ward_s1"
PARAMS = ["CELL_VOLUME_MU", "NECROTIC_FRACTION", "ACCURACY", "COMPRESSION_TOLERANCE",
          "SYNTHESIS_DURATION_MU", "BASAL_ENERGY_MU", "PROLIFERATION_ENERGY_MU",
          "MIGRATION_ENERGY_MU", "METABOLIC_PREFERENCE_MU", "CONVERSION_FRACTION_MU",
          "RATIO_GLUCOSE_PYRUVATE_MU", "LACTATE_RATE_MU", "AUTOPHAGY_RATE_MU",
          "GLUCOSE_UPTAKE_RATE_MU", "ATP_PRODUCTION_RATE_MU", "MIGRATORY_THRESHOLD_MU",
          "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION", "CAPILLARY_DENSITY"]
INTERPRETATION = (
    "The selected mode is a statistical density maximum of the corrected pm50 "
    "generation-4 ensemble. No biological-regime, mechanistic, functional or "
    "causal interpretation is made or tested by this campaign."
)


def canonical_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ selection
def load_cloud(output_root, run, generation):
    directory = Path(output_root) / run / f"iter_{generation}"
    parameters = pd.read_csv(directory / "all_param_df.csv")
    metrics = pd.read_csv(directory / "final_metrics.csv")
    weights = pd.read_csv(directory / "weights.csv").set_index("input_index")["weight"]
    if parameters["input_folder"].duplicated().any() or metrics["input_folder"].duplicated().any():
        raise ValueError("duplicate input_folder in the source cloud")
    metrics = metrics.set_index("input_folder").reindex(parameters["input_folder"]).reset_index()
    index = parameters["input_folder"].astype(str).str.extract(r"(\d+)").iloc[:, 0].astype(int)
    weight = index.map(weights).fillna(0.0).to_numpy(float)
    x = parameters[PARAMS].to_numpy(float)
    y = metrics[list(METRICS)].to_numpy(float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1) & (weight >= 0)
    x, y, weight = x[valid], y[valid], weight[valid]
    if weight.sum() <= 0:
        raise ValueError("non-positive total weight in the source cloud")
    return x, y, weight / weight.sum(), directory


def principal_basis(x):
    mean, sd = x.mean(axis=0), x.std(axis=0)
    z = (x - mean) / sd
    _, _, right = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    right = right[:2].copy()
    for row in range(2):
        column = np.argmax(np.abs(right[row]))
        if right[row, column] < 0:
            right[row] *= -1
    return ((x - mean) / sd - z.mean(axis=0)) @ right.T


def largest_mode(scores, weight, grid=200, window=5, threshold=0.05):
    """Weighted KDE maxima; the mode with the most members wins.

    On this cloud membership, weight mass and peak density all select the same
    mode, so the rule is unambiguous here. It is not unambiguous on every cloud
    and is therefore recorded explicitly in the manifest.
    """
    kde = gaussian_kde(scores.T, bw_method=BW, weights=weight)
    xs = np.linspace(scores[:, 0].min(), scores[:, 0].max(), grid)
    ys = np.linspace(scores[:, 1].min(), scores[:, 1].max(), grid)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    density = kde(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(grid, grid)
    hits = (density == maximum_filter(density, size=(window, window))) & (
        density > density.max() * threshold
    )
    yi, xi = np.nonzero(hits)
    centroids = np.column_stack([xs[xi], ys[yi]])
    labels = np.argmin(((scores[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
    counts = np.bincount(labels, minlength=len(centroids))
    chosen = int(np.argmax(counts))
    mass = np.array([weight[labels == index].sum() for index in range(len(centroids))])
    peak_density = kde(centroids.T)
    unanimous = bool(
        np.argmax(counts) == np.argmax(mass) == np.argmax(peak_density)
    )
    return centroids, labels, counts, mass, peak_density, chosen, unanimous


def local_correlation(x):
    correlation = np.corrcoef(x, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError("zero-variance parameter in the local sample")
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def ward_labels(correlation):
    return fcluster(linkage(correlation, method="ward"), WARD_THRESHOLD, criterion="distance")


def representatives(correlation, labels, score=None):
    keep = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) == 1:
            keep.append(int(members[0]))
        elif score is None:
            centroid = correlation[members].mean(axis=0)
            keep.append(int(members[np.argmin(((correlation[members] - centroid) ** 2).sum(axis=1))]))
        else:
            keep.append(int(members[np.argmax(score[members])]))
    return sorted(keep)


def lasso_scores(x, y):
    xz = (x - x.mean(axis=0)) / x.std(axis=0)
    total = np.zeros(x.shape[1])
    for column in range(y.shape[1]):
        yz = (y[:, column] - y[:, column].mean()) / y[:, column].std()
        coefficients = np.abs(LassoCV(cv=5, random_state=RNG_SEED, n_jobs=1).fit(xz, yz).coef_)
        total += coefficients / (coefficients.sum() or 1.0)
    return total / y.shape[1]


def sobol_scores(x, y):
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    bounds = []
    for column in range(x.shape[1]):
        low, high = float(x[:, column].min()), float(x[:, column].max())
        bounds.append([low, high if high > low else low + 1e-12])
    problem = {"num_vars": x.shape[1], "names": list(PARAMS), "bounds": bounds}
    design = sobol_sample.sample(problem, SOBOL_BASE, calc_second_order=False, seed=RNG_SEED)
    total, oob = np.zeros(x.shape[1]), []
    for column in range(y.shape[1]):
        model = RandomForestRegressor(
            n_estimators=TREES, random_state=RNG_SEED + column, oob_score=True, n_jobs=-1
        ).fit(x, y[:, column])
        oob.append(float(model.oob_score_))
        indices = np.asarray(sobol_analyze.analyze(
            problem, model.predict(design), calc_second_order=False, seed=RNG_SEED,
            num_resamples=100, print_to_console=False,
        )["ST"], float)
        indices = np.clip(np.nan_to_num(indices, nan=0.0), 0.0, None)
        total += indices / (indices.sum() or 1.0)
    return total / y.shape[1], oob


def freeze_selections(output_root):
    x, y, weight, directory = load_cloud(output_root, SOURCE_RUN, SOURCE_GENERATION)
    scores = principal_basis(x)
    centroids, labels, counts, mass, density, chosen, unanimous = largest_mode(scores, weight)
    members = np.flatnonzero(labels == chosen)
    local_x, local_y = x[members], y[members]

    correlation = local_correlation(local_x)
    clusters = ward_labels(correlation)
    k = int(len(np.unique(clusters)))

    lasso = lasso_scores(local_x, local_y)
    sobol, oob = sobol_scores(local_x, local_y)
    composite = (lasso / (lasso.sum() or 1.0) + sobol / (sobol.sum() or 1.0)) / 2.0

    keeps = {
        "ward": representatives(correlation, clusters),
        "hybrid": representatives(correlation, clusters, score=composite),
        "sobol": sorted(int(i) for i in np.argsort(-sobol)[:k]),
        "lasso": sorted(int(i) for i in np.argsort(-lasso)[:k]),
    }
    selections = {}
    for method, keep in keeps.items():
        kept = sorted(PARAMS[i] for i in keep)
        selections[method] = {
            "k": k,
            "keep": kept,
            "fix": sorted(set(PARAMS) - set(kept)),
        }

    manifest = {
        "schema_version": 1,
        "design": "R4.6 pm50 single-mode four-method selection",
        "interpretation": INTERPRETATION,
        "source_run": SOURCE_RUN,
        "generation": SOURCE_GENERATION,
        "parameters": list(PARAMS),
        "weighting": "unweighted; weights are used only to detect the density maxima",
        "mode_rule": "largest membership under nearest-maximum assignment",
        "mode_rule_unanimous": unanimous,
        "mode_index": chosen,
        "mode_centroid": [float(v) for v in centroids[chosen]],
        "mode_members": int(counts[chosen]),
        "mode_weight_mass": float(mass[chosen]),
        "modes_detected": int(len(centroids)),
        "mode_membership_all": [int(v) for v in counts],
        "driver_peak_name": DRIVER_PEAK_NAME,
        "ward_threshold": WARD_THRESHOLD,
        "k": k,
        "selection_metrics": list(METRICS),
        "importance_aggregation": "mean of per-metric normalized importances, matching the global campaign",
        "surrogate_oob_r2": dict(zip(METRICS, oob)),
        "scores": {"sobol": sobol.tolist(), "lasso": lasso.tolist(),
                   "hybrid_composite": composite.tolist()},
        "selections": selections,
        "source_files": {
            name: sha256_file(directory / name)
            for name in ("all_param_df.csv", "final_metrics.csv", "weights.csv")
        },
        "rng_seed": RNG_SEED,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


# ------------------------------------------------------------------ campaign
def validate_selection_manifest(manifest):
    recorded = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if recorded != canonical_hash(unsigned):
        raise ValueError("selection manifest hash mismatch")
    if "no biological-regime" not in manifest.get("interpretation", "").lower():
        raise ValueError("statistical-mode interpretation is not explicit")
    parameters = set(manifest["parameters"])
    if len(parameters) != 19:
        raise ValueError("selection manifest must contain 19 unique parameters")
    if set(manifest["selections"]) != set(METHODS):
        raise ValueError(f"selection manifest must contain exactly {len(METHODS)} sets")
    budgets = set()
    for key, selection in manifest["selections"].items():
        keep, fix, k = selection["keep"], selection["fix"], selection["k"]
        if len(keep) != k or len(set(keep)) != k:
            raise ValueError(f"{key}: invalid retained-set cardinality")
        if set(keep) & set(fix) or set(keep) | set(fix) != parameters:
            raise ValueError(f"{key}: keep and fix are not exact complements")
        budgets.add(k)
    if len(budgets) != 1:
        raise ValueError("all arms must retain the same number of parameters")
    return True


def build_campaign(manifest, template_config, output_dir, remote_campaign_root,
                   recalibration_seeds=RECALIBRATION_SEEDS, held_out_seeds=HELD_OUT_SEEDS,
                   arcade_output_root=os.path.join(os.environ.get("DED_HPC_ROOT", "/gscratch/cheme/chiu"), "ARCADE_OUTPUT")):
    validate_selection_manifest(manifest)
    recalibration_seeds = tuple(int(seed) for seed in recalibration_seeds)
    held_out_seeds = tuple(int(seed) for seed in held_out_seeds)
    if len(set(recalibration_seeds)) != 3:
        raise ValueError("exactly three unique recalibration seeds are required")
    if set(recalibration_seeds) & set(held_out_seeds):
        raise ValueError("recalibration and held-out seeds overlap")

    output_dir = Path(output_dir)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    driver_selections = {
        "schema_version": 1,
        "note": "keep = free parameters; fix = pinned at PARAMS_DEFAULTS",
        "source_manifest_sha256": manifest["manifest_sha256"],
        "arms": {},
    }
    arms = {}
    for method in METHODS:
        selection = manifest["selections"][method]
        for replicate, seed in enumerate(recalibration_seeds, start=1):
            arm_name = f"ml_{method}_s{replicate}"
            run_name = f"ABC_SMC_RF_N512_bench_{arm_name}"
            config = dict(template_config)
            config.update({
                "n_particles": 512,
                "n_iterations": 5,
                "n_cpu": 16,
                "random_state": seed,
                "run_name": run_name,
                "peak_name": manifest["driver_peak_name"],
                "preset_key": arm_name,
                "preset_file": f"{remote_campaign_root}/selections.json",
                "simplify_model": True,
                "simplify_method": "preset",
                "base_dir": f"{arcade_output_root}/{manifest['source_run']}",
                "base_s3_dir": None,
                "compression_tolerance_range": list(CT_RANGE),
            })
            (config_dir / f"{arm_name}.json").write_text(json.dumps(config, indent=2) + "\n")
            driver_selections["arms"][arm_name] = {
                **selection,
                "selection_key": method,
                "recalibration_seed": seed,
                "arm": method,
                "k_rule": f"ward_t{WARD_THRESHOLD:g}",
                "peak": manifest["mode_index"],
            }
            arms[arm_name] = {
                "selection_key": method,
                "method": method,
                "k": selection["k"],
                "replicate": replicate,
                "recalibration_seed": seed,
                "run_tag": f"r46ml_{method}_s{replicate}",
                "run_name": run_name,
                "short_job_name": f"r46ml_{method[:4]}s{replicate}",
                "config": f"configs/{arm_name}.json",
                "keep": selection["keep"],
                "fix": selection["fix"],
                "job_nice": 0,
            }

    (output_dir / "selections.json").write_text(json.dumps(driver_selections, indent=2) + "\n")
    campaign = {
        "schema_version": 1,
        "design": "R4.6 pm50 single-mode selection-method comparison: 1 mode x 4 methods x 3 seeds",
        "interpretation": manifest["interpretation"],
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "parameters": list(manifest["parameters"]),
        "k": manifest["k"],
        "source_run": manifest["source_run"],
        "mode_index": manifest["mode_index"],
        "mode_members": manifest["mode_members"],
        "recalibration_seeds": list(recalibration_seeds),
        "held_out_seeds": list(held_out_seeds),
        "held_out_policy": "fresh common evaluation seeds; submit only after all generation-4 recalibrations finish",
        "compression_tolerance_range": list(CT_RANGE),
        "n_particles": 512,
        "generations": [0, 1, 2, 3, 4],
        "submit_env": {"DED_STEP15": "0", "DED_ADAPTIVE_KERNEL": "0", "DED_N_SHARDS": "5"},
        "driver_selections": "selections.json",
        "remote_campaign_root": remote_campaign_root,
        "smoke_arm": SMOKE_ARM,
        "surrogate_oob_r2": manifest["surrogate_oob_r2"],
        "arms": arms,
    }
    campaign["campaign_sha256"] = canonical_hash(campaign)
    (output_dir / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    validate_campaign(campaign, output_dir)
    return campaign


def validate_campaign(campaign, output_dir):
    recorded = campaign.get("campaign_sha256")
    unsigned = {key: value for key, value in campaign.items() if key != "campaign_sha256"}
    if recorded != canonical_hash(unsigned):
        raise ValueError("campaign hash mismatch")
    output_dir = Path(output_dir)
    arms = campaign["arms"]
    expected = {f"ml_{method}_s{replicate}" for method in METHODS for replicate in (1, 2, 3)}
    if set(arms) != expected or len(arms) != 12:
        raise ValueError("campaign must contain exactly 12 arms")
    if campaign["smoke_arm"] not in arms:
        raise ValueError("smoke arm is not part of the campaign")
    if campaign["compression_tolerance_range"] != CT_RANGE:
        raise ValueError("COMPRESSION_TOLERANCE range mismatch")
    if campaign["submit_env"].get("DED_STEP15") != "0":
        raise ValueError("campaign must disable Step 15 explicitly")
    parameters = set(campaign["parameters"])
    seen_runs, seen_tags = set(), set()
    selections = json.loads((output_dir / "selections.json").read_text())["arms"]
    for arm_name, arm in arms.items():
        if arm["run_name"] in seen_runs or arm["run_tag"] in seen_tags:
            raise ValueError("duplicate run name or run tag")
        seen_runs.add(arm["run_name"])
        seen_tags.add(arm["run_tag"])
        if set(arm["keep"]) | set(arm["fix"]) != parameters or set(arm["keep"]) & set(arm["fix"]):
            raise ValueError(f"{arm_name}: keep and fix are not exact complements")
        if len(arm["keep"]) != campaign["k"]:
            raise ValueError(f"{arm_name}: retained-set size does not match the frozen k")
        config = json.loads((output_dir / arm["config"]).read_text())
        if config["preset_key"] != arm_name:
            raise ValueError(f"{arm_name}: preset_key does not match the arm name")
        if config["simplify_method"] != "preset" or not config["simplify_model"]:
            raise ValueError(f"{arm_name}: config does not use the preset path")
        if config["random_state"] != arm["recalibration_seed"]:
            raise ValueError(f"{arm_name}: config seed does not match the campaign record")
        if config["n_particles"] != 512 or config["n_iterations"] != 5:
            raise ValueError(f"{arm_name}: config budget mismatch")
        if selections[arm_name]["keep"] != arm["keep"]:
            raise ValueError(f"{arm_name}: driver selections disagree with the campaign")
    if len({tuple(arm["keep"]) for arm in arms.values()}) != len(METHODS):
        raise ValueError("the four methods must produce four distinct retained sets")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcade-output-root", default=os.path.join(os.environ.get("DED_HPC_ROOT", "/gscratch/cheme/chiu"), "ARCADE_OUTPUT"))
    parser.add_argument("--template-config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--remote-campaign-root", required=True)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = freeze_selections(args.arcade_output_root)
    (args.out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"modes detected: {manifest['modes_detected']}  membership {manifest['mode_membership_all']}")
    print(f"selected mode {manifest['mode_index']} with {manifest['mode_members']} members "
          f"({100 * manifest['mode_weight_mass']:.1f}% of weight); "
          f"three-criterion agreement: {manifest['mode_rule_unanimous']}")
    print(f"Ward k at threshold {WARD_THRESHOLD:g}: {manifest['k']}")
    print("surrogate OOB R2: " + "  ".join(
        f"{metric}={value:.3f}" for metric, value in manifest["surrogate_oob_r2"].items()))
    for method in METHODS:
        print(f"  {method:>7s}: " + ", ".join(manifest["selections"][method]["keep"]))
    print(f"manifest_sha256={manifest['manifest_sha256']}")
    if args.manifest_only:
        return

    template = json.loads(args.template_config.read_text())
    campaign = build_campaign(manifest, template, args.out, args.remote_campaign_root,
                              arcade_output_root=args.arcade_output_root)
    print(f"\nbuilt {len(campaign['arms'])} arms, k={campaign['k']}, smoke={campaign['smoke_arm']}")
    print(f"campaign_sha256={campaign['campaign_sha256']}")


if __name__ == "__main__":
    main()
