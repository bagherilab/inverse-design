"""Evaluate the frozen R4.6 campaign on common held-out ARCADE seeds."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


PRIMARY_METRICS = ("doub_time", "symmetry", "colony_growth")
ALL_TARGET_METRICS = PRIMARY_METRICS + ("doub_time_std", "symmetry_std")
N_PARTICLES = 512
N_SHARDS = 5
INPUT_PATTERN = re.compile(r"input_(\d+)\.xml$")


def canonical_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_indexed_files(paths):
    """Hash an indexed file collection without storing 512 hashes in the manifest."""
    digest = hashlib.sha256()
    for index, path in sorted(paths.items()):
        digest.update(f"{index}:".encode())
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _campaign(campaign_root):
    campaign_root = Path(campaign_root)
    campaign = json.loads((campaign_root / "campaign.json").read_text())
    unsigned = {key: value for key, value in campaign.items() if key != "campaign_sha256"}
    if campaign.get("campaign_sha256") != canonical_hash(unsigned):
        raise ValueError("campaign manifest hash mismatch")
    _campaign_profile(campaign)
    return campaign_root, campaign


def _campaign_profile(campaign):
    """Return the frozen evaluation profile and reject mixed campaign designs."""
    arms = campaign.get("arms", {})
    controls = campaign.get("full_controls", [])
    seeds = campaign.get("held_out_seeds", [])
    if len(arms) == 18 and len(controls) == 3 and seeds == list(range(91001, 91011)):
        return "partition_free"
    if len(arms) == 12 and not controls and seeds == list(range(93001, 93011)):
        methods = {arm.get("method") for arm in arms.values()}
        budgets = {arm.get("k") for arm in arms.values()}
        counts = {}
        for arm in arms.values():
            key = arm.get("selection_key")
            counts[key] = counts.get(key, 0) + 1
        if methods != {"sobol", "lasso"} or budgets != {8, 14}:
            raise ValueError("global comparator must contain Sobol/LASSO at k=8/14")
        if len(counts) != 4 or set(counts.values()) != {3}:
            raise ValueError("global comparator requires three arms per selection")
        return "global_comparator"
    if len(arms) == 6 and not controls and seeds == list(range(93001, 93011)):
        methods = {arm.get("method") for arm in arms.values()}
        budgets = {arm.get("k") for arm in arms.values()}
        counts = {}
        for arm in arms.values():
            key = arm.get("selection_key")
            counts[key] = counts.get(key, 0) + 1
        if methods not in ({"ward"}, {"hybrid_ward"}) or budgets != {8, 14}:
            raise ValueError("global Ward comparator must contain one Ward method at k=8/14")
        if len(counts) != 2 or set(counts.values()) != {3}:
            raise ValueError("global Ward comparator requires three arms per selection")
        return (
            "global_hybrid_ward_comparator"
            if methods == {"hybrid_ward"}
            else "global_ward_comparator"
        )
    if len(arms) == 24 and not controls and seeds == list(range(93001, 93011)):
        required = {"peak", "method", "k", "replicate", "selection_key", "run_name"}
        for arm_name, arm in arms.items():
            missing = required - set(arm)
            if missing:
                raise ValueError(f"{arm_name}: peak-local arm omits {sorted(missing)}")
        if {arm["peak"] for arm in arms.values()} != {1, 2, 3, 4}:
            raise ValueError("peak-local campaign must contain P1--P4")
        if {arm["method"] for arm in arms.values()} != {
            "original_ward",
            "hybrid_ward",
        }:
            raise ValueError("peak-local campaign must contain Original and Hybrid Ward")
        counts = {}
        for arm in arms.values():
            key = (arm["peak"], arm["method"])
            counts[key] = counts.get(key, 0) + 1
        if set(counts.values()) != {3} or len(counts) != 8:
            raise ValueError("each peak-local method requires exactly three recalibration arms")
        return "peak_local"
    raise ValueError(
        "campaign is not a supported frozen R4.6 held-out design"
    )


def _paths_from_config(campaign_root, campaign):
    first_arm = campaign["arms"][sorted(campaign["arms"])[0]]
    config = json.loads((campaign_root / first_arm["config"]).read_text())
    for field in ("base_input_dir", "base_output_dir"):
        if not config.get(field):
            raise ValueError(f"campaign config missing {field}")
    return Path(config["base_input_dir"]), Path(config["base_output_dir"])


def _evaluation_arms(campaign_root, campaign):
    input_root, output_root = _paths_from_config(campaign_root, campaign)
    arms = []
    for control in campaign.get("full_controls", []):
        arms.append(
            {
                "evaluation_id": f"full_{control['family']}",
                "kind": "control",
                "family": control["family"],
                "method": "full",
                "k": 19,
                "source_run": control["run"],
                "source_generation": control["generation"],
            }
        )
    for arm_name in sorted(campaign["arms"]):
        arm = campaign["arms"][arm_name]
        arms.append(
            {
                "evaluation_id": arm_name,
                "kind": "reduced",
                "selection_key": arm["selection_key"],
                "method": arm["method"],
                "k": arm["k"],
                "replicate": arm["replicate"],
                "recalibration_seed": arm["recalibration_seed"],
                "peak": arm.get("peak"),
                "source_run": arm["run_name"],
                "source_generation": 4,
            }
        )
    expected = len(campaign.get("full_controls", [])) + len(campaign["arms"])
    if len(arms) != expected or len({arm["evaluation_id"] for arm in arms}) != expected:
        raise ValueError("evaluation IDs must be complete and unique")
    for arm in arms:
        source_root = output_root / arm["source_run"]
        arm["source_input_dir"] = str(
            input_root / arm["source_run"].lower() / f"iter_{arm['source_generation']}"
        )
        arm["source_output_dir"] = str(source_root / f"iter_{arm['source_generation']}")
        arm["target_path"] = str(source_root / "targets.json")
    return arms


def _xml_inputs(input_dir):
    paths = {}
    for path in (Path(input_dir) / "inputs").glob("input_*.xml"):
        match = INPUT_PATTERN.search(path.name)
        if match:
            paths[int(match.group(1))] = path
    expected = set(range(1, N_PARTICLES + 1))
    if set(paths) != expected:
        missing = sorted(expected - set(paths))
        extra = sorted(set(paths) - expected)
        raise ValueError(
            f"{input_dir}: expected {N_PARTICLES} input XMLs; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return paths


def _timestamps_from_xml(path):
    root = ET.parse(path).getroot()
    series = [element for element in root.iter("series") if element.get("name") == "grid"]
    if len(series) != 1:
        raise ValueError(f"{path}: expected exactly one grid series")
    try:
        ticks = int(series[0].attrib["ticks"])
        interval = int(series[0].attrib["interval"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: invalid grid series timing") from error
    if ticks <= 0 or interval <= 0 or ticks % interval:
        raise ValueError(f"{path}: incompatible grid ticks/interval")
    return [f"{tick:06d}" for tick in range(0, ticks + 1, interval)]


def _complete_source(arm):
    output_dir = Path(arm["source_output_dir"])
    for filename in ("all_param_df.csv", "final_metrics.csv", "weights.csv"):
        path = output_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{arm['evaluation_id']}: missing source {path}")
    metrics = pd.read_csv(output_dir / "final_metrics.csv")
    parameters = pd.read_csv(output_dir / "all_param_df.csv")
    if len(metrics) != N_PARTICLES or len(parameters) != N_PARTICLES:
        raise ValueError(
            f"{arm['evaluation_id']}: source g4 requires {N_PARTICLES} metrics and parameters"
        )
    inputs = _xml_inputs(arm["source_input_dir"])
    return inputs


def build_evaluation_manifest(campaign_root):
    campaign_root, campaign = _campaign(campaign_root)
    profile = _campaign_profile(campaign)
    held_out_seeds = campaign.get("held_out_seeds", [])
    arms = _evaluation_arms(campaign_root, campaign)
    targets = None
    target_hash = None
    timestamps = None
    for arm in arms:
        inputs = _complete_source(arm)
        arm["source_files"] = {
            filename: sha256_file(Path(arm["source_output_dir"]) / filename)
            for filename in ("all_param_df.csv", "final_metrics.csv", "weights.csv")
        }
        arm["source_files"]["input_xml_tree"] = sha256_indexed_files(inputs)
        target_path = Path(arm["target_path"])
        if not target_path.is_file():
            raise ValueError(f"{arm['evaluation_id']}: missing targets {target_path}")
        current_targets = json.loads(target_path.read_text())
        if set(ALL_TARGET_METRICS) - set(current_targets):
            raise ValueError(f"{arm['evaluation_id']}: targets omit evaluation metrics")
        current_hash = sha256_file(target_path)
        if targets is None:
            targets, target_hash = current_targets, current_hash
        elif current_targets != targets:
            raise ValueError("evaluation arms have non-identical targets")
        current_timestamps = _timestamps_from_xml(inputs[1])
        if timestamps is None:
            timestamps = current_timestamps
        elif current_timestamps != timestamps:
            raise ValueError("evaluation arms have incompatible ARCADE time grids")

    payload = {
        "schema_version": 1,
        "design": (
            "R4.6 partition-free common-seed held-out evaluation"
            if profile == "partition_free"
            else (
                "R4.6 peak-local Original-vs-Hybrid Ward common-seed held-out evaluation"
                if profile == "peak_local"
                else (
                    "R4.6 global Sobol/LASSO fresh common-seed comparator"
                    if profile == "global_comparator"
                    else (
                        "R4.6 global Hybrid Ward fresh common-seed comparator"
                        if profile == "global_hybrid_ward_comparator"
                        else "R4.6 global Original Ward fresh common-seed comparator"
                    )
                )
            )
        ),
        "comparison_profile": profile,
        "campaign_sha256": campaign["campaign_sha256"],
        "held_out_seeds": held_out_seeds,
        "n_particles": N_PARTICLES,
        "n_shards": N_SHARDS,
        "timestamps": timestamps,
        "targets": targets,
        "targets_sha256": target_hash,
        "scoring": {
            "particle_aggregate": "median over the ten held-out ARCADE seeds",
            "posterior_aggregate": "unweighted IQR-filtered median over 512 particles",
            "primary_metrics": list(PRIMARY_METRICS),
            "secondary_metrics": list(ALL_TARGET_METRICS),
            "winner_rule": (
                "lowest mean three-chain primary MAE; report chain and seed variation"
                if profile in {
                    "partition_free",
                    "global_comparator",
                    "global_ward_comparator",
                    "global_hybrid_ward_comparator",
                }
                else (
                    "compare Original vs Hybrid within each peak using paired recalibration "
                    "chains; cross-peak method means are descriptive"
                )
            ),
        },
        "arms": arms,
    }
    payload["evaluation_sha256"] = canonical_hash(payload)
    return payload


def _evaluation_root(campaign_root):
    return Path(campaign_root) / "heldout"


def _write_heldout_xml(source, destination, held_out_seeds):
    root = ET.parse(source)
    series = [element for element in root.getroot().iter("series") if element.get("name") == "grid"]
    if len(series) != 1:
        raise ValueError(f"{source}: expected exactly one grid series")
    series[0].set("start", str(held_out_seeds[0]))
    series[0].set("end", str(held_out_seeds[-1]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    root.write(destination, encoding="utf-8", xml_declaration=True)


def _normalized_heldout_xml(path, held_out_seeds):
    root = ET.parse(path).getroot()
    series = [element for element in root.iter("series") if element.get("name") == "grid"]
    if len(series) != 1:
        raise ValueError(f"{path}: expected exactly one grid series")
    series[0].set("start", str(held_out_seeds[0]))
    series[0].set("end", str(held_out_seeds[-1]))
    return ET.tostring(root, encoding="utf-8")


def prepare_evaluation(campaign_root):
    campaign_root = Path(campaign_root)
    manifest = build_evaluation_manifest(campaign_root)
    evaluation_root = _evaluation_root(campaign_root)
    manifest_path = evaluation_root / "evaluation_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("evaluation_sha256") != manifest["evaluation_sha256"]:
            raise ValueError("existing held-out manifest does not match frozen campaign inputs")
    else:
        evaluation_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for arm in manifest["arms"]:
        source_inputs = _xml_inputs(arm["source_input_dir"])
        destination_root = evaluation_root / "inputs" / arm["evaluation_id"] / "inputs"
        for index, source in source_inputs.items():
            destination = destination_root / source.name
            if not destination.exists():
                _write_heldout_xml(source, destination, manifest["held_out_seeds"])
            if _normalized_heldout_xml(destination, manifest["held_out_seeds"]) != (
                _normalized_heldout_xml(source, manifest["held_out_seeds"])
            ):
                raise ValueError(
                    f"{destination}: differs from generation-4 XML beyond held-out seeds"
                )
            check = ET.parse(destination).getroot()
            series = [element for element in check.iter("series") if element.get("name") == "grid"]
            if len(series) != 1 or [
                series[0].get("start"), series[0].get("end")
            ] != [str(manifest["held_out_seeds"][0]), str(manifest["held_out_seeds"][-1])]:
                raise ValueError(f"{destination}: held-out seed schedule mismatch")
    print(
        "HELDOUT_PREPARED "
        + json.dumps(
            {
                "evaluation_sha256": manifest["evaluation_sha256"],
                "arms": len(manifest["arms"]),
                "particles_per_arm": manifest["n_particles"],
                "held_out_seeds": manifest["held_out_seeds"],
            },
            sort_keys=True,
        )
    )
    return manifest


def _evaluation_manifest(campaign_root):
    path = _evaluation_root(campaign_root) / "evaluation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text())
    unsigned = {key: value for key, value in manifest.items() if key != "evaluation_sha256"}
    if manifest.get("evaluation_sha256") != canonical_hash(unsigned):
        raise ValueError("held-out manifest hash mismatch")
    return manifest


def task_paths(campaign_root, task):
    manifest = _evaluation_manifest(campaign_root)
    task = int(task)
    maximum = len(manifest["arms"]) * manifest["n_shards"]
    if not 0 <= task < maximum:
        raise ValueError(f"task must be in [0, {maximum})")
    arm = manifest["arms"][task // manifest["n_shards"]]
    shard = task % manifest["n_shards"] + 1
    root = _evaluation_root(campaign_root)
    return arm, shard, root / "inputs" / arm["evaluation_id"], root / "outputs" / arm["evaluation_id"]


def _numeric_input_folders(output_dir):
    folders = []
    for path in (Path(output_dir) / "inputs").glob("input_*"):
        match = re.fullmatch(r"input_(\d+)", path.name)
        if match:
            folders.append((int(match.group(1)), path))
    folders.sort()
    expected = list(range(1, N_PARTICLES + 1))
    if [index for index, _ in folders] != expected:
        raise ValueError(f"{output_dir}: incomplete held-out simulation folders")
    return [path for _, path in folders]


def analyze_arm(campaign_root, arm_index):
    manifest = _evaluation_manifest(campaign_root)
    arm = manifest["arms"][int(arm_index)]
    evaluation_root = _evaluation_root(campaign_root)
    input_dir = evaluation_root / "inputs" / arm["evaluation_id"]
    output_dir = evaluation_root / "outputs" / arm["evaluation_id"]
    folders = _numeric_input_folders(output_dir)
    from inverse_design.analyze.save_aggregated_results import SimulationMetrics

    calculator = SimulationMetrics(str(output_dir), str(input_dir))
    metrics, _ = calculator.analyze_all_simulations(manifest["timestamps"], folders)
    seed_metrics = pd.read_csv(output_dir / "final_metrics_seed.csv")
    if len(metrics) != N_PARTICLES or len(seed_metrics) != N_PARTICLES * len(manifest["held_out_seeds"]):
        raise ValueError(
            f"{arm['evaluation_id']}: expected {N_PARTICLES} particle metrics and "
            f"{N_PARTICLES * len(manifest['held_out_seeds'])} seed metrics"
        )
    score = score_arm(metrics, manifest["targets"], arm["evaluation_id"])
    for path in (output_dir / "inputs").rglob("*.json"):
        path.unlink()
    marker_dir = evaluation_root / "markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "evaluation_sha256": manifest["evaluation_sha256"],
        "evaluation_id": arm["evaluation_id"],
        "particle_metrics": len(metrics),
        "seed_metrics": len(seed_metrics),
        "score": score,
        "final_metrics_sha256": sha256_file(output_dir / "final_metrics.csv"),
        "final_metrics_seed_sha256": sha256_file(output_dir / "final_metrics_seed.csv"),
    }
    (marker_dir / f"{arm['evaluation_id']}.json").write_text(json.dumps(marker, indent=2) + "\n")
    print("HELDOUT_ARM_OK " + json.dumps(marker, sort_keys=True))
    return marker


def summarize_evaluation(campaign_root):
    campaign_root = Path(campaign_root)
    manifest = _evaluation_manifest(campaign_root)
    root = _evaluation_root(campaign_root)
    scores = []
    metadata = []
    for arm in manifest["arms"]:
        marker_path = root / "markers" / f"{arm['evaluation_id']}.json"
        if not marker_path.is_file():
            raise ValueError(f"missing held-out completion marker: {arm['evaluation_id']}")
        marker = json.loads(marker_path.read_text())
        if marker.get("evaluation_sha256") != manifest["evaluation_sha256"]:
            raise ValueError(f"{arm['evaluation_id']}: completion marker manifest mismatch")
        scores.append(marker["score"])
        metadata.append(
            {
                "arm": arm["evaluation_id"],
                "kind": arm["kind"],
                "selection_key": arm.get("selection_key", "full"),
                "method": arm["method"],
                "k": arm["k"],
                "peak": arm.get("peak"),
                "replicate": arm.get("replicate"),
                "recalibration_seed": arm.get("recalibration_seed"),
            }
        )
    if manifest.get("comparison_profile") == "peak_local":
        (
            arm_scores,
            selection_summary,
            paired_chains,
            peak_comparison,
            method_summary,
        ) = summarize_peak_local_scores(scores, metadata)
        arm_scores.to_csv(root / "heldout_arm_scores.csv", index=False)
        selection_summary.to_csv(root / "heldout_selection_summary.csv", index=False)
        paired_chains.to_csv(root / "heldout_paired_chain_comparison.csv", index=False)
        peak_comparison.to_csv(root / "heldout_peak_comparison.csv", index=False)
        method_summary.to_csv(root / "heldout_method_summary.csv", index=False)
        report = {
            "evaluation_sha256": manifest["evaluation_sha256"],
            "comparison_profile": "peak_local",
            "interpretation": (
                "P1--P4 are statistical posterior modes used to estimate local covariance; "
                "no biological-regime claim is made"
            ),
            "scoring": manifest["scoring"],
            "primary_comparison": "paired Original-vs-Hybrid differences within peak and recalibration seed",
            "peak_comparison": peak_comparison.to_dict(orient="records"),
            "method_summary_descriptive": method_summary.to_dict(orient="records"),
            "selection_summary": selection_summary.to_dict(orient="records"),
        }
    else:
        arm_scores, selection_summary, winner = summarize_scores(scores, metadata)
        controls = arm_scores[arm_scores["kind"] == "control"]
        profile = manifest.get("comparison_profile", "partition_free")
        if profile == "partition_free":
            if len(controls) != 3:
                raise ValueError("expected three completed full-model controls")
            control_summary = {
                "controls": len(controls),
                "mean_primary_mae_pct": float(controls["primary_mae_pct"].mean()),
                "sd_primary_mae_pct": float(controls["primary_mae_pct"].std()),
                "mean_all_target_mae_pct": float(controls["all_target_mae_pct"].mean()),
                "mean_seed_sd_pct": float(controls["mean_seed_sd_pct"].mean()),
            }
        elif len(controls):
            raise ValueError("global comparator must not contain control arms")
        else:
            control_summary = None
        arm_scores.to_csv(root / "heldout_arm_scores.csv", index=False)
        selection_summary.to_csv(root / "heldout_selection_summary.csv", index=False)
        report = {
            "evaluation_sha256": manifest["evaluation_sha256"],
            "comparison_profile": profile,
            "scoring": manifest["scoring"],
            "winner": winner,
            "full_control_summary": control_summary,
            "selection_summary": selection_summary.to_dict(orient="records"),
        }
    (root / "heldout_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("HELDOUT_REPORT " + json.dumps(report, sort_keys=True))
    return report


def _iqr_filter(values):
    values = pd.Series(values, dtype=float).dropna()
    lower, upper = values.quantile([0.25, 0.75])
    iqr = upper - lower
    return values[(values >= lower - 1.5 * iqr) & (values <= upper + 1.5 * iqr)]


def score_arm(metrics, targets, arm):
    """Score one arm from its held-out per-particle aggregate metrics."""
    missing = set(ALL_TARGET_METRICS) - set(metrics.columns) - set(targets)
    if missing:
        raise ValueError(f"{arm}: missing target metric(s): {sorted(missing)}")

    row = {"arm": arm}
    errors = {}
    for metric in ALL_TARGET_METRICS:
        if metric not in metrics or metric not in targets:
            raise ValueError(f"{arm}: missing {metric}")
        filtered = _iqr_filter(metrics[metric])
        if filtered.empty:
            raise ValueError(f"{arm}: no finite {metric} values")
        prediction = float(filtered.median())
        target = float(targets[metric])
        if target == 0:
            raise ValueError(f"{arm}: target for {metric} is zero")
        error = 100.0 * abs(prediction - target) / abs(target)
        row[f"{metric}_prediction"] = prediction
        row[f"{metric}_mae_pct"] = error
        row[f"{metric}_particles"] = len(filtered)
        errors[metric] = error

    row["primary_mae_pct"] = float(np.mean([errors[metric] for metric in PRIMARY_METRICS]))
    row["all_target_mae_pct"] = float(
        np.mean([errors[metric] for metric in ALL_TARGET_METRICS])
    )
    for metric in PRIMARY_METRICS:
        std_column = f"{metric}_std"
        if std_column not in metrics:
            raise ValueError(f"{arm}: missing held-out seed spread {std_column}")
        spread = _iqr_filter(metrics[std_column])
        if spread.empty:
            raise ValueError(f"{arm}: no finite {std_column} values")
        row[f"{metric}_seed_sd_pct"] = 100.0 * float(spread.median()) / abs(
            float(targets[metric])
        )
    row["mean_seed_sd_pct"] = float(
        np.mean([row[f"{metric}_seed_sd_pct"] for metric in PRIMARY_METRICS])
    )
    return row


def summarize_scores(arm_scores, arm_metadata):
    """Aggregate chain-seed scores and rank the six frozen selections."""
    scores = pd.DataFrame(arm_scores)
    metadata = pd.DataFrame(arm_metadata)
    merged = scores.merge(metadata, on="arm", validate="one_to_one")
    reduced = merged[merged["kind"] == "reduced"].copy()
    counts = reduced.groupby("selection_key")["arm"].count()
    incomplete = counts[counts != 3]
    if not incomplete.empty:
        raise ValueError(
            "every reduced selection requires three recalibration arms: "
            f"{incomplete.to_dict()}"
        )

    summary = (
        reduced.groupby(["method", "k", "selection_key"], as_index=False)
        .agg(
            recalibration_arms=("arm", "count"),
            mean_primary_mae_pct=("primary_mae_pct", "mean"),
            sd_primary_mae_pct=("primary_mae_pct", "std"),
            mean_all_target_mae_pct=("all_target_mae_pct", "mean"),
            sd_all_target_mae_pct=("all_target_mae_pct", "std"),
            mean_seed_sd_pct=("mean_seed_sd_pct", "mean"),
        )
        .sort_values(["mean_primary_mae_pct", "sd_primary_mae_pct", "selection_key"])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    winner = summary.iloc[0].to_dict()
    return merged, summary, winner


def summarize_peak_local_scores(arm_scores, arm_metadata):
    """Compare Original and Hybrid Ward with chains paired within each peak."""
    scores = pd.DataFrame(arm_scores)
    metadata = pd.DataFrame(arm_metadata)
    merged = scores.merge(metadata, on="arm", validate="one_to_one")
    if set(merged["kind"]) != {"reduced"} or len(merged) != 24:
        raise ValueError("peak-local comparison requires exactly 24 reduced arms")
    if merged[["peak", "replicate", "recalibration_seed"]].isna().any().any():
        raise ValueError("peak-local comparison metadata is incomplete")

    counts = merged.groupby(["peak", "method"])["arm"].count()
    if len(counts) != 8 or set(counts) != {3}:
        raise ValueError("each peak and method requires three completed arms")

    selection_summary = (
        merged.groupby(["peak", "method", "k", "selection_key"], as_index=False)
        .agg(
            recalibration_arms=("arm", "count"),
            mean_primary_mae_pct=("primary_mae_pct", "mean"),
            sd_primary_mae_pct=("primary_mae_pct", "std"),
            mean_all_target_mae_pct=("all_target_mae_pct", "mean"),
            sd_all_target_mae_pct=("all_target_mae_pct", "std"),
            mean_seed_sd_pct=("mean_seed_sd_pct", "mean"),
        )
        .sort_values(["peak", "mean_primary_mae_pct", "method"])
        .reset_index(drop=True)
    )

    comparison_columns = [
        "arm",
        "peak",
        "k",
        "replicate",
        "recalibration_seed",
        "primary_mae_pct",
        "all_target_mae_pct",
        "mean_seed_sd_pct",
    ]
    original = merged[merged["method"] == "original_ward"][comparison_columns].copy()
    hybrid = merged[merged["method"] == "hybrid_ward"][comparison_columns].copy()
    paired = original.merge(
        hybrid,
        on=["peak", "replicate", "recalibration_seed"],
        suffixes=("_original", "_hybrid"),
        validate="one_to_one",
    )
    if len(paired) != 12 or not (paired["k_original"] == paired["k_hybrid"]).all():
        raise ValueError("Original and Hybrid arms are not pairable within peak")
    paired["k"] = paired.pop("k_original")
    paired = paired.drop(columns="k_hybrid")
    for metric in ("primary_mae_pct", "all_target_mae_pct", "mean_seed_sd_pct"):
        paired[f"delta_{metric}_hybrid_minus_original"] = (
            paired[f"{metric}_hybrid"] - paired[f"{metric}_original"]
        )
    paired["primary_winner"] = np.select(
        [
            paired["delta_primary_mae_pct_hybrid_minus_original"] < 0,
            paired["delta_primary_mae_pct_hybrid_minus_original"] > 0,
        ],
        ["hybrid_ward", "original_ward"],
        default="tie",
    )
    paired = paired.sort_values(["peak", "replicate"]).reset_index(drop=True)

    original_summary = selection_summary[
        selection_summary["method"] == "original_ward"
    ].copy()
    hybrid_summary = selection_summary[
        selection_summary["method"] == "hybrid_ward"
    ].copy()
    peak_comparison = original_summary.merge(
        hybrid_summary,
        on="peak",
        suffixes=("_original", "_hybrid"),
        validate="one_to_one",
    )
    if not (peak_comparison["k_original"] == peak_comparison["k_hybrid"]).all():
        raise ValueError("Original and Hybrid k differ within a peak")
    peak_comparison["k"] = peak_comparison.pop("k_original")
    peak_comparison = peak_comparison.drop(columns="k_hybrid")
    peak_comparison["delta_primary_mae_pct_hybrid_minus_original"] = (
        peak_comparison["mean_primary_mae_pct_hybrid"]
        - peak_comparison["mean_primary_mae_pct_original"]
    )
    denominator = peak_comparison["mean_primary_mae_pct_original"].replace(0, np.nan)
    peak_comparison["relative_primary_change_pct"] = (
        100.0
        * peak_comparison["delta_primary_mae_pct_hybrid_minus_original"]
        / denominator
    )
    paired_wins = paired.groupby("peak")["primary_winner"].apply(
        lambda values: int((values == "hybrid_ward").sum())
    )
    peak_comparison["hybrid_chain_wins_of_3"] = peak_comparison["peak"].map(paired_wins)
    peak_comparison["winner"] = np.select(
        [
            peak_comparison["delta_primary_mae_pct_hybrid_minus_original"] < 0,
            peak_comparison["delta_primary_mae_pct_hybrid_minus_original"] > 0,
        ],
        ["hybrid_ward", "original_ward"],
        default="tie",
    )
    keep = [
        "peak",
        "k",
        "mean_primary_mae_pct_original",
        "sd_primary_mae_pct_original",
        "mean_primary_mae_pct_hybrid",
        "sd_primary_mae_pct_hybrid",
        "delta_primary_mae_pct_hybrid_minus_original",
        "relative_primary_change_pct",
        "hybrid_chain_wins_of_3",
        "winner",
    ]
    peak_comparison = peak_comparison[keep].sort_values("peak").reset_index(drop=True)

    method_summary = (
        merged.groupby("method", as_index=False)
        .agg(
            peaks=("peak", "nunique"),
            recalibration_arms=("arm", "count"),
            mean_primary_mae_pct=("primary_mae_pct", "mean"),
            sd_primary_mae_pct=("primary_mae_pct", "std"),
            mean_all_target_mae_pct=("all_target_mae_pct", "mean"),
            sd_all_target_mae_pct=("all_target_mae_pct", "std"),
            mean_seed_sd_pct=("mean_seed_sd_pct", "mean"),
        )
        .sort_values(["mean_primary_mae_pct", "method"])
        .reset_index(drop=True)
    )
    return merged, selection_summary, paired, peak_comparison, method_summary


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--campaign-root", type=Path, required=True)
    task_parser = subparsers.add_parser("task-paths")
    task_parser.add_argument("--campaign-root", type=Path, required=True)
    task_parser.add_argument("--task", type=int, required=True)
    task_parser.add_argument("--format", choices=("json", "tsv"), default="json")
    analyze_parser = subparsers.add_parser("analyze-arm")
    analyze_parser.add_argument("--campaign-root", type=Path, required=True)
    analyze_parser.add_argument("--arm-index", type=int, required=True)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_evaluation(args.campaign_root)
    elif args.command == "task-paths":
        arm, shard, input_dir, output_dir = task_paths(args.campaign_root, args.task)
        payload = {
            "evaluation_id": arm["evaluation_id"],
            "shard": shard,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        }
        if args.format == "tsv":
            print("\t".join(str(payload[key]) for key in payload))
        else:
            print(json.dumps(payload, sort_keys=True))
    elif args.command == "analyze-arm":
        analyze_arm(args.campaign_root, args.arm_index)
    elif args.command == "summarize":
        summarize_evaluation(args.campaign_root)


if __name__ == "__main__":
    main()
