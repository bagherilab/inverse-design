"""Quantify genealogical concentration and geometric spread in the N=5000 run.

The R2.14/R2.16 ancestry diagnostic shows how many generation-0 founders remain,
but founder count alone does not measure how far repeated perturbations move the
descendants.  This script joins the exact g1--g4 lineage maps to the saved
parameter particles and reports both quantities.

Distances use the 18 independently Gaussian-perturbed continuous coordinates,
each divided by its configured prior width.  Thus a distance of one is one full
prior width along one coordinate (or the Euclidean equivalent across several
coordinates).  X/Y spacing is summarized separately because it uses a discrete
kernel; capillary density is derived from the spacing pair.

Expected input layout under ROOT::

    output/iter_0/all_param_df.csv ... output/iter_4/all_param_df.csv
    input/iter_1/lineage.csv ... input/iter_4/lineage.csv
    g4_peak_assignments.csv

Usage::

    python analysis_scripts/r214_lineage_geometry.py ROOT --out OUTDIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RANGES = {
    "CELL_VOLUME_MU": (1125.0, 3375.0),
    "NECROTIC_FRACTION": (0.0, 1.0),
    "ACCURACY": (0.0, 1.0),
    "COMPRESSION_TOLERANCE": (2.175, 6.525),
    "SYNTHESIS_DURATION_MU": (500.0, 774.0),
    "BASAL_ENERGY_MU": (0.0005, 0.0015),
    "PROLIFERATION_ENERGY_MU": (0.0005, 0.0015),
    "MIGRATION_ENERGY_MU": (0.0001, 0.0003),
    "METABOLIC_PREFERENCE_MU": (0.15, 0.45),
    "CONVERSION_FRACTION_MU": (0.125, 0.375),
    "RATIO_GLUCOSE_PYRUVATE_MU": (0.25, 0.75),
    "LACTATE_RATE_MU": (0.05, 0.15),
    "AUTOPHAGY_RATE_MU": (0.00005, 0.00015),
    "GLUCOSE_UPTAKE_RATE_MU": (0.56, 1.68),
    "ATP_PRODUCTION_RATE_MU": (4.4635, 13.3905),
    "MIGRATORY_THRESHOLD_MU": (5.0, 15.0),
    "GLUCOSE_CONCENTRATION": (0.001, 0.01),
    "OXYGEN_CONCENTRATION": (0.0, 70.0),
}
PARAMS = list(RANGES)
WIDTHS = np.array([RANGES[p][1] - RANGES[p][0] for p in PARAMS])
SCHEDULE = {1: 0.08, 2: 0.06, 3: 0.04, 4: 0.02}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_index(frame: pd.DataFrame) -> pd.Index:
    values = frame["input_folder"].astype(str).str.extract(r"(\d+)")[0]
    if values.isna().any():
        raise ValueError("Could not parse every input_folder index")
    return pd.Index(values.astype(int), name="input_index")


def load_particles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(PARAMS + ["X_SPACING", "Y_SPACING", "input_folder"]) - set(frame)
    if missing:
        raise ValueError(f"Missing particle columns in {path}: {sorted(missing)}")
    frame.index = input_index(frame)
    if not frame.index.is_unique:
        raise ValueError(f"Particle indices are not unique in {path}")
    return frame.sort_index()


def normalized(frame: pd.DataFrame) -> np.ndarray:
    values = frame[PARAMS].to_numpy(float)
    lower = np.array([RANGES[p][0] for p in PARAMS])
    return (values - lower) / WIDTHS


def spacing(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for name in ("X_SPACING", "Y_SPACING"):
        values = frame[name].astype(str).str.split(":").str[-1].astype(int)
        columns.append(values.to_numpy())
    return np.column_stack(columns)


def central_span(values: np.ndarray) -> np.ndarray:
    return np.quantile(values, 0.95, axis=0) - np.quantile(values, 0.05, axis=0)


def describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_max": float(np.max(values)),
    }


def validate_lineage(path: Path, n: int) -> pd.Series:
    frame = pd.read_csv(path)
    required = {"input_index", "parent_input_index"}
    if set(frame) != required:
        raise ValueError(f"Unexpected lineage schema in {path}: {list(frame)}")
    if len(frame) != n or frame["input_index"].nunique() != n:
        raise ValueError(f"Lineage file {path} does not contain {n} unique children")
    mapping = frame.set_index("input_index")["parent_input_index"].astype(int).sort_index()
    if mapping.min() < 1 or mapping.max() > n:
        raise ValueError(f"Parent indices outside 1..{n} in {path}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    particle_paths = {
        g: args.root / "output" / f"iter_{g}" / "all_param_df.csv"
        for g in range(5)
    }
    lineage_paths = {
        g: args.root / "input" / f"iter_{g}" / "lineage.csv"
        for g in range(1, 5)
    }
    assignment_path = args.root / "g4_peak_assignments.csv"
    required_paths = list(particle_paths.values()) + list(lineage_paths.values()) + [assignment_path]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    particles = {g: load_particles(path) for g, path in particle_paths.items()}
    sizes = {len(frame) for frame in particles.values()}
    if sizes != {5000}:
        raise ValueError(f"Expected 5000 particles per generation, found {sorted(sizes)}")
    lineage = {g: validate_lineage(path, 5000) for g, path in lineage_paths.items()}
    z = {g: normalized(frame) for g, frame in particles.items()}
    xy = {g: spacing(frame) for g, frame in particles.items()}

    transition_rows = []
    for g in range(1, 5):
        parents = lineage[g].to_numpy() - 1
        delta = z[g] - z[g - 1][parents]
        distance = np.linalg.norm(delta, axis=1)
        spacing_delta = xy[g] - xy[g - 1][parents]
        spacing_l1 = np.abs(spacing_delta).sum(axis=1)
        row = {
            "generation": g,
            "kernel_sd_fraction": SCHEDULE[g],
            "unique_continuous_vectors": int(np.unique(z[g], axis=0).shape[0]),
            "nonzero_continuous_step_fraction": float(np.mean(distance > 0)),
            "rms_step_per_dimension": float(np.sqrt(np.mean(delta**2))),
            "spacing_pair_changed_fraction": float(np.mean(spacing_l1 > 0)),
            "spacing_l1_median": float(np.median(spacing_l1)),
            "spacing_l1_p95": float(np.quantile(spacing_l1, 0.95)),
        }
        row.update(describe(distance, "normalized_euclidean_step"))
        transition_rows.append(row)
    transition = pd.DataFrame(transition_rows)

    # Trace every g4 particle back to its exact g0 founder.
    current = np.arange(1, 5001, dtype=int)
    for g in range(4, 0, -1):
        current = lineage[g].reindex(current).to_numpy(int)
    founders = current
    founder_z = z[0][founders - 1]
    cumulative_distance = np.linalg.norm(z[4] - founder_z, axis=1)

    assignments = pd.read_csv(assignment_path)
    assignments = assignments.set_index("input_index").sort_index()
    if not assignments.index.equals(pd.Index(np.arange(1, 5001), name="input_index")):
        raise ValueError("Peak assignments do not contain input_index 1..5000 in order")
    modes = assignments["mode_index"].to_numpy(int)

    founder_rows = []
    for founder in sorted(np.unique(founders)):
        hit = founders == founder
        group = z[4][hit]
        spans = central_span(group) if hit.sum() > 1 else np.zeros(len(PARAMS))
        mode_values = sorted(np.unique(modes[hit]).tolist())
        founder_rows.append(
            {
                "g0_founder_input_index": int(founder),
                "g4_descendants": int(hit.sum()),
                "distinct_g4_modes": len(mode_values),
                "g4_modes": ";".join(str(value) for value in mode_values),
                "cumulative_distance_median": float(np.median(cumulative_distance[hit])),
                "cumulative_distance_p95": float(np.quantile(cumulative_distance[hit], 0.95)),
                "central90_span_across_parameters_median": float(np.median(spans)),
                "central90_span_across_parameters_min": float(np.min(spans)),
                "central90_span_across_parameters_max": float(np.max(spans)),
            }
        )
    founder_geometry = pd.DataFrame(founder_rows).sort_values(
        ["g4_descendants", "g0_founder_input_index"], ascending=[False, True]
    )

    span_rows = []
    for g in range(5):
        spans = central_span(z[g])
        for parameter, value in zip(PARAMS, spans):
            span_rows.append(
                {
                    "generation": g,
                    "parameter": parameter,
                    "central90_span_fraction_of_prior_range": float(value),
                }
            )
    spans = pd.DataFrame(span_rows)
    g4_spans = spans.loc[spans["generation"] == 4, "central90_span_fraction_of_prior_range"].to_numpy()

    multi_mode_founders = set(
        founder_geometry.loc[founder_geometry["distinct_g4_modes"] > 1, "g0_founder_input_index"]
    )
    particle_multi = np.fromiter((founder in multi_mode_founders for founder in founders), bool)
    major_founders = founder_geometry[founder_geometry["g4_descendants"] >= 50]
    summary = {
        "schema_version": 1,
        "source_run": "ABC_SMC_RF_N5000_pm50",
        "n_particles_per_generation": 5000,
        "continuous_dimensions": len(PARAMS),
        "distance_definition": "Euclidean distance after dividing each of 18 independently Gaussian-perturbed continuous coordinates by its configured prior width",
        "g4_unique_continuous_vectors": int(np.unique(z[4], axis=0).shape[0]),
        "g4_distinct_g0_founders": int(np.unique(founders).size),
        "g0_founders_contributing_to_multiple_g4_modes": len(multi_mode_founders),
        "g4_particles_whose_founder_contributes_to_multiple_modes": int(particle_multi.sum()),
        "g4_particle_fraction_whose_founder_contributes_to_multiple_modes": float(particle_multi.mean()),
        "g0_founders_with_at_least_50_g4_descendants": int(len(major_founders)),
        "g4_particles_from_founders_with_at_least_50_descendants": int(major_founders["g4_descendants"].sum()),
        "major_founder_central90_span_across_parameters_median_of_medians": float(
            major_founders["central90_span_across_parameters_median"].median()
        ),
        "major_founder_central90_span_across_parameters_minimum_median": float(
            major_founders["central90_span_across_parameters_median"].min()
        ),
        "major_founder_central90_span_across_parameters_maximum_median": float(
            major_founders["central90_span_across_parameters_median"].max()
        ),
        "g4_central90_span_fraction_of_prior_range_across_parameters_min": float(g4_spans.min()),
        "g4_central90_span_fraction_of_prior_range_across_parameters_median": float(np.median(g4_spans)),
        "g4_central90_span_fraction_of_prior_range_across_parameters_max": float(g4_spans.max()),
    }
    summary.update(describe(cumulative_distance, "g0_to_g4_normalized_euclidean_distance"))

    args.out.mkdir(parents=True, exist_ok=True)
    transition.to_csv(args.out / "transition_geometry.csv", index=False)
    founder_geometry.to_csv(args.out / "founder_geometry.csv", index=False)
    spans.to_csv(args.out / "parameter_span_by_generation.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "source_files": {str(path.relative_to(args.root)): sha256(path) for path in required_paths},
        "configured_ranges": RANGES,
        "kernel_schedule": SCHEDULE,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("\nTransition geometry:")
    print(transition.to_string(index=False))
    print("\nFounder geometry:")
    print(founder_geometry.to_string(index=False))


if __name__ == "__main__":
    main()
