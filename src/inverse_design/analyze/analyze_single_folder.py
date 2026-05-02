from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from inverse_design.analyze.cell_metrics import CellMetrics
from inverse_design.analyze.population_metrics import PopulationMetrics
from inverse_design.config.metrics_config import CELLULAR_METRICS, POPULATION_METRICS


def analyze_single_folder(
    input_folder_path: str | Path,
    timestamps: List[str],
) -> Dict[str, Any]:
    """Analyze a single simulation input folder across multiple seeds and timestamps.

    Args:
        input_folder_path: Path to the simulation input folder (e.g. .../input_1)
        timestamps: List of timestamp strings in any order (e.g. ["000000", "010080"]).
                    Will be sorted ascending internally.

    Returns:
        Dictionary with three keys:
          - "metrics_by_timestamp_seed": raw per-seed metrics at every timestamp
                {timestamp: {metric_name: [val_seed0, val_seed1, ...]}}
          - "temporal_metrics": aggregated (median ± std) metrics at every timestamp
                {timestamp: {metric_name: float, metric_name_std: float, ...}}
          - "final_metrics": aggregated metrics at the last timestamp only (rounded)
                {metric_name: float, ...}
          - "per_seed_df": DataFrame with one row per seed (final timestamp only)
    """
    folder_path = Path(input_folder_path)
    timestamps = sorted(timestamps, key=lambda x: int(x))
    cell_metrics = CellMetrics()
    population_metrics = PopulationMetrics()

    # ------------------------------------------------------------------
    # 1. Collect raw per-seed metrics at every timestamp
    # ------------------------------------------------------------------
    metrics_by_timestamp_seed: Dict[str, Dict[str, List]] = {}

    for timestamp in timestamps:
        cells_data = cell_metrics.load_cells_data(folder_path, timestamp)
        locations_data = population_metrics.load_locations_data(folder_path, timestamp)
        metrics_dict: Dict[str, List] = {
            metric: [] for metric in CELLULAR_METRICS.keys() | POPULATION_METRICS.keys()
        }

        for (_, cells), (_, locations) in zip(cells_data, locations_data):
            for metric, config in CELLULAR_METRICS.items():
                metrics_dict[metric].append(getattr(cell_metrics, f"calculate_{metric}")(cells))
            for metric, config in POPULATION_METRICS.items():
                if config["spatial"]:
                    metrics_dict[metric].append(
                        getattr(population_metrics, f"calculate_{metric}")(cells, locations)
                    )
                else:
                    metrics_dict[metric].append(
                        getattr(population_metrics, f"calculate_{metric}")(cells)
                    )

        metrics_by_timestamp_seed[timestamp] = metrics_dict
    # ------------------------------------------------------------------
    # 2. Compute derived metrics (doubling time, colony growth)
    # ------------------------------------------------------------------
    n_cells_t1 = metrics_by_timestamp_seed[timestamps[0]]["n_cells"]
    n_cells_t2 = metrics_by_timestamp_seed[timestamps[-1]]["n_cells"]
    time_difference = int(timestamps[-1]) - int(timestamps[0])

    # Colony growth rate across all timestamps
    n_seeds = len(n_cells_t1)
    colony_diameters_over_time = {i: [] for i in range(n_seeds)}
    for ts in timestamps:
        for seed_idx, diam in enumerate(metrics_by_timestamp_seed[ts]["colony_diameter"]):
            colony_diameters_over_time[seed_idx].append(diam)

    timestamps_days = [int(ts) / 60 / 24 for ts in timestamps]
    colony_growth_results = population_metrics.calculate_colony_growth(
        colony_diameters_over_time, timestamps_days
    )

    doub_times = [
        population_metrics.calculate_doub_time(n1, n2, time_difference)
        for n1, n2 in zip(n_cells_t1, n_cells_t2)
    ]

    # Attach derived metrics to the final timestamp
    metrics_by_timestamp_seed[timestamps[-1]]["doub_time"] = doub_times
    metrics_by_timestamp_seed[timestamps[-1]]["colony_growth"] = colony_growth_results["slope"]
    metrics_by_timestamp_seed[timestamps[-1]]["colony_growth_r"] = colony_growth_results["r_value"]

    # ------------------------------------------------------------------
    # 3. Aggregate (median ± std) across seeds at every timestamp
    # ------------------------------------------------------------------
    def _aggregate(metrics_seed: Dict[str, Any]) -> Dict[str, Any]:
        aggregated: Dict[str, Any] = {}
        for metric_name, values in metrics_seed.items():
            if metric_name == "states":
                aggregated[metric_name] = {
                    state: int(np.median([s[state] for s in values])) for state in values[0].keys()
                }
            elif isinstance(values, list):
                aggregated[metric_name] = np.median(values)
                aggregated[metric_name + "_std"] = np.std(values)
            else:
                # scalar derived metric (e.g. colony_growth slope is already a scalar list)
                aggregated[metric_name] = values
        return aggregated

    temporal_metrics: Dict[str, Dict[str, Any]] = {
        ts: _aggregate(metrics_by_timestamp_seed[ts]) for ts in timestamps
    }

    # ------------------------------------------------------------------
    # 4. Final metrics (last timestamp, rounded)
    # ------------------------------------------------------------------
    final_metrics = {
        k: (round(v, 3) if isinstance(v, (float, np.floating)) else v)
        for k, v in temporal_metrics[timestamps[-1]].items()
    }

    # ------------------------------------------------------------------
    # 5. Per-seed DataFrame (final timestamp)
    # ------------------------------------------------------------------
    final_seed_data = metrics_by_timestamp_seed[timestamps[-1]]
    n_seeds = len(final_seed_data["n_cells"])
    metric_names = list(final_seed_data.keys())
    per_seed_rows = [{m: final_seed_data[m][i] for m in metric_names} for i in range(n_seeds)]
    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.insert(0, "seed", range(n_seeds))

    return {
        "metrics_by_timestamp_seed": metrics_by_timestamp_seed,
        "temporal_metrics": temporal_metrics,
        "final_metrics": final_metrics,
        "per_seed_df": per_seed_df,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
