from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import re
import numpy as np
from inverse_design.analyze.population_metrics import PopulationMetrics
from inverse_design.analyze.analyze_seed_results import SeedAnalyzer
from inverse_design.analyze.utils.analyze_utils import collect_parameter_data
from inverse_design.config.parameter_config import PARAMETER_LIST, SOURCE_PARAMS
from inverse_design.config.metrics_config import (
    CELLULAR_METRICS,
    POPULATION_METRICS,
    SUMMARY_METRICS,
)


class SimulationMetrics:
    def __init__(self, base_output_dir: str, input_base_dir: str):
        """Initialize the metrics calculator

        Args:
            base_output_dir: Base directory containing all simulation output folders
        """
        self.base_output_dir = Path(base_output_dir)
        self.input_base_dir = Path(input_base_dir)
        self.logger = logging.getLogger(__name__)
        self.population_metrics = PopulationMetrics()
        self.seed_analyzer = SeedAnalyzer(base_output_dir)

    def _round_metrics(self, metrics_dict: Dict[str, Any], decimals: int = 3) -> Dict[str, Any]:
        """Round all numeric values in a dictionary to specified decimals.

        Args:
            metrics_dict: Dictionary containing metrics
            decimals: Number of decimal places to round to

        Returns:
            Dictionary with rounded numeric values
        """
        rounded_dict = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (float, np.floating)):
                rounded_dict[key] = round(value, decimals)
            else:
                rounded_dict[key] = value
        return rounded_dict

    def analyze_simulation(self, folder_path: Path, timestamps: List[str]) -> Dict[str, Any]:
        """Analyze a single simulation folder with multiple seeds."""
        temporal_metrics = {}

        metrics_by_timestamp_seed = {
            timestamp: self.seed_analyzer.calculate_seed_metrics(folder_path, timestamp)
            for timestamp in timestamps
        }

        for key, value in metrics_by_timestamp_seed.items():
            print(key, value)

        n_cells_t1_seed_list = metrics_by_timestamp_seed[timestamps[0]]["n_cells"]
        n_cells_t2_seed_list = metrics_by_timestamp_seed[timestamps[-1]]["n_cells"]
        time_difference = int(timestamps[-1]) - int(timestamps[0])

        colony_diameters_over_time = self.seed_analyzer.collect_colony_diameters_over_time(
            metrics_by_timestamp_seed
        )
        timestamps_days = [int(timestamp) / 60 / 24 for timestamp in timestamps]
        colony_growth_rates_results = self.population_metrics.calculate_colony_growth(
            colony_diameters_over_time, timestamps_days
        )

        doub_times_seed = []
        for n1, n2 in zip(n_cells_t1_seed_list, n_cells_t2_seed_list):
            doub_time = self.population_metrics.calculate_doub_time(n1, n2, time_difference)
            doub_times_seed.append(doub_time)
        # Add doubling and colony growth rates to metrics_by_timestamp_seed
        metrics_by_timestamp_seed[timestamps[-1]]["doub_time"] = doub_times_seed
        metrics_by_timestamp_seed[timestamps[-1]]["colony_growth"] = colony_growth_rates_results[
            "slope"
        ]
        metrics_by_timestamp_seed[timestamps[-1]]["colony_growth_r"] = colony_growth_rates_results[
            "r_value"
        ]
        for timestamp, metrics_seed in metrics_by_timestamp_seed.items():
            temporal_metrics[timestamp] = self._aggregate_timestamp_metrics(metrics_seed)
        final_metrics = temporal_metrics[timestamps[-1]]
        final_metrics = self._round_metrics(final_metrics)

        return {
            "metrics_by_timestamp_seed": metrics_by_timestamp_seed,
            "temporal_metrics": temporal_metrics,
            "final_metrics": final_metrics,
        }

    def _aggregate_timestamp_metrics(
        self, metrics_for_timestamp: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics for a single timestamp."""
        aggregated_metrics = {}
        for key, value in metrics_for_timestamp.items():
            print(key, value)
        for metric_name, values in metrics_for_timestamp.items():
            aggregated_metrics[metric_name] = {}

            if metric_name == "states":
                state_medians = {}
                for state in values[0].keys():  # Use first seed's states as reference
                    state_values = [int(seed_states[state]) for seed_states in values]
                    state_medians[state] = int(np.median(state_values))
                aggregated_metrics[metric_name] = state_medians
            else:
                aggregated_metrics[metric_name] = np.median(values)
                aggregated_metrics[metric_name + "_std"] = np.std(values)
                # aggregated_metrics[metric_name + "_min"] = np.min(values)
                # aggregated_metrics[metric_name + "_max"] = np.max(values)
                # aggregated_metrics[metric_name + "_25"] = np.percentile(values, 25)
                # aggregated_metrics[metric_name + "_75"] = np.percentile(values, 75)

        return aggregated_metrics

    def analyze_all_simulations(
        self, timestamps: List[str], sim_folders: List[Path]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """Analyze all simulation folders and compile results.

        Args:
            timestamps: List of timestamp strings (e.g., ["000000", "000720", ...])
                       Will be sorted in ascending order.

        Returns:
            Tuple containing:
            - DataFrame with final metrics (from last timestamp)
            - Dictionary with temporal metrics for all timestamps
        """
        # Ensure timestamps are sorted
        timestamps = sorted(timestamps, key=lambda x: int(x))
        final_results = []
        final_results_seed = []
        temporal_results = {}

        for folder in sim_folders:
            try:
                folder_number = int(re.search(r"input_(\d+)", folder.name).group(1))
                if folder_number % 50 == 0:
                    print(f"Analyzing {folder.name} ({folder_number}/{len(sim_folders)})")
                metrics = self.analyze_simulation(folder, timestamps)
                final_metrics_flat = metrics["final_metrics"].copy()
                final_metrics_flat["input_folder"] = folder.name
                final_results.append(final_metrics_flat)
                final_results_seed_flat = metrics["metrics_by_timestamp_seed"][
                    timestamps[-1]
                ].copy()
                n_seeds = len(final_results_seed_flat["n_cells"])
                metrics_names = final_results_seed_flat.keys()
                for i in range(n_seeds):
                    seed_results = {
                        metric_name: final_results_seed_flat[metric_name][i]
                        for metric_name in metrics_names
                    }
                    seed_results["input_folder"] = folder.name
                    final_results_seed.append(seed_results)
                temporal_results[folder.name] = metrics["temporal_metrics"]
            except Exception as e:
                self.logger.error(f"Error processing {folder}: {str(e)}")
        df = pd.DataFrame(final_results)
        df_seed = pd.DataFrame(final_results_seed)
        df = self._reorder_columns(df)
        df_seed = self._reorder_columns(df_seed)
        # Save final metrics
        output_file = self.base_output_dir / "final_metrics.csv"
        df.to_csv(output_file, index=False)
        output_file_seed = self.base_output_dir / "final_metrics_seed.csv"
        df_seed.to_csv(output_file_seed, index=False)
        self.logger.info(
            f"Saved final metrics for {len(final_results)} simulations to {output_file}"
        )

        return df, temporal_results

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns.tolist()
        cols.remove("input_folder")
        cols.remove("states")
        cols.append("input_folder")
        cols.append("states")
        df = df[cols]
        return df

    def extract_and_save_parameters(
        self,
        sim_folders,
        param_list: List[str],
        source_param_list: List[str],
        save_file: str = "all_param_df.csv",
    ):
        all_param_df = collect_parameter_data(sim_folders, param_list)
        param_from_input_folder = self.input_base_dir / "parameter_log.csv"
        param_from_input_df = pd.read_csv(param_from_input_folder)
        added_columns = [
            column for column in param_from_input_df.columns if column in source_param_list
        ]
        if added_columns:
            all_param_df = pd.concat([all_param_df, param_from_input_df[added_columns]], axis=1)
        # Remove constant parameters
        constant_columns = [col for col in all_param_df.columns if all_param_df[col].nunique() == 1]
        print(f"constant_columns: {constant_columns}")
        all_param_df = all_param_df.drop(columns=constant_columns)

        all_param_df.to_csv(f"{str(self.base_output_dir)}/{save_file}", index=False)
        self.logger.info(
            f"Saved all parameters for {len(sim_folders)} simulations to {str(self.base_output_dir)}/all_param_df.csv"
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    case_name = "peak_4"
    parameter_base_folder = f"../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2/POSTERIOR_OUTPUTS/{case_name}/iter_0"
    # input_base_folder = "inputs/sensitivity_analysis/doub_time/"
    input_base_folder = f"../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2/posterior_inputs/{case_name}/iter_0"
    input_folder = parameter_base_folder + "/inputs"
    metrics_calculator = SimulationMetrics(parameter_base_folder, input_base_folder)

    timestamps = [
        "000000",
        "000720",
        "001440",
        "002160",
        "002880",
        "003600",
        "004320",
        "005040",
        "005760",
        "006480",
        "007200",
        "007920",
        "008640",
        "009360",
        "010080",
    ]

    sim_folders = sorted(
        [f for f in Path(input_folder).glob("input_*")],
        key=lambda x: int(re.search(r"input_(\d+)", x.name).group(1)),
    )
    metrics_calculator.analyze_all_simulations(timestamps=timestamps, sim_folders=sim_folders)
    # metrics_calculator.extract_and_save_parameters(sim_folders, PARAMETER_LIST, SOURCE_PARAMS)


def run_single_analysis():
    """Analyze a single hard-coded folder (developer helper; see :func:`main` for batch)."""
    timestamps = [
        "000000",
        "000720",
        "001440",
        "002160",
        "002880",
        "003600",
        "004320",
        "005040",
        "005760",
        "006480",
        "007200",
        "007920",
        "008640",
        "009360",
        "010080",
    ]
    case_name = "peak_4"
    parameter_base_folder = f"../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2/POSTERIOR_OUTPUTS/{case_name}/iter_0"
    input_base_folder = f"../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2/posterior_inputs/{case_name}/iter_0"
    metrics_calculator = SimulationMetrics(parameter_base_folder, input_base_folder)
    folder = Path(
        f"../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2/posterior_inputs/{case_name}/iter_0/input_1"
    )
    metrics_calculator.analyze_simulation(folder, timestamps)
