import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from inverse_design.analyze.utils.analyze_utils import analyze_metric_percentiles, remove_outliers
from typing import List, Dict
from inverse_design.analyze.cell_metrics import CellMetrics
from inverse_design.analyze.population_metrics import PopulationMetrics
from inverse_design.config.metrics_config import (
    CELLULAR_METRICS,
    POPULATION_METRICS,
    SUMMARY_METRICS,
)


class SeedAnalyzer:
    def __init__(self, base_output_dir: str):
        """Initialize the seed analyzer.

        Args:
            base_output_dir: Base directory containing all simulation output folders
        """
        self.base_output_dir = Path(base_output_dir)
        self.input_folder = Path(base_output_dir + "/inputs")
        self.cell_metrics = CellMetrics()
        self.population_metrics = PopulationMetrics()

    def filter_valid_metrics(self, metrics_list):
        """Filter out seeds with NaN or inf values from a list of metrics"""
        valid_indices = []
        for i in range(len(metrics_list[0])):
            has_invalid = any(np.isnan(metric[i]) or np.isinf(metric[i]) for metric in metrics_list)
            if not has_invalid:
                valid_indices.append(i)
        return valid_indices, [[metric[i] for i in valid_indices] for metric in metrics_list]

    def _calculate_seed_metrics_helper(
        self, cells: Dict, locations: Dict, metrics_dict: Dict[str, List]
    ) -> Dict[str, List]:
        """Calculate metrics for a single seed and append to metrics dictionary.

        Args:
            cells: Cell data for a single seed
            locations: Location data for a single seed
            metrics_dict: Dictionary to store metrics
        """
        # Calculate cellular metrics
        for metric, config in CELLULAR_METRICS.items():
            value = getattr(self.cell_metrics, f"calculate_{metric}")(cells)
            metrics_dict[metric].append(value)

        # Calculate population metrics
        for metric, config in POPULATION_METRICS.items():
            if config["spatial"]:
                value = getattr(self.population_metrics, f"calculate_{metric}")(cells, locations)
            else:
                value = getattr(self.population_metrics, f"calculate_{metric}")(cells)
            metrics_dict[metric].append(value)
        return metrics_dict

    def calculate_seed_metrics(self, folder_path, time_point) -> Dict[str, List]:
        """Calculate metrics for a single seed and append to metrics dictionary.

        Args:
            cells: Cell data for a single seed
            locations: Location data for a single seed
            metrics_dict: Dictionary to store metrics
        """
        cells_data = self.cell_metrics.load_cells_data(folder_path, time_point)
        locations_data = self.population_metrics.load_locations_data(folder_path, time_point)
        metrics_dict = {
            metric: [] for metric in CELLULAR_METRICS.keys() | POPULATION_METRICS.keys()
        }
        for (_, cells), (_, locations) in zip(cells_data, locations_data):
            metrics_dict = self._calculate_seed_metrics_helper(cells, locations, metrics_dict)
        return metrics_dict

    def process_parameter_seeds(self, folder_path, timestamps: List[str]):
        """
        Process metrics across different seeds for a single parameter set.

        Args:
            folder_path: Path to the simulation folder
            timestamps: List of timestamps to analyze

        Returns:
            Dictionary containing metrics for each experiment group and name
        """
        metrics_timestamps = {}
        for time_point in timestamps:
            metrics_timestamps[time_point] = self.calculate_seed_metrics(folder_path, time_point)

        n_cells_t1_seed_list = metrics_timestamps[timestamps[0]]["n_cells"]
        n_cells_t2_seed_list = metrics_timestamps[timestamps[-1]]["n_cells"]
        time_difference = int(timestamps[-1]) - int(timestamps[0])

        colony_diameters_over_time = self.collect_colony_diameters_over_time(metrics_timestamps)
        timestamps_days = [int(timestamp) / 60 / 24 for timestamp in timestamps]
        colony_growth_rates_results = self.population_metrics.calculate_colony_growth(
            colony_diameters_over_time, timestamps_days
        )

        doub_times_seed = []
        for n1, n2 in zip(n_cells_t1_seed_list, n_cells_t2_seed_list):
            doub_time = self.population_metrics.calculate_doub_time(n1, n2, time_difference)
            doub_times_seed.append(doub_time)

        for time_point, metrics in metrics_timestamps.items():
            # Filter invalid metrics
            for key, value in metrics.items():
                # Get all metrics except seed_count and states
                metrics_to_check = []
                metrics_keys = []
                for key, value in metrics.items():
                    if key != "states" and isinstance(value, list):
                        metrics_to_check.append(value)
                        metrics_keys.append(key)

            valid_indices, filtered_metrics = self.filter_valid_metrics(metrics_to_check)

            # Update all metrics with filtered values
            for idx, key in enumerate(metrics_keys):
                metrics_timestamps[time_point][key] = filtered_metrics[idx]

            metrics_timestamps[time_point]["seed_count"] = len(valid_indices)
        final_metrics = metrics_timestamps[timestamps[-1]]

        final_metrics["doub_time"] = doub_times_seed
        final_metrics["colony_growth"] = colony_growth_rates_results["slope"]
        final_metrics["colony_growth_r"] = colony_growth_rates_results["r_value"]

        return final_metrics

    def collect_colony_diameters_over_time(
        self, metrics_by_timestamp_seed: Dict[str, Dict[str, List]]
    ) -> Dict[int, List]:
        """Collect colony diameters for each seed across all timestamps.

        Args:
            metrics_by_timestamp_seed: Dictionary mapping timestamps to metrics dictionaries

        Returns:
            Dictionary mapping seed indices to lists of colony diameters over time
        """
        n_seeds = len(list(metrics_by_timestamp_seed.values())[0]["colony_diameter"])
        colony_diameters_over_time = {i: [] for i in range(n_seeds)}
        for _, metrics_seed in metrics_by_timestamp_seed.items():
            for seed_idx, seed_metrics in enumerate(metrics_seed["colony_diameter"]):
                colony_diameters_over_time[seed_idx].append(seed_metrics)

        return colony_diameters_over_time

    def analyze_seeds_for_folders(self, input_folders, timestamps: List[str]):
        """Analyze metrics across seeds for given input folders.

        Args:
            input_folders: List of input folder names to analyze
            timestamps: List of timestamps to analyze

        Returns:
            Dictionary containing metrics for each folder's seeds
        """
        folder_metrics_seed = {}

        for folder_name in input_folders:
            folder_path = self.input_folder / folder_name

            final_metrics = self.process_parameter_seeds(folder_path, timestamps)
            folder_metrics_seed[folder_name] = final_metrics
        return folder_metrics_seed

    def plot_seed_comparisons(
        self,
        csv_file: str,
        metric_name: str,
        percentile: float = 10,
        timestamps: List[str] = None,
        metrics_to_plot: List[str] = None,
        save_file: str = None,
        remove_outliers_flag: bool = True,
    ):
        """Plot violin plots comparing seed distributions for top and bottom performers.

        Args:
            csv_file: Path to aggregated metrics CSV file
            metric_name: Name of metric to analyze
            percentile: Percentile for top/bottom selection
            timestamps: List of timestamps for analysis
            metrics_to_plot: List of metrics to plot
            save_file: Path to save the plot (if None, display plot)
            remove_outliers_flag: Whether to remove outliers from the data
        """
        # Get top and bottom folders
        top_n_input_file, bottom_n_input_file, _ = analyze_metric_percentiles(
            csv_file, metric_name, percentile, verbose=False
        )
        # Get seed metrics for each group
        top_folder_metrics_seed = self.analyze_seeds_for_folders(top_n_input_file, timestamps)
        bottom_folder_metrics_seed = self.analyze_seeds_for_folders(bottom_n_input_file, timestamps)

        # Create plot data from seed metrics
        plot_data = []
        for group, folder_metrics_seed in [
            ("top", top_folder_metrics_seed),
            ("bottom", bottom_folder_metrics_seed),
        ]:
            for folder, metrics_seed in folder_metrics_seed.items():
                for metric in metrics_to_plot:
                    values = metrics_seed[metric]
                    print(f"values: {values}")
                    for value in values:
                        plot_data.append(
                            {"metric": metric, "value": value, "group": group, "folder": folder}
                        )

        original_plot_df = pd.DataFrame(plot_data)
        if remove_outliers_flag:
            cleaned_plot_data = []
            for metric in metrics_to_plot:
                metric_data = original_plot_df[original_plot_df["metric"] == metric].copy()
                cleaned_data, _, _ = remove_outliers(metric_data, verbose=True)
                cleaned_plot_data.append(cleaned_data)
            print(f"number of outliers removed: {len(original_plot_df) - len(cleaned_plot_data)}")
            plot_df = pd.concat(cleaned_plot_data, axis=0)
        else:
            plot_df = original_plot_df

        print(f"number of top {percentile}% folders: {len(top_n_input_file)}")
        print(f"number of bottom {percentile}% folders: {len(bottom_n_input_file)}")

        # Create plots
        plt.figure(figsize=(4 * len(metrics_to_plot), 8))

        for idx, metric in enumerate(metrics_to_plot):
            plt.subplot(1, len(metrics_to_plot), idx + 1)
            metric_data = plot_df[plot_df["metric"] == metric]

            # Create violin plot
            sns.violinplot(data=metric_data, x="group", y="value", order=["top", "bottom"])

            # Add individual points
            sns.stripplot(
                data=metric_data,
                x="group",
                y="value",
                order=["top", "bottom"],
                color="black",
                alpha=0.8,
                size=3,
                jitter=0.2,
            )

            plt.title(f"{metric} Distribution")
            plt.xticks(rotation=45)

            # Add statistics
            top_vals = metric_data[metric_data["group"] == "top"]["value"]
            bottom_vals = metric_data[metric_data["group"] == "bottom"]["value"]

            stats_text = (
                f"Top {percentile}%:\nMean: {top_vals.mean():.2f}\nStd: {top_vals.std():.2f}\n\n"
            )
            stats_text += f"Bottom {percentile}%:\nMean: {bottom_vals.mean():.2f}\nStd: {bottom_vals.std():.2f}"

            plt.text(
                0.95,
                0.95,
                stats_text,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        if save_file is not None:
            plt.savefig(save_file)
        else:
            plt.show()
