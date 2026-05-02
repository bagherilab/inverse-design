import logging
from typing import Dict
import pandas as pd
import numpy as np
from inverse_design.abc.abc_base import ABCBase


class ABCPrecomputed(ABCBase):
    def __init__(self, *args, param_file: str, metrics_file: str, **kwargs):
        """
        Initialize ABC for pre-computed results
        Args:
            para_file: Path to file containing model parameters
            metrics_file: Path to file containing model metrics
            *args, **kwargs: Arguments passed to ABCBase
        """
        super().__init__(*args, **kwargs)
        self.log = logging.getLogger(__name__)
        self.param_file = param_file
        self.metrics_file = metrics_file
        self.param_df = pd.read_csv(param_file)
        self.metrics_df = pd.read_csv(metrics_file)

        # Check lengths of param_df and metrics_df
        if len(self.param_df) != len(self.metrics_df):
            self.log.warning(
                "Length mismatch: param_df has %d samples, metrics_df has %d samples",
                len(self.param_df),
                len(self.metrics_df),
            )
            min_length = min(len(self.param_df), len(self.metrics_df))
            self.param_df = self.param_df.iloc[:min_length]
            self.metrics_df = self.metrics_df.iloc[:min_length]

        # Check if input_folder columns exist and verify order consistency
        if "input_folder" in self.param_df.columns and "input_folder" in self.metrics_df.columns:
            param_folders = self.param_df["input_folder"].values
            metrics_folders = self.metrics_df["input_folder"].values
            if not np.array_equal(param_folders, metrics_folders):
                self.log.warning("Input folder order mismatch between param_df and metrics_df")
                # Sort both DataFrames by input_folder to ensure consistency
                self.param_df = self.param_df.sort_values(
                    "input_folder",
                    key=lambda x: x.str.extract("input_(\d+)").iloc[:, 0].astype(int),
                )
                self.metrics_df = self.metrics_df.sort_values(
                    "input_folder",
                    key=lambda x: x.str.extract("input_(\d+)").iloc[:, 0].astype(int),
                )
                self.log.info("DataFrames have been sorted by input_folder")
        self.num_samples = len(self.param_df)
        # Calculate dynamic normalization factors for metrics not in static factors
        self._calculate_dynamic_normalization_factors()

    def _calculate_dynamic_normalization_factors(self):
        """Calculate normalization factors based on metrics range for undefined metrics"""
        for target in self.targets:
            if target.metric.value not in self.normalization_factors:
                metric_values = self.metrics_df[target.metric.value]
                metric_range = metric_values.max() - metric_values.min()
                if metric_range > 0:
                    self.dynamic_normalization_factors[target.metric.value] = metric_range
                else:
                    self.log.warning(
                        f"Zero range for metric {target.metric.value}, using 1.0 as normalization factor"
                    )
                    self.dynamic_normalization_factors[target.metric.value] = 1.0

    def run_inference(
        self,
    ) -> Dict:
        """Run ABC inference on pre-computed results"""
        target_str = ", ".join([f"{t.metric.value}: {t.value}" for t in self.targets])
        self.log.info(
            f"Starting ABC inference on {self.num_samples} pre-computed samples for targets: {target_str}"
        )

        for i, row in self.param_df.iterrows():
            # Drop non-numeric columns
            numeric_cols = row.index[row.apply(lambda x: isinstance(x, (int, float)))]
            params = row[numeric_cols].to_dict()

            # Convert metrics Series to dict, only including target metrics
            metrics_row = self.metrics_df.iloc[i]

            metrics = {target.metric: metrics_row[target.metric.value] for target in self.targets}

            distance = self.calculate_distance(metrics)
            accepted = distance < self.epsilon
            sample_data = self.parameter_handler.format_sample_data(
                params, metrics, distance=distance, accepted=accepted
            )
            self.param_metrics_distances_results.append(sample_data)

            if (i + 1) % self.output_frequency == 0:
                accepted_count = sum(
                    1 for sample in self.param_metrics_distances_results if sample["accepted"]
                )
                self.log.info(f"Processed {i + 1} samples, accepted {accepted_count}")
        accepted_count = sum(
            1 for sample in self.param_metrics_distances_results if sample["accepted"]
        )
        self.log.info(
            f"Finished ABC inference on {self.num_samples} pre-computed samples for targets: {target_str}"
        )
        self.log.info(f"Accepted {accepted_count} samples")
        return self.param_metrics_distances_results
