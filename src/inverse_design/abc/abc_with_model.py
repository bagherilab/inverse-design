import logging
from typing import Dict, Optional
import scipy
from inverse_design.abc.abc_base import ABCBase
from inverse_design.models.models import ModelFactory
from inverse_design.metrics.metrics import MetricsFactory


class ABCWithModel(ABCBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize Sobol sampling
        self.sobol_power = self.abc_config.sobol_power
        self.parameter_ranges = self.abc_config.parameter_ranges

        self.sobol_engine = scipy.stats.qmc.Sobol(
            d=len(self.parameter_ranges),
            scramble=True,
        )
        self.samples = self.sobol_engine.random_base2(m=self.sobol_power)
        self.num_samples = len(self.samples)
        self.current_sample_idx = 0

    def sample_parameters(self) -> Dict:
        """Sample parameters using Sobol sequence for better space-filling properties
        Returns:
            Dictionary of parameter names and their sampled values
        """
        if self.current_sample_idx >= len(self.samples):
            raise ValueError("All samples have been used. Initialize a new ABC instance if needed.")

        sample = self.samples[self.current_sample_idx]
        self.current_sample_idx += 1

        params = {}
        for (param_name, ranges), value in zip(self.parameter_ranges.items(), sample):
            min_val = ranges.min
            max_val = ranges.max
            scaled_value = min_val + (max_val - min_val) * value
            params[param_name] = scaled_value

        return params

    def run_inference(self):
        """Run ABC inference with model simulations"""
        # Create model once
        model = ModelFactory.create_model(self.model_type, self.model_config)

        # First phase: Calculate all metrics
        all_samples = []
        print("Phase 1: Calculating metrics for all samples...")

        for i in range(self.num_samples):
            params = self.sample_parameters()
            config = self.model_config.copy()
            config = self.parameter_handler.update_config(config, params)

            model.update_config(config)
            model_output = model.step()

            # Calculate metrics
            metrics_calculator = MetricsFactory.create_metrics(self.model_type, model_output)
            metrics = self.calculate_all_metrics(metrics_calculator)

            # Store parameters and metrics
            all_samples.append({"params": params, "metrics": metrics})

            if i % self.output_frequency == 0:
                print(f"Completed {i}/{self.num_samples} samples")

        # Second phase: Calculate normalization factors if not provided
        print("\nPhase 2: Computing normalization factors...")
        if self.normalization_factors is None:
            # Collect all values for each metric
            metric_values = {
                metric.value: [sample["metrics"][metric] for sample in all_samples]
                for metric in next(iter(all_samples))["metrics"].keys()
            }

            # Calculate ranges for normalization
            self.normalization_factors = {
                metric_name: max(values) - min(values) if len(values) > 0 else 1.0
                for metric_name, values in metric_values.items()
            }
            print(f"Computed normalization factors: {self.normalization_factors}")

        # Third phase: Calculate distances and format results
        print("\nPhase 3: Computing distances and formatting results...")
        self.param_metrics_distances_results = []

        for sample in all_samples:
            # Calculate distance using normalized metrics
            distance = self.calculate_distance(sample["metrics"])

            # Format and store results
            sample_data = self.parameter_handler.format_sample_data(
                sample["params"], sample["metrics"], distance, distance <= self.epsilon
            )
            self.param_metrics_distances_results.append(sample_data)

        print("\nABC inference completed!")
        return self.param_metrics_distances_results
