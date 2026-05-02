from typing import Dict, List, Union
import pandas as pd
from abc import ABC
from inverse_design.conf.config import BDMConfig, ABCConfig, ARCADEConfig
from inverse_design.common.enum import Metric, Target
from inverse_design.utils.utils import get_samples_data
from inverse_design.models.parameters import ParameterFactory


class ABCBase(ABC):
    def __init__(
        self,
        model_config: Union[BDMConfig, ARCADEConfig],
        abc_config: ABCConfig,
        targets: Union[Target, List[Target]],
    ):
        """Common initialization for ABC classes"""
        self.model_config = model_config
        self.abc_config = abc_config
        self.targets = [targets] if isinstance(targets, Target) else targets

        # Get ABC parameters from config
        self.epsilon = abc_config.epsilon
        self.output_frequency = abc_config.output_frequency
        self.model_type = abc_config.model_type
        self.max_time = self.model_config.output.max_time
        self.parameter_handler = ParameterFactory.create_parameter_handler(self.model_type)
        self.param_metrics_distances_results = []
        self.normalization_factors = self._get_normalization_factors()
        self.dynamic_normalization_factors = {}

    def _get_normalization_factors(self) -> Dict[str, float]:
        """Get normalization factors for distance calculation based on model type and targets"""
        if self.model_type == "BDM":
            return {
                Metric.get("density").value: 100.0,
                Metric.get("time_to_equilibrium").value: self.max_time,
            }
        elif self.model_type == "ARCADE":
            return {
                Metric.get("growth_rate").value: 1.0,
                Metric.get("symmetry").value: 1.0,
                Metric.get("activity").value: 1.0,
            }
        else:
            return None

    def calculate_distance(self, metrics: Dict[Metric, float]) -> float:
        """Calculate weighted distance between calculated metrics and targets"""
        total_distance = 0.0
        total_weight = sum(target.weight for target in self.targets)

        for target in self.targets:
            value = metrics[target.metric]
            norm_factor = self.normalization_factors.get(
                target.metric.value,
                self.dynamic_normalization_factors.get(target.metric.value, 1.0),
            )
            normalized_distance = abs(value - target.value) / norm_factor
            total_distance += (normalized_distance * target.weight) / total_weight

        return total_distance

    def calculate_all_metrics(self, metrics_calculator=None) -> Dict[Metric, float]:
        """Calculate all required metrics from grid states
        Args:
            metrics_calculator: Instance of Metrics class to calculate metrics
        Returns:
            Dictionary mapping metrics to their calculated values
        """
        if metrics_calculator is None:
            raise ValueError("Metrics calculator must be provided")

        metrics = {}
        for target in self.targets:
            try:
                metrics[target.metric] = metrics_calculator.calculate_metric(target.metric)
            except ValueError as e:
                raise ValueError(
                    f"Cannot calculate {target.metric.value} for model type {self.model_type}: {str(e)}"
                )

        return metrics

    def save_results(self, save_path: str = "all_samples_metrics.csv") -> pd.DataFrame:
        """Save all samples data to a CSV file

        Args:
            save_path: Path where to save the CSV file

        Returns:
            pandas DataFrame containing all samples data
        """
        return get_samples_data(
            param_metrics_distances_results=self.param_metrics_distances_results,
            model_type=self.model_type,
            save_path=save_path,
        )

    def update_targets(self, new_targets: Union[Target, List[Target]]):
        """Update targets and recalculate normalization factors

        Args:
            new_targets: New target or list of targets to use
        """
        self.targets = [new_targets] if isinstance(new_targets, Target) else new_targets

        self.param_metrics_distances_results = []
        if hasattr(self, "dynamic_normalization_factors"):
            self.dynamic_normalization_factors = {}
            if hasattr(self, "_calculate_dynamic_normalization_factors"):
                self._calculate_dynamic_normalization_factors()

        target_str = ", ".join([f"{t.metric.value}: {t.value}" for t in self.targets])
        print(f"Updated targets to: {target_str}")
