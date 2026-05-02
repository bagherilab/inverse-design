from typing import List, Optional, Set
import numpy as np
from inverse_design.common.enum import Metric


class Metrics:
    """Abstract base class for model metrics calculations"""

    def get_available_metrics(self) -> Set[Metric]:
        """Returns set of metrics available for this model"""
        pass

    def __init__(self, model_output: dict):
        """
        Initialize Metrics calculator
        Args:
            model_output (dict): Output from model
        """
        self._available_metrics = self.get_available_metrics()
        self.model_output = model_output

    def calculate_metric(self, metric: Metric) -> float:
        """Calculate specified metric with given parameters"""

        if metric not in self._available_metrics:
            raise ValueError(f"Metric {metric.value} is not available for this model")

        # Call the appropriate calculation method based on the metric
        method_name = f"calculate_{metric.value}"
        if not hasattr(self, method_name):
            raise NotImplementedError(f"Calculation method for {metric.value} not implemented")

        calculation_method = getattr(self, method_name)
        return calculation_method()


class MetricsBDM(Metrics):
    """Class for calculating various metrics from BDM (Biocellular Dynamic Model) grid states.

    This class implements specific metric calculations for the BDM model, including density
    and time to equilibrium measurements.
    """

    def get_available_metrics(self) -> Set[Metric]:
        """Returns the set of metrics that can be calculated for BDM simulations.

        Returns:
            Set[Metric]: Available metrics for BDM model (density and time to equilibrium)
        """
        metrics = {Metric.DENSITY, Metric.TIME_TO_EQUILIBRIUM}
        return metrics

    def __init__(self, model_output: dict):
        """Initialize MetricsBDM

        Args:
            model_output (dict): Output from model
        """
        self._available_metrics = self.get_available_metrics()
        self.model_output = model_output

    def calculate_density(self, target_time: Optional[float] = None) -> float:
        """Calculate the cell density at a specific time point or at the final state.

        Args:
            target_time (Optional[float]): Time point at which to calculate density.
                If None, uses the final state.

        Returns:
            float: Cell density as a percentage (0-100) of occupied grid points.
        """

        if target_time is None:
            target_time = self.model_output["time_points"][-1]
        target_time_idx = np.argmin(
            np.abs(np.array(self.model_output["time_points"]) - target_time)
        )

        grid = self.model_output["grid_states"][target_time_idx]
        return grid.num_cells / (grid.lattice_size**2) * 100

    def calculate_time_to_equilibrium(self, threshold: float = 0.05) -> float:
        """Calculate the time required for the system to reach equilibrium.

        Equilibrium is determined by measuring density fluctuations over a sliding window.
        The system is considered at equilibrium when relative density fluctuations fall
        below the threshold.

        Args:
            threshold (float): Maximum allowed relative density fluctuation to consider
                the system at equilibrium. Defaults to 0.05 (5%).

        Returns:
            float: Time at which equilibrium was reached, or final time point if
                equilibrium was not reached.
        """
        densities = []
        for i in range(len(self.model_output["grid_states"])):
            density = self.calculate_density(self.model_output["time_points"][i])
            densities.append(density)

        window_size = 5  # Number of time points to check for stability
        for i in range(window_size, len(self.model_output["time_points"])):
            window = densities[i - window_size : i]
            mean_density = np.mean(window)

            # Handle the case where mean_density is 0
            if mean_density == 0:
                fluctuation = np.max(np.abs(np.array(window)))
            else:
                fluctuation = np.max(np.abs(np.array(window) - mean_density) / mean_density)

            if fluctuation < threshold:
                return self.model_output["time_points"][i]

        # If equilibrium is not reached, return the final time point
        return self.model_output["time_points"][-1]


class MetricsARCADE(Metrics):
    """Metrics for ARCADE model outputs (growth rate, symmetry, activity)."""

    def get_available_metrics(self) -> Set[Metric]:
        return {Metric.GROWTH_RATE, Metric.SYMMETRY, Metric.ACTIVITY}

    def __init__(self, model_output: dict):
        super().__init__(model_output)

    def calculate_growth_rate(self) -> float:
        if "growth_rate" not in self.model_output:
            raise KeyError("model_output must include 'growth_rate' for this metric")
        return float(self.model_output["growth_rate"])

    def calculate_symmetry(self) -> float:
        if "symmetry" not in self.model_output:
            raise KeyError("model_output must include 'symmetry' for this metric")
        return float(self.model_output["symmetry"])

    def calculate_activity(self) -> float:
        if "activity" not in self.model_output:
            raise KeyError("model_output must include 'activity' for this metric")
        return float(self.model_output["activity"])


class MetricsFactory:
    """Factory class for creating appropriate metrics calculator based on model type."""

    @staticmethod
    def create_metrics(model_type: str, model_output: dict) -> Metrics:
        """Create and return appropriate metrics calculator for the given model type.

        Args:
            model_type (str): Type of model ('BDM' or 'ARCADE')
            model_output (dict): Output from model

        Returns:
            Metrics: Appropriate metrics calculator instance

        Raises:
            ValueError: If model_type is not recognized
        """
        if model_type == "BDM":
            return MetricsBDM(model_output)
        elif model_type == "ARCADE":
            return MetricsARCADE(model_output)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
