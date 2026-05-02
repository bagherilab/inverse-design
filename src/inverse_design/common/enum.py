from enum import Enum
from dataclasses import dataclass


class Metric:
    """
    Dynamic Metric class with default metrics and support for custom metrics from file
    """

    _metrics = {}  # Dictionary to store all metric instances

    # Define default metrics as class attributes
    DENSITY = None
    TIME_TO_EQUILIBRIUM = None
    GROWTH_RATE = None
    SYMMETRY = None
    ACTIVITY = None

    def __init__(self, value):
        self.value = value
        self.name = value.upper()
        Metric._metrics[value] = self

    @classmethod
    def _initialize_default_metrics(cls):
        """Initialize the default metrics if they haven't been created yet"""
        if cls.DENSITY is None:
            # Only initialize if we're not in Custom mode
            if len(cls._metrics) == 0:
                cls.DENSITY = cls("density")
                cls.TIME_TO_EQUILIBRIUM = cls("time_to_equilibrium")
                cls.GROWTH_RATE = cls("growth_rate")
                cls.SYMMETRY = cls("symmetry")
                cls.ACTIVITY = cls("activity")

    @classmethod
    def load_from_file(cls, file_path):
        """Load metrics from a CSV file, clearing all existing metrics first"""
        import pandas as pd

        try:
            # Clear all existing metrics
            cls._metrics.clear()

            # Load and process the metrics file
            metrics_df = pd.read_csv(file_path)
            numerical_columns = metrics_df.select_dtypes(include=["number"]).columns

            # Add new metrics (skip columns ending with '_std' and non-numeric columns)
            for col in numerical_columns:
                if not col.endswith("_std"):
                    cls(col)  # Create new metric

        except Exception as e:
            print(f"Error loading metrics from file: {e}")

    @classmethod
    def get_all_metrics(cls):
        """Return all available metrics"""
        return list(cls._metrics.values())

    @classmethod
    def get_default_bdm_metrics(cls):
        """Get default metrics for BDM model"""
        cls._initialize_default_metrics()
        return [cls.DENSITY, cls.TIME_TO_EQUILIBRIUM]

    @classmethod
    def get_default_arcade_metrics(cls):
        """Get default metrics for ARCADE model"""
        cls._initialize_default_metrics()
        return [cls.GROWTH_RATE, cls.SYMMETRY, cls.ACTIVITY]

    def __eq__(self, other):
        if isinstance(other, Metric):
            return self.value == other.value
        return False

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return self.value

    @classmethod
    def get(cls, value):
        """Get a metric by value, create if it doesn't exist"""
        cls._initialize_default_metrics()
        return cls._metrics.get(value) or cls(value)


@dataclass
class Target:
    """
    Target class for multi-objective optimization
    Attributes:
        metric: The metric to target (density, time to equilibrium, etc.)
        value: The desired value for this metric
        weight: Weight for multi-objective optimization (default: 1.0)
    """

    metric: Metric
    value: float
    weight: float = 1.0
