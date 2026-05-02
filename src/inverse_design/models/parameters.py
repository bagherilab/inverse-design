from typing import Dict, Any
from inverse_design.common.enum import Metric
from abc import ABC, abstractmethod


class ModelParameters(ABC):
    """Abstract base class for model-specific parameter handling"""

    @abstractmethod
    def update_config(self, config: Any, params: Dict) -> Any:
        """Update configuration with sampled parameters"""
        pass

    @abstractmethod
    def format_sample_data(
        self, params: Dict, metrics: Dict, distance: float = None, accepted: bool = False
    ) -> Dict:
        """Format parameters and metrics for storage"""
        pass


class BDMParameters(ModelParameters):
    def update_config(self, config: Any, params: Dict) -> Any:
        config.rates.proliferate = params["proliferate"]
        config.rates.death = params["death"]
        config.rates.migrate = params["migrate"]
        return config

    def format_sample_data(
        self, params: Dict, metrics: Dict, distance: float = None, accepted: bool = False
    ) -> Dict:
        return {
            "proliferate": round(params.get("proliferate", None), 3),
            "death": round(params.get("death", None), 3),
            "migrate": round(params.get("migrate", None), 3),
            "cell_density": round(metrics.get(Metric.DENSITY, None), 3),
            "time_to_eq": round(metrics.get(Metric.TIME_TO_EQUILIBRIUM, None), 3),
            "distance": round(distance, 3) if distance is not None else None,
            "accepted": accepted,
        }


class ARCADEParameters(ModelParameters):
    def update_config(self, config: Any, params: Dict) -> Any:
        config.rates.division = params["division"]
        config.rates.death = params["death"]
        config.rates.motility = params["motility"]
        config.rates.adhesion = params["adhesion"]
        return config

    def format_sample_data(
        self, params: Dict, metrics: Dict, distance: float = None, accepted: bool = False
    ) -> Dict:
        formatted_data = {key: value for key, value in params.items()}

        formatted_data.update({key.value: value for key, value in metrics.items()})

        formatted_data.update(
            {
                "distance": distance if distance is not None else None,
                "accepted": accepted,
            }
        )

        return formatted_data


class ParameterFactory:
    @staticmethod
    def create_parameter_handler(model_type: str) -> ModelParameters:
        if model_type == "BDM":
            return BDMParameters()
        elif model_type == "ARCADE":
            return ARCADEParameters()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
