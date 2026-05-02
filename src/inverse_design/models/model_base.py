from abc import ABC, abstractmethod
from typing import Dict, Any, Set, List, Type
from inverse_design.common.enum import Metric, Target
from inverse_design.conf.config import BDMConfig, ARCADEConfig


class ModelBase(ABC):
    """Base class for all models"""

    @abstractmethod
    def get_available_metrics(self) -> Set[Metric]:
        """Returns set of metrics supported by this model"""
        pass

    @abstractmethod
    def get_parameter_keys(self) -> List[str]:
        """Returns list of parameter names required by this model"""
        pass

    @abstractmethod
    def get_metric_keys(self) -> Dict[str, str]:
        """Returns mapping of display names to internal metric keys"""
        pass

    @abstractmethod
    def get_default_targets(self) -> List[Target]:
        """Returns default target values for this model"""
        pass

    @abstractmethod
    def get_config_class(self):
        """Returns the configuration class for this model"""
        pass


class ModelRegistry:
    """Registry for model types"""

    _models = {}

    @classmethod
    def register(cls, name: str, model_class: Type[ModelBase]):
        """Register a new model type"""
        cls._models[name] = model_class

    @classmethod
    def get_model(cls, name: str) -> ModelBase:
        """Get model by name"""
        if name not in cls._models:
            raise ValueError(
                f"Unknown model type: {name}. Available models: {list(cls._models.keys())}"
            )
        return cls._models[name]()


# Example implementation for BDM
class BDMModel(ModelBase):
    def get_available_metrics(self) -> Set[Metric]:
        return {Metric.DENSITY, Metric.TIME_TO_EQUILIBRIUM}

    def get_parameter_keys(self) -> List[str]:
        return ["proliferate", "death", "migrate"]

    def get_metric_keys(self) -> Dict[str, str]:
        return {"cell_density": "cell_density", "time_to_eq": "time_to_eq"}

    def get_default_targets(self) -> List[Target]:
        return [Target(Metric.DENSITY, 70.0), Target(Metric.TIME_TO_EQUILIBRIUM, 1400.0)]

    def get_config_class(self):
        return BDMConfig


class ARCADEModel(ModelBase):
    def get_available_metrics(self) -> Set[Metric]:
        return {Metric.ACTIVITY}

    def get_parameter_keys(self) -> List[str]:
        return ["proliferate", "death", "migrate"]

    def get_metric_keys(self) -> Dict[str, str]:
        return {"cell_density": "cell_density", "time_to_eq": "time_to_eq"}

    def get_default_targets(self) -> List[Target]:
        return [Target(Metric.DENSITY, 70.0), Target(Metric.TIME_TO_EQUILIBRIUM, 1400.0)]

    def get_config_class(self):
        return ARCADEConfig


# Register built-in models
ModelRegistry.register("BDM", BDMModel)
ModelRegistry.register("ARCADE", ARCADEModel)
