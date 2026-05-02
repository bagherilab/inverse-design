from inverse_design.conf.config import BDMConfig, ARCADEConfig
from inverse_design.models.bdm.bdm import BDM
from inverse_design.models.arcade.arcade import ARCADE
from typing import Union


class ModelFactory:
    """Factory class for creating model instances"""

    _model_instances = {}  # Cache for model instances

    @classmethod
    def create_model(cls, model_type: str, config: Union[BDMConfig, ARCADEConfig]):
        """Create or get cached model instance and update its configuration"""
        if model_type not in cls._model_instances:
            if model_type == "BDM":
                cls._model_instances[model_type] = BDM(config)
            elif model_type == "ARCADE":
                cls._model_instances[model_type] = ARCADE(config)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        else:
            # Update existing model's configuration
            cls._model_instances[model_type].update_config(config)

        return cls._model_instances[model_type]
