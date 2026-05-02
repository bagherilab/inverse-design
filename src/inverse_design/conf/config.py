# config file for the BDM model
from dataclasses import dataclass
from typing import Any, Dict
import hydra
from omegaconf import DictConfig, OmegaConf
import copy


@dataclass
class LatticeConfig:
    # These must now be specified in config.yaml
    size: int = 30  # Default value specified
    initial_density: float = 0.05  # Default value specified

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class RatesConfig:
    # These must now be specified in config.yaml
    proliferate: float = 0.01  # Default value specified
    death: float = 0.0025  # Default value specified
    migrate: float = 0.1  # Default value specified

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class OutputConfig:
    frequency: float = 100.0  # Default value specified
    max_time: float = 2500.0  # Default value specified

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class MetricsConfig:
    equilibrium_threshold: float = 0.05

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class BDMConfig:
    lattice: LatticeConfig
    rates: RatesConfig
    output: OutputConfig
    metrics: MetricsConfig
    verbose: bool

    def copy(self):
        return BDMConfig(
            lattice=self.lattice.copy(),
            rates=self.rates.copy(),
            output=self.output.copy(),
            metrics=self.metrics.copy(),
            verbose=self.verbose,
        )

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> "BDMConfig":
        return cls(
            lattice=LatticeConfig(**cfg.lattice),
            rates=RatesConfig(**cfg.rates),
            output=OutputConfig(**cfg.output),
            metrics=MetricsConfig(**cfg.metrics),
            verbose=cfg.verbose,
        )


@dataclass
class CellularConfig:
    volume_mu: float
    volume_sigma: float
    apop_age_mu: float
    apop_age_sigma: float
    necrotic_fraction: float
    accuracy: float
    affinity: float
    compression_tolerance: float

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class ARCADEConfig:
    cellular: CellularConfig
    output: OutputConfig

    def copy(self):
        return ARCADEConfig(
            cellular=self.cellular.copy(),
            output=self.output.copy(),
        )

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> "ARCADEConfig":
        return cls(
            cellular=CellularConfig(**cfg.cellular),
            output=OutputConfig(**cfg.output),
        )


@dataclass
class ParameterRange:
    min: float
    max: float


@dataclass
class ABCConfig:
    sobol_power: int
    epsilon: float
    parameter_ranges: Dict[str, ParameterRange]
    output_frequency: int
    model_type: str

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> "ABCConfig":
        parameter_ranges = {
            name: ParameterRange(**ranges) for name, ranges in cfg.parameter_ranges.items()
        }
        return cls(
            epsilon=cfg.epsilon,
            sobol_power=cfg.sobol_power,
            parameter_ranges=parameter_ranges,
            output_frequency=cfg.output_frequency,
            model_type=cfg.model_type,
        )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="bdm_config", node=BDMConfig)
cs.store(name="arcade_config", node=ARCADEConfig)
cs.store(name="abc_config", node=ABCConfig)


@hydra.main(version_base=None, config_name="bdm_config")
def get_config(cfg: DictConfig) -> Any:
    return OmegaConf.to_container(cfg, resolve=True)


if __name__ == "__main__":
    get_config()
