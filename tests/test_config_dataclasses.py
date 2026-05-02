"""Tests for Hydra/OmegaConf dataclass construction (conf.config)."""

from omegaconf import OmegaConf

from inverse_design.conf.config import ABCConfig, BDMConfig, ParameterRange


def test_abc_config_from_dictconfig():
    cfg = OmegaConf.create(
        {
            "epsilon": 0.05,
            "sobol_power": 9,
            "output_frequency": 100,
            "model_type": "BDM",
            "parameter_ranges": {
                "proliferate": {"min": 0.001, "max": 0.02},
                "death": {"min": 0.0001, "max": 0.01},
            },
        }
    )
    abc = ABCConfig.from_dictconfig(cfg)
    assert abc.epsilon == 0.05
    assert abc.sobol_power == 9
    assert abc.model_type == "BDM"
    assert isinstance(abc.parameter_ranges["proliferate"], ParameterRange)
    assert abc.parameter_ranges["proliferate"].min == 0.001


def test_bdm_config_from_dictconfig():
    cfg = OmegaConf.create(
        {
            "lattice": {"size": 20, "initial_density": 0.1},
            "rates": {"proliferate": 0.01, "death": 0.0025, "migrate": 0.1},
            "output": {"frequency": 100.0, "max_time": 2000.0},
            "metrics": {"equilibrium_threshold": 0.05},
            "verbose": False,
        }
    )
    bdm = BDMConfig.from_dictconfig(cfg)
    assert bdm.lattice.size == 20
    assert bdm.rates.death == 0.0025
    assert bdm.verbose is False
