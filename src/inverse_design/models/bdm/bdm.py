# Implementation of a birth-death-migration model using the Gillespie algorithm

import hydra
import copy
import time
import os
import logging
from omegaconf import DictConfig, OmegaConf
from typing import List
import numpy as np

from inverse_design.models.algorithms.gillespie import Gillespie
from inverse_design.vis.vis import plot_grid, plot_cell_density, plot_combined_grid_and_density
from inverse_design.conf.config import BDMConfig
from inverse_design.models.bdm.cell import Cell
from inverse_design.models.bdm.grid import Grid


class BDM:
    def __init__(self, config: BDMConfig):
        """
        Initialize BDM model using Hydra config
        Args:
            config: Hydra configuration object
        """
        self.lattice_size = config.lattice.size
        self.initial_density = config.lattice.initial_density
        self.proliferate_rate = config.rates.proliferate
        self.death_rate = config.rates.death
        self.migrate_rate = config.rates.migrate
        self.output_frequency = config.output.frequency
        self.max_time = config.output.max_time
        self.verbose = config.verbose
        self.random_seed = 42
        self.current_time = 0.0
        self.grid = self.initialize()
        self.gillespie = Gillespie(config)
        if self.verbose:
            self.log_config(config)

    def log_config(self, config: BDMConfig):
        """
        Log model parameters and save configuration file
        Args:
            config: Hydra configuration object
        """
        log = logging.getLogger(__name__)

        # Log model parameters only
        log.info("Model Parameters:")
        log.info(f"- Lattice size: {self.lattice_size}")
        log.info(f"- Initial density: {self.initial_density}")
        log.info(f"- Proliferation rate: {self.proliferate_rate}")
        log.info(f"- Death rate: {self.death_rate}")
        log.info(f"- Migration rate: {self.migrate_rate}")

        # Save config to YAML file in Hydra's output directory
        if hydra.core.hydra_config.HydraConfig.initialized():
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            config_path = os.path.join(output_dir, "config_log.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config=config, f=f)
            log.info(f"Configuration saved to {config_path}")

    def initialize(self):
        """
        Initialize lattice with agents randomly placed according to initial_density
        Returns: Grid object with agents placed
        """
        # Create grid with empty cells
        grid = Grid(self.lattice_size)

        # Calculate number of agents to place
        total_sites = self.lattice_size * self.lattice_size
        num_agents = int(self.initial_density * total_sites)

        # Randomly select positions for agents
        flat_indices = np.random.choice(total_sites, size=num_agents, replace=False)
        row_indices = flat_indices // self.lattice_size
        col_indices = flat_indices % self.lattice_size

        # Place agents at selected positions
        for i, j in zip(row_indices, col_indices):
            location = grid.get_location(i, j)
            cell = Cell(status="cell", location=location)
            location.set_cell(cell)

        self.original_grid = copy.deepcopy(grid)
        return grid

    def get_grid(self):
        """Return the current grid state"""
        return self.grid

    def step(self) -> tuple[List[float], List[tuple[str, int, int]], List[Grid]]:
        """
        Step the simulation forward by a specified time duration using Gillespie algorithm
        Args:
            max_time: Duration to simulate (default: 1.0)
        Returns:
            time_points: List of time points when events occurred
            events: List of (event_type, i, j) tuples describing what happened
            grid_states: List of grids at each time point
        """
        time_points, events, grid_states = self.gillespie.run(
            self.grid, self.max_time, frequency=self.output_frequency
        )
        self.current_time += max(time_points)
        model_output = {
            "time_points": time_points,
            "events": events,
            "grid_states": grid_states,
        }
        return model_output

    def evaluate(self, max_time: float = 2500.0) -> float:
        """
        Evaluate the model by running a simulation and returning the final cell density
        Args:
            max_time: Duration to simulate (default: 2500.0)
        Returns:
            final_density: Final cell density as a percentage
        """
        _, _, grid_states = self.step(max_time)
        final_density = grid_states[-1].num_cells / (self.lattice_size**2) * 100
        return final_density

    def update_config(self, config: BDMConfig):
        """Update the model configuration"""
        self.lattice_size = config.lattice.size
        self.initial_density = config.lattice.initial_density
        self.proliferate_rate = config.rates.proliferate
        self.death_rate = config.rates.death
        self.migrate_rate = config.rates.migrate


@hydra.main(version_base=None, config_path="conf/bdm", config_name="default")
def main(cfg: DictConfig) -> None:
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(
                    hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                    "bdm.log",
                )
            ),
            logging.StreamHandler(),
        ],
    )

    bdm = BDM(cfg)
    log = logging.getLogger(__name__)
    target_time_point = 1200.0
    target_density = 80.0
    # Run complete simulation
    log.info("Starting simulation...")
    start_time = time.time()
    print("Starting model step...")
    time_points, events, grid_states = bdm.step()
    end_time = time.time()
    print(f"Model step completed in {end_time - start_time:.2f} seconds")
    cell_densities = [grid.num_cells / (bdm.lattice_size**2) * 100 for grid in grid_states]

    if 1 == 0:
        plot_grid(bdm.original_grid, 0)
        for time_point, grid in zip(
            time_points[1:], grid_states[1:]
        ):  # Skip first since we already plotted original
            plot_grid(grid, np.round(time_point, 2))

        for i in range(len(time_points)):
            plot_cell_density(time_points, cell_densities, time_points[i], cell_densities[i])

    if 1 == 1:
        for red_dot_time_point, grid in zip(time_points, grid_states):
            plot_combined_grid_and_density(
                grid,
                time_points,
                cell_densities,
                red_dot_time_point,
                target_density,
                target_time_point,
            )

    log.info(f"Simulation completed with {len(events)} total events")


if __name__ == "__main__":
    """
    Example usage:
    python bdm.py
    python bdm.py --config=experiments/exp1_config.yaml
    python bdm.py lattice.size=200 rates.proliferate=2.0
    """
    main()
