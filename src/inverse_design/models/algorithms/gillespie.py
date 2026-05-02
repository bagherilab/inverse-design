# Gillespie algorithm for BDM model

import copy
import warnings
from typing import Tuple, List
import numpy as np
from inverse_design.conf.config import BDMConfig
from inverse_design.models.bdm.grid import Grid
from inverse_design.models.bdm.cell import Cell
from inverse_design.models.bdm.location import Location


class Gillespie:
    def __init__(self, config: BDMConfig):
        """
        Initialize Gillespie algorithm for BDM model
        Args:
            config: Configuration object containing rates
        """
        self.config = config
        self.proliferate_rate = config.rates.proliferate
        self.death_rate = config.rates.death
        self.migrate_rate = config.rates.migrate
        self.verbose = config.verbose

    def calculate_propensities(self, grid: Grid) -> Tuple[float, Cell]:
        """
        Calculate propensities for all possible events
        Args:
            grid: Current grid state
        Returns:
            total_propensity: Sum of all event propensities
            action_cell: Cell to perform the action on
        """
        total_propensity = self.death_rate + self.migrate_rate + self.proliferate_rate
        total_propensity *= grid.get_num_cells()
        action_cell = grid.get_random_cell()
        if action_cell is None:
            raise ValueError("No cell to perform the action on")
        return total_propensity, action_cell

    def _get_random_empty_neighbor(self, grid: Grid, location: Location) -> Location | None:
        """
        Get a random empty neighbor from all von Neumann neighbors
        Args:
            grid: Current grid state
            location: Location to get neighbors from
        Returns:
            target_location if found, None otherwise
        """
        all_neighbors = location.get_von_neumann_neighbors()
        if all_neighbors:
            new_i, new_j = all_neighbors[np.random.randint(len(all_neighbors))]
            target_location = grid.get_location(new_i, new_j)
            if not target_location.has_cell():
                return target_location
        return None

    def _move_cell(self, from_location, to_location):
        """
        Move a cell from one location to another
        Args:
            from_location: Source location containing the cell
            to_location: Target location to move the cell to
        """
        cell = from_location.get_cell()
        if cell is None:
            return
        from_location.remove_cell()
        to_location.set_cell(cell)

    def _migrate_cell(self, grid: Grid, location: Location):
        """
        Attempt to migrate a cell to a random neighboring site
        Args:
            grid: Current grid state
            location: Location of the cell to migrate
        """
        neighbor_result = self._get_random_empty_neighbor(grid, location)
        if neighbor_result is not None:
            target_location = neighbor_result
            self._move_cell(location, target_location)

    def _proliferate_cell(self, grid: Grid, location: Location):
        """
        Attempt to create a new cell in a random neighboring site
        Args:
            grid: Current grid state
            location: Location of the parent cell
        """
        neighbor_result = self._get_random_empty_neighbor(grid, location)
        if neighbor_result is not None:
            target_location = neighbor_result
            new_cell = Cell(status="cell", location=target_location)
            target_location.set_cell(new_cell)

    def execute_event(self, grid: Grid, event_type: str, cell: Cell):
        """
        Execute the chosen event
        Args:
            grid: Current grid state
            event_type: Type of event to execute
            cell: Cell to perform the action on
        """
        if event_type == "death":
            cell.get_location().remove_cell()

        elif event_type == "migrate":
            self._migrate_cell(grid, cell.get_location())

        elif event_type == "proliferate":
            self._proliferate_cell(grid, cell.get_location())
        else:
            raise ValueError(f"Invalid event type: {event_type}")

    def run(
        self, grid: Grid, max_time: float = 1.0, frequency: float = 0.1
    ) -> tuple[List[float], List[tuple[str, int, int]]]:
        """
        Run the Gillespie algorithm until max_time
        Args:
            grid: Initial grid state
            max_time: Maximum simulation time
            frequency: How often to print the current time (in time units)
        Returns:
            time_points: List of time points when states were saved
            executed_events: List of events that were executed
            grid_states: List of grid states at each saved time point
        """
        current_time = 0
        next_print_time = frequency  # Initialize next time to print
        next_save_time = frequency  # Initialize next time to save grid state
        time_points = []
        executed_events = []
        grid_states = []

        # Save initial state
        time_points.append(current_time)
        grid_states.append(copy.deepcopy(grid))

        while current_time < max_time:
            time_increment = self.step(grid)
            executed_events.append(self.last_event)
            if time_increment == 0:
                break
            current_time += time_increment

            # Print current time when we pass the next print threshold
            while current_time >= next_print_time:
                if self.verbose:
                    print(f"Current time: {next_print_time:.3f} / {max_time:.3f}")
                next_print_time += frequency

            # Save grid state only when we pass the save threshold
            while current_time >= next_save_time and next_save_time <= max_time:
                time_points.append(current_time)
                grid_states.append(copy.deepcopy(grid))
                next_save_time += frequency

        return time_points, executed_events, grid_states

    def step(self, grid: Grid) -> float:
        """
        Execute one step of the Gillespie algorithm
        Args:
            grid: Current grid state
        Returns:
            time_increment: Time increment for this step, returns 0 if simulation should stop
        """
        # Calculate propensities
        total_propensity, action_cell = self.calculate_propensities(grid)

        if total_propensity == 0:
            return 0

        # Generate random numbers
        r1, r2 = np.random.random(2)
        event_rate = r2 * total_propensity
        # Calculate time increment
        time_increment = -np.log(r1) / total_propensity

        # Choose event
        if event_rate < self.proliferate_rate * grid.get_num_cells():
            event_type = "proliferate"
        elif event_rate < (self.proliferate_rate + self.migrate_rate) * grid.get_num_cells():
            event_type = "migrate"
        else:
            event_type = "death"

        # Store the event for logging
        self.last_event = event_type

        # Execute event
        self.execute_event(grid, event_type, action_cell)

        # Check if all cells have died
        if grid.get_num_cells() == 0:
            warnings.warn("All cells have died", RuntimeWarning)
            return 0

        return time_increment
