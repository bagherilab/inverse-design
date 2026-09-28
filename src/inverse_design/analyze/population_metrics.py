from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import re
import logging
import warnings
import numpy as np
from scipy import stats
from .utils.file_utils import FileParser, list_simulation_files


class PopulationMetrics:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_parser = FileParser()

    @staticmethod
    def calculate_doub_time(n1: float, n2: float, time_difference: float) -> float:
        """Calculate cell population doubling time based on initial and final cell counts

        Args:
            n1: Initial cell count
            n2: Final cell count
            time_difference: Time elapsed between counts (minutes)

        Returns:
            Doubling time in hours. Returns float('inf') if no growth or negative growth.
        """
        if n2 <= n1 or n1 <= 0:
            return float("inf")
        doubling_time = time_difference * np.log(2) / np.log(n2 / n1)
        return doubling_time / 60

    @staticmethod
    def calculate_colony_diameter(
        cells: List[Dict[str, Any]], locations: List[Dict[str, Any]], C: float = 30.0
    ) -> float:
        """Calculate the colony diameter based on cell coordinates.

        Args:
            cells: no used
            locations: List of location dictionaries containing cell coordinates
            C: Scaling factor (default = 30)

        Returns:
            Colony diameter
        """

        if not locations:
            return 0.0

        u_values = [location["coordinate"][0] for location in locations]
        v_values = [location["coordinate"][1] for location in locations]
        w_values = [location["coordinate"][2] for location in locations]

        umax, umin = max(u_values), min(u_values)
        vmax, vmin = max(v_values), min(v_values)
        wmax, wmin = max(w_values), min(w_values)

        diameter = (
            C * (max(umax - umin + 1, 0) + max(vmax - vmin + 1, 0) + max(wmax - wmin + 1, 0)) / 3
        )
        return diameter

    @staticmethod
    def calculate_symmetry(cells: List[Dict[str, Any]], locations: List[Dict[str, Any]]) -> float:
        """Calculate colony symmetry based on hexagonal coordinate system.

        For each occupied location (u,v,w), checks if the 5 corresponding symmetric positions
        are also occupied: (-w,-u,-v), (v,w,u), (-u,-v,-w), (w,u,v), and (-v,-w,-u).
        The symmetry score is calculated as 1 - (average proportion of missing positions).
        Args:
            cells: not used
            locations: List of location dictionaries containing cell coordinates in hex system

        Returns:
            Symmetry score between 0 and 1, where:
            1 = perfect symmetry (all symmetric positions are occupied)
            0 = no symmetry
        """
        if not locations:
            return 0.0

        # Convert locations to set of tuples for efficient lookup
        occupied = {tuple(loc["coordinate"][:3]) for loc in locations}

        # Track unique locations to avoid counting duplicates
        processed = set()
        total_missing = 0
        unique_locations = 0

        for loc in locations:
            u, v, w, _ = loc["coordinate"]
            base_pos = (u, v, w)

            if base_pos in processed:
                continue

            processed.add(base_pos)
            unique_locations += 1

            # Generate the 5 symmetric positions
            symmetric_positions = [(-w, -u, -v), (v, w, u), (-u, -v, -w), (w, u, v), (-v, -w, -u)]

            # Count missing symmetric positions
            missing = sum(1 for pos in symmetric_positions if pos not in occupied)
            total_missing += missing

        if unique_locations == 0:
            return 0.0

        # Calculate symmetry score: 1 - (average proportion of missing positions)
        symmetry = 1 - (total_missing / (5 * unique_locations))
        return max(0.0, min(1.0, symmetry))  # Ensure result is between 0 and 1

    @staticmethod
    def calculate_shannon(cells: List[Dict[str, Any]]) -> float:
        """Calculate Shannon diversity index based on cell states.

        Args:
            cells: List of cell dictionaries

        Returns:
            Shannon diversity index
        """
        if not cells:
            return 0.0

        # Count states
        state_counts = {}
        for cell in cells:
            state = cell["state"]
            state_counts[state] = state_counts.get(state, 0) + 1

        # Calculate Shannon index
        total_cells = len(cells)
        shannon_index = 0.0
        for count in state_counts.values():
            p = count / total_cells
            shannon_index -= p * np.log(p)

        return shannon_index

    @staticmethod
    def calculate_act_ratio(cells: List[Dict[str, Any]]) -> float:
        """Calculate the number of active cells over total number of cells"""
        if not cells:
            return 0.0

        return len(
            [cell for cell in cells if cell["state"] in ["PROLIFERATIVE", "MIGRATORY"]]
        ) / len(cells)

    @staticmethod
    def calculate_activity(cells: List[Dict[str, Any]]) -> float:
        """Calculate the number of active cells over total number of cells"""
        if not cells:
            return 0.0
        n_active = len([cell for cell in cells if cell["state"] in ["PROLIFERATIVE", "MIGRATORY"]])
        n_inactive = len([cell for cell in cells if cell["state"] in ["NECROTIC", "APOPTOTIC"]])
        if n_active + n_inactive == 0:
            return np.nan
        return 2 * n_active / (n_inactive + n_active) - 1

    @staticmethod
    def calculate_n_cells(cells: List[Dict[str, Any]]) -> float:
        """Calculate the number of cells in the colony"""
        if not cells:
            return 0.0

        return len(cells)

    def load_locations_data(
        self, folder_path: Path, timestamp: str
    ) -> List[Tuple[Dict[str, str], List[Dict[str, Any]]]]:
        """Load location data from a specific timestamp for all seeds"""
        results = []
        try:
            location_files = list_simulation_files(folder_path, f"_{timestamp}.LOCATIONS.json")

            for location_file in location_files:
                file_info = self.file_parser.parse_simulation_file(location_file.name, "LOCATIONS")
                if file_info:
                    with open(location_file, "r") as f:
                        location_data = json.load(f)
                        results.append((file_info, location_data))

        except Exception as e:
            self.logger.error(f"Error loading locations data from {folder_path}: {str(e)}")

        return results

    @staticmethod
    def calculate_colony_growth(
        colony_diameters: Dict[int, list[float]],
        timestamps: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """Fit colony diameter growth to linear function (y = mx + c)

        Args:
            colony_diameters: Dictionary of diameters structured as {seed: [diameters]}
            timestamps: List of timepoints in days

        Returns:
            Dictionary with structure:
            {
                "slope": float (um^2/day),
                "slope_std": float,
                "r_value": float
            }
        """
        results = {}
        slopes = []
        r_values = []

        for diameters in colony_diameters.values():
            if len(diameters) != len(timestamps):
                warnings.warn(
                    f"Length of diameters ({len(diameters)}) does not match length of timestamps ({len(timestamps)})"
                )
                shorter_length = min(len(diameters), len(timestamps))
                diameters = diameters[:shorter_length]
                timestamps = timestamps[:shorter_length]

            slope, intercept, r_value, p_value, slope_std = stats.linregress(timestamps, diameters)
            slopes.append(slope)
            r_values.append(r_value)
        results = {"slope": slopes, "r_value": r_values}

        return results
