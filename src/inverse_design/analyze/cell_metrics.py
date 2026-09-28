from typing import List, Dict, Any, Tuple
import math
import numpy as np
import json
import re
import logging
from pathlib import Path
from .utils.file_utils import FileParser, list_simulation_files


class CellMetrics:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_parser = FileParser()

    @staticmethod
    def calculate_vol(cells: List[Dict[str, Any]]) -> float:
        """Calculate average cell volume"""
        if not cells:
            return 0.0
        non_necrotic_cells = [cell for cell in cells if cell["state"] not in ["NECROTIC"]]
        if len(non_necrotic_cells) == 0:
            return 0.0
        return sum([cell["volume"] for cell in non_necrotic_cells]) / len(non_necrotic_cells)

    @staticmethod
    def calculate_doubling_time(n1: float, n2: float, time_difference: float) -> float:
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
    def calculate_states(cells: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate the number of cells in each state

        Args:
            cells: List of cell dictionaries

        Returns:
            Dictionary with counts for each cell state
        """
        states = {
            "UNDEFINED": 0,
            "QUIESCENT": 0,
            "MIGRATORY": 0,
            "PROLIFERATIVE": 0,
            "APOPTOTIC": 0,
            "NECROTIC": 0,
            "SENESCENT": 0,
        }

        if not cells:
            return states

        for cell in cells:
            state = cell["state"]
            if state in states:
                states[state] += 1

        return states

    @staticmethod
    def calculate_age(cells: List[Dict[str, Any]]) -> float:
        """Calculate the average age of the cells"""
        if not cells:
            return 0.0
        non_necrotic_cells = [cell for cell in cells if cell["state"] not in ["NECROTIC"]]
        if len(non_necrotic_cells) == 0:
            return 0.0
        return sum([cell["age"] for cell in non_necrotic_cells]) / len(non_necrotic_cells)

    @staticmethod
    def calculate_cycle_length(cells: List[Dict[str, Any]]) -> float:
        """Calculate average cell cycle length across all cells.

        For each cell, averages its recorded cycle lengths (time between entering
        proliferative state and successful division, in minutes), then averages
        across all cells.

        Args:
            cells: List of cell dictionaries containing cycle length data

        Returns:
            Average cycle length in hours. Returns 0.0 if no cycle data available.
        """
        if not cells:
            return 0.0

        # Calculate average cycle length for each cell that has cycles
        cell_averages = []
        for cell in cells:
            cycles = cell.get("cycles", [])
            if cycles:  # Only include cells that have completed at least one cycle
                cycles = [int(cycle) for cycle in cycles]
                cell_avg = sum(cycles) / len(cycles)
                cell_averages.append(cell_avg)

        # Calculate average across all cells
        if not cell_averages:
            return 0.0

        # Convert from minutes to hours
        return (sum(cell_averages) / len(cell_averages)) / 60

    def load_cells_data(
        self, folder_path: Path, timestamp: str
    ) -> List[Tuple[Dict[str, str], List[Dict[str, Any]]]]:
        """Load cell data from a specific timestamp for all seeds

        Args:
            folder_path: Path to simulation folder
            timestamp: Timestamp string (e.g., '000000' or '000720')

        Returns:
            List of tuples (file_info, cell_data)
        """
        results = []
        try:
            cell_files = list_simulation_files(folder_path, f"_{timestamp}.CELLS.json")
            for cell_file in cell_files:
                file_info = self.file_parser.parse_simulation_file(cell_file.name, "CELLS")
                if file_info:
                    with open(cell_file, "r") as f:
                        cell_data = json.load(f)
                        results.append((file_info, cell_data))

        except Exception as e:
            self.logger.error(f"Error loading cells data from {folder_path}: {str(e)}")

        return results
