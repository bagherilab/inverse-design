# Implementation of a birth-death-migration model using the Gillespie algorithm
import random
import numpy as np
from .location import Location


class Grid:
    def __init__(self, lattice_size):
        self.lattice_size = lattice_size
        self.initialize(lattice_size)

    def initialize(self, lattice_size):
        self.lattice = np.full((lattice_size, lattice_size), None, dtype=object)
        for i in range(lattice_size):
            for j in range(lattice_size):
                self.lattice[i, j] = Location(i, j, grid=self)

        for i in range(lattice_size):
            for j in range(lattice_size):
                location = self.lattice[i, j]
                location.calculate_valid_neighbors(lattice_size)
        self.cells = self.get_cells()
        self.num_cells = self.get_num_cells()

    def add_cell(self, cell):
        """Called when a cell is added to a location"""
        if cell not in self.cells:
            self.cells.append(cell)
            self.num_cells += 1

    def remove_cell(self, cell):
        """Called when a cell is removed from a location"""
        if cell in self.cells:
            self.cells.remove(cell)
            self.num_cells -= 1

    def get_lattice(self):
        return self.lattice

    def get_lattice_size(self):
        return self.lattice_size

    def get_location(self, i, j):
        return self.lattice[i, j]

    def get_cells(self):
        """Return a list of all cells in the grid"""
        cells = []
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                if self.lattice[i, j].has_cell():
                    cells.append(self.lattice[i, j].get_cell())
        return cells

    def get_random_cell(self):
        """Return a random cell in the grid"""
        assert self.num_cells > 0, "No cells in the grid"
        return random.choice(self.cells)

    def get_num_cells(self):
        """Return the total number of cells in the grid"""
        return len(self.cells)

    def get_empty_von_neumann_neighbors(self, i, j):
        location = self.get_location(i, j)
        empty_neighbors = []
        for ni, nj in location.get_von_neumann_neighbors():
            neighbor = self.get_location(ni, nj)
            if not neighbor.has_cell():
                empty_neighbors.append((ni, nj))
        return empty_neighbors
