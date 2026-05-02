# Implementation of a location in a grid


class Location:
    def __init__(self, i, j, grid=None):
        self.i = i
        self.j = j
        self.cell = None
        self.valid_neighbors = []
        self.grid = grid  # Store reference to grid

    @property
    def coordinates(self):
        """Return coordinates as a tuple"""
        return (self.i, self.j)

    def get_neighbor_coordinates(self):
        """Get potential neighbor coordinates without bounds checking"""
        return [
            (self.i - 1, self.j),
            (self.i + 1, self.j),
            (self.i, self.j - 1),
            (self.i, self.j + 1),
        ]

    def calculate_valid_neighbors(self, lattice_size):
        """Calculate and store valid neighbor coordinates"""
        potential_neighbors = self.get_neighbor_coordinates()
        self.valid_neighbors = [
            n
            for n in potential_neighbors
            if (0 <= n[0] < lattice_size and 0 <= n[1] < lattice_size)
        ]

    def get_cell(self):
        return self.cell

    def set_cell(self, cell):
        if cell is not None:
            old_cell = self.cell
            if old_cell is not None:
                self.remove_cell()  # Clean up old cell if exists

            self.cell = cell
            cell.location = self  # Update cell's location reference
            if self.grid:
                self.grid.add_cell(cell)

    def get_von_neumann_neighbors(self):
        return self.valid_neighbors

    def has_cell(self):
        return self.cell is not None

    def remove_cell(self):
        if self.cell is not None:
            old_cell = self.cell
            self.cell = None
            old_cell.location = None  # Clear cell's location reference
            if self.grid:
                self.grid.remove_cell(old_cell)
