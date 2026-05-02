# Implementation of a cell class

import numpy as np


class Cell:
    def __init__(self, status="cell", location=None):
        self.status = status
        self.location = None  # Initialize to None
        if location is not None:
            self.set_location(location)  # Use setter to properly set up relationship

    def get_status(self):
        return self.status

    def set_status(self, status):
        """Set the status of the cell
        Available statuses: "cell"
        """
        self.status = status

    def set_location(self, location):
        self.location = location
        if location is not None:
            location.set_cell(self)  # Establish bidirectional relationship

    def get_location(self):
        return self.location
