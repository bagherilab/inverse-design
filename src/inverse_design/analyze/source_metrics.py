# calculate the capillary density for a hexagonal grid-source layout

import numpy as np

MICRON_to_MM = 1e-3


def calculate_capillary_density(radius_bound, length_spacing, width_spacing, side_length):
    """
    Calculate capillary density for a hexagonal grid-source layout

    Parameters:
    radius_bound (float): Simulation radius + margin
    length_spacing (float): spacing in length dimension
    width_spacing (float): spacing in width dimension
    side_length (float): Length of one side of the triangular lattice (in microns)

    Returns:
    float: Capillary density (capillaries/mm^2)
    """
    # Calculate grid dimensions
    length = 6 * radius_bound - 3
    width = 4 * radius_bound - 2

    # Calculate total number of capillaries (N)
    num_capillaries = (length // length_spacing) * (width // width_spacing)

    # Calculate area (A)
    area = ((length + 1) / 2 * side_length) * (width * side_length)

    # Calculate capillary density

    area = area * MICRON_to_MM**2
    density = num_capillaries / area

    return density


def calculate_distance_between_points(
    point1, point2, side_length, radius_bounds, depth_bounds=1, height_offset=0
):
    """
    Calculate the distance between two points in a hexagonal grid.

    Args:
        point1: numpy array [x, y, z]
        point2: numpy array [x, y, z]
        side_length: float, side length of the hexagonal grid [microns]

    Returns:
        float, distance between the two points [mm]
    """

    u1, v1, w1, z1 = translate_xyz_to_uvwz(point1, radius_bounds, depth_bounds, height_offset)
    u2, v2, w2, z2 = translate_xyz_to_uvwz(point2, radius_bounds, depth_bounds, height_offset)
    difference = np.array([u1, v1, w1, z1]) - np.array([u2, v2, w2, z2])
    return np.sum(np.abs(difference)) / 2 * side_length * np.sqrt(3) * MICRON_to_MM


def translate_xyz_to_uvwz(coordinate, radius_bounds, depth_bounds=1, height_offset=0):
    """
    Translate XYZ coordinates to UVWZ coordinates.

    Args:
        coordinate: numpy array [x, y, z]
        radius_bounds: int, radius bounds
        depth_bounds: int, depth bounds (default=1)
        height_offset: int, height offset (default=0)

    Returns:
        numpy array [u, v, w, z] or None if out of bounds
    """

    x, y, z = coordinate
    z = z - depth_bounds + 1
    zo = abs(height_offset + z) % 3

    uu = (x - (-1 if zo == 2 else zo) + 2) / 3.0 - radius_bounds
    u = round(round(uu))

    vw = y - 2 * radius_bounds + 2 - (0 if zo == 0 else 1)
    v = -(vw + u) // 2
    w = -(u + v)

    if abs(v) > radius_bounds or abs(w) > radius_bounds:
        return None

    return np.array([u, v, w, z])


def find_spacing_from_density(density, radius_bound, side_length, MICRON_to_MM=1e-3):
    """
    Find length_spacing and width_spacing that produce the density closest to the target.
    Returns the combination where length_spacing and width_spacing are closest to each other
    among all combinations that produce the closest density to the target.
    Constraint: length_spacing <= width_spacing

    Parameters:
    density (float): Target capillary density (capillaries/mm^2)
    radius_bound (float): Simulation radius + margin
    side_length (float): Length of one side of the triangular lattice (in microns)
    MICRON_to_MM (float): Conversion factor from microns to mm

    Returns:
    tuple: (length_spacing, width_spacing) that produces the closest density to target
           where length_spacing <= width_spacing
    """
    # Calculate grid dimensions
    length = 6 * radius_bound - 3
    width = 4 * radius_bound - 2

    # Calculate area in mm^2
    area_microns = ((length + 1) / 2 * side_length) * (width * side_length)
    area_mm2 = area_microns * MICRON_to_MM**2

    # Find all possible integer combinations of length_spacing and width_spacing
    # and calculate their resulting densities
    best_combination = None
    min_density_diff = float("inf")
    min_spacing_diff = float("inf")

    # Search through reasonable ranges of spacing values
    max_length_spacing = int(length) + 1
    max_width_spacing = int(width) + 1

    for length_spacing in range(1, max_length_spacing):
        for width_spacing in range(
            length_spacing, max_width_spacing
        ):  # Ensure width_spacing >= length_spacing
            # Calculate number of capillaries for this combination
            calculated_capillaries = (length // length_spacing) * (width // width_spacing)

            # Calculate the resulting density
            calculated_density = calculated_capillaries / area_mm2

            # Calculate the difference from target density
            density_diff = abs(calculated_density - density)

            # Calculate how close length_spacing and width_spacing are to each other
            spacing_diff = abs(length_spacing - width_spacing)

            # Prioritize spacing values being close first, then density accuracy
            if (spacing_diff < min_spacing_diff) or (
                spacing_diff == min_spacing_diff and density_diff < min_density_diff
            ):
                min_density_diff = density_diff
                min_spacing_diff = spacing_diff
                best_combination = (length_spacing, width_spacing)

    return best_combination


def verify_density(length_spacing, width_spacing, radius_bound, side_length, MICRON_to_MM=1e-3):
    """
    Verify the density calculation with the found spacing values.

    Parameters:
    length_spacing (float): spacing in length dimension
    width_spacing (float): spacing in width dimension
    radius_bound (float): Simulation radius + margin
    side_length (float): Length of one side of the triangular lattice (in microns)
    MICRON_to_MM (float): Conversion factor from microns to mm

    Returns:
    float: Calculated capillary density (capillaries/mm^2)
    """
    # Calculate grid dimensions
    length = 6 * radius_bound - 3
    width = 4 * radius_bound - 2

    # Calculate total number of capillaries (N)
    num_capillaries = (length // length_spacing) * (width // width_spacing)

    # Calculate area (A)
    area = ((length + 1) / 2 * side_length) * (width * side_length)

    # Calculate capillary density
    area = area * MICRON_to_MM**2
    density = num_capillaries / area

    return density


# Example usage
