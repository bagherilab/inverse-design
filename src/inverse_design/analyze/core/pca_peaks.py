"""PCA on standardized data + 2D KDE peak detection (no matplotlib)."""

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from inverse_design.plotting.colormap import PCA_CLUSTER_COLORS


def find_density_peaks(Z, X, Y, threshold_ratio=0.1, neighborhood_size=5):
    """
    Find local maxima in a 2D density grid.

    Args:
        Z: 2D array of density values
        X, Y: 2D coordinate meshes matching Z
        threshold_ratio: Peak cutoff as a fraction of max(Z)
        neighborhood_size: Footprint size for local-maximum filter

    Returns:
        peak_positions: (n_peaks, 2) array of (x, y) peak coordinates
    """
    neighborhood = np.ones((neighborhood_size, neighborhood_size))
    local_maxima = maximum_filter(Z, footprint=neighborhood) == Z
    threshold = threshold_ratio * np.max(Z)
    peaks_mask = local_maxima & (Z > threshold)
    peak_coords = np.where(peaks_mask)
    peak_positions = np.column_stack((X[peak_coords], Y[peak_coords]))
    return peak_positions


def perform_pca_and_find_peaks(
    data,
    n_components=2,
    threshold_ratio=0.1,
    neighborhood_size=5,
    random_state=0,
):
    """
    Standardize data, run PCA, KDE in PC1–PC2 plane, and assign points to peaks.

    Returns:
        (pca_result, peak_positions, point_colors, pca, Z, X, Y, peak_points)
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca_result = pca.fit_transform(data_scaled)

    x = pca_result[:, 0]
    y = pca_result[:, 1]

    x_grid = np.linspace(x.min() - 2, x.max() + 2, 100)
    y_grid = np.linspace(y.min() - 1, y.max() + 1, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    kernel = gaussian_kde(np.vstack([x, y]))
    Z = np.reshape(kernel(positions).T, X.shape)

    peak_positions = find_density_peaks(Z, X, Y, threshold_ratio, neighborhood_size)
    point_colors = []
    peak_point_indices = []
    for point in pca_result[:, :2]:
        distances = np.sqrt(np.sum((peak_positions - point) ** 2, axis=1))
        closest_peak = np.argmin(distances)
        point_colors.append(PCA_CLUSTER_COLORS[closest_peak % len(PCA_CLUSTER_COLORS)])
        peak_point_indices.append(closest_peak)
    peak_point_indices = np.array(peak_point_indices)
    peak_points = []
    for i, _peak in enumerate(peak_positions):
        peak_points.append(data[peak_point_indices == i])

    return pca_result, peak_positions, point_colors, pca, Z, X, Y, peak_points
