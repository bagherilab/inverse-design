"""Backward-compatibility re-export shim for the vis package.

All implementations now live in their dedicated modules:
    - ``vis.vis_posterior_overview``   – posterior / prior comparison plots
    - ``vis.vis_peak_diagnostics``     – peak-focused plots
    - ``vis.vis_parameter_comparison`` – parameter comparison bar charts
    - ``vis.vis_metric_embedding``     – metric-contour & parallel-coordinate plots
    - ``vis.utils``                    – shared constants and utility helpers

Prefer importing from the specific modules above.  This shim exists only so
that code that previously did ``from inverse_design.vis.sandbox import X``
continues to work without modification.
"""

# ruff: noqa: F401  (all imports are intentional re-exports)

from inverse_design.vis.vis_posterior_overview import (
    create_corner_plot,
    plot_correlation_matrix,
    plot_correlation_matrix_for_peaks,
    plot_marginal_distributions,
    plot_pca_analysis,
    plot_posterior_pair_density,
    plot_pca_with_peaks,
    plot_summary_comparison,
    plot_trace_analysis,
)
from inverse_design.vis.vis_peak_diagnostics import (
    plot_all_peak_percentage_changes,
    plot_all_peak_percentage_changes_grouped,
)
from inverse_design.vis.vis_parameter_comparison import (
    find_mode_bin,
    plot_best_sample_percentage_changes_split,
    plot_multi_boxcharts,
)
from inverse_design.vis.vis_metric_embedding import (
    create_parameter_order,
    find_best_sample_for_metric,
    load_redundancy_analysis,
    plot_parallel_coordinates_multiple_groups_vs_posterior,
    plot_parallel_coordinates_with_change,
    plot_pca_with_metric_contour,
)
from inverse_design.analyze.core import perform_pca_and_find_peaks
