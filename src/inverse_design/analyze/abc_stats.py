"""Numerical ABC metric comparison API.

The implementation remains in ``abc_metrics_comparison`` for compatibility while
callers migrate to this numerics-focused module.
"""

from inverse_design.analyze.abc_metrics_comparison import (  # noqa: F401
    ABCMetricsComparison,
    analyze_iteration,
    analyze_iterations,
)

__all__ = ["ABCMetricsComparison", "analyze_iteration", "analyze_iterations"]
