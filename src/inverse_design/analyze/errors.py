"""Shared error metrics used by analysis and figure scripts."""

from __future__ import annotations

import numpy as np


def signed_percent_error(values, target):
    """Return signed percent error relative to ``target``."""
    return (values - target) / abs(target) * 100.0


def absolute_percent_error(values, target):
    """Return absolute percent error relative to ``target``."""
    return np.abs(values - target) / abs(target) * 100.0
