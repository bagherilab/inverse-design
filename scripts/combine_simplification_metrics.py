#!/usr/bin/env python3
"""Thin wrapper for :mod:`inverse_design.figures.combine_simplification_metrics`."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inverse_design.figures import combine_simplification_metrics as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
main = _impl.main


if __name__ == "__main__":
    main()
