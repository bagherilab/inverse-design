"""Bootstrap helpers for repository-local scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def init_script(headless: bool = True) -> Path:
    """Configure script defaults and return the repository root."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")
    repo_root = Path(__file__).resolve().parents[3]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if headless:
        import matplotlib

        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg", force=False)
    return repo_root
