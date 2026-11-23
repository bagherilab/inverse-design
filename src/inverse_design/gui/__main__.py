"""``python -m inverse_design.gui [template.xml | run_spec.json]`` launches the setup GUI."""

import sys

try:
    from inverse_design.gui.app import main
except ModuleNotFoundError as exc:  # PySide6 is optional: only the GUI needs it
    if exc.name and exc.name.startswith("PySide6"):
        sys.exit(
            "The ARCADE setup GUI needs PySide6, which is not installed.\n"
            "Install it in this environment with:  pip install 'PySide6>=6.6'"
        )
    raise

if __name__ == "__main__":
    raise SystemExit(main())
