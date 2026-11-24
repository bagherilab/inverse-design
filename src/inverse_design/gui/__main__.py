"""``python -m inverse_design.gui [template.xml | run_spec.json]`` launches the setup GUI."""

import sys

try:
    from inverse_design.gui.app import main
except ModuleNotFoundError as exc:  # e.g. Python >= 3.15, where PySide6 is not installed
    if exc.name and exc.name.startswith("PySide6"):
        sys.exit(
            "The ARCADE setup GUI needs PySide6, which is not installed.\n"
            "Run `poetry install` on Python < 3.15, or in this environment:\n"
            "  pip install 'pyside6-essentials>=6.6'"
        )
    raise

if __name__ == "__main__":
    raise SystemExit(main())
