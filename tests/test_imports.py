"""Smoke tests: critical packages and entry modules import without side effects."""


def test_import_package():
    import inverse_design  # noqa: F401


def test_import_plotting_theme():
    from inverse_design.plotting import apply_publication_style, metric_color

    assert metric_color("symmetry")


def test_import_analyze_core_pca():
    from inverse_design.analyze.core import perform_pca_and_find_peaks  # noqa: F401


def test_import_analyze_scenarios():
    from inverse_design.io.scenarios import combined_grid_n512_breast  # noqa: F401

    assert combined_grid_n512_breast().base_original_dir


def test_import_io_arcade_layout():
    from pathlib import Path

    from inverse_design.io import ArcadeRunLayout

    assert isinstance(ArcadeRunLayout.from_root(".").run_root, Path)


def test_import_abc_precomputed():
    from inverse_design.abc.abc_precomputed import ABCPrecomputed  # noqa: F401


def test_import_arcade_rf():
    from inverse_design.rf.abc_smc_rf_arcade import ABCSMCRF  # noqa: F401


def test_import_arcade_example_module():
    import inverse_design.rf.arcade_example as arcade_example  # noqa: F401

    assert hasattr(arcade_example, "run_example")
    assert hasattr(arcade_example, "parse_arguments")


def test_import_vis_sandbox():
    from inverse_design.vis import sandbox as vis_sandbox  # noqa: F401

    assert hasattr(vis_sandbox, "perform_pca_and_find_peaks")
