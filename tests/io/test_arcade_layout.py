"""ArcadeRunLayout matches filesystem conventions in test_arcade_path_conventions."""

from pathlib import Path

from inverse_design.io import (
    ALL_PARAM_DF_CSV,
    ArcadeRunLayout,
    FINAL_METRICS_CSV,
    TARGETS_JSON,
)


def test_layout_paths_resolve_under_run_root(tmp_path: Path):
    run = tmp_path / "ABC_run"
    layout = ArcadeRunLayout.from_root(run)
    assert layout.final_metrics_csv("iter_2") == run / "iter_2" / FINAL_METRICS_CSV
    assert layout.all_param_df_csv("iter_2") == run / "iter_2" / ALL_PARAM_DF_CSV
    assert layout.targets_json() == run / TARGETS_JSON


def test_iter_k_dir_matches_string_postfix(tmp_path: Path):
    run = tmp_path / "r"
    layout = ArcadeRunLayout.from_root(run)
    assert layout.iter_k_dir(4) == layout.iteration_subdir("iter_4")
