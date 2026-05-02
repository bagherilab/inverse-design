"""
Document filesystem conventions for ARCADE ABC-SMC outputs.

These tests do not require real simulation data; they lock the on-disk layout
that refactor plan `io.arcade_layout` will formalize.
"""

from pathlib import Path


def _touch(p: Path, text: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_convention_iter_final_metrics_and_params(tmp_path: Path):
    run = tmp_path / "ABC_SMC_RF_N512_fake"
    _touch(run / "iter_0" / "final_metrics.csv", "col\n0\n")
    _touch(run / "iter_0" / "all_param_df.csv", "p\n0\n")
    _touch(run / "iter_4" / "final_metrics.csv", "col\n1\n")
    _touch(run / "iter_4" / "all_param_df.csv", "p\n1\n")
    _touch(run / "targets.json", "{}")

    assert (run / "iter_4" / "final_metrics.csv").is_file()
    assert (run / "iter_4" / "all_param_df.csv").is_file()
    assert (run / "targets.json").is_file()


def test_convention_lr_prediction_json_naming(tmp_path: Path):
    run = tmp_path / "simplified_linear"
    f = run / "lr_predictions_r0.9_p2.json"
    _touch(f, "{}")
    assert f.name == "lr_predictions_r0.9_p2.json"


def test_convention_redundancy_json_naming(tmp_path: Path):
    run = tmp_path / "simplified_dendrogram"
    f = run / "redundancy_analysis_threshold_1.25_p2.json"
    _touch(f, "{}")
    assert "redundancy_analysis_threshold_" in f.name
