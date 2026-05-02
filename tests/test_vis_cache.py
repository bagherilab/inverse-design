from pathlib import Path

from inverse_design.io.scenarios.load import CombinedGridN512Paths
from inverse_design.vis.vis_cache import (
    cache_payload_path,
    load_cache_payload,
    select_simplified_dirs,
)


def _sample_paths() -> CombinedGridN512Paths:
    return CombinedGridN512Paths(
        base_original_dir="base",
        linear_model_dirs=tuple(f"lin{i}" for i in range(12)),
        dendrogram_threshold_dirs=tuple(f"den{i}" for i in range(12)),
        ext_linear_dirs=tuple(f"elin{i}" for i in range(4)),
        ext_dendrogram_threshold_dirs=tuple(f"eden{i}" for i in range(4)),
    )


def test_select_simplified_dirs_linear_peak_2():
    selected = select_simplified_dirs(_sample_paths(), simplified_method="linear", peak_idx=2)
    assert selected == ["lin6", "lin7", "lin8", "elin2"]


def test_select_simplified_dirs_threshold_peak_1():
    selected = select_simplified_dirs(_sample_paths(), simplified_method="threshold", peak_idx=1)
    assert selected == ["den3", "den4", "den5", "eden1"]


def test_cache_payload_roundtrip(tmp_path: Path):
    payload_path = cache_payload_path(tmp_path, "linear", 3)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        '{"peak_idx": 3, "best_fit_samples_idx_list": [9, 2]}', encoding="utf-8"
    )
    loaded = load_cache_payload(tmp_path, "linear", 3)
    assert loaded["peak_idx"] == 3
    assert loaded["best_fit_samples_idx_list"] == [9, 2]
