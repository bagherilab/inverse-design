"""Tests for zip_directory and factor-pair helper in utils.utils."""

import zipfile

import pytest

from inverse_design.utils import utils as utils_mod


def test_all_factor_pairs_positive_integer():
    pairs = utils_mod._all_factor_pairs(12)
    assert (1, 12) in pairs
    assert (3, 4) in pairs


def test_all_factor_pairs_nonpositive_raises():
    with pytest.raises(ValueError, match="positive integer"):
        utils_mod._all_factor_pairs(0)


def test_zip_directory_include_root(tmp_path):
    src = tmp_path / "folder"
    src.mkdir()
    (src / "a.txt").write_text("hello", encoding="utf-8")
    zip_path = tmp_path / "out.zip"
    utils_mod.zip_directory(str(src), str(zip_path), include_root_folder=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert any("a.txt" in n for n in names)
