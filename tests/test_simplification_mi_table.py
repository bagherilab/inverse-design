from scripts.combine_simplification_mi_table import _format_mi_direction


def test_format_mi_direction_adds_spearman_arrow():
    assert _format_mi_direction(0.456, 1) == "0.46↑"
    assert _format_mi_direction(0.456, -1) == "0.46↓"
    assert _format_mi_direction(0.456, 0) == "0.46"
