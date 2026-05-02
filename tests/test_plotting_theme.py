"""plotting.theme helpers."""

import matplotlib.pyplot as plt

from inverse_design.plotting.theme import (
    apply_journal_style_nature_baseline,
    apply_publication_style,
    publication_rc_params,
)


def test_publication_rc_params_defaults():
    d = publication_rc_params()
    assert d["font.family"] == "sans-serif"
    assert d["axes.edgecolor"] == "black"
    assert d["font.size"] == 12
    assert d["xtick.major.width"] == 1.5


def test_publication_rc_params_tick_major_width():
    d = publication_rc_params(tick_major_width=1.0)
    assert d["xtick.major.width"] == 1.0
    assert d["ytick.major.width"] == 1.0


def test_apply_publication_style_updates_rcparams():
    with plt.rc_context():
        apply_publication_style(font_size=11, axes_linewidth=1.25)
        assert float(plt.rcParams["font.size"]) == 11.0
        assert float(plt.rcParams["axes.linewidth"]) == 1.25


def test_apply_journal_style_nature_baseline_runs():
    with plt.rc_context():
        apply_journal_style_nature_baseline()
        assert plt.rcParams["legend.frameon"] is False
