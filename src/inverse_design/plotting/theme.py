"""Shared matplotlib rcParams for publication-style figures."""

from __future__ import annotations

from typing import Any, Dict


def publication_rc_params(
    *,
    font_size: float = 12,
    axes_linewidth: float = 1.0,
    tick_major_width: float = 1.5,
) -> Dict[str, Any]:
    """Return a dict suitable for ``matplotlib.pyplot.rcParams.update``."""
    return {
        "font.size": font_size,
        "font.family": "sans-serif",
        "axes.linewidth": axes_linewidth,
        "axes.edgecolor": "black",
        "xtick.major.width": tick_major_width,
        "ytick.major.width": tick_major_width,
        "xtick.color": "black",
        "ytick.color": "black",
    }


def apply_publication_style(
    *,
    font_size: float = 12,
    axes_linewidth: float = 1.0,
    tick_major_width: float = 1.5,
    **extra: Any,
) -> None:
    """Apply publication defaults globally via ``rcParams.update``."""
    import matplotlib.pyplot as plt

    params = publication_rc_params(
        font_size=font_size,
        axes_linewidth=axes_linewidth,
        tick_major_width=tick_major_width,
    )
    params.update(extra)
    plt.rcParams.update(params)


def apply_journal_style_nature_baseline() -> None:
    """Spine/tick defaults used by ``vis_metrics_hyperparam`` (Nature-style baseline)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 3,
            "xtick.minor.size": 1.5,
            "ytick.major.size": 3,
            "ytick.minor.size": 1.5,
            "legend.frameon": False,
            "figure.dpi": 300,
        }
    )


NATURE_COMMUNICATIONS_RCPARAMS: Dict[str, Any] = {
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "mathtext.fontset": "dejavusans",
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_nature_communications_style() -> None:
    """Apply compact Nature Communications-style rcParams."""
    apply_publication_style(
        font_size=6,
        axes_linewidth=0.5,
        tick_major_width=0.5,
        **NATURE_COMMUNICATIONS_RCPARAMS,
    )
