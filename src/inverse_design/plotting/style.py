"""Shared Nature Communications-style figure defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from inverse_design.plotting.theme import apply_publication_style

FONT_BASE: Final[int] = 6
FONT_LABEL: Final[int] = 7
FONT_TITLE: Final[int] = 7
FONT_PANEL: Final[int] = 8
FONT_METRIC: Final[int] = 8  # mathtext metric names on SIR panels C/D (above axis tick size)

LINE_AXIS: Final[float] = 0.5
LINE_PLOT: Final[float] = 1.0
DPI: Final[int] = 600

WIDTH_SINGLE: Final[float] = 3.50
WIDTH_1P5: Final[float] = 4.72
WIDTH_DOUBLE: Final[float] = 7.20

PALETTE: Final[dict[str, str]] = {
    "doub_time": "#486b45",
    "doub_time_std": "#679A63",
    "symmetry": "#bb883b",
    "symmetry_std": "#ddc39d",
    "act_ratio": "#af1b0a",
    "colony_growth": "#545aab",
    "peak_1": "#e41a1c",
    "peak_2": "#377eb8",
    "peak_3": "#4daf4a",
    "peak_4": "#984ea3",
    "sir_s": "#545aab",
    "sir_i": "#af1b0a",
    "sir_r": "#486b45",
}


def apply_style() -> None:
    """Apply shared publication rcParams.

    The function is intentionally idempotent: repeated calls overwrite the same
    rcParams with the same values and do not depend on prior matplotlib state.
    """
    apply_publication_style(
        font_size=FONT_BASE,
        axes_linewidth=LINE_AXIS,
        tick_major_width=LINE_AXIS,
        **{
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": FONT_LABEL,
            "axes.titlesize": FONT_TITLE,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": FONT_LABEL,
            "legend.frameon": False,
            "xtick.labelsize": FONT_BASE,
            "ytick.labelsize": FONT_BASE,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "lines.linewidth": LINE_PLOT,
            "patch.linewidth": LINE_AXIS,
            "mathtext.fontset": "dejavusans",
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def _stem_from_path(path_stem: str | Path) -> Path:
    path = Path(path_stem)
    if path.suffix.lower() in {".png", ".pdf"}:
        return path.with_suffix("")
    return path


def savefig_both(fig, path_stem: str | Path, **kwargs) -> tuple[Path, Path]:
    """Save a figure as 600-dpi PNG and editable-font PDF.

    ``path_stem`` may be either a stem or an existing ``.png`` / ``.pdf`` path;
    both outputs are written next to that stem.
    """
    stem = _stem_from_path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    save_kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.02}
    save_kwargs.update(kwargs)
    fig.savefig(png_path, dpi=DPI, **save_kwargs)
    fig.savefig(pdf_path, **save_kwargs)
    return png_path, pdf_path


def panel_letter(target, text: str, x: float, y: float, **kwargs) -> None:
    """Draw a consistent bold panel letter on a figure or axes."""
    default_kwargs = {
        "fontsize": FONT_PANEL,
        "fontweight": "bold",
        "ha": "left",
        "va": "bottom",
    }
    default_kwargs.update(kwargs)
    if hasattr(target, "transAxes"):
        target.text(x, y, text, transform=target.transAxes, clip_on=False, **default_kwargs)
    else:
        target.text(x, y, text, clip_on=False, **default_kwargs)
