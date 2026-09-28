"""Shared PLOS styling and inspection helpers for reviewer-response figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PLOS_DOUBLE_WIDTH = 6.81
PNG_DPI = 600
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 7.0
AXIS_LABEL_SIZE = 7.5
PANEL_LABEL_SIZE = 9.0
LINE_WIDTH = 0.9
SPINE_WIDTH = 0.7

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#777777",
}

# Preserve the manuscript's existing peak mapping while keeping P4 visible on white.
PEAK_COLORS = ["#009E73", "#CC79A7", "#E69F00", "#B79F00"]
PEAK_MARKERS = ["^", "D", "*", "p"]
PEAK_LINESTYLES = ["-", "--", "-.", ":"]


def apply_plos_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": PNG_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Arial", "Helvetica"],
            "font.size": FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.linewidth": SPINE_WIDTH,
            "xtick.major.width": SPINE_WIDTH,
            "ytick.major.width": SPINE_WIDTH,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": LINE_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def four_spines(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(SPINE_WIDTH)
    axis.tick_params(direction="out", colors="black", width=SPINE_WIDTH)


def panel_label(axis: plt.Axes, label: str, x: float = -0.16, y: float = 1.04) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        clip_on=False,
    )


def in_axis_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.025,
        0.965,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=FONT_SIZE,
        bbox={"boxstyle": "square,pad=0.22", "facecolor": "black", "edgecolor": "black"},
        zorder=20,
    )


def save_figure(fig: plt.Figure, output: Path, *, split_formats: bool = False) -> None:
    if split_formats:
        png_output = output.parent / "png" / output.with_suffix(".png").name
        pdf_output = output.parent / "pdf" / output.with_suffix(".pdf").name
    else:
        png_output = output.with_suffix(".png")
        pdf_output = output.with_suffix(".pdf")
    png_output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_output,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        pdf_output,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )


def assert_text_inside_figure(fig: plt.Figure, tolerance_px: float = 12.0) -> None:
    """Fail if visible text falls outside the tight export bounding box."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Every reviewer figure is exported with ``bbox_inches='tight'``.  Inspect
    # that actual export box rather than the pre-export interactive canvas,
    # which legitimately excludes outward tick labels and panel letters.
    canvas = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    for text in fig.findobj(match=lambda artist: isinstance(artist, mpl.text.Text)):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer=renderer)
        if box.width == 0 or box.height == 0:
            continue
        # Matplotlib tick-label antialiasing can extend a few pixels beyond the
        # nominal canvas before ``bbox_inches='tight'`` expands the export.
        if (
            box.x0 < canvas.x0 - tolerance_px
            or box.y0 < canvas.y0 - tolerance_px
            or box.x1 > canvas.x1 + tolerance_px
            or box.y1 > canvas.y1 + tolerance_px
        ):
            raise RuntimeError(
                f"Text outside tight export box: {text.get_text()!r} at {box.bounds}; "
                f"export={canvas.bounds}"
            )
