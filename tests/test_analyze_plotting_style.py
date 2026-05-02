import matplotlib.pyplot as plt

from inverse_design.plotting.style import (
    DPI,
    FONT_BASE,
    FONT_LABEL,
    FONT_PANEL,
    LINE_AXIS,
    apply_style,
    panel_letter,
    savefig_both,
)


def test_apply_style_sets_expected_rcparams():
    with plt.rc_context():
        apply_style()
        assert plt.rcParams["font.family"] == ["sans-serif"]
        assert plt.rcParams["font.sans-serif"][:3] == ["Arial", "Helvetica", "DejaVu Sans"]
        assert float(plt.rcParams["font.size"]) == FONT_BASE
        assert float(plt.rcParams["axes.labelsize"]) == FONT_LABEL
        assert float(plt.rcParams["axes.linewidth"]) == LINE_AXIS
        assert int(plt.rcParams["savefig.dpi"]) == DPI
        assert int(plt.rcParams["pdf.fonttype"]) == 42
        assert plt.rcParams["xtick.direction"] == "out"


def test_savefig_both_writes_png_and_pdf(tmp_path):
    with plt.rc_context():
        apply_style()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        png_path, pdf_path = savefig_both(fig, tmp_path / "figure.png")
        assert png_path.name == "figure.png"
        assert pdf_path.name == "figure.pdf"
        assert png_path.exists()
        assert pdf_path.exists()
        plt.close(fig)


def test_panel_letter_uses_shared_size():
    fig, ax = plt.subplots()
    panel_letter(ax, "A", -0.1, 1.0)
    assert ax.texts[0].get_fontsize() == FONT_PANEL
    assert ax.texts[0].get_fontweight() == "bold"
    plt.close(fig)
