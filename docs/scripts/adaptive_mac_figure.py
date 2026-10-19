"""Make the figure in docs/masking.md that shows how the adaptive box size changes
the minimum absolute clip (MAC). From the root of the repository run:

    uv run --with matplotlib python docs/scripts/adaptive_mac_figure.py

which writes a light and a dark version of the figure to docs/_static, one for each
of the furo themes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from eye_patch.mac import minimum_absolute_clip, minimum_absolute_clip_threshold

OUTPUT_DIR = Path(__file__).parent.parent / "_static"

# Surfaces of the furo themes, with chart colours validated against each
THEMES = {
    "light": {
        "surface": "#ffffff",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "fixed": "#2a78d6",
        "adaptive": "#eb6834",
    },
    "dark": {
        "surface": "#131416",
        "ink": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "fixed": "#3987e5",
        "adaptive": "#d95926",
    },
}

SHAPE = (512, 512)
CUT_ROW = 256
COMPACT_X = 90  # The compact source along the cut
WIDE_X, WIDE_RADIUS = 330, 90  # The source that is wider than the box
BOX_SIZE = 75  # The default mac_box_size
INCREASE_FACTOR = 4.0  # The default base_clip
ADAPTIVE_MAX_DEPTH = 3

# Line widths in points, which are about 2 and 1 screen pixels at the docs width
LINE_WIDTH = 1.44
HAIRLINE_WIDTH = 0.72
# Text placed on a masked span is one of these, whichever contrasts more with it
LABEL_INKS = ("#ffffff", "#0b0b0b")


def make_image() -> np.ndarray:
    """Unit noise with a compact source, and a bright source wider than the box"""
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    radius = np.hypot(xx - WIDE_X, yy - CUT_ROW)
    wide = 30.0 / (1 + np.exp((radius - WIDE_RADIUS) / 4))  # Flat topped, soft edged
    compact = 50.0 * np.exp(-((xx - COMPACT_X) ** 2 + (yy - CUT_ROW) ** 2) / (2 * 3**2))
    noise = np.random.default_rng(1).normal(size=SHAPE)
    return noise + wide + compact


def masked_spans(mask_row: np.ndarray) -> list[tuple[int, int]]:
    """The (start, width) of each run of masked pixels"""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask_row.astype(int), [0]])))
    return [
        (int(start), int(stop - start))
        for start, stop in zip(edges[::2], edges[1::2], strict=True)
    ]


def span_name(start: int, width: int) -> str:
    """Describe the part of the image a masked span covers"""
    if start <= COMPACT_X < start + width:
        return "compact source"
    if start <= WIDE_X < start + width:
        return "whole wide source"
    return "edge"


def _relative_luminance(colour: str) -> float:
    """The WCAG relative luminance of a hex colour"""
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    red, green, blue = (
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    )
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(first: str, second: str) -> float:
    """The WCAG contrast ratio of two hex colours"""
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def label_spans(
    ax,
    spans: list[tuple[int, int]],
    row: int,
    fill: str,
    colours: dict[str, str],
    renderer,
) -> None:
    """Name each masked span. Should a name not fit inside its span with some
    padding it is placed just past the end of the span instead."""
    padding = 6  # Display pixels either side of a name inside a span
    for start, width in spans:
        name = span_name(start=start, width=width)
        label = ax.text(
            start + width / 2,
            row,
            name,
            ha="center",
            va="center",
            fontsize=8,
            color=max(LABEL_INKS, key=lambda ink: _contrast(ink, fill)),
            in_layout=False,
        )
        left, right = ax.transData.transform([(start, row), (start + width, row)])
        if label.get_window_extent(renderer).width + 2 * padding > right[0] - left[0]:
            label.remove()
            ax.annotate(
                name,
                xy=(start + width, row),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color=colours["secondary"],
                annotation_clip=False,
                in_layout=False,
            )


def plot(colours: dict[str, str], path: Path) -> None:
    """Plot a cut through the image with the clip levels of a fixed and an adaptive
    box, and the mask each makes"""
    image = make_image()
    adaptive_options = {"box_size": BOX_SIZE, "adaptive_max_depth": ADAPTIVE_MAX_DEPTH}
    clip_levels = {
        "fixed": INCREASE_FACTOR
        * minimum_absolute_clip_threshold(image=image, box_size=BOX_SIZE),
        "adaptive": INCREASE_FACTOR
        * minimum_absolute_clip_threshold(image=image, **adaptive_options),
    }
    masks = {
        "fixed": minimum_absolute_clip(
            image=image, increase_factor=INCREASE_FACTOR, box_size=BOX_SIZE
        ),
        "adaptive": minimum_absolute_clip(
            image=image, increase_factor=INCREASE_FACTOR, **adaptive_options
        ),
    }
    x = np.arange(SHAPE[1])

    fig = Figure(figsize=(7.5, 4.8), facecolor=colours["surface"], layout="constrained")
    profile_ax, mask_ax = fig.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": (4, 1.1)}
    )
    for ax in (profile_ax, mask_ax):
        ax.set_facecolor(colours["surface"])
        ax.tick_params(colors=colours["secondary"], labelsize=9, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(colours["axis"])
            ax.spines[side].set_linewidth(HAIRLINE_WIDTH)

    # The top panel states the rule that the bottom panel shows the result of
    profile_ax.set_title(
        "A pixel is included in the clean mask where the image is above the clip level",
        loc="left",
        fontsize=10,
        color=colours["ink"],
    )
    profile_ax.grid(axis="y", color=colours["grid"], linewidth=HAIRLINE_WIDTH)
    profile_ax.set_axisbelow(True)
    profile_ax.plot(
        x,
        image[CUT_ROW],
        color=colours["muted"],
        linewidth=HAIRLINE_WIDTH,
        label="Image",
    )
    for name, label in (
        ("fixed", f"Clip level, fixed {BOX_SIZE}-pixel box"),
        ("adaptive", "Clip level, adaptive box"),
    ):
        profile_ax.plot(
            x,
            clip_levels[name][CUT_ROW],
            color=colours[name],
            linewidth=LINE_WIDTH,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=label,
        )
    profile_ax.legend(
        loc="upper left", frameon=False, fontsize=9, labelcolor=colours["ink"]
    )
    # Direct labels where the clip levels separate, in text rather than series colours
    profile_ax.text(
        WIDE_X,
        clip_levels["fixed"][CUT_ROW, WIDE_X] + 3,
        "fixed box",
        ha="center",
        va="bottom",
        fontsize=9,
        color=colours["secondary"],
    )
    profile_ax.text(
        WIDE_X,
        clip_levels["adaptive"][CUT_ROW, WIDE_X] + 2,
        "adaptive box",
        ha="center",
        va="bottom",
        fontsize=9,
        color=colours["secondary"],
    )
    # Point at the image where it falls below the clip level of the fixed box
    if not masks["fixed"][CUT_ROW, WIDE_X]:
        profile_ax.annotate(
            "missed by the\nfixed box",
            xy=(WIDE_X, image[CUT_ROW, WIDE_X] + 2),
            xytext=(WIDE_X, 66),
            ha="center",
            va="center",
            fontsize=8,
            color=colours["secondary"],
            arrowprops={
                "arrowstyle": "-|>",
                "color": colours["secondary"],
                "linewidth": HAIRLINE_WIDTH,
                "shrinkA": 4,
                "shrinkB": 0,
            },
        )
    profile_ax.set_ylim(-6, 120)
    profile_ax.set_ylabel(
        r"Intensity (noise $\sigma$)", fontsize=10, color=colours["secondary"]
    )

    rows = {"adaptive": 0, "fixed": 1}
    for name, row in rows.items():
        mask_ax.broken_barh(
            masked_spans(masks[name][CUT_ROW]),
            (row - 0.3, 0.6),
            facecolors=colours[name],
        )
    mask_ax.set_yticks(list(rows.values()), ["Adaptive box", "Fixed box"])
    mask_ax.set_ylim(-0.6, 1.6)
    mask_ax.spines["left"].set_visible(False)
    mask_ax.set_title(
        "Mask along the cut (bars are True, gaps are False)",
        loc="left",
        fontsize=9,
        color=colours["secondary"],
    )
    mask_ax.set_xlim(0, SHAPE[1] - 1)
    mask_ax.set_xlabel(
        "Position along the cut (pixels)", fontsize=10, color=colours["secondary"]
    )
    # Name the gap the fixed box leaves in the middle of the wide source
    if not masks["fixed"][CUT_ROW, WIDE_X]:
        mask_ax.text(
            WIDE_X,
            rows["fixed"],
            "False: hole",
            ha="center",
            va="center",
            fontsize=8,
            color=colours["secondary"],
        )

    # The span names are fitted to their spans once the layout is final
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    for name, row in rows.items():
        label_spans(
            ax=mask_ax,
            spans=masked_spans(masks[name][CUT_ROW]),
            row=row,
            fill=colours[name],
            colours=colours,
            renderer=canvas.get_renderer(),
        )

    fig.savefig(path, dpi=200, facecolor=colours["surface"])


if __name__ == "__main__":
    for theme, colours in THEMES.items():
        plot(colours=colours, path=OUTPUT_DIR / f"adaptive_mac_{theme}.png")
