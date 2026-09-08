"""Shared figure style for the documents.

`docs/tex/` is written as an academic letter, and its figures follow: a serif
face matching the body text, one-column sizing, recessive axes, and no
ornament that does not carry information. Every QA script draws inside
:func:`letter_style` so the document reads as one set of figures rather than
one per author.

Colour
------
`PALETTE` is the Okabe--Ito colourblind-safe qualitative set, in a fixed
order that is never cycled. Validated with the visualization skill's
checker against a light print surface: adjacent-pair CVD separation
``dE 11.0`` (deuteranopia), normal-vision floor ``dE 16.4``, all four inside
the lightness band and above the chroma floor. Over all pairs rather than
adjacent ones, ``purple/green`` falls to ``dE 7.6``, inside the band that is
legal only with secondary encoding -- which is why `MARKERS` and
`LINESTYLES` exist and why figures label series directly. Colour never
carries identity alone.

A fifth series is not a new hue. Facet, or fold into an aggregate.

Categorical *states* -- Potts states, hidden states, the coupled model's
labels -- are a different object from series and take `STATE_PALETTE`, the
full eight-colour Okabe--Ito set (issue #312): a state is read off a
legend or a colourbar, never traced across a plot, so the adjacent-pair
separation the series palette was validated for is not what it needs. A
ninth state is a facet, as a fifth series is.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.typing import ColorType

# Okabe-Ito, fixed order. See the module docstring for the validation result.
PALETTE: tuple[str, ...] = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")

# Secondary encoding, index-matched to PALETTE: identity survives greyscale
# printing and colour-vision deficiency without relying on hue.
MARKERS: tuple[str, ...] = ("o", "s", "^", "D")
LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")

# The full Okabe-Ito set, for categorical states rather than series. Order
# is the published one, with black omitted because INK is the text colour.
STATE_PALETTE: tuple[str, ...] = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
)

# Ink, not series colour. Text never wears a series hue.
INK = "#1a1a1a"
INK_MUTED = "#5c5c5c"
RULE = "#c8c8c8"

# Widths chosen for a single text column at the document's 11pt body size.
ONE_COLUMN = (5.4, 3.4)
ONE_COLUMN_WIDE = (5.4, 4.2)
ONE_COLUMN_SHORT = (5.4, 2.6)


def series_style(index: int) -> dict[str, str]:
    """Colour, marker and linestyle for series ``index``.

    Parameters
    ----------
    index : int
        Zero-based series index.

    Returns
    -------
    dict[str, str]
        ``color``, ``marker`` and ``linestyle`` keyword arguments.

    Raises
    ------
    IndexError
        If ``index`` is beyond the palette. A fifth hue is not generated:
        facet the figure or aggregate instead.
    """
    if not 0 <= index < len(PALETTE):
        msg = (
            f"series index {index} outside the {len(PALETTE)}-colour palette; "
            "facet or aggregate rather than adding a hue"
        )
        raise IndexError(msg)
    return {
        "color": PALETTE[index],
        "marker": MARKERS[index],
        "linestyle": LINESTYLES[index],
    }


def blend_with_white(
    color: ColorType, alpha: float
) -> tuple[float, float, float, float]:
    """``color`` mixed with white: ``alpha = 1`` is the colour, ``alpha = 0`` is white.

    A fill lighter than its edge without transparency, so it prints the same
    over any background and does not compound where fills overlap.
    """
    if not 0.0 <= alpha <= 1.0:
        msg = f"alpha must be in [0, 1], got {alpha}"
        raise ValueError(msg)
    r, g, b, _ = mcolors.to_rgba(color)
    return (1 - alpha * (1 - r), 1 - alpha * (1 - g), 1 - alpha * (1 - b), 1.0)


def with_opacity(colors: Sequence[ColorType], opacities: Sequence[float]) -> np.ndarray:
    """One RGBA row per colour, its alpha the matching opacity clipped to ``[0, 1]``.

    For a scatter whose points carry a weight -- a posterior probability, a
    support -- as opacity, with the colour carrying the state.
    """
    if len(colors) != len(opacities):
        msg = f"{len(colors)} colours but {len(opacities)} opacities"
        raise ValueError(msg)
    rgba = np.zeros((len(colors), 4))
    for row, (color, alpha) in enumerate(zip(colors, opacities, strict=True)):
        rgba[row, :3] = mcolors.to_rgb(color)
        rgba[row, 3] = float(np.clip(alpha, 0.0, 1.0))
    return rgba


def discrete_palette(
    n_states: int, *, highlight: int | None = None, highlight_color: str = INK
) -> dict[int, str]:
    """State index to hex colour from `STATE_PALETTE`, with one state optionally drawn in ``highlight_color``.

    Raises
    ------
    IndexError
        If ``n_states`` exceeds the eight states the palette carries, or
        ``highlight`` names a state outside them. A ninth state is a facet.
    """
    if not 1 <= n_states <= len(STATE_PALETTE):
        msg = (
            f"{n_states} states outside the {len(STATE_PALETTE)}-colour state palette; "
            "facet rather than adding a hue"
        )
        raise IndexError(msg)
    palette = dict(enumerate(STATE_PALETTE[:n_states]))
    if highlight is not None:
        if highlight not in palette:
            msg = f"highlight {highlight} is not one of the {n_states} states"
            raise IndexError(msg)
        palette[highlight] = highlight_color
    return palette


@contextmanager
def notebook_style() -> Iterator[None]:
    """The letter style at screen resolution, for the notebooks under ``docs/nb/``.

    The document's 200 dpi is what the committed PDFs are byte-compared at
    and stays in :func:`letter_style`; a notebook renders inline and is not
    compared, so it draws at 150.
    """
    with letter_style(), mpl.rc_context({"figure.dpi": 150}):
        yield


@contextmanager
def letter_style() -> Iterator[None]:
    """Draw inside the documents' figure style.

    The rcParams are written as a literal rather than built and passed, so
    mypy checks each key against matplotlib's ``Literal`` key type -- a
    misspelled rcParam is a type error here rather than a silently ignored
    setting at runtime.

    Yields
    ------
    None
        A context in which matplotlib's rcParams carry the letter style.
        Scoped rather than global, so importing this module does not change
        plotting behaviour for anything else in the process.
    """
    with mpl.rc_context(
        {
            # Serif, to sit with the document body rather than against it.
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            # Recessive frame: two spines, thin, muted. The data is the ink.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK_MUTED,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.grid": True,
            "grid.color": RULE,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Type 42 for the same reason figure.py sets it: GitHub's PDF
            # viewer cannot render Type 3. Set here too so a script that
            # previews a figure without write_qa_figure still gets
            # embeddable text.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        yield


def tolerance_note(values: Sequence[float], tolerance: float) -> str:
    """State the largest deviation against the tolerance it was checked at.

    Parameters
    ----------
    values : Sequence[float]
        Absolute deviations from a reference.
    tolerance : float
        The tolerance the deviations are checked against.

    Returns
    -------
    str
        A caption fragment giving the realized maximum and the tolerance --
        the PR template's "realized value alongside the reference and
        tolerance" rule, in figure form.
    """
    worst = max(values) if values else 0.0
    return f"largest deviation {worst:.2e}, checked at {tolerance:.0e}"
