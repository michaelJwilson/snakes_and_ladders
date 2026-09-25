"""How a figure is arranged, beside `style.py`'s marks (issue #312).

A **discrete legend** names states by colour with round markers, off to the
side. It takes and returns matplotlib objects and draws inside whatever rc
context the caller opened, so the document's style applies without the
layout knowing it.

Four compositions were written here and one is called. The spatial
grid, the grouped tracks, the joint distribution and the marker-size rule
were rendered by no figure and no notebook, and issue #864 removed them with
their tests: a composition nothing composes is a claim about what the
figures need that the figures do not make.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D


def discrete_legend(
    ax: Axes,
    categories: Sequence[object],
    colors: Sequence[str],
    *,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] = (1.02, 1.0),
) -> None:
    """A legend of round markers naming ``categories`` by ``colors``, placed outside the axes by default."""
    if len(categories) != len(colors):
        msg = f"{len(categories)} categories but {len(colors)} colours"
        raise ValueError(msg)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markersize=6,
            label=str(category),
        )
        for category, color in zip(categories, colors, strict=True)
    ]
    options: dict[str, Any] = {
        "loc": loc,
        "bbox_to_anchor": bbox_to_anchor,
        "frameon": False,
        "borderaxespad": 0.0,
    }
    ax.legend(handles=handles, **options)


__all__ = [
    "discrete_legend",
]
