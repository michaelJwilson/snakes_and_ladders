"""Shared figure layouts: the compositions more than one figure needs, built once (issue #312).

`style.py` says how a mark looks; this module says how a figure is arranged.
Four compositions recur across the problem classes. A **spatial grid** puts
one panel per feature over the same 2-D coordinates -- a lattice's
labelling beside its marginals, the coupled model's fields over its sites.
**Grouped tracks** stack one axis per track with a gap between groups -- an
HMM's posterior per state along the positions, one group per chain. A
**joint distribution** is a scatter with outlined marginal histograms per
group -- a surrogate's value against the exact one, or a distance against a
likelihood gap, coloured by which alignment or which move produced it. A
**discrete legend** names states by colour with round markers, off to the
side. Every function takes and returns matplotlib objects and draws inside
whatever rc context the caller opened, so the document's style applies
without the layout knowing it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from snakes_and_ladders.qa.style import INK, RULE

# Gap between track groups, as a fraction of one track's height.
TRACK_GAP = 0.25


def marker_size(n_points: int) -> float:
    """Scatter marker area that shrinks with point count, clamped to ``[0.1, 25]``."""
    if n_points < 1:
        msg = f"n_points must be positive, got {n_points}"
        raise ValueError(msg)
    return float(np.clip(12_000.0 / n_points, 0.1, 25.0))


def spatial_grid(
    coords: np.ndarray,
    features: Mapping[str, np.ndarray],
    *,
    base_size: float = 3.0,
    max_cols: int = 3,
    cmap: str = "viridis",
) -> tuple[Figure, list[Axes]]:
    """One panel per feature over shared 2-D coordinates, zeros drawn hollow, a colourbar per panel.

    Parameters
    ----------
    coords : np.ndarray
        Shape ``(n_points, 2)``.
    features : Mapping[str, np.ndarray]
        Name to one value per point; the name titles its panel and labels
        its colourbar. Panels are filled row by row, ``max_cols`` per row,
        and any axis left over is switched off.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        msg = f"coords must be (n_points, 2), got {coords.shape}"
        raise ValueError(msg)
    if not features:
        msg = "spatial_grid needs at least one feature"
        raise ValueError(msg)
    n_features = len(features)
    cols = min(n_features, max_cols)
    rows = math.ceil(n_features / cols)
    fig, axes_grid = plt.subplots(
        rows, cols, figsize=(base_size * cols, base_size * rows), squeeze=False
    )
    axes: list[Axes] = list(axes_grid.flatten())
    size = marker_size(coords.shape[0])
    for ax, (name, raw) in zip(axes, features.items(), strict=False):
        values = np.asarray(raw, dtype=float)
        if values.shape != (coords.shape[0],):
            msg = f"feature {name!r} has shape {values.shape}, expected ({coords.shape[0]},)"
            raise ValueError(msg)
        zero = values == 0.0
        if zero.any():
            ax.scatter(
                coords[zero, 0],
                coords[zero, 1],
                c="white",
                s=size,
                edgecolor=INK,
                linewidth=0.1,
            )
        if (~zero).any():
            drawn = ax.scatter(
                coords[~zero, 0],
                coords[~zero, 1],
                c=values[~zero],
                s=size,
                cmap=cmap,
                edgecolor="none",
                rasterized=True,
            )
            fig.colorbar(drawn, ax=ax, fraction=0.046, pad=0.04).set_label(name)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes[n_features:]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def grouped_tracks(
    n_groups: int,
    tracks_per_group: int,
    *,
    width: float = 5.4,
    track_height: float = 0.8,
    title: str | None = None,
) -> tuple[Figure, list[Axes]]:
    """``n_groups * tracks_per_group`` axes stacked vertically, a gap of `TRACK_GAP` between groups.

    Axes are returned group by group, top to bottom; within a group the
    tracks abut, and a group is separated from the next by an empty row of
    the grid so the eye reads the grouping without a frame.
    """
    if n_groups < 1 or tracks_per_group < 1:
        msg = f"need positive counts, got {n_groups} groups of {tracks_per_group}"
        raise ValueError(msg)
    ratios: list[float] = []
    for group in range(n_groups):
        ratios.extend([1.0] * tracks_per_group)
        if group < n_groups - 1:
            ratios.append(TRACK_GAP)
    fig = plt.figure(figsize=(width, track_height * sum(ratios)))
    grid = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.0)
    axes: list[Axes] = []
    row = 0
    for group in range(n_groups):
        for _ in range(tracks_per_group):
            axes.append(fig.add_subplot(grid[row, 0]))
            row += 1
        if group < n_groups - 1:
            row += 1
    if title:
        fig.suptitle(title)
    return fig, axes


def format_track_axis(
    ax: Axes,
    ylabel: str,
    ylim: tuple[float, float],
    yticks: Sequence[float],
    *,
    remove_xticks: bool = True,
    max_x: float | None = None,
) -> None:
    """Label, limits and gridlines a track shares with its neighbours, so a column of tracks reads as one."""
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([f"{tick:.1f}" for tick in yticks])
    if max_x is not None:
        ax.set_xlim(0.0, max_x)
    if remove_xticks:
        ax.set_xticks([])
    for tick in yticks:
        ax.axhline(tick, color=RULE, linewidth=0.5, zorder=0)


def joint_distribution(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    palette: Mapping[int, str],
    *,
    x_label: str = "x",
    y_label: str = "y",
    bins: int = 50,
    size: float = 5.4,
) -> tuple[Figure, dict[str, Axes]]:
    """A scatter with an outlined marginal histogram per group above and to the right.

    Non-finite points are dropped from both the scatter and the marginals.
    The axes come back by name -- ``joint``, ``top``, ``right`` -- so a
    caller can add a reference line to the joint panel.
    """
    x, y, groups = (
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(groups),
    )
    if not x.shape == y.shape == groups.shape:
        msg = f"x, y and groups must share a shape, got {x.shape}, {y.shape}, {groups.shape}"
        raise ValueError(msg)
    keep = np.isfinite(x) & np.isfinite(y)
    if not keep.any():
        msg = "joint_distribution needs at least one finite point"
        raise ValueError(msg)
    fig = plt.figure(figsize=(size, size))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=0.05, hspace=0.05
    )
    joint = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=joint)
    right = fig.add_subplot(grid[1, 1], sharey=joint)
    bins_x = np.histogram_bin_edges(x[keep], bins=bins)
    bins_y = np.histogram_bin_edges(y[keep], bins=bins)
    handles = []
    for group in np.unique(groups[keep]):
        mask = keep & (groups == group)
        color = palette.get(int(group), INK)
        joint.scatter(
            x[mask], y[mask], s=4, marker=".", alpha=0.6, color=color, edgecolor="none"
        )
        top.hist(
            x[mask], bins=bins_x.tolist(), histtype="step", color=color, linewidth=1.0
        )
        right.hist(
            y[mask],
            bins=bins_y.tolist(),
            histtype="step",
            color=color,
            linewidth=1.0,
            orientation="horizontal",
        )
        handles.append(Patch(facecolor="none", edgecolor=color, label=str(group)))
    joint.set_xlabel(x_label)
    joint.set_ylabel(y_label)
    joint.legend(handles=handles, loc="upper left", framealpha=0.0)
    for marginal in (top, right):
        marginal.axis("off")
    return fig, {"joint": joint, "top": top, "right": right}


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
    "TRACK_GAP",
    "discrete_legend",
    "format_track_axis",
    "grouped_tracks",
    "joint_distribution",
    "marker_size",
    "spatial_grid",
]
