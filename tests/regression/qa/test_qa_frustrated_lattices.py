"""Regression tests for snakes_and_ladders.qa.frustrated_lattices.

The lattice drawing is pinned to the geometry it claims -- every bond at
unit length, every wrapped bond ending at a translate of its target -- and
the ground state it colours to the closed form; the glass panel to the
gauge argument that makes frustration zero trivial.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from snakes_and_ladders.qa.frustrated_lattices import (
    FRUSTRATIONS,
    LATTICE_SHAPE,
    GlassScan,
    build_figure,
    edge_segments,
    glass_scan,
    ground_state,
    lattice_layout,
    main,
)
from snakes_and_ladders.sim.canonical import (
    frustrated_triangular_lattice,
    minimum_frustrated_edges,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph


@pytest.mark.oracle
def test_the_coloured_ground_state_agrees_on_exactly_one_bond_in_three() -> None:
    graph = frustrated_triangular_lattice(LATTICE_SHAPE)
    labelling, agreeing = ground_state(graph)

    assert agreeing == minimum_frustrated_edges(graph) == graph.n_nodes
    assert (
        sum(labelling[first] == labelling[second] for first, second in graph.edges)
        == agreeing
    )


@pytest.mark.mathematical
def test_every_drawn_bond_has_unit_length_and_wraps_only_across_the_torus() -> None:
    graph = frustrated_triangular_lattice(LATTICE_SHAPE)
    layout = lattice_layout(graph)
    rows, columns = LATTICE_SHAPE
    row_height = rows * np.sqrt(3.0) / 2.0

    wrapped = 0
    for (_, second), (start, end, wraps) in zip(
        graph.edges, edge_segments(graph), strict=True
    ):
        assert np.isclose(np.linalg.norm(end - start), 1.0)
        if wraps:
            wrapped += 1
            # The ghost end is the target translated by whole lattice periods.
            shift = end - layout[second]
            row_periods = shift[1] / row_height
            column_periods = (shift[0] + row_periods * rows / 2.0) / columns
            steps = np.array([row_periods, column_periods])
            assert np.allclose(steps, np.round(steps))
            assert np.any(np.round(steps) != 0)
        else:
            assert np.allclose(end, layout[second])
    # An open 3x3 triangular lattice has 6 row, 6 column and 4 diagonal bonds
    # of the periodic lattice's 27; the other 11 wrap.
    assert wrapped == 27 - 16


@pytest.mark.edge_case
def test_a_non_lattice_graph_is_refused_by_the_layout() -> None:
    chain = lattice_graph((4,), BoundaryCondition.OPEN, 1.0)
    with pytest.raises(ValueError, match="2-D lattice"):
        lattice_layout(chain)
    with pytest.raises(ValueError, match="2-D lattice"):
        edge_segments(chain)


@pytest.mark.oracle
def test_descent_reaches_the_planted_energy_where_nothing_is_frustrated() -> None:
    # At frustration zero the instance is a gauge transform of the
    # ferromagnet, so the planted state is the ground state and descent from
    # any start reaches its energy; above zero descent may only tie or beat
    # the planted energy, never sit above it after twenty restarts on this
    # instance size.
    scan = glass_scan(seed=20260908)

    assert FRUSTRATIONS[0] == 0.0
    assert scan.descended[0] == scan.planted[0]
    assert scan.planted.shape == scan.descended.shape == (len(FRUSTRATIONS),)


@pytest.mark.structural
def test_the_caption_counts_matches_and_undercuts(tmp_path: Path) -> None:
    graph = frustrated_triangular_lattice(LATTICE_SHAPE)
    labelling, agreeing = ground_state(graph)
    scan = GlassScan(
        planted=np.array([-6.0, -5.0, -4.0] + [0.0] * (len(FRUSTRATIONS) - 3)),
        descended=np.array([-6.0, -7.0, -4.0] + [-1.0] * (len(FRUSTRATIONS) - 3)),
    )
    _, caption = build_figure(graph, labelling, agreeing, scan)

    assert "9 of 27" in caption
    assert f"at 2 of {len(FRUSTRATIONS)} frustrations and goes below it at 6" in caption

    qa_figure = main(["--output-dir", str(tmp_path)])
    assert qa_figure.figure_path.is_file()
    assert "seed 20260908" in qa_figure.caption
