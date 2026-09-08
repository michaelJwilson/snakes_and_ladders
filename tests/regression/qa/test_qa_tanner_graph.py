"""Regression tests for snakes_and_ladders.qa.tanner_graph.

The drawing is pinned where an input stamp cannot reach: node coordinates
come from the code and from nothing else, so two renders of one tree place
every node in the same place and the committed figure survives a re-render.
The graph the coordinates are joined by is pinned to the ensemble the code
was drawn from -- one edge per nonzero, the declared degrees, and the first
band covering consecutive bits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from snakes_and_ladders.qa.tanner_graph import (
    CHECK_ROW_HEIGHT,
    build_figure,
    main,
    tanner_layout,
)
from snakes_and_ladders.sim.fixtures import fixture

FIXTURE = fixture("ldpc", "ci")
PARAMS = FIXTURE.params


@pytest.mark.mathematical
def test_the_layout_joins_every_nonzero_and_nothing_else() -> None:
    # The picture is the matrix: one segment per nonzero, from the bit's
    # coordinate to the check's, and no segment anywhere else.
    code = PARAMS.code()
    layout = tanner_layout(code, PARAMS.row_weight)

    assert layout.edges.shape == (code.n_edges, 2)
    assert code.n_edges == PARAMS.n_bits * PARAMS.column_weight
    assert code.n_edges == code.n_checks * PARAMS.row_weight
    drawn = np.zeros((code.n_checks, code.n_bits), dtype=np.uint8)
    drawn[layout.edges[:, 1], layout.edges[:, 0]] = 1
    np.testing.assert_array_equal(drawn, code.dense())


@pytest.mark.structural
def test_the_node_coordinates_are_a_function_of_the_code_alone() -> None:
    # The failure this prevents: a randomized graph layout, which renders
    # differently on each run, so the committed figure disagrees with CI's
    # re-render and no input stamp can see why.
    code = PARAMS.code()
    first = tanner_layout(code, PARAMS.row_weight)
    second = tanner_layout(PARAMS.code(), PARAMS.row_weight)

    np.testing.assert_array_equal(first.variables, second.variables)
    np.testing.assert_array_equal(first.checks, second.checks)
    np.testing.assert_array_equal(first.edges, second.edges)

    # Bits sit at their index on the baseline; checks sit one per column of
    # an even division of the same span, so no two are drawn on top of one
    # another and the row order is the mean bit each check covers.
    np.testing.assert_array_equal(
        first.variables, np.column_stack([np.arange(12), np.zeros(12)])
    )
    assert np.all(first.checks[:, 1] == CHECK_ROW_HEIGHT)
    np.testing.assert_allclose(
        np.sort(first.checks[:, 0]), 11.0 * (np.arange(6) + 0.5) / 6.0
    )
    mean_bit = np.array(
        [first.edges[first.edges[:, 1] == check, 0].mean() for check in range(6)]
    )
    assert np.array_equal(np.argsort(first.checks[:, 0]), np.argsort(mean_bit))


@pytest.mark.mathematical
def test_the_heavy_band_is_the_one_covering_consecutive_bits() -> None:
    # Gallager's construction: the first band's row i covers bits
    # i * row_weight upward, and every later band is that band under a
    # permutation of the columns. The figure draws band zero heavy, so which
    # edges are heavy is a claim about the construction and not a style.
    code = PARAMS.code()
    layout = tanner_layout(code, PARAMS.row_weight)
    rows_per_band = PARAMS.n_bits // PARAMS.row_weight

    assert set(layout.band.tolist()) == set(range(PARAMS.column_weight))
    for check in range(rows_per_band):
        bits = np.sort(layout.edges[layout.edges[:, 1] == check, 0])
        expected = np.arange(check * PARAMS.row_weight, (check + 1) * PARAMS.row_weight)
        np.testing.assert_array_equal(bits, expected)
        assert np.all(layout.band[layout.edges[:, 1] == check] == 0)
    # Every band covers every bit exactly once, which is what makes the
    # column weight exact.
    for band in range(PARAMS.column_weight):
        covered = np.sort(layout.edges[layout.band == band, 0])
        np.testing.assert_array_equal(covered, np.arange(PARAMS.n_bits))


@pytest.mark.edge_case
def test_a_row_weight_that_forms_no_bands_is_refused() -> None:
    code = PARAMS.code()

    with pytest.raises(ValueError, match="form no bands"):
        tanner_layout(code, 5)


@pytest.mark.structural
def test_the_caption_names_the_instance_that_was_drawn(tmp_path: Path) -> None:
    _, caption = build_figure(PARAMS)

    assert f"seed {PARAMS.seed}" in caption
    assert "(3, 6) Gallager ensemble" in caption
    assert "12 bits" in caption
    assert "36 of them" in caption

    qa_figure = main(["--params", str(FIXTURE.path), "--output-dir", str(tmp_path)])
    assert qa_figure.figure_path.is_file()
    assert qa_figure.caption == caption
