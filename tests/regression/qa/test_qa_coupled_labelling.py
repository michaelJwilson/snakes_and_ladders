"""Regression tests for snakes_and_ladders.qa.coupled_labelling.

The three panels share one set of coordinates, and those coordinates are the
lattice's own: a randomized graph layout would place the nodes differently on
each run, so the committed figure would disagree with a re-render and no
input stamp could say why. What the panels carry is pinned too --- the
planting against the one the suite plants on this instance, the permutation
the second panel is drawn under against the accuracy the fit is measured by,
and the recovery itself against the planted truth.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from snakes_and_ladders.qa.coupled_labelling import (
    DRAW_SEED,
    FIT_SEED,
    best_permutation,
    build_figure,
    lattice_coordinates,
    main,
    planted_labelling,
    recover,
)
from snakes_and_ladders.search.spatio_sequential import label_accuracy
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

FIXTURE = fixture("spatio_sequential", "stress")
PARAMS = FIXTURE.params
SIDE = PARAMS.graph.shape[0]


@pytest.mark.structural
def test_the_coordinates_are_the_lattice_and_nothing_random() -> None:
    # The failure this prevents: node positions from a randomized layout,
    # redrawn differently on every render.
    coords = lattice_coordinates(PARAMS.graph)

    np.testing.assert_array_equal(coords, lattice_coordinates(PARAMS.graph))
    assert coords.shape == (SIDE * SIDE, 2)
    index = np.arange(SIDE * SIDE)
    np.testing.assert_array_equal(coords[:, 0], index % SIDE)
    np.testing.assert_array_equal(coords[:, 1], -(index // SIDE))
    # One node per drawn position, so nothing is hidden under anything.
    assert len({tuple(point) for point in coords}) == SIDE * SIDE


@pytest.mark.structural
def test_the_planting_is_the_one_the_suite_plants() -> None:
    # The figure draws the instance the solvers are measured on, so it plants
    # what they are measured against rather than a planting of its own.
    planted = planted_labelling(PARAMS)

    np.testing.assert_array_equal(
        planted, (np.arange(SIDE * SIDE) % SIDE < SIDE // 2).astype(np.int64)
    )
    assert np.bincount(planted).tolist() == [
        SIDE * (SIDE - SIDE // 2),
        SIDE * (SIDE // 2),
    ]


@pytest.mark.mathematical
@pytest.mark.parametrize("flipped", [0, 7, 100])
def test_the_permutation_the_panels_are_drawn_under_is_the_accuracy_one(
    flipped: int,
) -> None:
    # A fit names its classes in whatever order it found them. Panels (b) and
    # (c) are drawn in the planted classes' palette and sign, so the
    # permutation they are drawn under must be the one the reported accuracy
    # is achieved by --- else the pictures and the number disagree.
    planted = planted_labelling(PARAMS)
    fitted = planted.copy()
    fitted[:flipped] = 1 - fitted[:flipped]
    mapping = best_permutation(fitted, planted, 2)

    assert sorted(mapping.tolist()) == [0, 1]
    assert float((mapping[fitted] == planted).mean()) == label_accuracy(
        fitted, planted, 2
    )


@pytest.mark.simulated_truth
def test_the_recovered_labelling_agrees_with_the_planted_one() -> None:
    # The claim the figure makes: on this replicate the annealed start
    # recovers the planting. The suite measures a mean of 0.97 over six
    # replicates; this is the first of them, and it is exact.
    planted = planted_labelling(PARAMS)
    recovery = recover(PARAMS, planted, DRAW_SEED, FIT_SEED)

    assert recovery.accuracy == 1.0
    np.testing.assert_array_equal(recovery.labels, planted)
    assert recovery.accuracy == label_accuracy(recovery.fit.labels, planted, 2)
    assert recovery.observations.shape == (PARAMS.n_positions, SIDE * SIDE)
    assert recovery.field.shape == (SIDE * SIDE, PARAMS.n_classes)
    # Panel (c) is the field under the same permutation as panel (b), so its
    # columns are the planted classes and the sign of the margin says which
    # class the observations at a node favour on their own.
    np.testing.assert_array_equal(
        recovery.field_labelling, (recovery.margin > 0).astype(np.int64)
    )
    np.testing.assert_array_equal(
        np.sort(recovery.field, axis=1), np.sort(recovery.fit.field, axis=1)
    )
    # The claim panel (c) is in the figure to make: the field alone does not
    # separate the classes at this many nodes, and the spatial prior does.
    assert int((recovery.field_labelling != planted).sum()) == 15
    # The generators are seeded, so a second call draws and fits the same
    # replicate: the figure is a function of the fixture and the two seeds.
    again = recover(PARAMS, planted, DRAW_SEED, FIT_SEED)
    np.testing.assert_array_equal(again.observations, recovery.observations)
    np.testing.assert_array_equal(again.labels, recovery.labels)


@pytest.mark.edge_case
def test_a_lattice_the_figure_cannot_draw_is_refused() -> None:
    chain = lattice_graph((4,), BoundaryCondition.OPEN, 1.0)
    with pytest.raises(ValueError, match="2-D lattice"):
        lattice_coordinates(chain)

    oblong = replace(PARAMS, graph=lattice_graph((2, 3), BoundaryCondition.OPEN, 1.0))
    with pytest.raises(ValueError, match="square lattice"):
        planted_labelling(oblong)


@pytest.mark.structural
def test_the_caption_names_the_instance_the_seeds_and_the_agreement(
    tmp_path: Path,
) -> None:
    _, caption = build_figure(PARAMS, draw_seed=DRAW_SEED, fit_seed=FIT_SEED)

    assert f"{SIDE}x{SIDE} open lattice" in caption
    assert f"seed {DRAW_SEED}" in caption
    assert f"fitted from seed {FIT_SEED}" in caption
    assert f"0 of {SIDE * SIDE} nodes are labelled otherwise" in caption
    assert "agreement of 1.00" in caption
    assert f"misses the planting at 15 of {SIDE * SIDE} nodes" in caption

    qa_figure = main(["--params", str(FIXTURE.path), "--output-dir", str(tmp_path)])
    assert qa_figure.figure_path.is_file()
    assert qa_figure.caption == caption
