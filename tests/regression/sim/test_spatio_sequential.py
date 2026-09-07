"""The coupled model's simulator, held to the distributions it composes.

Labels against the enumerated Potts prior, transitions against the circulant
rate, emissions against the families' own tables -- each a goodness-of-fit at
the significance the sampler tests use, over independent draws under one
generator. A simulator that ran and returned the right shapes would pass a
weaker test and prove nothing (root `CLAUDE.md`).
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
from snakes_and_ladders.emissions import CategoricalEmission
from snakes_and_ladders.search.statistics import chi_square_p_value
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.spatio_sequential import (
    SimulatedSpatioSequential,
    SpatioSequentialParams,
    canonical_spatio_sequential,
    circulant_transition,
    simulate_spatio_sequential,
)

SIGNIFICANCE = 0.001
N_DRAWS = 400
BURN_IN = 200


def _draws(n: int) -> list[SimulatedSpatioSequential]:
    params = canonical_spatio_sequential()
    rng = np.random.default_rng(7)
    return [simulate_spatio_sequential(params, rng, burn_in=BURN_IN) for _ in range(n)]


@pytest.mark.simulated_truth
def test_the_labels_are_drawn_from_the_potts_prior() -> None:
    params = canonical_spatio_sequential()
    configurations = list(product(range(params.n_classes), repeat=4))
    energies = np.zeros(len(configurations))
    for index, labelling in enumerate(configurations):
        for (first, second), coupling in params.graph.weighted_edges():
            energies[index] += (
                params.beta * coupling * (labelling[first] == labelling[second])
            )
    expected = np.exp(energies - energies.max())
    expected = N_DRAWS * expected / expected.sum()

    observed = np.zeros(len(configurations))
    for draw in _draws(N_DRAWS):
        observed[configurations.index(tuple(int(v) for v in draw.labels))] += 1

    assert chi_square_p_value(observed, expected) > SIGNIFICANCE


@pytest.mark.simulated_truth
def test_the_chains_follow_the_circulant_transition_and_the_initial() -> None:
    params = canonical_spatio_sequential()
    draws = _draws(N_DRAWS)
    stays = sum(
        int((draw.states[:, 1:] == draw.states[:, :-1]).sum()) for draw in draws
    )
    total = N_DRAWS * params.n_classes * (params.n_positions - 1)
    expected = np.array([params.self_transition, 1 - params.self_transition]) * total

    assert chi_square_p_value(np.array([stays, total - stays]), expected) > SIGNIFICANCE

    for m in range(params.n_classes):
        first = np.array([draw.states[m, 0] for draw in draws])
        observed = np.bincount(first, minlength=params.n_states).astype(float)
        assert (
            chi_square_p_value(observed, N_DRAWS * params.initial[m]) > SIGNIFICANCE
        ), m


@pytest.mark.simulated_truth
def test_the_observations_come_from_the_class_of_the_node_at_the_state_of_its_chain() -> (
    None
):
    params = canonical_spatio_sequential()
    draws = _draws(N_DRAWS)
    n_symbols = 3
    for m, family in enumerate(params.emissions):
        assert isinstance(family, CategoricalEmission)
        for k in range(params.n_states):
            counts = np.zeros(n_symbols)
            for draw in draws:
                nodes = np.flatnonzero(draw.labels == m)
                positions = np.flatnonzero(draw.states[m] == k)
                block = draw.observations[np.ix_(positions, nodes)]
                counts += np.bincount(block.ravel(), minlength=n_symbols)
            expected = counts.sum() * family.matrix[k].numpy()
            assert chi_square_p_value(counts, expected) > SIGNIFICANCE, (m, k)


@pytest.mark.mathematical
def test_the_circulant_transition_is_row_stochastic_with_the_declared_diagonal() -> (
    None
):
    transition = circulant_transition(4, 0.55)

    np.testing.assert_allclose(transition.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.diag(transition), 0.55)
    off = transition[~np.eye(4, dtype=bool)]
    np.testing.assert_allclose(off, off[0])


@pytest.mark.structural
def test_planted_labels_are_kept_and_the_generator_reproduces_the_draw() -> None:
    params = canonical_spatio_sequential()
    planted = np.array([0, 1, 1, 0])

    first = simulate_spatio_sequential(params, np.random.default_rng(3), labels=planted)
    second = simulate_spatio_sequential(
        params, np.random.default_rng(3), labels=planted
    )

    assert np.array_equal(first.labels, planted)
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.observations, second.observations)
    assert first.observations.shape == (params.n_positions, params.graph.n_nodes)


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            {
                "graph": PottsGraph(
                    n_nodes=4,
                    edges=((0, 1), (1, 2), (2, 3), (3, 0)),
                    coupling=(1, 1, -1, 1),
                )
            },
            "ferromagnetic",
        ),
        ({"self_transition": 1.0}, "self_transition"),
        ({"initial": np.array([[0.6, 0.4]])}, "shape"),
        ({"initial": np.array([[0.6, 0.6], [0.3, 0.7]])}, "distribution"),
        (
            {
                "emissions": (
                    CategoricalEmission(np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])),
                )
            },
            "families",
        ),
        (
            {
                "emissions": (
                    CategoricalEmission(np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])),
                    CategoricalEmission(np.array([[0.5, 0.5]])),
                )
            },
            "states",
        ),
        ({"beta": -0.1}, "beta"),
    ],
)
def test_an_inconsistent_truth_is_refused(change: dict[str, Any], match: str) -> None:
    params = canonical_spatio_sequential()
    with pytest.raises(ValueError, match=match):
        replace(params, **change)


@pytest.mark.edge_case
def test_planted_labels_of_the_wrong_shape_or_range_are_refused() -> None:
    params = canonical_spatio_sequential()
    with pytest.raises(ValueError, match="planted labels"):
        simulate_spatio_sequential(
            params, np.random.default_rng(0), labels=np.array([0, 1])
        )
    with pytest.raises(ValueError, match="planted labels"):
        simulate_spatio_sequential(
            params, np.random.default_rng(0), labels=np.array([0, 1, 2, 0])
        )


@pytest.mark.edge_case
def test_the_open_lattice_is_the_graph_the_canonical_instance_declares() -> None:
    params = canonical_spatio_sequential()
    assert params.graph == lattice_graph((2, 2), BoundaryCondition.OPEN, 1.0)
    assert isinstance(params, SpatioSequentialParams)
