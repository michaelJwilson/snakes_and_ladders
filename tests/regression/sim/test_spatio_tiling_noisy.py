"""The noisy, strongly coupled tiling fixture and its held-out validation set (issue #1074).

`sim.potts.smoothed_noise` is judged against its definition, one neighbour
average written here over the edge list; `search.potts_starts.held_tiles`
against the same margin and outside coupling recomputed site by site, and
against `recovery_bound` where the field is noise-free. The fixture is pinned
by a recorded digest. On the held-out instances the fixture declares,
alpha-expansion's energy is compared with ICM's on the paired difference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from sal.search.alpha_expansion import alpha_expansion
from sal.search.icm import iterated_conditional_modes
from sal.search.potts_starts import held_tiles, recovery_bound, tiling_rung
from sal.search.trws import trws
from sal.sim.fixtures import fixture
from sal.sim.graph import BoundaryCondition, triangular_lattice_graph
from sal.sim.potts import SpatioTilingParams, smoothed_noise

#: The two declared tiers, loaded once each.
CI = fixture("spatio_tiling_noisy", "ci")
RELEASE = fixture("spatio_tiling_noisy", "release")


def _declared(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        loaded: dict[str, Any] = yaml.safe_load(handle)
    return loaded


def _argmax_icm(params: SpatioTilingParams) -> float:
    """ICM's energy from the field's argmax, run to convergence."""
    run = iterated_conditional_modes(
        params.graph,
        params.field,
        params.n_states,
        np.random.default_rng(0),
        start=params.field.argmax(axis=1).astype(np.int64),
        max_sweeps=100_000,
    )
    assert run.termination is not None
    assert run.termination.converged
    return run.energy


@pytest.mark.smoke
@pytest.mark.snapshot
@pytest.mark.parametrize("loaded", [CI, RELEASE], ids=["ci", "release"])
def test_the_field_reproduces_from_its_seeds(loaded: Any) -> None:
    digest = hashlib.sha256(loaded.params.field.tobytes()).hexdigest()[:16]

    assert digest == _declared(loaded.path)["digest"]


@pytest.mark.oracle
@pytest.mark.parametrize("rounds", [0, 1, 3])
def test_the_noise_is_its_neighbour_average(rounds: int) -> None:
    graph = triangular_lattice_graph((7, 6), BoundaryCondition.OPEN, 1.0)
    white = np.random.default_rng(5).standard_normal((graph.n_nodes, 4))
    expected = white.copy()
    for _ in range(rounds):
        total = expected.copy()
        count = np.ones(graph.n_nodes)
        for left, right in graph.edges:
            total[left] += expected[right]
            total[right] += expected[left]
            count[left] += 1
            count[right] += 1
        expected = total / count[:, np.newaxis]
    expected /= expected.std()

    drawn = smoothed_noise(graph, 4, rounds, np.random.default_rng(5))

    np.testing.assert_allclose(drawn, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("loaded", [CI, RELEASE], ids=["ci", "release"])
def test_held_tiles_is_the_site_margin_over_the_outside_coupling(
    loaded: Any,
) -> None:
    params: SpatioTilingParams = loaded.params
    rung = tiling_rung(params, "noisy")
    slack = np.full(params.n_tiles, np.inf)
    outside = np.zeros(params.graph.n_nodes)
    for (left, right), coupling in zip(
        params.graph.edges, params.graph.edge_coupling, strict=True
    ):
        if params.tiles[left] != params.tiles[right]:
            outside[left] += coupling
            outside[right] += coupling
    for node in range(params.graph.n_nodes):
        tile = params.tiles[node]
        row = params.field[node]
        favoured = params.states[tile]
        margin = row[favoured] - max(
            row[state] for state in range(params.n_states) if state != favoured
        )
        slack[tile] = min(slack[tile], margin - outside[node])

    assert np.array_equal(held_tiles(rung), slack > 0.0)


@pytest.mark.oracle
@pytest.mark.parametrize("tier", ["ci", "release"])
def test_held_tiles_is_the_recovery_bound_on_a_noise_free_field(tier: str) -> None:
    rung = tiling_rung(fixture("spatio_tiling", tier).params, tier)

    assert np.array_equal(held_tiles(rung), rung.strengths > recovery_bound(rung))


@pytest.mark.end2end
def test_expansion_labels_every_held_tile_its_planted_state() -> None:
    params: SpatioTilingParams = CI.params
    rung = tiling_rung(params, "ci")
    labelling = alpha_expansion(params.graph, params.field, params.n_states).labelling
    held = held_tiles(rung)

    assert held.any()
    for tile in np.flatnonzero(held):
        assert np.all(labelling[params.tiles == tile] == params.states[tile])


@pytest.mark.experiment
def test_expansion_is_below_icm_and_above_the_bound_at_ci() -> None:
    params: SpatioTilingParams = CI.params
    bound = trws(params.graph, params.field).bound
    expansion = alpha_expansion(params.graph, params.field, params.n_states).energy

    assert bound <= expansion + 1e-9
    assert expansion < _argmax_icm(params)


@pytest.mark.release
@pytest.mark.experiment
def test_expansion_beats_icm_on_the_held_out_set_by_more_than_two_se() -> None:
    params: SpatioTilingParams = RELEASE.params
    validation = _declared(RELEASE.path)["validation"]
    recorded = np.asarray(validation["recorded"], dtype=float)
    rows = []
    for index in range(int(validation["count"])):
        held_out = params.redrawn(
            int(validation["tiling_seed"]) + index,
            int(validation["noise_seed"]) + index,
        )
        rows.append(
            [
                trws(held_out.graph, held_out.field).bound,
                alpha_expansion(
                    held_out.graph, held_out.field, held_out.n_states
                ).energy,
                _argmax_icm(held_out),
            ]
        )
    measured = np.asarray(rows)
    difference = measured[:, 2] - measured[:, 1]
    standard_error = difference.std(ddof=1) / np.sqrt(difference.size)

    np.testing.assert_allclose(measured, recorded, rtol=1e-12)
    assert bool(np.all(measured[:, 0] <= measured[:, 1] + 1e-9))
    assert difference.mean() > 2.0 * standard_error
