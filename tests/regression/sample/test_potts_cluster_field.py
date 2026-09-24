"""The cluster moves that read the field, held to the enumerated Gibbs law at 3x3 and three labels (issue #1041).

Ghost-spin Swendsen-Wang, label-directed Swendsen-Wang and Swendsen-Wang
tempering with Houdayer moves between two temperatures. The referee is
`tests._chains.enumerated_law`: all 3^9 = 19,683 labellings of a 3x3 open
lattice in a per-site field, weighted exactly, and sharing no bond, cluster or
accept step with the kernels.

**The statistic is total variation, and its bound is its own sampling
error.** A chi-square over 19,683 cells has most expected counts below one,
so the distance between the chain's frequencies and the law is read
instead, against the distance :data:`RECORDED` independent draws from the law
itself reach: the 99.9th percentile of :data:`REPLICATES` such draws, times
:data:`SLACK` for what thinning leaves correlated. Each ablation --- the
Hastings term dropped, the field's negative part dropped from the ghost
bonds --- is refuted by the same bound, which is the evidence it has power.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.sample.potts_mcmc import (
    PottsMove,
    cluster_tempering,
    sample_potts,
    sweeps,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

from tests._chains import cell_counts, enumerated_law

SHAPE = (3, 3)
N_STATES = 3
COUPLING = 0.8
#: A per-site field, one row per site, drawn once from a declared seed.
FIELD = np.random.default_rng(1041).normal(0.0, 0.6, (9, N_STATES))
#: Recorded states per chain, and the thinning between them. At 20,000 the
#: ablations sit 46% and 67% above the bound; the tempering test records
#: 7,000, at its own bound, to stay inside the per-PR duration cap: two
#: replicas and a Houdayer move a step took 9.4 s at 10,000.
RECORDED = 20_000
TEMPERED_RECORDED = 7_000
THINNING = 5
#: Independent draws from the law behind the bound.
REPLICATES = 1_000
#: What the thinned chain's residual correlation is allowed over the
#: independent draws' 99.9th percentile.
SLACK = 1.25
SEED = 1041


def _graph() -> PottsGraph:
    return lattice_graph(SHAPE, BoundaryCondition.OPEN, COUPLING)


def _total_variation(probability: np.ndarray, counts: np.ndarray) -> float:
    return 0.5 * float(np.abs(counts / counts.sum() - probability).sum())


def _bound(probability: np.ndarray, recorded: int = RECORDED) -> float:
    """The 99.9th percentile of ``recorded`` independent draws' distance, times :data:`SLACK`."""
    rng = np.random.default_rng(SEED)
    distances = [
        _total_variation(probability, rng.multinomial(recorded, probability))
        for _ in range(REPLICATES)
    ]
    return SLACK * float(np.quantile(distances, 0.999))


def _chain_distance(move: PottsMove, temperature: float = 1.0) -> tuple[float, float]:
    """The chain's distance from the enumerated law, and the bound."""
    graph = _graph()
    index, probability = enumerated_law(graph, FIELD, temperature)
    chain = sample_potts(
        graph,
        FIELD,
        move,
        np.random.default_rng(SEED),
        RECORDED,
        burn_in=RECORDED // 10,
        thin=THINNING,
        temperature=temperature,
        # The compiled Swendsen-Wang pass for the control; the other two
        # kernels' chains are the same on either union-find.
        cluster_backend=Backend.RUST,
    )
    counts = cell_counts(index, chain.states)
    return _total_variation(probability, counts), _bound(probability)


@pytest.mark.oracle
@pytest.mark.parametrize("move", [PottsMove.GHOST_SPIN, PottsMove.LABEL_DIRECTED])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_the_chain_draws_from_the_enumerated_gibbs_law(
    move: PottsMove, temperature: float
) -> None:
    distance, bound = _chain_distance(move, temperature)

    assert distance < bound, f"total variation {distance:.4f} >= {bound:.4f}"


@pytest.mark.oracle
@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG])
def test_the_control_move_meets_the_same_bound(move: PottsMove) -> None:
    # Swendsen-Wang, exact by the chi-square of `test_potts_mcmc.py`, read by
    # the same statistic: the bound admits a correct cluster move.
    distance, bound = _chain_distance(move)

    assert distance < bound, f"total variation {distance:.4f} >= {bound:.4f}"


@pytest.mark.oracle
def test_dropping_the_hastings_term_is_refuted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Onto the target is proposed from every other label and off it with
    # probability (q - 1) / q, so without the log q term the target is
    # over-weighted.
    monkeypatch.setattr(sweeps, "_label_hastings", lambda _: 0.0)

    distance, bound = _chain_distance(PottsMove.LABEL_DIRECTED)

    assert distance > bound, f"total variation {distance:.4f} <= {bound:.4f}"


@pytest.mark.oracle
def test_dropping_the_fields_negative_part_is_refuted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `max(0, h)` without the shift: a label a site's field penalizes is
    # left as likely as the zero of the field, and nothing accepts it back.
    monkeypatch.setattr(sweeps, "ghost_couplings", lambda rows: np.maximum(rows, 0.0))

    distance, bound = _chain_distance(PottsMove.GHOST_SPIN)

    assert distance > bound, f"total variation {distance:.4f} <= {bound:.4f}"


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("move", [PottsMove.GHOST_SPIN, PottsMove.LABEL_DIRECTED])
def test_the_compiled_union_find_is_the_same_chain(move: PottsMove) -> None:
    # The kernels differ between backends in the union-find alone, whose
    # roots `test_bond_roots.py` pins bitwise, so the chains are equal.
    graph = lattice_graph((6, 6), BoundaryCondition.OPEN, COUPLING)
    field = np.random.default_rng(7).normal(0.0, 0.6, (graph.n_nodes, N_STATES))
    runs = [
        sample_potts(
            graph,
            field,
            move,
            np.random.default_rng(SEED),
            200,
            cluster_backend=backend,
        ).states
        for backend in (Backend.PYTHON, Backend.RUST)
    ]

    assert np.array_equal(runs[0], runs[1])


@pytest.mark.oracle
def test_tempering_with_houdayer_draws_each_replica_from_its_law() -> None:
    # Two temperatures and a Houdayer move between them every step, which
    # needs the accept step across the pair; each replica is read against the
    # law at its own temperature.
    graph = _graph()
    temperatures = (1.0, 1.4)
    run = cluster_tempering(
        graph,
        FIELD,
        temperatures,
        np.random.default_rng(SEED),
        TEMPERED_RECORDED,
        burn_in=TEMPERED_RECORDED // 10,
        thin=THINNING,
        record=True,
    )

    assert len(run.houdayer_sizes) > TEMPERED_RECORDED
    assert 0.0 < run.houdayer_acceptance[0] < 1.0
    for replica, temperature in enumerate(temperatures):
        index, probability = enumerated_law(graph, FIELD, temperature)
        counts = cell_counts(index, run.states[:, replica])
        distance = _total_variation(probability, counts)
        bound = _bound(probability, TEMPERED_RECORDED)
        assert distance < bound, f"replica {replica}: {distance:.4f} >= {bound:.4f}"


@pytest.mark.smoke
def test_a_ladder_not_coldest_first_is_refused() -> None:
    with pytest.raises(ValueError, match="increasing"):
        cluster_tempering(_graph(), FIELD, (1.4, 1.0), np.random.default_rng(0), 1)
