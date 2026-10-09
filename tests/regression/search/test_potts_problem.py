"""`sal.search.potts_problem.PottsProblem` (issue #1413): pinned to the solvers it replaces, and by hand where the semantics are new."""

from __future__ import annotations

import numpy as np
import pytest
from sal.opt.termination import Stop
from sal.sample.potts_mcmc import PottsMove, anneal_potts
from sal.sample.potts_mcmc.chains import merge_labels
from sal.sample.schedule import Polish
from sal.search.ground_state import ANNEAL_SCHEDULE
from sal.search.icm import (
    FloorPolicy,
    SweepOrder,
    _floor_smallest_first,
    iterated_conditional_modes,
)
from sal.search.potts_problem import FloorAt, MergeGain, MergePairs, PottsProblem
from sal.sim.fixtures import fixture
from sal.sim.graph import PottsGraph
from sal.sim.potts import energy
from sal.sim.potts_cell import PlantedPotts


def _cell() -> PlantedPotts:
    cell: PlantedPotts = fixture("potts_labelling", "ci").params.instance()
    return cell


def _starts(cell: PlantedPotts) -> list[np.ndarray]:
    field = np.asarray(cell.field)
    return [
        np.argmax(field, axis=1).astype(np.int64),
        np.random.default_rng(1).integers(0, cell.n_states, cell.graph.n_nodes),
    ]


@pytest.mark.oracle
@pytest.mark.backend
def test_index_icm_is_iterated_conditional_modes_bitwise() -> None:
    cell = _cell()
    problem = PottsProblem(cell.graph, cell.field)
    for start in _starts(cell):
        want = iterated_conditional_modes(
            cell.graph, cell.field, np.random.default_rng(0), start=start
        )
        got = problem.icm(np.random.default_rng(0), start=start)
        np.testing.assert_array_equal(got.labelling, want.labelling)
        assert (got.sweeps, got.termination) == (want.sweeps, want.termination)
        # The Rust loop's energy, each edge from both ends and halved.
        assert got.energy == pytest.approx(want.energy, rel=1e-12)


@pytest.mark.oracle
@pytest.mark.backend
def test_skipping_clean_sites_is_the_full_sweep_bitwise() -> None:
    from sal import oxisal

    cell = _cell()
    offsets, neighbours, couplings = cell.graph.compressed_adjacency()
    field = np.ascontiguousarray(cell.field, dtype=np.float64)
    for compact in (True, False):
        held = oxisal.PottsProblem(offsets, neighbours, couplings, field, compact)
        for start in _starts(cell):
            for order in (0, 1, 2):
                runs = [
                    held.icm(start, order, 0, 0, 0, 200, 7, skip)
                    for skip in (True, False)
                ]
                np.testing.assert_array_equal(runs[0][0], runs[1][0])
                assert runs[0][1:] == runs[1][1:]


def _sweep_then_floor(
    graph: PottsGraph, field: np.ndarray, start: np.ndarray, min_sites: int
) -> tuple[np.ndarray, int]:
    """The oracle of ``SMALLEST_FIRST_BEST_FIELD`` at ``SWEEP``: one index sweep, then the Python floor, until neither moves a site."""
    labels, sweeps = start.copy(), 0
    while sweeps < 200:
        sweeps += 1
        swept = iterated_conditional_modes(
            graph, field, np.random.default_rng(0), start=labels, max_iterations=1
        ).labelling
        changed = not np.array_equal(swept, labels)
        labels = np.asarray(swept).copy()
        if _floor_smallest_first(labels, field, min_sites) > 0:
            changed = True
        if not changed:
            break
    return labels, sweeps


@pytest.mark.oracle
@pytest.mark.backend
def test_the_smallest_first_floor_is_the_python_floor_after_each_sweep() -> None:
    cell = _cell()
    field = np.asarray(cell.field, dtype=float)
    problem = PottsProblem(cell.graph, field)
    for start in _starts(cell):
        want, sweeps = _sweep_then_floor(cell.graph, field, start, 30)
        got = problem.icm(
            np.random.default_rng(0),
            start=start,
            min_sites=30,
            policy=FloorPolicy.SMALLEST_FIRST_BEST_FIELD,
        )
        np.testing.assert_array_equal(got.labelling, want)
        assert got.sweeps == sweeps


@pytest.mark.oracle
@pytest.mark.backend
def test_the_full_merge_is_merge_labels_bitwise() -> None:
    cell = _cell()
    field = np.asarray(cell.field, dtype=float)
    problem = PottsProblem(cell.graph, field)
    descended = [
        iterated_conditional_modes(
            cell.graph, field, np.random.default_rng(0), start=s
        ).labelling
        for s in _starts(cell)
    ]
    for labelling in [*_starts(cell), *descended]:
        want, rounds = merge_labels(cell.graph, field, labelling)
        got = problem.merge(labelling)
        np.testing.assert_array_equal(got.labelling, want)
        assert got.sweeps == rounds
        assert got.energy == pytest.approx(energy(cell.graph, field, want), rel=1e-12)


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("polish", [None, Polish.ICM, Polish.ICM_MERGE], ids=str)
def test_the_anneal_is_anneal_potts_bitwise(polish: Polish | None) -> None:
    cell = _cell()
    schedule = ANNEAL_SCHEDULE.build(60)
    start = _starts(cell)[1]
    want = anneal_potts(
        cell.graph,
        cell.field,
        schedule,
        np.random.default_rng(4),
        move=(PottsMove.SINGLE_SITE,),
        start=start,
        polish=polish,
    )
    got = PottsProblem(cell.graph, cell.field).anneal(
        schedule,
        np.random.default_rng(4),
        start=start,
        move=(PottsMove.SINGLE_SITE,),
        polish=polish,
    )
    np.testing.assert_array_equal(got.labelling, want.best)
    assert got.termination == want.termination
    assert got.energy == pytest.approx(want.energy, rel=1e-12)


def _path(field: list[list[float]]) -> PottsProblem:
    """A path over ``len(field)`` sites at unit coupling."""
    n = len(field)
    edges = tuple((i, i + 1) for i in range(n - 1))
    graph = PottsGraph(n_nodes=n, edges=edges, coupling=(1.0,) * len(edges))
    return PottsProblem(graph, np.array(field, dtype=float))


@pytest.mark.analytic
def test_the_halved_gain_misses_the_merge_the_full_gain_takes() -> None:
    # 0 0 | 1 1 on a path: merging 1 into 0 gains the one unit bond and loses
    # 0.3 per site of label 1, so -1 + 0.6 under the full gain, -0.5 + 0.6
    # under the halved.
    problem = _path([[0.0, -5.0], [0.0, -5.0], [0.0, 0.3], [0.0, 0.3]])
    split = np.array([0, 0, 1, 1])
    full = problem.merge(split)
    np.testing.assert_array_equal(full.labelling, [0, 0, 0, 0])
    assert full.energy == pytest.approx(-3.0)
    halved = problem.merge(split, gain=MergeGain.HALVED)
    np.testing.assert_array_equal(halved.labelling, split)
    assert halved.sweeps == 1


@pytest.mark.analytic
def test_adjacent_pairs_refuse_a_merge_across_a_third_label() -> None:
    # 0 | 1 | 2: label 2's site prefers label 0 by 2, no bond between them;
    # every merge pays the bond it gains, so only the far pair (0, 2) pays.
    field = [[0.0, -5.0, -5.0], [-5.0, 0.0, -5.0], [2.0, -5.0, 0.0]]
    problem = _path(field)
    labels = np.array([0, 1, 2])
    every = problem.merge(labels)
    np.testing.assert_array_equal(every.labelling, [0, 1, 0])
    adjacent = problem.merge(labels, pairs=MergePairs.ADJACENT)
    np.testing.assert_array_equal(adjacent.labelling, labels)


@pytest.mark.analytic
@pytest.mark.parametrize(
    "order", [SweepOrder.WORKLIST, SweepOrder.RANDOM, SweepOrder.CHECKERBOARD], ids=str
)
def test_every_order_stops_at_a_labelling_an_index_sweep_leaves(
    order: SweepOrder,
) -> None:
    cell = _cell()
    problem = PottsProblem(cell.graph, cell.field)
    for start in _starts(cell):
        got = problem.icm(np.random.default_rng(3), start=start, order=order)
        assert got.termination.reason is Stop.CONVERGED
        again = iterated_conditional_modes(
            cell.graph, cell.field, np.random.default_rng(0), start=got.labelling
        )
        np.testing.assert_array_equal(again.labelling, got.labelling)
        assert again.sweeps == 1


@pytest.mark.analytic
@pytest.mark.parametrize("policy", list(FloorPolicy), ids=str)
@pytest.mark.parametrize("floor_at", list(FloorAt), ids=str)
def test_a_converged_floor_leaves_no_state_below_it(
    policy: FloorPolicy, floor_at: FloorAt
) -> None:
    cell = _cell()
    problem = PottsProblem(cell.graph, cell.field)
    for start in _starts(cell):
        got = problem.icm(
            np.random.default_rng(5),
            start=start,
            min_sites=30,
            policy=policy,
            floor_at=floor_at,
        )
        counts = np.bincount(got.labelling, minlength=cell.n_states)
        if got.termination.reason is Stop.CONVERGED:
            assert not ((counts > 0) & (counts < 30)).any()
        assert got.energy == pytest.approx(
            energy(cell.graph, cell.field, got.labelling), rel=1e-12
        )


@pytest.mark.analytic
def test_set_field_is_a_new_problem() -> None:
    cell = _cell()
    other = np.asarray(cell.field)[::-1].copy()
    held = PottsProblem(cell.graph, cell.field)
    held.set_field(other)
    fresh = PottsProblem(cell.graph, other)
    start = _starts(cell)[0]
    got, want = (p.icm(np.random.default_rng(0), start=start) for p in (held, fresh))
    np.testing.assert_array_equal(got.labelling, want.labelling)
    np.testing.assert_array_equal(held.argmax().labelling, np.argmax(other, axis=1))


@pytest.mark.experiment
def test_the_floor_oscillates_on_the_hard_variant_as_experiment_040_records() -> None:
    # Experiment 040: on `hard` the floored ICM reaches its 200-sweep budget.
    params = fixture("potts_labelling", "stress").params
    cell = params.variant("hard").instance()
    problem = PottsProblem(cell.graph, cell.field)
    floor = round(50 / 3000 * cell.graph.n_nodes)
    start = np.argmax(np.asarray(cell.field), axis=1)
    got = problem.icm(np.random.default_rng(0), start=start, min_sites=floor)
    assert (got.sweeps, got.termination.reason) == (200, Stop.BUDGET)
    plain = problem.icm(np.random.default_rng(0), start=start)
    assert plain.termination.reason is Stop.CONVERGED
