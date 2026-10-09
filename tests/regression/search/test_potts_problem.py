"""`sal.search.potts_problem.PottsProblem` (issue #1413): pinned to the solvers it replaces, and by hand where the semantics are new."""

from __future__ import annotations

import numpy as np
import pytest
from sal.opt.termination import Stop
from sal.sample.potts_mcmc import PottsMove, anneal_potts
from sal.sample.potts_mcmc.chains import merge_labels
from sal.sample.schedule import Polish
from sal.search.alpha_expansion import alpha_expansion
from sal.search.ground_state import ANNEAL_SCHEDULE
from sal.search.icm import (
    FloorPolicy,
    SweepOrder,
    _floor_smallest_first,
    iterated_conditional_modes,
)
from sal.search.potts_problem import FloorAt, MergeGain, MergePairs, PottsProblem
from sal.search.trws import trws
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


@pytest.mark.oracle
@pytest.mark.backend
def test_alpha_expansion_is_the_function_bitwise() -> None:
    cell = _cell()
    field = np.asarray(cell.field, dtype=float).copy()
    # A forbidden label at three sites: the cut reads the penalized stand-in.
    forbidden = field.copy()
    forbidden[:3, 0] = -np.inf
    for rows in (field, forbidden):
        problem = PottsProblem(cell.graph, rows)
        for start in (None, _starts(cell)[1]):
            want = alpha_expansion(cell.graph, rows, start=start)
            got = problem.alpha_expansion(start=start)
            np.testing.assert_array_equal(got.labelling, want.labelling)
            assert (got.cycles, got.moves) == (want.cycles, want.moves)
            assert got.termination == want.termination
            # The Rust loop's energy, each edge from both ends and halved.
            assert got.energy == pytest.approx(want.energy, rel=1e-12)
    assert problem.footprint()["cut"] > 0


@pytest.mark.oracle
@pytest.mark.backend
def test_alpha_expansion_then_icm_is_the_two_calls_bitwise() -> None:
    cell = _cell()
    problem = PottsProblem(cell.graph, cell.field)
    for start in _starts(cell):
        expanded = alpha_expansion(cell.graph, cell.field, start=start)
        want = iterated_conditional_modes(
            cell.graph, cell.field, np.random.default_rng(0), start=expanded.labelling
        )
        got = problem.alpha_expansion(start=start, then_icm=True)
        np.testing.assert_array_equal(got.labelling, want.labelling)
        assert got.termination == want.termination


@pytest.mark.oracle
@pytest.mark.backend
def test_trws_is_the_numba_kernel_bitwise() -> None:
    cell = _cell()
    problem = PottsProblem(cell.graph, cell.field)
    for iterations in (3, 5000):
        want = trws(cell.graph, cell.field, max_iterations=iterations)
        got = problem.trws(max_iterations=iterations)
        np.testing.assert_array_equal(got.trace, want.trace)
        np.testing.assert_array_equal(got.labelling, want.labelling)
        assert (got.bound, got.termination) == (want.bound, want.termination)
        # The decode's energy: unaries then edges, against `sim.potts.energy`.
        assert got.energy == pytest.approx(want.energy, rel=1e-12)


def _floored_path(labels: list[int]) -> PottsProblem:
    """Eight sites on a unit path, three labels, each site pinned to its label in ``labels`` by 10."""
    field = np.zeros((8, 3))
    field[np.arange(8), labels] = 10.0
    edges = tuple((i, i + 1) for i in range(7))
    graph = PottsGraph(n_nodes=8, edges=edges, coupling=(1.0,) * 7)
    return PottsProblem(graph, field)


@pytest.mark.analytic
def test_the_guarded_floor_stops_on_its_eleventh_firing() -> None:
    # Label 2 holds 2 < 3 sites with every label in use, so the floor fires
    # after each sweep (labels 0 and 1 at the floor: the guard counts it),
    # moves both sites to label 0, and the sweep pins them back by 10.
    start = np.array([0, 0, 0, 2, 2, 1, 1, 1])
    problem = _floored_path(start.tolist())
    got = problem.icm(
        np.random.default_rng(0),
        start=start,
        min_sites=3,
        policy=FloorPolicy.SMALLEST_FIRST_BEST_FIELD,
        floor_at=FloorAt.EPOCH_GUARDED,
    )
    assert (got.sweeps, got.termination.reason) == (11, Stop.BUDGET)
    np.testing.assert_array_equal(got.labelling, [0, 0, 0, 0, 0, 1, 1, 1])


@pytest.mark.analytic
def test_the_worklist_does_not_revisit_a_site_the_floor_moved() -> None:
    # 0 0 0 2 1 1 1 1: the floor moves site 3 to label 0 and queues its row
    # (sites 2 and 4), not the site; neither changes, so epoch 2 is clean,
    # label 2 is empty and the floor no longer fires. Index sweeps revisit
    # site 3, which returns to label 2, and the guard stops them.
    start = np.array([0, 0, 0, 2, 1, 1, 1, 1])
    problem = _floored_path(start.tolist())
    runs = {
        order: problem.icm(
            np.random.default_rng(0),
            start=start,
            order=order,
            min_sites=3,
            policy=FloorPolicy.SMALLEST_FIRST_BEST_FIELD,
            floor_at=FloorAt.EPOCH_GUARDED,
        )
        for order in (SweepOrder.WORKLIST, SweepOrder.INDEX)
    }
    queued = runs[SweepOrder.WORKLIST]
    np.testing.assert_array_equal(queued.labelling, [0, 0, 0, 0, 1, 1, 1, 1])
    # Site 3 would return to label 2: not a fixed point.
    assert (queued.sweeps, queued.termination.reason) == (2, Stop.BUDGET)
    assert runs[SweepOrder.INDEX].sweeps == 11


@pytest.mark.analytic
def test_the_guarded_floor_waits_while_a_label_is_empty() -> None:
    # 0 0 0 0 1 1 on a unit path, label 2 empty: SWEEP dissolves label 1
    # (2 < 3 sites); EPOCH_GUARDED does not fire, as the downstream floor
    # does not while a label holds no site.
    field = np.zeros((6, 3))
    field[4:, 1] = 0.1
    edges = tuple((i, i + 1) for i in range(5))
    graph = PottsGraph(n_nodes=6, edges=edges, coupling=(1.0,) * 5)
    problem = PottsProblem(graph, field)
    start = np.array([0, 0, 0, 0, 1, 1])
    runs = {
        at: problem.icm(
            np.random.default_rng(0),
            start=start,
            min_sites=3,
            policy=FloorPolicy.SMALLEST_FIRST_BEST_FIELD,
            floor_at=at,
        )
        for at in FloorAt
    }
    np.testing.assert_array_equal(runs[FloorAt.SWEEP].labelling, [0] * 6)
    np.testing.assert_array_equal(runs[FloorAt.EPOCH_GUARDED].labelling, start)
    assert runs[FloorAt.EPOCH_GUARDED].sweeps == 1


@pytest.mark.analytic
def test_directed_rows_descend_on_the_row_and_score_the_symmetric_energy() -> None:
    # Row 0 names site 1 at 2; row 1 names nothing. Site 0 prefers label 1
    # by 1.5; site 1 holds label 0 by 5. On its row, site 0 reads 2 for
    # label 0 against 1.5; on the symmetric coupling (2 + 0) / 2 = 1, it
    # reads 1 against 1.5 and moves.
    indptr, indices, weights = np.array([0, 1, 1]), np.array([1]), np.array([2.0])
    field = np.array([[0.0, 1.5], [5.0, 0.0]])
    directed = PottsProblem.from_directed_csr(indptr, indices, weights, field)
    symmetric = PottsProblem(directed.graph, field)
    start = np.array([0, 0])
    for order in (SweepOrder.INDEX, SweepOrder.WORKLIST):
        got = directed.icm(np.random.default_rng(0), start=start, order=order)
        np.testing.assert_array_equal(got.labelling, [0, 0])
        # -5 field, one bond at the symmetric coupling 1.
        assert got.energy == -6.0
        moved = symmetric.icm(np.random.default_rng(0), start=start, order=order)
        np.testing.assert_array_equal(moved.labelling, [1, 0])
    with pytest.raises(ValueError, match="directed"):
        directed.anneal(ANNEAL_SCHEDULE.build(4), np.random.default_rng(0), start=start)
