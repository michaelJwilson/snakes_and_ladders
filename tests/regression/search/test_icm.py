"""Iterated conditional modes (`search.icm`): the descent, its compiled kernel, and the minimum-sites floor (issue #1055).

Referees: the descent settles where no single site improves, and its local
deltas are the recomputing loop's argmin; the compiled kernel is the Python
oracle bitwise, floor included, from the same draws; with ``min_sites = 0``
every caller's run is the one recorded before the floor existed, as sha256
digests; after every floored sweep no state holds ``0 < count < min_sites``,
each dissolved site lands on ``surviving[floor(u * m)]``, and the picks split
uniformly among the survivors by chi-square; the kernel releases the GIL.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest
from numba import njit
from scipy.stats import chisquare
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.sample.kernels import icm_sweeps, icm_sweeps_checked
from snakes_and_ladders.search.ground_state import (
    FLOORED,
    METHODS,
    descend,
    ground_state,
)
from snakes_and_ladders.search.icm import SweepOrder, iterated_conditional_modes
from snakes_and_ladders.search.potts_starts import spatio_rung
from snakes_and_ladders.search.spatio_sequential import (
    LabelSolver,
    fit_spatio_sequential,
    label_step,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import energy, site_field
from snakes_and_ladders.sim.spatio_sequential import simulate_spatio_sequential

from tests._rows import every_value


@pytest.mark.analytic
def test_single_site_descent_settles_at_a_local_minimum() -> None:
    # A baseline that stopped early would make beating it say nothing. On
    # termination no single site can improve, which defines the move set it
    # represents.
    rng = np.random.default_rng(21)
    graph = lattice_graph((5, 5), BoundaryCondition.OPEN, 0.9)
    field_values = rng.normal(size=(graph.n_nodes, 3))

    labelling, settled = iterated_conditional_modes(
        graph, field_values, 3, np.random.default_rng(1)
    )

    for node in range(graph.n_nodes):
        for label in range(3):
            candidate = labelling.copy()
            candidate[node] = label

            assert energy(graph, field_values, candidate) >= settled - 1e-12


# --- the descent and its compiled kernel, moved from test_alpha_expansion.py ---


@pytest.mark.critical
@pytest.mark.oracle
def test_the_numba_descent_reproduces_the_python_one_bitwise() -> None:
    # Same update, order and tie rule (#264): identical, not close. A rugged
    # per-node field makes the start decide the optimum; 20 of 20 agree at 32x32.
    def check(seed: int) -> None:
        graph = lattice_graph((8, 8), BoundaryCondition.PERIODIC, 0.5)
        field = np.random.default_rng(100 + seed).normal(size=(graph.n_nodes, 3))

        python = iterated_conditional_modes(
            graph, field, 3, np.random.default_rng(seed), backend=Backend.PYTHON
        )
        compiled = iterated_conditional_modes(
            graph, field, 3, np.random.default_rng(seed), backend=Backend.NUMBA
        )

        assert np.array_equal(python.labelling, compiled.labelling)
        assert python.energy == compiled.energy

    every_value(range(6), check)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("sweep_order", [SweepOrder.RANDOM, SweepOrder.INDEX])
def test_the_numba_descent_in_any_order_every_sweep_run_is_the_python_one(
    sweep_order: SweepOrder,
) -> None:
    # Issue #923: ICM in a random order with every sweep run, the heat bath
    # at T = 0 (`ground_state.run_icm_random`). The compiled sweep takes the
    # permutations the Python sweep draws, in its order, so the labelling,
    # the energy and the generator's state after are the Python sweep's.
    def check(seed: int) -> None:
        graph = lattice_graph((8, 8), BoundaryCondition.PERIODIC, 0.5)
        field = np.random.default_rng(300 + seed).normal(size=(graph.n_nodes, 3))
        streams = [np.random.default_rng(seed) for _ in range(2)]
        python, compiled = (
            iterated_conditional_modes(
                graph,
                field,
                3,
                stream,
                max_sweeps=12,
                sweep_order=sweep_order,
                stop_when_clean=False,
                backend=backend,
            )
            for stream, backend in zip(
                streams, (Backend.PYTHON, Backend.NUMBA), strict=True
            )
        )
        assert np.array_equal(python.labelling, compiled.labelling)
        assert python.energy == compiled.energy
        assert streams[0].integers(1 << 62) == streams[1].integers(1 << 62)

    every_value(range(6), check)


@pytest.mark.smoke
def test_descent_has_no_rust_backend() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)

    with pytest.raises(ValueError, match="runs on numba or python, not rust"):
        iterated_conditional_modes(
            graph, np.zeros(3), 3, np.random.default_rng(0), backend=Backend.RUST
        )


def _recomputing_descent(
    graph: PottsGraph,
    values: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
    start: np.ndarray,
) -> np.ndarray:
    """The loop `search.spatio_sequential.label_step` ran, kept as the sweep's oracle.

    A full energy per candidate label: the same argmin as the local delta, the long way.
    """
    current = np.asarray(start, dtype=np.int64).copy()
    best = energy(graph, values, current)
    for _ in range(200):
        moved = False
        for node in rng.permutation(graph.n_nodes):
            for label in range(n_states):
                if label == current[node]:
                    continue
                trial = current.copy()
                trial[node] = label
                value = energy(graph, values, trial)
                if value < best - 1e-12:
                    best, current, moved = value, trial, True
        if not moved:
            break
    return current


@pytest.mark.critical
@pytest.mark.oracle
def test_the_local_delta_sweep_is_the_recomputing_descent() -> None:
    # What lets `label_step` call the sweep (#858): the same site order from
    # the same start, and the argmin read off the site's own field and
    # incident edges rather than off `O(n_edges)` of energy per candidate.
    # Realized: 6 of 6 seeds agree site for site, at 3 labels on 36 sites.
    def check(seed: int) -> None:
        graph = lattice_graph((6, 6), BoundaryCondition.OPEN, 0.7)
        field = np.random.default_rng(700 + seed).normal(size=(graph.n_nodes, 3))
        start = np.random.default_rng(seed).integers(0, 3, size=graph.n_nodes)

        swept, _ = iterated_conditional_modes(
            graph,
            field,
            3,
            np.random.default_rng(seed),
            start=start,
            sweep_order=SweepOrder.RANDOM,
            backend=Backend.PYTHON,
        )
        recomputed = _recomputing_descent(
            graph,
            site_field(field, graph.n_nodes),
            3,
            np.random.default_rng(seed),
            start,
        )

        assert np.array_equal(swept, recomputed)

    every_value(range(6), check)


@pytest.mark.analytic
def test_a_random_order_runs_every_sweep_when_a_clean_one_does_not_end_it() -> None:
    # Gibbs at T = 0 is charged a fixed budget, so it spends every sweep; the
    # descent is still monotone, since each visit takes an argmin.
    graph = lattice_graph((5, 5), BoundaryCondition.PERIODIC, 0.9)
    field = np.random.default_rng(31).normal(size=(graph.n_nodes, 3))

    labelling, value = iterated_conditional_modes(
        graph,
        field,
        3,
        np.random.default_rng(4),
        max_sweeps=25,
        sweep_order=SweepOrder.RANDOM,
        stop_when_clean=False,
        backend=Backend.PYTHON,
    )
    settled, settled_value = iterated_conditional_modes(
        graph,
        field,
        3,
        np.random.default_rng(4),
        max_sweeps=25,
        sweep_order=SweepOrder.RANDOM,
        backend=Backend.PYTHON,
    )

    assert value <= energy(graph, field, labelling) + 1e-12
    assert settled_value <= value + 1e-12
    assert settled.shape == labelling.shape


@pytest.mark.smoke
def test_the_compiled_sweep_refuses_an_order_it_does_not_walk() -> None:
    # Since #923 the compiled sweep runs any order when every sweep runs; a
    # random order that stops on a clean sweep would spend the generator
    # past the stop, and is refused.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)

    with pytest.raises(ValueError, match="stop_when_clean=True needs python"):
        iterated_conditional_modes(
            graph,
            np.zeros(3),
            3,
            np.random.default_rng(0),
            sweep_order=SweepOrder.RANDOM,
        )


# --- min_sites = 0 is the descent before the floor, bitwise -------------------

#: The notebook's reported seeds.
SEEDS = (0, 1, 2, 3, 4)
#: Sweeps per tier, `test_ground_state_start.py`'s.
SWEEPS = {"ci": 50, "release": 40}

#: Each cell's run on `SEEDS` at the commit before #1055 (f0aba7f5's tree,
#: `origin/main` 44fed650), as sha256 over the labelling's bytes, the
#: energy's repr and the spend, first 16 hex digits. The `python` cells were
#: `iterated_conditional_modes(..., backend=PYTHON)` with the arguments
#: `run_icm` and `run_icm_random` pass, since `ground_state` took no backend;
#: they equal the `numba` cells, which `test_ground_state_start.py` also pins.
#: `descend` hashes the labelling and the sweep count; the `spatio_sequential`
#: cells are the ICM label step from a uniform draw and a four-block ICM fit
#: (labels and log-likelihoods) on `spatio_sequential/ci`, data seeded 3.
BEFORE = {
    "ci/icm/numba": "eba7c65d69aeb916",
    "ci/icm/python": "eba7c65d69aeb916",
    "ci/icm-random/numba": "f61e867f262cd619",
    "ci/icm-random/python": "f61e867f262cd619",
    "ci/descend": "c13dc0d1eea90bd2",
    "release/icm/numba": "0b81269cf9db1847",
    "release/icm/python": "0b81269cf9db1847",
    "release/icm-random/numba": "354122e191105cb6",
    "release/icm-random/python": "354122e191105cb6",
    "release/descend": "5bfbc113246ec726",
}
LABEL_STEP_BEFORE = "b11b990fd1c0db10"
FIT_BEFORE = "efa028be58f518e4"


def _rung_cells() -> list[str]:
    return [cell for cell in BEFORE if not cell.endswith("descend")]


@pytest.mark.analytic
@pytest.mark.snapshot
@pytest.mark.parametrize("cell", _rung_cells())
def test_no_floor_is_the_ground_state_run_before_the_floor(cell: str) -> None:
    tier, method, backend = cell.split("/")
    rung = spatio_rung(fixture("spatio_only", tier).params, tier)
    budget = Budget(Cost.SITE_VISITS, SWEEPS[tier] * rung.visits_per_sweep)
    digest = hashlib.sha256()
    for seed in SEEDS:
        run = ground_state(
            rung.graph,
            rung.field,
            method,
            budget,
            np.random.default_rng(seed),
            backend=Backend(backend),
            min_sites=0,
        )
        digest.update(np.asarray(run.labelling, dtype=np.int64).tobytes())
        digest.update(repr(float(run.energy)).encode())
        digest.update(str(int(run.spent)).encode())
    assert digest.hexdigest()[:16] == BEFORE[cell]


@pytest.mark.analytic
@pytest.mark.snapshot
@pytest.mark.parametrize("tier", ["ci", "release"])
def test_no_floor_is_the_descent_before_the_floor(tier: str) -> None:
    rung = spatio_rung(fixture("spatio_only", tier).params, tier)
    digest = hashlib.sha256()
    for seed in SEEDS:
        labelling, sweeps = descend(
            rung, np.random.default_rng(seed), SWEEPS[tier], min_sites=0
        )
        digest.update(np.asarray(labelling, dtype=np.int64).tobytes())
        digest.update(str(int(sweeps)).encode())
    assert digest.hexdigest()[:16] == BEFORE[f"{tier}/descend"]


@pytest.mark.analytic
@pytest.mark.snapshot
def test_no_floor_is_the_label_step_and_the_fit_before_the_floor() -> None:
    params = fixture("spatio_sequential", "ci").params
    data = simulate_spatio_sequential(params, np.random.default_rng(3))
    steps, fits = hashlib.sha256(), hashlib.sha256()
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, params.n_classes, size=params.graph.n_nodes)
        step = label_step(
            params, data.observations, labels, rng, LabelSolver.ICM, min_sites=0
        )
        steps.update(np.asarray(step, dtype=np.int64).tobytes())
        fit = fit_spatio_sequential(
            params,
            data.observations,
            np.random.default_rng(seed),
            solver=LabelSolver.ICM,
            n_blocks=4,
        )
        fits.update(np.asarray(fit.labels, dtype=np.int64).tobytes())
        fits.update(fit.log_likelihoods.tobytes())
    assert steps.hexdigest()[:16] == LABEL_STEP_BEFORE
    assert fits.hexdigest()[:16] == FIT_BEFORE


# --- the floor: the kernel against the oracle --------------------------------

#: Six states on 64 sites, coupled and rugged enough that floors of 3 and 10
#: dissolve: in 12 and 66 of the 72 sweeps below, seeds 0 to 5.
FLOOR_STATES = 6
FLOORS = (0, 1, 3, 64 // FLOOR_STATES)
ORDERS = [
    (SweepOrder.INDEX, True),
    (SweepOrder.INDEX, False),
    (SweepOrder.RANDOM, False),
]


def _floor_problem(seed: int) -> tuple[PottsGraph, np.ndarray]:
    graph = lattice_graph((8, 8), BoundaryCondition.PERIODIC, 1.0)
    field = np.random.default_rng(1055 + seed).normal(
        size=(graph.n_nodes, FLOOR_STATES)
    )
    return graph, field


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("min_sites", FLOORS)
@pytest.mark.parametrize(("sweep_order", "stop_when_clean"), ORDERS)
def test_the_floored_kernel_is_the_python_oracle_bitwise(
    min_sites: int, sweep_order: SweepOrder, stop_when_clean: bool
) -> None:
    # Same update, order, tie rule and floor from the same draws: the
    # labelling, the energy and the generator's state after are equal.
    def check(seed: int) -> None:
        graph, field = _floor_problem(seed)
        streams = [np.random.default_rng(seed) for _ in range(2)]
        python, compiled = (
            iterated_conditional_modes(
                graph,
                field,
                FLOOR_STATES,
                stream,
                max_sweeps=12,
                sweep_order=sweep_order,
                stop_when_clean=stop_when_clean,
                min_sites=min_sites,
                backend=backend,
            )
            for stream, backend in zip(
                streams, (Backend.PYTHON, Backend.NUMBA), strict=True
            )
        )
        np.testing.assert_array_equal(python.labelling, compiled.labelling)
        assert python.energy == compiled.energy
        assert streams[0].integers(1 << 62) == streams[1].integers(1 << 62)

    every_value(range(6), check)


@pytest.mark.analytic
@pytest.mark.parametrize("backend", [Backend.NUMBA, Backend.PYTHON], ids=str)
@pytest.mark.parametrize("min_sites", FLOORS[2:])
def test_every_floored_sweep_dissolves_onto_the_survivors_by_its_draw(
    backend: Backend, min_sites: int
) -> None:
    # One sweep at a time from the last: the unfloored sweep gives the
    # counts the floor reads, and the floored one must differ from it at
    # exactly the dissolved states' sites, each moved to surviving[floor(u m)].
    dissolved_sweeps = 0
    for seed in range(6):
        graph, field = _floor_problem(seed)
        labelling = np.random.default_rng(7).integers(0, FLOOR_STATES, graph.n_nodes)
        for sweep in range(12):
            swept = iterated_conditional_modes(
                graph,
                field,
                FLOOR_STATES,
                np.random.default_rng(sweep),
                start=labelling,
                max_sweeps=1,
                backend=backend,
            ).labelling
            floored = iterated_conditional_modes(
                graph,
                field,
                FLOOR_STATES,
                np.random.default_rng(sweep),
                start=labelling,
                max_sweeps=1,
                min_sites=min_sites,
                backend=backend,
            ).labelling
            uniforms = np.random.default_rng(sweep).random(graph.n_nodes)
            counts = np.bincount(swept, minlength=FLOOR_STATES)
            surviving = np.flatnonzero(counts >= min_sites)
            below = (counts > 0) & (counts < min_sites)
            moved = below[swept]
            expected = swept.copy()
            expected[moved] = surviving[
                np.minimum(
                    (uniforms[moved] * surviving.size).astype(int), surviving.size - 1
                )
            ]
            np.testing.assert_array_equal(floored, expected)
            assert np.isin(floored[moved], surviving).all()
            after = np.bincount(floored, minlength=FLOOR_STATES)
            assert not ((after > 0) & (after < min_sites)).any()
            dissolved_sweeps += int(moved.any())
            labelling = floored
    # The floor bound on this instance, so the checks above were not vacuous.
    assert dissolved_sweeps > 0


@pytest.mark.analytic
def test_dissolved_sites_split_uniformly_among_the_survivors() -> None:
    # No coupling, so one sweep puts every site at its field's argmax: 5 of
    # 100 sites in state 3, under a floor of 10, and the other three states
    # each over it. The 5 are recoloured over {0, 1, 2}; over 400 seeds, 2,000
    # picks. Uniformity is rejected at the 0.1% level. Realized: 651, 633 and
    # 716, p = 0.057.
    graph = lattice_graph((10, 10), BoundaryCondition.OPEN, 0.0)
    field = np.zeros((graph.n_nodes, 4))
    field[np.arange(graph.n_nodes), np.arange(graph.n_nodes) % 3] = 1.0
    field[:5] = 0.0
    field[:5, 3] = 1.0
    picks = np.zeros(3, dtype=np.int64)
    for seed in range(400):
        run = iterated_conditional_modes(
            graph, field, 4, np.random.default_rng(seed), max_sweeps=1, min_sites=10
        )
        assert int(np.sum(run.labelling == 3)) == 0
        picks += np.bincount(run.labelling[:5], minlength=3)
    assert picks.sum() == 2000
    assert chisquare(picks).pvalue > 1e-3


# --- the kernel's shape checks and the GIL -----------------------------------


def _kernel_inputs(
    side: int, n_states: int, n_sweeps: int, min_sites: int
) -> tuple[np.ndarray, ...]:
    graph = lattice_graph((side, side), BoundaryCondition.PERIODIC, 0.6)
    rng = np.random.default_rng(1055)
    field = rng.normal(size=(graph.n_nodes, n_states))
    state = rng.integers(0, n_states, graph.n_nodes)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    draws = rng.random(n_sweeps * graph.n_nodes) if min_sites else np.empty(0)
    return state, field, offsets, neighbours, couplings, draws


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"field": np.zeros(16)},
            r"field must be 2-D, \(n_nodes, n_states\).*got shape \(16,\)",
        ),
        ({"field": np.zeros((15, 3))}, "field has 15 rows and state has 16 sites"),
        ({"field": np.zeros((16, 0))}, "field is empty"),
        (
            {"offsets": np.zeros(16, dtype=np.int64)},
            r"offsets has length 16, expected 17",
        ),
        ({"offsets": -np.ones(17, dtype=np.int64)}, "offsets must be >= 0"),
        (
            {"couplings": np.zeros(3)},
            r"neighbours has length \(64,\) and couplings \(3,\)",
        ),
        (
            {"orders": np.zeros((2, 16), dtype=np.int64)},
            r"orders has shape \(2, 16\), expected \(4, 16\)",
        ),
        ({"draws": np.zeros(10)}, r"draws has length 10, expected 4 \* 16"),
        (
            {"state": np.full(16, 3, dtype=np.int64)},
            r"state at node 0 is 3, expected \[0, 3\)",
        ),
        (
            {"state": np.zeros(16, dtype=np.int32)},
            "state must be a 1-D C-contiguous int64 array.*got int32",
        ),
        ({"min_sites": -1}, "min_sites must be >= 0, got -1"),
    ],
)
def test_each_kernel_shape_error_names_wanted_and_given(
    change: dict[str, object], message: str
) -> None:
    state, field, offsets, neighbours, couplings, draws = _kernel_inputs(4, 3, 4, 2)
    arguments: dict[str, object] = {
        "state": state,
        "field": field,
        "offsets": offsets,
        "neighbours": neighbours,
        "couplings": couplings,
        "orders": np.empty((0, 16), dtype=np.int64),
        "draws": draws,
        "n_sweeps": 4,
        "stop_when_clean": True,
        "min_sites": 2,
    }
    arguments.update(change)
    with pytest.raises(ValueError, match=message):
        icm_sweeps_checked(**arguments)  # type: ignore[arg-type]


#: Sweeps the GIL tests run, every one: about 60 ms at 5,041 sites and ten
#: states on this host, long beside a scheduler's quantum.
GIL_SWEEPS = 240


def _gil_inputs() -> tuple[tuple[object, ...], np.ndarray]:
    state, field, offsets, neighbours, couplings, draws = _kernel_inputs(
        71, 10, GIL_SWEEPS, 40
    )
    arguments = (
        field,
        offsets,
        neighbours,
        couplings,
        np.empty((0, state.size), dtype=np.int64),
        draws,
        GIL_SWEEPS,
        False,
        40,
    )
    icm_sweeps(state.copy(), *arguments)  # compiled before any clock starts
    return arguments, state


def _wall(
    kernel: Callable[..., int],
    arguments: tuple[object, ...],
    state: np.ndarray,
    n_threads: int,
) -> float:
    threads = [
        threading.Thread(target=kernel, args=(state.copy(), *arguments))
        for _ in range(n_threads)
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.perf_counter() - started


@pytest.mark.smoke
def test_the_kernel_lets_the_calling_thread_run() -> None:
    # A kernel holding the GIL stops the main thread for its whole call, so
    # the longest pause between two of the main thread's clock reads is the
    # call's length: 62.0 ms of a 66.2 ms call, measured on a copy compiled
    # without `nogil`. Released, the pause is the scheduler's: 4.2 ms under a
    # load average of 3.4 on four cores.
    arguments, state = _gil_inputs()
    alone = min(_wall(icm_sweeps, arguments, state, 1) for _ in range(3))
    worker = threading.Thread(target=icm_sweeps, args=(state.copy(), *arguments))
    last = time.perf_counter()
    pause = 0.0
    worker.start()
    while worker.is_alive():
        now = time.perf_counter()
        pause, last = max(pause, now - last), now
    worker.join()
    assert pause < 0.5 * alone


@pytest.mark.smoke
@pytest.mark.release
def test_four_threads_on_the_kernel_beat_the_serialized_control() -> None:
    # #604's measure: four threads on a kernel holding the GIL take 4x one
    # thread's wall. The control is this kernel compiled without `nogil`;
    # measured 3.66x against the kernel's 2.90x under a load average of 3.4
    # on four cores, where the other agents held the cores the threads would
    # have taken. The bound is the control's, so a loaded host moves both.
    arguments, state = _gil_inputs()
    held: Callable[..., int] = njit(cache=False)(icm_sweeps.py_func)
    held(state.copy(), *arguments)
    ratios = [
        min(_wall(kernel, arguments, state, 4) for _ in range(3))
        / min(_wall(kernel, arguments, state, 1) for _ in range(3))
        for kernel in (icm_sweeps, held)
    ]
    assert ratios[0] < 0.9 * ratios[1]


# --- refusals ----------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("backend", [Backend.NUMBA, Backend.PYTHON], ids=str)
def test_an_unsatisfiable_floor_raises(backend: Backend) -> None:
    graph = lattice_graph((2, 2), BoundaryCondition.OPEN, 0.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="min_sites=5 exceeds the 4 sites"):
        iterated_conditional_modes(
            graph, np.zeros(2), 2, rng, min_sites=5, backend=backend
        )
    with pytest.raises(ValueError, match="min_sites must be >= 0, got -1"):
        iterated_conditional_modes(
            graph, np.zeros(2), 2, rng, min_sites=-1, backend=backend
        )
    # Every site prefers its own state: one site each, none at a floor of 2.
    field = 5.0 * np.eye(4)
    with pytest.raises(ValueError, match="sweep 1 left no state holding min_sites=2"):
        iterated_conditional_modes(graph, field, 4, rng, min_sites=2, backend=backend)


@pytest.mark.smoke
def test_the_compiled_sweep_still_refuses_a_random_order_that_stops_clean() -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.5)
    with pytest.raises(ValueError, match="stop_when_clean=True needs python"):
        iterated_conditional_modes(
            graph,
            np.zeros(3),
            3,
            np.random.default_rng(0),
            sweep_order=SweepOrder.RANDOM,
            min_sites=2,
        )


@pytest.mark.smoke
@pytest.mark.parametrize("method", sorted(set(METHODS) - FLOORED))
def test_a_method_with_no_descent_refuses_a_floor_or_a_backend_by_name(
    method: str,
) -> None:
    rung = spatio_rung(fixture("spatio_only", "ci").params, "ci")
    budget = Budget(Cost.SITE_VISITS, 5 * rung.visits_per_sweep)
    for keywords in ({"min_sites": 2}, {"backend": Backend.PYTHON}):
        with pytest.raises(ValueError, match=f"'{method}' runs no single-site descent"):
            ground_state(
                rung.graph,
                rung.field,
                method,
                budget,
                np.random.default_rng(0),
                **keywords,
            )


@pytest.mark.smoke
def test_the_floor_reaches_every_descent_and_the_other_label_steps_refuse_it() -> None:
    rung = spatio_rung(fixture("spatio_only", "ci").params, "ci")
    budget = Budget(Cost.SITE_VISITS, 5 * rung.visits_per_sweep)
    for method in sorted(FLOORED):
        run = ground_state(
            rung.graph,
            rung.field,
            method,
            budget,
            np.random.default_rng(0),
            min_sites=3,
        )
        counts = np.bincount(run.labelling, minlength=rung.n_states)
        assert not ((counts > 0) & (counts < 3)).any()
    labelling, _ = descend(rung, np.random.default_rng(0), 5, min_sites=3)
    counts = np.bincount(labelling, minlength=rung.n_states)
    assert not ((counts > 0) & (counts < 3)).any()
    params = fixture("spatio_sequential", "ci").params
    data = simulate_spatio_sequential(params, np.random.default_rng(3))
    labels = np.zeros(params.graph.n_nodes, dtype=np.int64)
    with pytest.raises(
        ValueError, match="the alpha_expansion label step has no min_sites"
    ):
        label_step(
            params,
            data.observations,
            labels,
            np.random.default_rng(0),
            LabelSolver.ALPHA_EXPANSION,
            min_sites=2,
        )
