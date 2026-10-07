"""One field behaviour and move sets on every Potts entry point, held to enumeration (issue #1317, stage 1).

The fixture is a 2x3 open lattice at ``q = 3`` with a per-site field drawn
from a declared seed and label 2 forbidden at site 0: 3^6 = 729 labellings,
486 of them allowed. The referees share no bond, cluster or accept step with
the kernels:

* :func:`tests._chains.enumerated_law` for the Boltzmann law, against which
  every recolour x cluster move, alone and composed with single-site Gibbs,
  is fitted by chi-square at :data:`SIGNIFICANCE`, declared here before the
  run, over cells pooled to an expected count of at least :data:`MIN_EXPECTED`;
* the Wolff heat-bath kernel and the Gibbs sweep built as transition matrices
  from their definitions --- every seed site, every bond configuration, every
  label --- for irreducibility, ``pi P = pi``, and the ``beta -> 0`` limit;
* :func:`~sal.search.icm.iterated_conditional_modes`' index-order sweep for
  the ``beta -> infinity`` limit of the Gibbs sweep.
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable

import numpy as np
import pytest
from sal.backend import Backend
from sal.sample import annealed, potts_keyed, potts_mcmc, tempered
from sal.sample.potts_mcmc import (
    PottsMove,
    Recolour,
    move_set,
    sample_potts,
    sweep_at,
)
from sal.sample.statistics import chi_square_p_value
from sal.search.icm import SweepOrder, iterated_conditional_modes
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph

from tests._chains import cell_counts, enumerated_law

SHAPE = (2, 3)
N_STATES = 3
COUPLING = 0.6
SEED = 1317
#: Per-site field, one row per site, from a declared seed; label 2 at site 0
#: is forbidden.
FIELD = np.random.default_rng(SEED).normal(0.0, 0.5, (6, N_STATES))
FIELD[0, 2] = -np.inf
#: Declared before the run: a correct sampler fails one fit in a thousand.
SIGNIFICANCE = 1e-3
#: Cells are pooled until each expected count reaches this.
MIN_EXPECTED = 5.0
RECORDED = 10_000
THINNING = 10
#: The ``beta -> 0`` agreement the issue asks for.
LIMIT_TOLERANCE = 1e-15


def _graph(coupling: float = COUPLING) -> PottsGraph:
    return lattice_graph(SHAPE, BoundaryCondition.OPEN, coupling)


def _pooled_p_value(probability: np.ndarray, counts: np.ndarray) -> float:
    """Chi-square over cells sorted by probability, merged to :data:`MIN_EXPECTED`."""
    expected = probability * counts.sum()
    order = np.argsort(expected)
    observed_cells, expected_cells = [], []
    run_obs, run_exp = 0.0, 0.0
    for cell in order:
        run_obs += counts[cell]
        run_exp += expected[cell]
        if run_exp >= MIN_EXPECTED:
            observed_cells.append(run_obs)
            expected_cells.append(run_exp)
            run_obs, run_exp = 0.0, 0.0
    observed_cells[-1] += run_obs
    expected_cells[-1] += run_exp
    return chi_square_p_value(np.array(observed_cells), np.array(expected_cells))


CHAINS = [
    (moves, recolour)
    for recolour in (Recolour.HEAT_BATH, Recolour.UNIFORM)
    for cluster in (PottsMove.WOLFF, PottsMove.SWENDSEN_WANG)
    for moves in ((cluster,), (cluster, PottsMove.SINGLE_SITE))
] + [
    # Issue #1323: the defaults, a bare move under ``PER_MOVE``, which
    # resolves to the heat-bath move composed with a Gibbs sweep.
    (cluster, Recolour.PER_MOVE)
    for cluster in (PottsMove.WOLFF, PottsMove.SWENDSEN_WANG)
]


@pytest.mark.oracle
@pytest.mark.parametrize(("moves", "recolour"), CHAINS)
def test_each_recolour_and_move_set_leaves_the_boltzmann_law_invariant(
    moves: PottsMove | tuple[PottsMove, ...], recolour: Recolour
) -> None:
    """Chi-square of the thinned chain against the enumerated law, with no visit off its support."""
    graph = _graph()
    index, probability = enumerated_law(graph, FIELD)
    chain = sample_potts(
        graph,
        FIELD,
        moves,
        np.random.default_rng(SEED),
        RECORDED,
        burn_in=RECORDED // 10,
        thin=THINNING,
        recolour=recolour,
    )
    counts = cell_counts(index, chain.states)
    support = probability > 0
    assert counts[~support].sum() == 0
    assert _pooled_p_value(probability[support], counts[support]) > SIGNIFICANCE


# -- the kernels as matrices, from their definitions --------------------------


def _states() -> np.ndarray:
    return np.array(list(itertools.product(range(N_STATES), repeat=6)), dtype=np.int64)


def _log_weight(graph: PottsGraph, state: np.ndarray, beta: float) -> float:
    field = FIELD[np.arange(6), state].sum()
    bonds = sum(
        coupling * (state[i] == state[j])
        for (i, j), coupling in zip(graph.edges, graph.coupling, strict=True)
    )
    return float(beta * (field + bonds))


def _neighbours(graph: PottsGraph) -> list[list[int]]:
    adjacent: list[list[int]] = [[] for _ in range(6)]
    for i, j in graph.edges:
        adjacent[i].append(j)
        adjacent[j].append(i)
    return adjacent


def _heat_bath(members: list[int], beta: float) -> np.ndarray:
    log = beta * FIELD[members].sum(axis=0)
    weight = np.exp(log - log[np.isfinite(log)].max())
    return np.asarray(weight / weight.sum())


def _wolff_heat_bath_matrix(graph: PottsGraph, beta: float) -> np.ndarray:
    """Wolff with heat-bath recolouring: every seed, every like-bond subset, every label."""
    states = _states()
    where = {tuple(row): k for k, row in enumerate(states)}
    adjacent = _neighbours(graph)
    coupling = dict(zip(graph.edges, graph.coupling, strict=True))
    matrix = np.zeros((len(states), len(states)))
    for k, state in enumerate(states):
        if state[0] == 2:
            continue  # off the support: the row stays empty
        like = [edge for edge in graph.edges if state[edge[0]] == state[edge[1]]]
        for open_bits in itertools.product((False, True), repeat=len(like)):
            chance = 1.0
            active = set()
            for edge, bit in zip(like, open_bits, strict=True):
                p = 1.0 - np.exp(-beta * coupling[edge])
                chance *= p if bit else 1.0 - p
                if bit:
                    active.add(edge)
                    active.add(edge[::-1])
            if chance == 0.0:
                continue
            for seed in range(6):
                members, frontier = {seed}, [seed]
                while frontier:
                    site = frontier.pop()
                    for other in adjacent[site]:
                        if other not in members and (site, other) in active:
                            members.add(other)
                            frontier.append(other)
                labels = _heat_bath(sorted(members), beta)
                for label in range(N_STATES):
                    if labels[label] == 0.0:
                        continue
                    moved = state.copy()
                    moved[sorted(members)] = label
                    matrix[k, where[tuple(moved)]] += chance / 6 * labels[label]
    return matrix


def _gibbs_site_matrix(graph: PottsGraph, beta: float, site: int) -> np.ndarray:
    states = _states()
    where = {tuple(row): k for k, row in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    for k, state in enumerate(states):
        if state[0] == 2:
            continue  # off the support: the row stays empty
        log = np.array(
            [
                _log_weight(graph, np.where(np.arange(6) == site, label, state), beta)
                for label in range(N_STATES)
            ]
        )
        weight = np.exp(log - log[np.isfinite(log)].max())
        weight /= weight.sum()
        for label in range(N_STATES):
            moved = state.copy()
            moved[site] = label
            matrix[k, where[tuple(moved)]] += weight[label]
    return matrix


def _boltzmann(graph: PottsGraph, beta: float) -> np.ndarray:
    log = np.array([_log_weight(graph, row, beta) for row in _states()])
    law = np.exp(log - log.max())
    return np.asarray(law / law.sum())


@pytest.mark.oracle
def test_the_composed_kernel_is_irreducible_and_keeps_the_law() -> None:
    """Wolff heat bath then a Gibbs sweep: ``pi P = pi`` to 1e-12, and ``P^k > 0`` on the support."""
    graph = _graph()
    wolff = _wolff_heat_bath_matrix(graph, 1.0)
    sweep = np.linalg.multi_dot([_gibbs_site_matrix(graph, 1.0, s) for s in range(6)])
    composed = wolff @ sweep
    law = _boltzmann(graph, 1.0)
    support = law > 0
    np.testing.assert_allclose(law @ wolff, law, rtol=0, atol=1e-12)
    np.testing.assert_allclose(law @ composed, law, rtol=0, atol=1e-12)
    on = composed[np.ix_(support, support)]
    reach = np.linalg.matrix_power(on, 6)
    assert (reach > 0).all()


@pytest.mark.oracle
def test_wolff_heat_bath_is_random_scan_gibbs_as_beta_goes_to_zero() -> None:
    """At zero coupling every cluster is one site: Wolff heat bath is random-scan Gibbs to 1e-15."""
    graph = _graph(coupling=0.0)
    wolff = _wolff_heat_bath_matrix(graph, 1.0)
    gibbs = sum(_gibbs_site_matrix(graph, 1.0, s) for s in range(6)) / 6
    assert float(np.abs(wolff - gibbs).max()) <= LIMIT_TOLERANCE


@pytest.mark.oracle
def test_the_wolff_heat_bath_code_matches_its_matrix_at_zero_coupling() -> None:
    """The kernel's one-step law from a fixed state against the enumerated row, by chi-square."""
    graph = _graph(coupling=0.0)
    states = _states()
    start = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    row = (_wolff_heat_bath_matrix(graph, 1.0))[
        {tuple(r): k for k, r in enumerate(states)}[tuple(start)]
    ]
    offsets, neighbours, couplings = graph.compressed_adjacency()
    rows = np.ascontiguousarray(FIELD)
    step = potts_mcmc.sweep_for(
        [PottsMove.WOLFF],
        graph,
        rows,
        offsets,
        neighbours,
        couplings,
        Backend.PYTHON,
        recolour=Recolour.HEAT_BATH,
    )
    rng = np.random.default_rng(SEED)
    index = {tuple(r): k for k, r in enumerate(states)}
    counts = np.zeros(len(states))
    for _ in range(20_000):
        state = start.copy()
        step(state, rng, 1.0)
        counts[index[tuple(state)]] += 1
    support = row > 0
    assert counts[~support].sum() == 0
    assert _pooled_p_value(row[support], counts[support]) > SIGNIFICANCE


@pytest.mark.oracle
def test_gibbs_is_the_icm_step_as_beta_goes_to_infinity() -> None:
    """At ``beta = 1e12`` an index-order Gibbs sweep is ICM's sweep, from every allowed start."""
    graph = _graph()
    offsets, neighbours, couplings = graph.compressed_adjacency()
    sweep = sweep_at(
        np.ascontiguousarray(FIELD), offsets, neighbours, couplings, Backend.PYTHON
    )
    rng = np.random.default_rng(SEED)
    checked = 0
    for start in _states():
        if start[0] == 2:
            continue
        gibbs = start.copy()
        sweep(gibbs, rng, 1e12)
        icm = iterated_conditional_modes(
            graph,
            FIELD,
            rng,
            start=start,
            max_iterations=1,
            sweep_order=SweepOrder.INDEX,
            stop_when_clean=False,
            backend=Backend.PYTHON,
        )
        np.testing.assert_array_equal(gibbs, np.asarray(icm.labelling))
        checked += 1
    assert checked == 486


# -- the API: aliases, move sets, refusals, the signature guard ---------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("alias", "move"),
    [
        (PottsMove.WOLFF_HEAT_BATH, PottsMove.WOLFF),
        (PottsMove.SWENDSEN_WANG_HEAT_BATH, PottsMove.SWENDSEN_WANG),
    ],
)
def test_the_deprecated_members_are_the_move_with_heat_bath_bitwise(
    alias: PottsMove, move: PottsMove
) -> None:
    graph = _graph()
    run: Callable[..., np.ndarray] = lambda m, **kw: sample_potts(  # noqa: E731
        graph, FIELD, m, np.random.default_rng(SEED), 200, **kw
    ).states
    np.testing.assert_array_equal(run(alias), run([move], recolour=Recolour.HEAT_BATH))
    assert move_set(alias, Recolour.UNIFORM) == (alias,)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "move", [PottsMove.SINGLE_SITE, PottsMove.WOLFF, PottsMove.SWENDSEN_WANG]
)
def test_a_single_move_is_its_resolved_move_set_bitwise(move: PottsMove) -> None:
    """A bare move runs bitwise as the sequence :func:`move_set` resolves it to (#1323)."""
    graph = _graph()
    one = sample_potts(graph, FIELD, move, np.random.default_rng(SEED), 200)
    seq = sample_potts(graph, FIELD, move_set(move), np.random.default_rng(SEED), 200)
    np.testing.assert_array_equal(one.states, seq.states)
    assert one.mean_cluster_size == seq.mean_cluster_size


@pytest.mark.smoke
@pytest.mark.parametrize(
    "move", [PottsMove.NIEDERMAYER, PottsMove.GHOST_SPIN, PottsMove.LABEL_DIRECTED]
)
def test_a_move_without_a_heat_bath_recolouring_is_refused_by_name(
    move: PottsMove,
) -> None:
    with pytest.raises(ValueError, match=str(move)):
        sample_potts(
            _graph(),
            FIELD,
            [PottsMove.SINGLE_SITE, move],
            np.random.default_rng(SEED),
            1,
            recolour=Recolour.HEAT_BATH,
        )


#: The public Potts entry points: every function that runs a Potts move set.
ENTRY_POINTS = [
    potts_mcmc.sample_potts,
    potts_mcmc.sample_potts_pair,
    potts_mcmc.sample_potts_starts,
    potts_mcmc.anneal_potts,
    potts_mcmc.parallel_tempering,
    potts_mcmc.sweep_for,
    annealed.annealed_importance_sampling,
    annealed.population_annealing,
    annealed.simulated_tempering,
    tempered.tempered_potts_pair,
    potts_keyed.cluster_moves,
]

#: The annealed Potts entry points and the name of their schedule argument:
#: ``schedule`` where it is temperatures, ``betas`` where the ladder is
#: inverse temperatures (issue #1317, stage 2).
ANNEALED_ENTRY_POINTS = [
    (potts_mcmc.anneal_potts, "schedule"),
    (annealed.annealed_importance_sampling, "betas"),
    (annealed.population_annealing, "betas"),
    (annealed.simulated_tempering, "betas"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda f: f.__name__)
def test_every_potts_entry_point_takes_a_move_set_and_recolour(
    entry: Callable[..., object],
) -> None:
    parameters = inspect.signature(entry).parameters
    assert "recolour" in parameters
    assert parameters["recolour"].default is Recolour.PER_MOVE
    annotation = str(parameters["move"].annotation)
    assert annotation in {"PottsMoves", "RungMoves"}, annotation


#: Each cluster move's default: what a bare move resolves to under the
#: entry points' default ``recolour`` (issue #1323). Wolff and Swendsen-Wang
#: take the heat bath and a Gibbs sweep; the moves with no heat-bath form keep
#: the uniform proposal and run alone.
DEFAULTS = {
    PottsMove.WOLFF: (PottsMove.WOLFF_HEAT_BATH, PottsMove.SINGLE_SITE),
    PottsMove.SWENDSEN_WANG: (
        PottsMove.SWENDSEN_WANG_HEAT_BATH,
        PottsMove.SINGLE_SITE,
    ),
    PottsMove.NIEDERMAYER: (PottsMove.NIEDERMAYER,),
    PottsMove.GHOST_SPIN: (PottsMove.GHOST_SPIN,),
    PottsMove.LABEL_DIRECTED: (PottsMove.LABEL_DIRECTED,),
}


@pytest.mark.analytic
@pytest.mark.parametrize(("move", "resolved"), DEFAULTS.items(), ids=str)
def test_each_cluster_move_has_its_stated_default_recolour_and_composition(
    move: PottsMove, resolved: tuple[PottsMove, ...]
) -> None:
    """The default per move, and the move before #1323 one flag away."""
    default = inspect.signature(move_set).parameters["recolour"].default
    assert default is Recolour.PER_MOVE
    assert move_set(move) == resolved
    assert move_set([move], Recolour.UNIFORM) == (move,)


@pytest.mark.analytic
def test_the_keyed_defaults_keep_wolff_uniform_and_heat_bath_swendsen_wang() -> None:
    """A keyed Wolff action names its label, so ``PER_MOVE`` keeps it uniform (#1323)."""
    built = potts_keyed.cluster_moves(_graph(), FIELD)
    uniform = potts_keyed.cluster_moves(_graph(), FIELD, recolour=Recolour.UNIFORM)
    for moves, recolour in ((built, Recolour.HEAT_BATH), (uniform, Recolour.UNIFORM)):
        swendsen_wang = moves[potts_mcmc.MoveKind.SWENDSEN_WANG]
        assert isinstance(swendsen_wang, potts_keyed.SwendsenWangMove)
        assert swendsen_wang._recolour is recolour
    assert set(built) == set(uniform)
    with pytest.raises(ValueError, match="names the label"):
        potts_keyed.cluster_moves(_graph(), FIELD, recolour=Recolour.HEAT_BATH)


@pytest.mark.smoke
def test_the_guard_lists_every_exported_entry_point_that_takes_a_move() -> None:
    """A new entry point in ``potts_mcmc.__all__`` that takes ``move`` must be listed above."""
    listed = {f.__name__ for f in ENTRY_POINTS}
    for name in potts_mcmc.__all__:
        value = getattr(potts_mcmc, name)
        if inspect.isfunction(value) and "move" in inspect.signature(value).parameters:
            if name in {"refuse_negative_coupling", "move_set", "moves_per_rung"}:
                continue
            if "graph" in inspect.signature(value).parameters:
                assert name in listed, name


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("entry", "name"), ANNEALED_ENTRY_POINTS, ids=lambda f: getattr(f, "__name__", f)
)
def test_every_annealed_entry_point_takes_an_auto_schedule_and_its_tuning(
    entry: Callable[..., object], name: str
) -> None:
    parameters = inspect.signature(entry).parameters
    assert (
        "auto" in str(parameters[name].annotation)
        or str(parameters[name].annotation) == "Schedule"
    ), parameters[name].annotation
    assert parameters["tuning"].default is None
    assert str(parameters["tuning"].annotation) == "ScheduleTuning | None"
