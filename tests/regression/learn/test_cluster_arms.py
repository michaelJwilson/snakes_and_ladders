"""The two cluster control arms: the oracle's moves, keyed, and what they cost.

Issue #706, plan steps 3 and 4. The moves are ``potts_mcmc``'s: above zero
temperature ``propose`` equals a direct ``wolff_sweep`` or
``swendsen_wang_sweep`` bitwise. At zero temperature a cluster is a
like-coloured component, computed here by breadth-first search, never the
moves' union-find, and a recolouring is kept only if the score does not fall.
The arm is the classical run: over 32 seeds the declared ladder is
indistinguishable from ``run_swendsen_wang`` and ``run_wolff`` in energy and
site visits (``release``). A pass costs ``visits_per_sweep``, a Wolff step its
cluster, and each arm's features obey `learn.environment.Environment`'s gauge rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.learn.keyed import KeyedMove, keyed_generator
from snakes_and_ladders.learn.potts_nd import (
    CLUSTER_KINDS,
    PARAMETRIC_KINDS,
    FeatureColumn,
    PottsAction,
    PottsNDEnvironment,
)
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.sample.potts_keyed import (
    NiedermayerMove,
    SwendsenWangMove,
    WolffMove,
    cluster_moves,
)
from snakes_and_ladders.sample.potts_mcmc import (
    MoveKind,
    niedermayer_sweep,
    swendsen_wang_sweep,
    wolff_sweep,
)
from snakes_and_ladders.sample.schedule import ExponentialTempSchedule
from snakes_and_ladders.search.ground_state import (
    ANNEAL_END,
    ANNEAL_START,
    Rung,
    run_swendsen_wang,
    run_wolff,
)
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import critical_coupling, energies, spatio_only_field

from tests._rows import every_value

#: The class ladder the fixtures tilt by, as `test_potts_nd.py` builds it.
ALPHA = (-1.0, 0.0, 1.0)

#: Seeds behind the `release` comparisons against the classical runs. Enough
#: that a Welch test has power against a difference of one standard deviation,
#: which is the size a wrong temperature or a wrong root would make.
COMPARISON_SEEDS = 32

#: Significance the two comparisons are read at. A *failure to reject* is what
#: is wanted here, which is why the arm's charge is asserted beside its energy:
#: two methods can reach the same energy at different prices, and then only one
#: of them is the control.
SIGNIFICANCE = 0.01


def _lattice(
    side: int, n_states: int, *, seed: int = 706, zero_field: bool = False
) -> tuple[PottsGraph, np.ndarray, float]:
    """A lattice at the critical coupling and its per-site field."""
    coupling = critical_coupling(n_states)
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, coupling)
    if zero_field:
        return graph, np.zeros((side * side, n_states)), coupling
    ladder = np.asarray(ALPHA if n_states == 3 else (ALPHA[0], ALPHA[-1]))
    sizes = np.random.default_rng(seed).lognormal(0.0, 0.6, size=side * side)
    return graph, spatio_only_field(ladder, sizes), coupling


def _arm(
    side: int,
    n_states: int,
    kinds: tuple[MoveKind, ...],
    *,
    ladder: tuple[float, ...] | None = None,
    seed: int = 706,
) -> tuple[PottsNDEnvironment, PottsGraph, np.ndarray]:
    """One arm, its lattice and its field."""
    graph, field, coupling = _lattice(side, n_states, seed=seed)
    kwargs = {} if ladder is None else {"ladder": ladder}
    environment = PottsNDEnvironment(
        edges=list(graph.edges),
        n_nodes=side * side,
        coupling=coupling,
        field=field,
        kinds=kinds,
        moves=cluster_moves(graph, field),
        generator=np.random.default_rng(seed),
        **kwargs,  # type: ignore[arg-type]
    )
    return environment, graph, field


def _component(labels: np.ndarray, graph: PottsGraph, root: int) -> set[int]:
    """The root's like-coloured connected component, by breadth-first search.

    Not ``monochrome_partition``, the moves' own union-find.
    """
    incident: dict[int, list[int]] = {node: [] for node in range(graph.n_nodes)}
    for first, second in graph.edges:
        incident[first].append(second)
        incident[second].append(first)
    colour = int(labels[root])
    seen, frontier = {root}, [root]
    while frontier:
        node = frontier.pop()
        for neighbour in incident[node]:
            if neighbour not in seen and int(labels[neighbour]) == colour:
                seen.add(neighbour)
                frontier.append(neighbour)
    return seen


# --- the seam ------------------------------------------------------------


@pytest.mark.smoke
def test_both_moves_satisfy_the_seam_the_environment_takes() -> None:
    """The protocol is what joins two packages that may not import each other."""
    graph, field, _ = _lattice(4, 3)
    for move in cluster_moves(graph, field).values():
        assert isinstance(move, KeyedMove)


@pytest.mark.smoke
def test_a_cluster_kind_without_its_move_is_refused_at_construction() -> None:
    """An arm that cannot take half its actions is a misconfigured experiment.

    Refused at construction, where the mismatch is knowable.
    """
    graph, field, coupling = _lattice(4, 3)
    with pytest.raises(ValueError, match="needs its move passed"):
        PottsNDEnvironment(
            edges=list(graph.edges),
            n_nodes=16,
            coupling=coupling,
            field=field,
            kinds=(MoveKind.WOLFF,),
            generator=np.random.default_rng(0),
        )


@pytest.mark.smoke
def test_a_move_built_on_another_lattice_is_refused() -> None:
    """Two callers scoring one problem is the property this ticket exists for.

    A move on another lattice would disagree on energies with no error.
    """
    graph, field, coupling = _lattice(4, 3)
    other, other_field, _ = _lattice(6, 3)
    with pytest.raises(ValueError, match="would score different problems"):
        PottsNDEnvironment(
            edges=list(graph.edges),
            n_nodes=16,
            coupling=coupling,
            field=field,
            kinds=(MoveKind.SWENDSEN_WANG,),
            moves={MoveKind.SWENDSEN_WANG: SwendsenWangMove(other, other_field)},
            generator=np.random.default_rng(0),
        )


@pytest.mark.smoke
def test_a_move_and_a_field_of_different_heights_are_refused() -> None:
    """The same check on the move's own side, where the field arrives."""
    graph, _, _ = _lattice(4, 3)
    for kind in (WolffMove, SwendsenWangMove, NiedermayerMove):
        with pytest.raises(ValueError, match="a move and its environment score"):
            kind(graph, np.zeros((9, 3)))


# --- the moves are the oracle's ------------------------------------------


@pytest.mark.oracle
def test_a_wolff_step_is_potts_mcmcs_own_sweep_bitwise() -> None:
    """Above zero temperature the move *is* ``wolff_sweep``, root and colour aside.

    The wrapper adds the root, the colour and the rung's ``beta``; this catches either.
    """

    def check(temperature: float) -> None:
        graph, field, _ = _lattice(4, 2)
        move = WolffMove(graph, field)
        offsets, neighbours, couplings = graph.compressed_adjacency()
        state = np.ascontiguousarray(
            np.random.default_rng(3).integers(0, 2, size=graph.n_nodes), dtype=np.int64
        )

        keyed, charge = move.propose(
            state,
            temperature=temperature,
            site=5,
            label=1,
            rng=np.random.default_rng(99),
        )
        direct = state.copy()
        size = wolff_sweep(
            direct,
            field,
            offsets,
            neighbours,
            couplings,
            np.random.default_rng(99),
            beta=1.0 / temperature,
            root=5,
            proposed=1,
        )

        assert np.array_equal(keyed, direct)
        assert charge == size * (1 + 2 * len(graph.edges) // graph.n_nodes)

    every_value([0.25, 1.0, 4.0], check)


@pytest.mark.oracle
def test_a_niedermayer_step_is_potts_mcmcs_own_sweep_bitwise() -> None:
    """The same for issue #756's arm, and at ``T = 0`` as well.

    ``math.inf`` passes through `niedermayer_sweep`: one kernel, four temperatures.
    """

    def check(temperature: float) -> None:
        graph, field, _ = _lattice(4, 2)
        move = NiedermayerMove(graph, field)
        offsets, neighbours, couplings = graph.compressed_adjacency()
        state = np.ascontiguousarray(
            np.random.default_rng(3).integers(0, 2, size=graph.n_nodes), dtype=np.int64
        )

        keyed, charge = move.propose(
            state,
            temperature=temperature,
            site=5,
            label=1,
            rng=np.random.default_rng(99),
        )
        direct = state.copy()
        size = niedermayer_sweep(
            direct,
            field,
            offsets,
            neighbours,
            couplings,
            np.random.default_rng(99),
            beta=math.inf if temperature == 0.0 else 1.0 / temperature,
            threshold=move.threshold,
            root=5,
            partner=1,
        )

        assert np.array_equal(keyed, direct)
        assert charge == size * (1 + 2 * len(graph.edges) // graph.n_nodes)

    every_value([0.0, 0.25, 1.0, 4.0], check)


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST])
def test_a_swendsen_wang_pass_is_potts_mcmcs_own_sweep_bitwise(
    backend: Backend,
) -> None:
    """The same for the bond pass, which the action does not parameterize at all.

    On both routes; #754's Rust pass is the sweep's backend, not a second kernel.
    """

    def check(temperature: float) -> None:
        graph, field, _ = _lattice(4, 3)
        move = SwendsenWangMove(graph, field, backend)
        state = np.ascontiguousarray(
            np.random.default_rng(4).integers(0, 3, size=graph.n_nodes), dtype=np.int64
        )

        keyed, charge = move.propose(
            state,
            temperature=temperature,
            site=-1,
            label=-1,
            rng=np.random.default_rng(7),
        )
        direct = state.copy()
        swendsen_wang_sweep(
            direct,
            graph,
            field,
            np.random.default_rng(7),
            None,
            1.0 / temperature,
            backend,
        )

        assert np.array_equal(keyed, direct)
        assert charge == graph.n_nodes + 2 * len(graph.edges)

    every_value([0.25, 1.0, 4.0], check)


@pytest.mark.analytic
def test_keys_draw_the_same_cluster_size_law_as_seeds() -> None:
    """A ``blake2b`` digest used as a seed does not bias the move it feeds.

    Cluster sizes over 4,000 of each: total variation 0.005, chi-square p = 0.78.
    """
    graph, field, _ = _lattice(4, 2)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    state = np.ascontiguousarray(
        np.random.default_rng(3).integers(0, 2, size=graph.n_nodes), dtype=np.int64
    )

    def sizes(generators: list[np.random.Generator]) -> np.ndarray:
        return np.asarray(
            [
                wolff_sweep(
                    state.copy(),
                    field,
                    offsets,
                    neighbours,
                    couplings,
                    generator,
                    beta=1.0,
                    root=5,
                    proposed=1,
                )
                for generator in generators
            ]
        )

    draws = 4_000
    keyed = sizes([keyed_generator(key, state, 17) for key in range(draws)])
    seeded = sizes([np.random.default_rng(1_000_000 + key) for key in range(draws)])
    top = int(max(keyed.max(), seeded.max())) + 1
    first, second = (
        np.bincount(keyed, minlength=top),
        np.bincount(seeded, minlength=top),
    )
    variation = 0.5 * float(np.abs(first / draws - second / draws).sum())
    keep = (first + second) >= 10
    expected = second[keep] * first[keep].sum() / second[keep].sum()

    assert variation < 0.02
    assert float(stats.chisquare(first[keep], expected).pvalue) > SIGNIFICANCE


# --- zero temperature is exact -------------------------------------------


@pytest.mark.oracle
def test_a_wolff_step_at_zero_flips_the_roots_whole_component() -> None:
    """The ``T -> 0`` limit: the cluster is the component, and the recolouring
    survives only if it does not lower the score, by this file's own search.
    """

    def check(side: int) -> None:
        environment, graph, field = _arm(side, 3, (MoveKind.WOLFF,))
        rng = np.random.default_rng(0)

        for _ in range(40):
            state = environment.reset(rng)
            labels = np.asarray(state, dtype=np.int64)
            root, label = int(rng.integers(side * side)), int(rng.integers(3))
            successor, reward = environment.step(
                state, PottsAction(MoveKind.WOLFF, root, label, 0)
            )
            members = sorted(_component(labels, graph, root))
            gain = float(
                field[members, label].sum() - field[members, labels[root]].sum()
            )
            changed = set(np.flatnonzero(np.asarray(successor) != labels).tolist())

            if label != labels[root] and gain >= 0.0:
                assert changed == set(members)
            else:
                assert changed == set()
            assert reward >= 0.0

    every_value([4, 6], check)


@pytest.mark.oracle
def test_swendsen_wang_at_zero_recolours_every_component_as_a_block() -> None:
    """Every cluster is a like-coloured component, so it moves or it does not.

    Each pre-move component carries one label after; nothing outside it moved.
    """

    def check(side: int) -> None:
        environment, graph, _ = _arm(side, 3, (MoveKind.SWENDSEN_WANG,))
        rng = np.random.default_rng(1)

        for _ in range(40):
            state = environment.reset(rng)
            labels = np.asarray(state, dtype=np.int64)
            successor, reward = environment.step(
                state, PottsAction(MoveKind.SWENDSEN_WANG, -1, -1, 0)
            )
            after = np.asarray(successor)
            for root in range(side * side):
                members = sorted(_component(labels, graph, root))
                assert len(set(after[members].tolist())) == 1
            assert reward >= 0.0

    every_value([4, 6], check)


@pytest.mark.analytic
def test_the_bond_probability_is_the_oracles_and_exactly_one_at_zero() -> None:
    """``1 - exp(-J / T)``, and the limit taken rather than divided.

    ``1 / 0`` is not a float, so `potts_keyed` writes zero temperature out.
    """
    ladder = (0.0, 0.25, 1.0, 4.0)
    environment, _, _ = _arm(4, 3, (MoveKind.SWENDSEN_WANG,), ladder=ladder)
    coupling = critical_coupling(3)

    assert environment.bond_probability(0) == 1.0
    for rung, temperature in enumerate(ladder[1:], start=1):
        assert_allclose(
            environment.bond_probability(rung),
            1.0 - np.exp(-coupling / temperature),
            rtol=1e-15,
        )


# --- one energy, and the arm's prices ------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("kind", CLUSTER_KINDS)
def test_the_score_after_a_cluster_move_is_the_negated_energy_bitwise(
    kind: MoveKind,
) -> None:
    """A cluster move lands on a labelling ``sim.potts.energies`` scores alike.

    Over the labellings cluster moves reach, not drawn ones (`test_potts_nd.py`).
    """
    side = 6
    environment, graph, field = _arm(side, 3, (kind,))
    rng = np.random.default_rng(2)

    for _ in range(100):
        state = environment.reset(rng)
        action = PottsAction(
            kind,
            int(rng.integers(side * side)) if kind in PARAMETRIC_KINDS else -1,
            int(rng.integers(3)) if kind in PARAMETRIC_KINDS else -1,
            int(rng.integers(len(environment._ladder))),
        )
        successor, _ = environment.step(state, action)
        reference = -float(energies(graph, field, np.asarray(successor)[None])[0])
        assert environment.score(successor) == reference


@pytest.mark.analytic
def test_a_swendsen_wang_pass_costs_one_sweep_and_a_wolff_step_its_cluster() -> None:
    """`ground_state`'s units, move by move.

    The Wolff charge is read against this file's own component at ``T = 0``.
    """
    side = 6
    environment, graph, _ = _arm(side, 3, tuple(MoveKind))
    per_sweep = graph.n_nodes + 2 * len(graph.edges)
    per_member = 1 + 2 * len(graph.edges) // graph.n_nodes
    rng = np.random.default_rng(3)
    state = environment.reset(rng)
    labels = np.asarray(state, dtype=np.int64)

    assert (
        environment.visits(state, PottsAction(MoveKind.SWENDSEN_WANG, -1, -1, 3))
        == per_sweep
    )
    assert (
        environment.visits(state, PottsAction(MoveKind.SWEEP, -1, -1, 3)) == per_sweep
    )
    for root in (0, 7, side * side - 1):
        charge = environment.visits(state, PottsAction(MoveKind.WOLFF, root, 1, 0))
        assert charge == len(_component(labels, graph, root)) * per_member


@pytest.mark.smoke
@pytest.mark.parametrize("kind", CLUSTER_KINDS)
def test_visits_is_the_charge_the_move_itself_reports(kind: MoveKind) -> None:
    """The realized charge, not an average of it.

    Keyed on ``(state, action)``, both readings grow the same cluster.
    """
    side = 6
    environment, graph, field = _arm(side, 3, (kind,))
    move = cluster_moves(graph, field)[kind]
    rng = np.random.default_rng(4)

    for rung in range(len(environment._ladder)):
        state = environment.reset(rng)
        action = PottsAction(
            kind,
            int(rng.integers(side * side)) if kind in PARAMETRIC_KINDS else -1,
            int(rng.integers(3)) if kind in PARAMETRIC_KINDS else -1,
            rung,
        )
        _, charge = move.propose(
            state=np.asarray(state, dtype=np.int64),
            temperature=environment._ladder[rung],
            site=action.site,
            label=action.label,
            rng=environment._move_generator(state, action),
        )
        assert environment.visits(state, action) == charge


@pytest.mark.analytic
def test_the_vectorized_gain_is_the_scalar_one_bitwise() -> None:
    """``_flip_gains`` equals ``_flip_gain`` to the last bit, over every pair.

    The gather made a Wolff decision 9.1x cheaper; counts under a mask sum alike.
    """
    side = 6
    environment, _, _ = _arm(side, 3, (MoveKind.FLIP, MoveKind.SWEEP))
    rng = np.random.default_rng(11)

    for _ in range(5):
        state = environment.reset(rng)
        labels = np.asarray(state, dtype=np.int64)
        sites = np.repeat(np.arange(side * side), 2)
        targets = (labels[sites] + np.tile((1, 2), side * side)) % 3
        together = environment._flip_gains(labels, sites, targets)
        apart = [
            environment._flip_gain(
                labels, PottsAction(MoveKind.FLIP, int(site), int(label), 0)
            )
            for site, label in zip(sites, targets, strict=True)
        ]
        assert together.tolist() == apart


# --- keying, on the moves that needed it ---------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("kind", CLUSTER_KINDS)
def test_a_cluster_move_replays_and_another_key_does_not(kind: MoveKind) -> None:
    """``step`` is a pure function, and not by being constant.

    200 replays give one successor (#706); other key material moves at least one state.
    """
    side = 6
    environment, graph, field = _arm(side, 3, (kind,))
    other = PottsNDEnvironment(
        edges=list(graph.edges),
        n_nodes=side * side,
        coupling=critical_coupling(3),
        field=field,
        kinds=(kind,),
        moves=cluster_moves(graph, field),
        generator=np.random.default_rng(31_337),
    )
    rng = np.random.default_rng(5)
    state = environment.reset(rng)
    action = PottsAction(
        kind,
        7 if kind in PARAMETRIC_KINDS else -1,
        1 if kind in PARAMETRIC_KINDS else -1,
        3,
    )

    first, reward = environment.step(state, action)
    for _ in range(200):
        again, again_reward = environment.step(state, action)
        assert again == first
        assert again_reward == reward

    assert any(
        other.step(drawn, action)[0] != environment.step(drawn, action)[0]
        for drawn in (environment.reset(rng) for _ in range(20))
    )


# --- the feature map is the arm's ----------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kinds", "expected"),
    [
        (
            (MoveKind.FLIP, MoveKind.SWEEP),
            (FeatureColumn.GAIN, FeatureColumn.TEMPERATURE, FeatureColumn.PRICE),
        ),
        (
            (MoveKind.WOLFF,),
            (
                FeatureColumn.GAIN,
                FeatureColumn.TEMPERATURE,
                FeatureColumn.BOND,
                FeatureColumn.PRICE,
            ),
        ),
        ((MoveKind.SWENDSEN_WANG,), (FeatureColumn.TEMPERATURE, FeatureColumn.BOND)),
        (
            tuple(MoveKind),
            (
                FeatureColumn.GAIN,
                FeatureColumn.TEMPERATURE,
                FeatureColumn.BOND,
                FeatureColumn.PRICE,
            ),
        ),
    ],
)
def test_each_arm_carries_the_columns_that_can_vary_in_it(
    kinds: tuple[MoveKind, ...], expected: tuple[FeatureColumn, ...]
) -> None:
    """The gauge rule, held per arm rather than on average.

    Swendsen-Wang actions differ in rung alone, so a gain column would be constant.
    """
    environment, _, _ = _arm(6, 3, kinds)
    state = environment.reset(np.random.default_rng(6))
    offered = environment.actions(state)
    readings = environment.features(state, offered)

    assert environment._columns == expected
    assert environment.n_features() == len(expected)
    assert readings.shape == (len(offered), len(expected))
    assert np.all(np.ptp(readings, axis=0) > 0.0)


@pytest.mark.smoke
def test_an_arm_with_nothing_to_prefer_is_refused() -> None:
    """One move at one rung offers one action, so there is nothing to learn.

    Refused at configuration: such a run can neither fail nor succeed.
    """
    graph, field, coupling = _lattice(4, 3)
    with pytest.raises(ValueError, match="no feature column that varies"):
        PottsNDEnvironment(
            edges=list(graph.edges),
            n_nodes=16,
            coupling=coupling,
            field=field,
            ladder=(1.0,),
            kinds=(MoveKind.SWENDSEN_WANG,),
            moves=cluster_moves(graph, field),
            generator=np.random.default_rng(0),
        )


@pytest.mark.smoke
def test_the_mixed_arm_offers_every_kind_it_admits() -> None:
    """Step 6b's arm exists as a configuration before it is measured as one."""
    environment, _, _ = _arm(6, 3, tuple(MoveKind))
    state = environment.reset(np.random.default_rng(7))
    assert {action.kind for action in environment.actions(state)} == set(MoveKind)


# --- the arm is the classical run ----------------------------------------


def _declared_control(
    kind: MoveKind, side: int, n_states: int
) -> tuple[list[float], list[int], Rung, Budget]:
    """The arm walking `ground_state`'s own exponential ladder, over seeds.

    ``ExponentialTempSchedule(2.0, 0.05, steps)(s)``, not ``DEFAULT_LADDER``.
    """
    graph, field, coupling = _lattice(side, n_states)
    alpha = np.asarray(ALPHA if n_states == 3 else (ALPHA[0], ALPHA[-1]))
    sizes = np.random.default_rng(706).lognormal(0.0, 0.6, size=side * side)
    rung = Rung(
        name=f"{kind}-arm",
        graph=graph,
        field=field,
        alpha=alpha,
        sizes=sizes,
        n_states=n_states,
        optimum=None,
    )
    budget = Budget(unit=Cost.SITE_VISITS, size=200 * rung.visits_per_sweep)
    steps = budget.size // rung.visits_per_sweep
    schedule = ExponentialTempSchedule(ANNEAL_START, ANNEAL_END, steps)
    ladder = tuple(sorted({float(schedule(step)) for step in range(steps)}))
    order = [ladder.index(float(schedule(step))) for step in range(steps)]
    environment = PottsNDEnvironment(
        edges=list(graph.edges),
        n_nodes=side * side,
        coupling=coupling,
        field=field,
        ladder=ladder,
        kinds=(kind,),
        moves=cluster_moves(graph, field),
        generator=np.random.default_rng(706),
    )

    energy, spend = [], []
    for seed in range(COMPARISON_SEEDS):
        rng = np.random.default_rng(seed)
        state = environment.reset(rng)
        best, spent = environment.score(state), 0
        for rung_index in order:
            action = PottsAction(
                kind,
                int(rng.integers(side * side)) if kind in PARAMETRIC_KINDS else -1,
                int(rng.integers(n_states)) if kind in PARAMETRIC_KINDS else -1,
                rung_index,
            )
            spent += environment.visits(state, action)
            state, _ = environment.step(state, action)
            best = max(best, environment.score(state))
        energy.append(-best)
        spend.append(spent)
    return energy, spend, rung, budget


@pytest.mark.release
@pytest.mark.oracle
@pytest.mark.parametrize(
    ("kind", "runner", "fixed_charge"),
    [
        (MoveKind.SWENDSEN_WANG, run_swendsen_wang, True),
        (MoveKind.WOLFF, run_wolff, False),
    ],
)
def test_the_arm_walking_the_declared_ladder_is_its_classical_run(
    kind: MoveKind, runner: object, fixed_charge: bool
) -> None:
    """The control *is* the classical method, in energy and in what it spends.

    144 sites, `q = 3`, 200 sweeps, :data:`COMPARISON_SEEDS` seeds.
    Swendsen-Wang: -263.5 +- 5.2 against -262.3 +- 5.8 (Welch p = 0.38) at an
    identical 134,400 visits, so visits are asserted equal (Welch would divide
    by zero). Wolff: -203.6 +- 13.2 against -206.5 +- 17.8 (p = 0.47) at
    25,393 +- 7,584 against 26,328 +- 11,998 visits (p = 0.72), a
    five-sixths underspend that ``ground_state.run_annealed`` reports.
    """
    energy, spend, rung, budget = _declared_control(kind, 12, 3)
    runs = [
        runner(rung, budget, np.random.default_rng(seed))  # type: ignore[operator]
        for seed in range(COMPARISON_SEEDS)
    ]
    theirs = [float(run.energy) for run in runs]
    their_spend = [int(run.spent) for run in runs]

    assert float(stats.ttest_ind(energy, theirs, equal_var=False).pvalue) > SIGNIFICANCE
    if fixed_charge:
        assert set(spend) == set(their_spend) == {200 * rung.visits_per_sweep}
    else:
        assert (
            float(stats.ttest_ind(spend, their_spend, equal_var=False).pvalue)
            > SIGNIFICANCE
        )
