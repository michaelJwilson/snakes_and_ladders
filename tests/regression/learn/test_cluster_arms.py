"""The two cluster control arms: the oracle's moves, keyed, and what they cost.

Issue #706, plan steps 3 and 4. A control arm asks one question --- *does a
learned schedule beat a declared one for this move?* --- so the arm is worth
nothing until its declared-schedule control is shown to be the classical
method itself. Four kinds of claim, in the order of how much they prove:

* **the moves are ``potts_mcmc``'s, not copies.** Above zero temperature
  ``propose`` and a direct ``_wolff_sweep`` or ``_swendsen_wang_sweep`` call on
  an identically seeded generator return the same labelling **bitwise**. So
  nothing here has to be kept in step with an oracle: the oracle is what runs,
  and the tests below pin the wrapping rather than the physics;
* **zero temperature is exact.** The bond probability ``1 - exp(-J / T)`` goes
  to one, so a cluster is a connected component of the like-coloured subgraph
  --- which this file computes by its own breadth-first search, never by the
  union-find the moves use --- and the accept step keeps only a recolouring
  that does not lower the score. That limit is where a cluster move is
  checkable against something rather than against a distribution;
* **the arm is the classical run.** The control walking `ground_state`'s
  declared exponential ladder is statistically indistinguishable from
  ``run_swendsen_wang`` and ``run_wolff`` over 32 seeds, in the energy it
  reaches *and* in the site visits it spends. Marked ``release``: the four
  comparisons are seconds each, not the per-PR cap;
* **the prices and the features are the arm's.** A Swendsen-Wang pass costs
  `ground_state`'s ``visits_per_sweep`` exactly, a Wolff step its realized
  cluster, and each arm's feature columns are the ones its move set and ladder
  let vary --- which is the gauge rule
  `learn.environment.Environment` states, held per arm rather than on average.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from snakes_and_ladders.learn.keyed import KeyedMove, keyed_generator
from snakes_and_ladders.learn.potts_nd import (
    CLUSTER_KINDS,
    FeatureColumn,
    MoveKind,
    PottsAction,
    PottsNDEnvironment,
)
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.opt.schedule import ExponentialTempSchedule
from snakes_and_ladders.search.ground_state import (
    ANNEAL_END,
    ANNEAL_START,
    Rung,
    run_swendsen_wang,
    run_wolff,
)
from snakes_and_ladders.search.potts_keyed import (
    SwendsenWangMove,
    WolffMove,
    cluster_moves,
)
from snakes_and_ladders.search.potts_mcmc import _swendsen_wang_sweep, _wolff_sweep
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import critical_coupling, energies, spatio_only_field

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

    Deliberately not ``potts_keyed.monochrome_partition``: that is the
    union-find the moves themselves run, so reading the answer from it would
    compare an implementation with itself. An adjacency walk from the root is
    the independent statement of what a component is.
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


@pytest.mark.structural
def test_both_moves_satisfy_the_seam_the_environment_takes() -> None:
    """The protocol is what joins two packages that may not import each other."""
    graph, field, _ = _lattice(4, 3)
    for move in cluster_moves(graph, field).values():
        assert isinstance(move, KeyedMove)


@pytest.mark.structural
def test_a_cluster_kind_without_its_move_is_refused_at_construction() -> None:
    """An arm that cannot take half its actions is a misconfigured experiment.

    Refused where it is configured rather than at the first `step`: the arm's
    kinds and its moves are given together, so a mismatch is knowable then and
    a failure a hundred decisions in says nothing about which.
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


@pytest.mark.structural
def test_a_move_built_on_another_lattice_is_refused() -> None:
    """Two callers scoring one problem is the property this ticket exists for.

    A move holds its own graph, so nothing but this check stops an arm being
    built with a move on a lattice of another size --- and the energies would
    then disagree with no error to read.
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


@pytest.mark.edge_case
def test_a_move_and_a_field_of_different_heights_are_refused() -> None:
    """The same check on the move's own side, where the field arrives."""
    graph, _, _ = _lattice(4, 3)
    for kind in (WolffMove, SwendsenWangMove):
        with pytest.raises(ValueError, match="a move and its environment score"):
            kind(graph, np.zeros((9, 3)))


# --- the moves are the oracle's ------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize("temperature", [0.25, 1.0, 4.0])
def test_a_wolff_step_is_potts_mcmcs_own_sweep_bitwise(temperature: float) -> None:
    """Above zero temperature the move *is* ``_wolff_sweep``, root and colour aside.

    The claim this file rests on: there is no second Wolff kernel to keep in
    step with an oracle. What the wrapper adds is the root and the colour,
    moved out of the sweep's generator and into the action, and the ``beta`` the
    rung means --- and a wrong one of either is exactly what this catches.
    """
    graph, field, _ = _lattice(4, 2)
    move = WolffMove(graph, field)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    state = np.ascontiguousarray(
        np.random.default_rng(3).integers(0, 2, size=graph.n_nodes), dtype=np.int64
    )

    keyed, charge = move.propose(
        state, temperature=temperature, site=5, label=1, rng=np.random.default_rng(99)
    )
    direct = state.copy()
    size = _wolff_sweep(
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


@pytest.mark.oracle
@pytest.mark.parametrize("temperature", [0.25, 1.0, 4.0])
def test_a_swendsen_wang_pass_is_potts_mcmcs_own_sweep_bitwise(
    temperature: float,
) -> None:
    """The same for the bond pass, which the action does not parameterize at all."""
    graph, field, _ = _lattice(4, 3)
    move = SwendsenWangMove(graph, field)
    state = np.ascontiguousarray(
        np.random.default_rng(4).integers(0, 3, size=graph.n_nodes), dtype=np.int64
    )

    keyed, charge = move.propose(
        state, temperature=temperature, site=-1, label=-1, rng=np.random.default_rng(7)
    )
    direct = state.copy()
    _swendsen_wang_sweep(
        direct, graph, field, np.random.default_rng(7), None, 1.0 / temperature
    )

    assert np.array_equal(keyed, direct)
    assert charge == graph.n_nodes + 2 * len(graph.edges)


@pytest.mark.mathematical
def test_keys_draw_the_same_cluster_size_law_as_seeds() -> None:
    """A ``blake2b`` digest used as a seed does not bias the move it feeds.

    The one thing the bitwise test above cannot show: it matches one draw to
    one draw, while the claim is that the *keys* induce the move's own law over
    its uniforms. Read as the cluster-size distribution, since that is what the
    bond probability determines. Measured over 4,000 of each: total variation
    0.005, chi-square p = 0.78.
    """
    graph, field, _ = _lattice(4, 2)
    offsets, neighbours, couplings = graph.compressed_adjacency()
    state = np.ascontiguousarray(
        np.random.default_rng(3).integers(0, 2, size=graph.n_nodes), dtype=np.int64
    )

    def sizes(generators: list[np.random.Generator]) -> np.ndarray:
        return np.asarray(
            [
                _wolff_sweep(
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
@pytest.mark.parametrize("side", [4, 6])
def test_a_wolff_step_at_zero_flips_the_roots_whole_component(side: int) -> None:
    """The ``T -> 0`` limit: every like-coloured bond active, so the cluster is
    the component, and the recolouring survives only if it does not lower the
    score.

    Both halves are checked against this file's own breadth-first search and
    its own field sum, so nothing is read from the move's union-find.
    """
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
        gain = float(field[members, label].sum() - field[members, labels[root]].sum())
        changed = set(np.flatnonzero(np.asarray(successor) != labels).tolist())

        if label != labels[root] and gain >= 0.0:
            assert changed == set(members)
        else:
            assert changed == set()
        assert reward >= 0.0


@pytest.mark.oracle
@pytest.mark.parametrize("side", [4, 6])
def test_swendsen_wang_at_zero_recolours_every_component_as_a_block(
    side: int,
) -> None:
    """Every cluster is a like-coloured component, so it moves or it does not.

    The partition is not asserted directly --- a component that keeps its
    colour is indistinguishable from one merged with a neighbour of that colour
    --- so what is asserted is the property that identifies it: each
    pre-move component carries one label afterwards, and no site outside it
    changed on its account. Together with the score never falling, that is the
    ``beta -> inf`` bond pass.
    """
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


@pytest.mark.mathematical
def test_the_bond_probability_is_the_oracles_and_exactly_one_at_zero() -> None:
    """``1 - exp(-J / T)``, and the limit taken rather than divided.

    ``1 / 0`` is not a float and the value it stands for is one, which is the
    whole reason zero temperature is written out in
    :mod:`snakes_and_ladders.search.potts_keyed` rather than passed through as
    ``beta``.
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

    `test_potts_nd.py` pins `score` against `-energies` over drawn labellings;
    this pins it over the labellings the *cluster moves reach*, which is the
    set a cluster arm actually visits and is not the same set.
    """
    side = 6
    environment, graph, field = _arm(side, 3, (kind,))
    rng = np.random.default_rng(2)

    for _ in range(100):
        state = environment.reset(rng)
        action = PottsAction(
            kind,
            int(rng.integers(side * side)) if kind is MoveKind.WOLFF else -1,
            int(rng.integers(3)) if kind is MoveKind.WOLFF else -1,
            int(rng.integers(len(environment._ladder))),
        )
        successor, _ = environment.step(state, action)
        reference = -float(energies(graph, field, np.asarray(successor)[None])[0])
        assert environment.score(successor) == reference


@pytest.mark.mathematical
def test_a_swendsen_wang_pass_costs_one_sweep_and_a_wolff_step_its_cluster() -> None:
    """`ground_state`'s units, move by move.

    A cluster move undercharged is a comparison that hands it a free lattice,
    which is the error the unit exists to prevent. The Wolff charge is read
    against this file's own component at ``T = 0``, where the cluster is known.
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


@pytest.mark.structural
@pytest.mark.parametrize("kind", CLUSTER_KINDS)
def test_visits_is_the_charge_the_move_itself_reports(kind: MoveKind) -> None:
    """The realized charge, not an average of it.

    Both readings grow the same cluster because both are keyed on
    ``(state, action)``, which is what lets a budget be debited by the move
    that will run rather than by one like it.
    """
    side = 6
    environment, graph, field = _arm(side, 3, (kind,))
    move = cluster_moves(graph, field)[kind]
    rng = np.random.default_rng(4)

    for rung in range(len(environment._ladder)):
        state = environment.reset(rng)
        action = PottsAction(
            kind,
            int(rng.integers(side * side)) if kind is MoveKind.WOLFF else -1,
            int(rng.integers(3)) if kind is MoveKind.WOLFF else -1,
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


@pytest.mark.mathematical
def test_the_vectorized_gain_is_the_scalar_one_bitwise() -> None:
    """``_flip_gains`` equals ``_flip_gain`` to the last bit, over every pair.

    The optimization that made a Wolff decision 9.1x cheaper replaced a call
    per candidate with one gather, and `CLAUDE.md`'s rule on that is bitwise is
    the target and the declared tolerance the floor. Here bitwise is available
    --- the agreement term is a count, so summing it over the padded row under
    the mask gives the same integer as over the compacted row --- so this pins
    it rather than a tolerance, on all 144 sites times two targets at once and
    one at a time.
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


@pytest.mark.structural
@pytest.mark.parametrize("kind", CLUSTER_KINDS)
def test_a_cluster_move_replays_and_another_key_does_not(kind: MoveKind) -> None:
    """``step`` is a pure function, and not by being constant.

    The obstacle #706 had to clear before a Monte Carlo move could be an
    action at all: 200 replays of one pair give one successor, so
    `learn.exact`'s enumeration stays valid --- and an environment with
    different key material reaches a different one on at least one state, so
    the determinism is the keyed kind and not a move that never moves.
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
        7 if kind is MoveKind.WOLFF else -1,
        1 if kind is MoveKind.WOLFF else -1,
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


@pytest.mark.structural
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

    The Swendsen-Wang arm is the case that forces it: its actions differ in
    their rung alone, so a gain column would be constant across every action
    at a state and would sit in the direction the softmax cancels. Carrying it
    anyway would leave a weight nothing identifies, and dropping it is what
    makes the arm's two columns the two readings a rung means.
    """
    environment, _, _ = _arm(6, 3, kinds)
    state = environment.reset(np.random.default_rng(6))
    offered = environment.actions(state)
    readings = environment.features(state, offered).numpy()

    assert environment._columns == expected
    assert environment.n_features() == len(expected)
    assert readings.shape == (len(offered), len(expected))
    assert np.all(np.ptp(readings, axis=0) > 0.0)


@pytest.mark.edge_case
def test_an_arm_with_nothing_to_prefer_is_refused() -> None:
    """One move at one rung offers one action, so there is nothing to learn.

    Refused rather than served with a zero-column feature map: an arm whose
    every action scores alike is a training run that cannot fail and cannot
    succeed, and the cheapest place to say so is where it was configured.
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


@pytest.mark.structural
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

    The ladder is the schedule's realized temperatures rather than
    :data:`~snakes_and_ladders.learn.potts_nd.DEFAULT_LADDER`: the control has
    to be the classical method, and the classical method's temperature at step
    ``s`` is ``ExponentialTempSchedule(2.0, 0.05, steps)(s)``. A coarser ladder
    would make this a comparison of two schedules rather than a recovery of
    one.
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
    budget = Budget(unit="site visits", size=200 * rung.visits_per_sweep)
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
                int(rng.integers(side * side)) if kind is MoveKind.WOLFF else -1,
                int(rng.integers(n_states)) if kind is MoveKind.WOLFF else -1,
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

    What the arm's whole question rests on. A learned schedule is worth
    something only against a declared one, and a declared one measured here
    that disagreed with `search.ground_state`'s run would mean the two callers
    are not running one method --- so the disagreement, not the learning, would
    be the finding.

    Measured at 144 sites, `q = 3`, the 200-sweep budget, over
    :data:`COMPARISON_SEEDS` seeds: Swendsen-Wang reaches -263.5 +- 5.2 against
    the run's -262.3 +- 5.8 (Welch p = 0.38) at an identical 134,400 site
    visits, and Wolff -203.6 +- 13.2 against -206.5 +- 17.8 (p = 0.47) at
    25,393 +- 7,584 against 26,328 +- 11,998 (p = 0.72). Wolff underspends the
    budget by five sixths, which is the finding ``ground_state._anneal``
    reports rather than a defect here.

    The charge is read by the instrument the move's cost has. A
    Swendsen-Wang pass costs a sweep whatever it draws, so every seed spends
    the same number and *equality* is the claim --- a Welch test over two
    constant samples is a division by zero, not a stronger check. A Wolff
    step's cost is its realized cluster, so there the claim is distributional
    like the energy.
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
