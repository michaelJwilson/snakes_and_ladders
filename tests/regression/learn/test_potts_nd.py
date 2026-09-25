"""The N-D Potts environment: one energy, a keyed move, and ICM inside the class.

Issue #706, the single-site arm. `PottsNDEnvironment.score` is
`-sim.potts.energies` bitwise on 200 labellings at three instances. `step` is
deterministic though the moves are Monte Carlo: keyed on state and action.
The rung-zero sweep policy is `iterated_conditional_modes` to the label (the
fine action at `T = 0` is steepest ascent, kept apart). A sweep costs
`search.ground_state`'s `visits_per_sweep`. Oracles (#704): `-J|E|` in zero
field, the exact cut at two labels, enumeration at nine sites; #729 adds a
planted field recovered at both grains.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from sal.learn.keyed import keyed_generator, state_key
from sal.learn.policy import EpsilonGreedyPolicy, LinearPolicy
from sal.learn.potts_nd import (
    DEFAULT_LADDER,
    PottsAction,
    PottsNDEnvironment,
)
from sal.learn.rollout import rollout
from sal.sample.potts_mcmc import MoveKind
from sal.search.ground_state import Rung
from sal.search.icm import iterated_conditional_modes
from sal.search.maxflow import ising_ground_state
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.potts import critical_coupling, energies, spatio_only_field

from tests._rows import every_row, every_value

#: The class ladder the fixtures tilt by, `search.ground_state.SWEEP_ALPHA`'s
#: shape: three equally spaced values so the null class sits in the middle.
ALPHA = (-1.0, 0.0, 1.0)


def _instance(
    side: int, n_states: int, *, seed: int = 706, zero_field: bool = False
) -> tuple[PottsNDEnvironment, PottsGraph, np.ndarray]:
    """An environment and the graph and field a classical method would take."""
    coupling = critical_coupling(n_states)
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, coupling)
    if zero_field:
        field = np.zeros((side * side, n_states))
    else:
        ladder = np.asarray(ALPHA if n_states == 3 else (ALPHA[0], ALPHA[-1]))
        sizes = np.random.default_rng(seed).lognormal(0.0, 0.6, size=side * side)
        field = spatio_only_field(ladder, sizes)
    environment = PottsNDEnvironment(
        edges=list(graph.edges),
        n_nodes=side * side,
        coupling=coupling,
        field=field,
        generator=np.random.default_rng(seed),
    )
    return environment, graph, field


@pytest.mark.oracle
def test_the_score_is_the_negated_energy_bitwise() -> None:
    """`score` equals `-energies` to the last bit on 200 seeded labellings.

    Same order of arithmetic as `sim.potts.energies` (a gather and a pairwise
    sum over the edges); reassociation moves a bit.
    """

    def check(side: int, n_states: int) -> None:
        environment, graph, field = _instance(side, n_states)
        rng = np.random.default_rng(0)

        for _ in range(200):
            state = environment.reset(rng)
            theirs = -float(energies(graph, field, np.asarray(state)[None])[0])
            assert environment.score(state) == theirs

    every_row([(3, 3), (6, 3), (8, 2)], check)


@pytest.mark.analytic
def test_a_step_is_the_score_difference() -> None:
    """Every action's reward equals the change in `score` it caused.

    To 1e-12: the local delta reassociates the whole-lattice sum.
    """
    environment, _, _ = _instance(4, 3)
    rng = np.random.default_rng(1)

    for _ in range(20):
        state = environment.reset(rng)
        for action in environment.actions(state):
            successor, reward = environment.step(state, action)
            assert reward == pytest.approx(
                environment.score(successor) - environment.score(state), abs=1e-12
            )


@pytest.mark.smoke
def test_a_keyed_move_replays_and_is_not_degenerate() -> None:
    """The same state and action give the same successor; a different state does not.

    A move ignoring its key passes the first half and fails the second.
    """
    environment, _, _ = _instance(4, 3)
    rng = np.random.default_rng(2)
    state = environment.reset(rng)
    hot = PottsAction(MoveKind.SWEEP, -1, -1, len(DEFAULT_LADDER) - 1)

    first = environment.step(state, hot)
    for _ in range(50):
        assert environment.step(state, hot) == first

    others = {environment.step(environment.reset(rng), hot)[0] for _ in range(20)}
    assert len(others) > 1


@pytest.mark.smoke
def test_the_key_is_stable_across_processes() -> None:
    """The key is a digest of the bytes, not `hash`, so it survives a restart.

    The literal is `blake2b`'s; `hash` is salted per process.
    """
    state = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

    assert state_key(706, state, 3) == state_key(706, state.copy(), 3)
    assert state_key(706, state, 3) != state_key(707, state, 3)
    assert state_key(706, state, 3) != state_key(706, state, 4)
    assert keyed_generator(706, state, 3).random() == pytest.approx(
        keyed_generator(706, state, 3).random(), abs=0.0
    )


@pytest.mark.oracle
def test_the_sweep_at_zero_is_one_icm_sweep() -> None:
    """Repeating the rung-zero sweep reproduces ICM's labelling, label for label.

    Same site order and the same first `integers` draw for the start.
    """

    def check(side: int) -> None:
        n_states = 3
        environment, graph, field = _instance(side, n_states)

        state = tuple(
            int(value)
            for value in np.random.default_rng(0).integers(
                0, n_states, size=side * side
            )
        )
        for _ in range(200):
            successor, _ = environment.step(state, environment.icm_action())
            if successor == state:
                break
            state = successor

        theirs, value, *_ = iterated_conditional_modes(
            graph, field, n_states, np.random.default_rng(0)
        )

        assert state == tuple(int(label) for label in theirs)
        assert environment.score(state) == pytest.approx(-value, abs=1e-9)

    every_value([4, 6], check)


@pytest.mark.analytic
def test_steepest_ascent_is_a_different_baseline() -> None:
    """The best flip anywhere is not ICM, and it converges to its own optimum.

    No improving flip remains; measured 61.87 against ICM's 57.33, two starts.
    """
    environment, _, field = _instance(6, 3)

    state = tuple(int(value) for value in field.argmax(axis=1))
    for _ in range(50 * environment.n_nodes):
        successor, reward = environment.step(state, environment.steepest_action(state))
        if reward <= 0.0:
            break
        state = successor

    gains = [
        environment.step(state, action)[1]
        for action in environment.actions(state)
        if action.kind is MoveKind.FLIP and action.rung == 0
    ]
    assert max(gains) <= 0.0


@pytest.mark.oracle
def test_the_zero_field_optimum_is_the_closed_form() -> None:
    """In zero field the best score is `J |E|`, the closed form #704 established.

    Reached by the control policy from the uniform start, with no search in it.
    """
    environment, graph, _ = _instance(6, 3, zero_field=True)
    closed = environment.coupling * len(graph.edges)

    uniform = tuple([0] * environment.n_nodes)

    assert environment.score(uniform) == pytest.approx(closed, rel=1e-12)


@pytest.mark.oracle
def test_two_labels_agree_with_the_exact_cut() -> None:
    """At `q = 2` the graph cut is exact, so it is the optimum to compare against.

    The cut's labelling scores its own value through this environment (#704).
    """
    environment, graph, field = _instance(8, 2)
    labelling, cut = ising_ground_state(graph, field)

    scored = environment.score(tuple(int(value) for value in labelling))

    assert scored == pytest.approx(-cut, abs=1e-9)


@pytest.mark.oracle
def test_nine_sites_enumerate_to_the_same_optimum() -> None:
    """At nine sites and three labels the optimum is one of 19,683, and it is found.

    A sign error or dropped term would move the maximum.
    """
    environment, graph, field = _instance(3, 3)

    best = max(
        environment.score(state)
        for state in itertools.product(range(3), repeat=environment.n_nodes)
    )
    _, exact = max(
        (
            (state, -float(energies(graph, field, np.asarray(state)[None])[0]))
            for state in itertools.product(range(3), repeat=environment.n_nodes)
        ),
        key=lambda pair: pair[1],
    )

    assert best == exact


@pytest.mark.smoke
def test_a_sweep_costs_what_the_suite_charges() -> None:
    """A sweep is `n_nodes + 2 n_edges` visits, and a flip its own degree plus one.

    `search.ground_state`'s unit exactly.
    """
    side, n_states = 6, 3
    environment, graph, field = _instance(side, n_states)
    rung = Rung(
        name="price",
        graph=graph,
        field=field,
        alpha=np.asarray(ALPHA),
        sizes=np.ones(side * side),
        n_states=n_states,
        optimum=None,
    )
    state = environment.reset(np.random.default_rng(3))

    sweep = PottsAction(MoveKind.SWEEP, -1, -1, 0)
    assert environment.visits(state, sweep) == rung.visits_per_sweep
    assert environment.sweep_visits == rung.visits_per_sweep

    flips = [a for a in environment.actions(state) if a.kind is MoveKind.FLIP]
    assert all(2 <= environment.visits(state, a) <= 5 for a in flips)


@pytest.mark.smoke
def test_the_scored_subset_belongs_to_the_state() -> None:
    """The same state offers the same candidates every time it is reached.

    So `learn.exact` enumerates the actions the policy scored.
    """
    environment, _, _ = _instance(12, 3)
    rng = np.random.default_rng(4)
    state = environment.reset(rng)

    first = list(environment.actions(state))
    assert first == list(environment.actions(state))
    assert len(first) <= 128 + len(DEFAULT_LADDER)

    other = environment.reset(rng)
    assert [a for a in environment.actions(other) if a.kind is MoveKind.FLIP] != [
        a for a in first if a.kind is MoveKind.FLIP
    ]


@pytest.mark.smoke
def test_the_features_carry_no_constant_column() -> None:
    """None of the three columns is constant across a state's actions.

    Gain varies by site, temperature by rung, price by kind: the softmax gauge.
    """
    environment, _, _ = _instance(6, 3)
    state = environment.reset(np.random.default_rng(5))

    features = environment.features(state, environment.actions(state))

    assert features.shape[1] == environment.n_features() == 3
    assert (features.std(axis=0) > 0.0).all()


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_nodes": 1}, "at least two sites"),
        ({"coupling": -1.0}, "coupling must be non-negative"),
        ({"ladder": ()}, "non-empty and non-negative"),
        ({"ladder": (1.0, 0.0)}, "ascending"),
        ({"candidates": 0}, "candidates must be >= 1"),
        ({"kinds": ()}, "at least one move kind"),
    ],
)
def test_an_unusable_construction_is_refused(
    kwargs: dict[str, object], message: str
) -> None:
    """Each range is refused with its reason, not clamped.

    An unsorted ladder would make "rung 0" a different temperature per instance.
    """
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 1.0)
    base: dict[str, object] = {
        "edges": list(graph.edges),
        "n_nodes": 16,
        "coupling": 1.0,
        "field": np.zeros((16, 3)),
        "generator": np.random.default_rng(0),
    }
    with pytest.raises(ValueError, match=message):
        PottsNDEnvironment(**{**base, **kwargs})  # type: ignore[arg-type]


@pytest.mark.smoke
def test_a_uniform_field_is_refused_by_shape() -> None:
    """A `(n_states,)` field is the old environment's, and this one refuses it.

    A uniform field makes the ground state `argmax(h)` everywhere.
    """
    graph = lattice_graph((4, 4), BoundaryCondition.OPEN, 1.0)
    with pytest.raises(ValueError, match="a per-site field"):
        PottsNDEnvironment(
            edges=list(graph.edges),
            n_nodes=16,
            coupling=1.0,
            field=np.zeros(3),
            generator=np.random.default_rng(0),
        )


# --- The planted ground state ---------------------------------------------


def _planted(
    side: int, n_states: int, *, seed: int = 729
) -> tuple[PottsNDEnvironment, PottsGraph, np.ndarray, tuple[int, ...], float]:
    """An instance whose ground state is the labelling its field was built from.

    `margin = J |E| + 1`: `k` changed sites lose `k margin` and regain `< margin`.
    """
    rng = np.random.default_rng(seed)
    coupling = critical_coupling(n_states)
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, coupling)
    n_nodes = side * side
    planted = tuple(int(value) for value in rng.integers(n_states, size=n_nodes))
    margin = coupling * len(graph.edges) + 1.0
    field = np.zeros((n_nodes, n_states))
    field[np.arange(n_nodes), planted] = margin
    environment = PottsNDEnvironment(
        edges=list(graph.edges),
        n_nodes=n_nodes,
        coupling=coupling,
        field=field,
        generator=np.random.default_rng(seed),
    )
    return environment, graph, field, planted, margin


@pytest.mark.critical
@pytest.mark.oracle
def test_the_planted_labelling_is_the_enumerated_maximum() -> None:
    """At nine sites the plant is the best of 19,683, and the only one.

    Via `sim.potts.energies`: 119.5557792617, no tie; margin 13.0606304649 per site.
    """
    environment, graph, field, planted, margin = _planted(3, 3)

    states = np.array(list(itertools.product(range(3), repeat=9)), dtype=np.int64)
    scores = -energies(graph, field, states)
    best = int(np.argmax(scores))

    assert tuple(int(value) for value in states[best]) == planted
    assert int((scores == scores[best]).sum()) == 1, "the plant is the only maximum"
    assert environment.score(planted) == float(scores[best])
    assert margin > environment.coupling * len(graph.edges)


@pytest.mark.critical
@pytest.mark.end2end
def test_both_grains_recover_the_planted_ground_state() -> None:
    """A learned schedule and steepest ascent each return the planted labelling.

    From eight seeded starts, label by label: the coarse grain (a greedy
    `LinearPolicy`, i.e. ICM) in its first sweep; steepest ascent in 5 to 9
    decisions and 19 to 32 visits at nine sites, 21 to 29 and 91 to 125 at
    thirty-six, against a sweep's 33 and 156.
    """

    def check(side: int) -> None:
        environment, _, _, planted, _ = _planted(side, 3)
        policy = LinearPolicy(environment.n_features())
        policy.set_weights(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64))
        schedule = EpsilonGreedyPolicy(policy, 0.0)

        for start in range(8):
            episode = rollout(
                environment,
                schedule,
                np.random.default_rng(1000 + start),
                max_steps=3,
                stop_at_local_optimum=False,
            )
            assert [action.kind for action in episode.actions] == [MoveKind.SWEEP] * 3
            assert episode.rewards[1:] == (0.0, 0.0), "the first sweep did all of it"
            assert max(episode.states, key=environment.score) == planted

            state = environment.reset(np.random.default_rng(2000 + start))
            spent = 0
            for _ in range(4 * environment.n_nodes):
                action = environment.steepest_action(state)
                spent += environment.visits(state, action)
                state, gain = environment.step(state, action)
                if gain == 0.0:
                    break
            assert state == planted
            assert spent >= environment.n_nodes

        assert not environment.is_terminal(planted), "a sampler never stops itself"

    every_value([3, 6], check)
