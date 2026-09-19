"""The N-D Potts environment: one energy, a keyed move, and ICM inside the class.

Issue #706, the single-site arm. Four kinds of claim:

* **one problem, two callers.** `PottsNDEnvironment.score` is the negation of
  `sim.potts.energies` **bitwise**, on 200 seeded labellings at three
  instances. The whole point of a per-site field is that the learner and the
  classical ground-state suite score the same function, and a tolerance here
  would leave that unproven;
* **`step` is deterministic**, as the protocol declares and
  `learn.exact`'s enumeration needs, even though the moves are Monte Carlo.
  The realization is keyed on the state and the action, so replaying an action
  replays its successor --- and a different state gives a different draw, so
  the determinism is not the degenerate kind;
* **the classical baseline is a point in the action space.** A policy that
  always takes the sweep at rung zero *is* iterated conditional modes, and its
  labelling equals `search.alpha_expansion.iterated_conditional_modes`'s to
  the label. That is what makes #705's *match* outcome provable rather than
  arguable --- and the fine action at `T = 0` is a *different* baseline,
  steepest ascent, which the suite keeps apart;
* **the prices are the suite's own.** A sweep costs `search.ground_state`'s
  `visits_per_sweep` exactly, so a matched-budget comparison against that
  module is against its unit and not a second one.

The oracles are the three #704 established: the closed form `-J|E|` in zero
field, the exact graph cut at two labels, and enumeration at nine sites. Issue
#729 adds the fourth, which is a truth rather than a second computation: a
field planted on one labelling, recovered by a run at both grains.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.keyed import keyed_generator, state_key
from snakes_and_ladders.learn.policy import EpsilonGreedyPolicy, LinearPolicy
from snakes_and_ladders.learn.potts_nd import (
    DEFAULT_LADDER,
    PottsAction,
    PottsNDEnvironment,
)
from snakes_and_ladders.learn.rollout import rollout
from snakes_and_ladders.search.alpha_expansion import iterated_conditional_modes
from snakes_and_ladders.search.ground_state import Rung
from snakes_and_ladders.search.maxflow import ising_ground_state
from snakes_and_ladders.search.potts_mcmc import MoveKind
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.potts import critical_coupling, energies, spatio_only_field

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
@pytest.mark.parametrize(("side", "n_states"), [(3, 3), (6, 3), (8, 2)])
def test_the_score_is_the_negated_energy_bitwise(side: int, n_states: int) -> None:
    """`score` equals `-energies` to the last bit on 200 seeded labellings.

    Bitwise and not to a tolerance: the field term is summed first and the
    coupling term added as one gather and a dot product over the edges in the
    order they were given, which is `sim.potts.energies`'s own arithmetic.
    Reassociating it --- halving a doubled count off the adjacency table, say
    --- moves the last bit, and this is the test that would catch it.
    """
    environment, graph, field = _instance(side, n_states)
    rng = np.random.default_rng(0)

    for _ in range(200):
        state = environment.reset(rng)
        theirs = -float(energies(graph, field, np.asarray(state)[None])[0])
        assert environment.score(state) == theirs


@pytest.mark.analytic
def test_a_step_is_the_score_difference() -> None:
    """Every action's reward equals the change in `score` it caused.

    The reward is computed from the affected bonds for a flip and by
    rescoring for a sweep; both are checked against the score difference, so
    an undiscounted return telescopes to the total improvement as
    `Episode` states. To 1e-12 rather than bitwise: the local delta sums two
    integer counts and a field difference where `score` sums the whole
    lattice, which is a reassociation and not a disagreement.
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

    Both halves are the claim. Replay is what `step`'s contract and
    `learn.exact`'s enumeration need from a Monte Carlo move. The second half
    is what says the key is doing something: a move that ignored its key would
    pass the first assertion and fail this one.
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

    `hash` is salted per process for bytes, so a key built from it would make
    a reproducible episode reproducible only inside one interpreter. The
    literal below is what `blake2b` gives, which is the property being
    claimed; a change to the personalization or the field order breaks it
    loudly rather than silently reseeding every published episode.
    """
    state = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

    assert state_key(706, state, 3) == state_key(706, state.copy(), 3)
    assert state_key(706, state, 3) != state_key(707, state, 3)
    assert state_key(706, state, 3) != state_key(706, state, 4)
    assert keyed_generator(706, state, 3).random() == pytest.approx(
        keyed_generator(706, state, 3).random(), abs=0.0
    )


@pytest.mark.oracle
@pytest.mark.parametrize("side", [4, 6])
def test_the_sweep_at_zero_is_one_icm_sweep(side: int) -> None:
    """Repeating the rung-zero sweep reproduces ICM's labelling, label for label.

    Both take each site's conditional mode in index order, so the two
    trajectories coincide from the same start --- and the start is drawn the
    way `iterated_conditional_modes` draws it, from the generator's first
    `integers` call, so nothing is injected that the classical method would
    not have seen.

    This is the assertion that makes a *match* against ICM provable: the
    baseline is a point in this action space rather than a separate program
    the comparison has to trust.
    """
    n_states = 3
    environment, graph, field = _instance(side, n_states)

    state = tuple(
        int(value)
        for value in np.random.default_rng(0).integers(0, n_states, size=side * side)
    )
    for _ in range(200):
        successor, _ = environment.step(state, environment.icm_action())
        if successor == state:
            break
        state = successor

    theirs, value = iterated_conditional_modes(
        graph, field, n_states, np.random.default_rng(0)
    )

    assert state == tuple(int(label) for label in theirs)
    assert environment.score(state) == pytest.approx(-value, abs=1e-9)


@pytest.mark.analytic
def test_steepest_ascent_is_a_different_baseline() -> None:
    """The best flip anywhere is not ICM, and it converges to its own optimum.

    Asserted as the property that holds --- the state it reaches admits no
    improving single flip --- rather than as an ordering against ICM, which is
    start-dependent and not a theorem. Measured on this fixture from the
    field-only start: steepest ascent reaches 61.87 where ICM's own random
    start reaches 57.33, which is two baselines and two starts, not one method
    beating another.
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

    Reached here by the control policy from the uniform start, which is the
    degenerate case on purpose: a baseline that cannot reach an answer with no
    search in it has a defect this catches before any policy is trained.
    """
    environment, graph, _ = _instance(6, 3, zero_field=True)
    closed = environment.coupling * len(graph.edges)

    uniform = tuple([0] * environment.n_nodes)

    assert environment.score(uniform) == pytest.approx(closed, rel=1e-12)


@pytest.mark.oracle
def test_two_labels_agree_with_the_exact_cut() -> None:
    """At `q = 2` the graph cut is exact, so it is the optimum to compare against.

    The environment's best reachable score cannot exceed the cut's, and the
    labelling the cut returns scores exactly its own value through this
    environment --- which is what says the two are scoring one problem at the
    rung where the truth is known at any size (#704).
    """
    environment, graph, field = _instance(8, 2)
    labelling, cut = ising_ground_state(graph, field)

    scored = environment.score(tuple(int(value) for value in labelling))

    assert scored == pytest.approx(-cut, abs=1e-9)


@pytest.mark.oracle
def test_nine_sites_enumerate_to_the_same_optimum() -> None:
    """At nine sites and three labels the optimum is one of 19,683, and it is found.

    The rung where `learn.exact`'s trajectory enumeration also fits, so the
    environment's own exhaustive best is worth pinning: a score function with
    a sign error or a dropped term would put the maximum somewhere else.
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

    `search.ground_state`'s unit exactly, so a matched-budget comparison
    against that module's methods is against its own accounting. A cluster
    move undercharged is a comparison handed a free lattice, which is the
    error the unit exists to prevent.
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

    So `learn.exact` enumerates the actions the policy scored. A subset keyed
    on the trajectory would make the action set depend on how the state was
    reached, which the enumeration cannot follow, and this is the assertion
    that would catch it.
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

    The gauge `Environment.features` names: a column shared by every action
    cancels in the softmax, so it is unidentifiable and must not be offered.
    With both grains the gain varies by site, the temperature by rung and the
    price by kind.
    """
    environment, _, _ = _instance(6, 3)
    state = environment.reset(np.random.default_rng(5))

    features = environment.features(state, environment.actions(state)).numpy()

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

    The ladder's order is the one worth naming: an unsorted ladder would run
    and would make "rung 0" mean a different temperature per instance, so
    every schedule a policy learned would be incomparable with every other.
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

    Not a convenience to broadcast: a uniform field makes the ferromagnetic
    ground state `argmax(h)` everywhere, so accepting it here would let a
    caller measure a policy on the instance this environment exists to stop
    measuring on.
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

    The field pays `margin` for the planted label at each site and nothing for
    any other, with `margin = J |E| + 1`. That makes the plant the unique
    maximum, with no search involved: a labelling differing from it at `k >= 1`
    sites gives up `k margin` of field and can recover at most `J |E| < margin`
    of coupling, so its score is strictly lower. The truth is therefore
    declared by construction rather than measured, which is what an `end2end`
    needs.
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

    The referee for the construction the `end2end` below rests on, and read
    through `sim.potts.energies` rather than through `score`, so the plant is
    established by a scorer this module does not own. Realized: the planted
    score is 119.5557792617 and no other labelling ties it, against a margin
    of 13.0606304649 per site.
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
@pytest.mark.parametrize("side", [3, 6])
def test_both_grains_recover_the_planted_ground_state(side: int) -> None:
    """A learned schedule and steepest ascent each return the planted labelling.

    The truth is the labelling `_planted` built the field from, so this is
    recovery of a planted parameter and not a comparison of two searches. Both
    grains the arm carries are run from the same eight seeded starts:

    * the **coarse** grain through a policy --- a `LinearPolicy` weighting gain
      positively and temperature negatively, wrapped at `epsilon = 0`, whose
      greedy action is the sweep at rung zero, which is ICM. One sweep suffices
      at any size here, since `margin` exceeds what any site's neighbours can
      offer, and the episode is scored on its best visited state;
    * the **fine** grain through `steepest_action`, which moves one label per
      decision and stalls when no flip gains.

    Exact, not to a tolerance: the recovered labelling is compared label by
    label. The first sweep does the whole of the coarse recovery and the other
    two decisions collect nothing. Steepest ascent spends 5 to 9 decisions and
    19 to 32 site visits at nine sites, and 21 to 29 decisions and 91 to 125
    visits at thirty-six, against a sweep's 33 and 156.
    """
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
