"""The Potts environment over a graph rather than a chain.

`PottsEnvironment.on_graph` is a second constructor and not a second class: the
energy, the move set, the features and the reward are shared with the chain,
and only the adjacency differs. So most of what could break is already
covered by `test_learn_potts.py`, and what is checked here is the part that
is new --- that the adjacency is read correctly, that the local reward still
matches a full evaluation when a site has more than two neighbours, and that
a chain built as a graph is the chain.

The graph arrives as plain edge indices. `learn/CLAUDE.md` forbids importing
`snakes_and_ladders.sim`, so a `PottsGraph` is unpacked by the caller; these tests do that
inline, which is also the demonstration that the adaptation is a two-field
read rather than a layer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.learn.exact import (
    exact_policy_gradient,
    finite_difference_gradient,
)
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import (
    PottsEnvironment,
    enumerate_configurations,
    optimum,
)
from snakes_and_ladders.learn.rollout import greedy_rollout
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

from tests.regression.learn.conftest import COUPLING, FIELD

_GRADIENT_TOLERANCE = 1e-8


def _lattice(shape: tuple[int, ...], boundary: BoundaryCondition) -> PottsEnvironment:
    """The environment over a lattice, with the graph unpacked at the boundary."""
    graph = lattice_graph(shape, boundary=boundary, coupling=COUPLING)
    return PottsEnvironment.on_graph(COUPLING, FIELD, graph.edges, graph.n_nodes)


@pytest.mark.oracle
def test_a_chain_built_as_a_graph_is_the_chain() -> None:
    # The claim that justifies one class rather than two: if the two
    # constructors disagreed anywhere, the generalization would have changed
    # the reference instance every existing result rests on.
    length = 5
    chain = PottsEnvironment(COUPLING, FIELD, length)
    as_graph = _lattice((length,), BoundaryCondition.OPEN)

    for state in enumerate_configurations(3, length):
        assert chain.energy(state) == pytest.approx(as_graph.energy(state), abs=1e-12)
        assert chain.actions(state) == as_graph.actions(state)
        assert chain.is_terminal(state) == as_graph.is_terminal(state)


@pytest.mark.oracle
def test_the_local_reward_matches_re_evaluating_the_energy_on_a_lattice() -> None:
    # A 3x3 open lattice has interior sites with four neighbours, which the
    # chain never exercises: its delta only ever sums two terms. An O(degree)
    # update that dropped a neighbour would pass every chain test.
    environment = _lattice((3, 3), BoundaryCondition.OPEN)
    rng = np.random.default_rng(0)
    for _ in range(20):
        state = environment.reset(rng)
        for action in environment.actions(state):
            successor, reward = environment.step(state, action)
            assert reward == pytest.approx(
                environment.energy(successor) - environment.energy(state), abs=1e-12
            )


@pytest.mark.oracle
def test_the_reward_matches_a_full_evaluation_under_a_periodic_boundary() -> None:
    # Periodic wrapping gives every site the same degree and makes a
    # 2-extent dimension list the same pair twice, as a doubled bond. Both
    # are adjacency the chain cannot produce.
    environment = _lattice((2, 3), BoundaryCondition.PERIODIC)
    rng = np.random.default_rng(1)
    for _ in range(20):
        state = environment.reset(rng)
        for action in environment.actions(state):
            successor, reward = environment.step(state, action)
            assert reward == pytest.approx(
                environment.energy(successor) - environment.energy(state), abs=1e-12
            )


@pytest.mark.analytic
def test_the_features_span_the_reward_on_a_lattice() -> None:
    environment = _lattice((3, 3), BoundaryCondition.OPEN)
    state = environment.reset(np.random.default_rng(2))
    actions = environment.actions(state)

    scored = (
        environment.features(state, actions) @ environment.greedy_weights()
    ).numpy()
    rewards = np.array([environment.step(state, action)[1] for action in actions])

    assert_allclose(scored, rewards, atol=1e-12)


@pytest.mark.oracle
def test_hill_climbing_reaches_the_enumerated_optimum_on_a_lattice() -> None:
    # 3**9 = 19,683 configurations, the same size #170's simulator validates
    # against, so the best configuration is an enumerated fact.
    environment = _lattice((3, 3), BoundaryCondition.OPEN)
    _, best = optimum(environment)
    rng = np.random.default_rng(3)
    reached = [
        environment.energy(
            greedy_rollout(environment, environment.reset(rng), 30).states[-1]
        )
        for _ in range(30)
    ]

    assert max(reached) == pytest.approx(best)


@pytest.mark.analytic
@pytest.mark.oracle
def test_the_enumerated_gradient_matches_central_differences_on_a_lattice() -> None:
    # The oracle that makes this an instance rather than a lookalike:
    # `snakes_and_ladders.learn.exact` carries it unchanged from the chain. A 2x2 lattice
    # keeps |A| ** horizon affordable at 8 actions and horizon 2.
    environment = _lattice((2, 2), BoundaryCondition.OPEN)
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor([0.5, 0.25], dtype=torch.float64))
    start = (0, 1, 2, 0)
    assert not environment.is_terminal(start), "a terminal start has no gradient"

    analytic = exact_policy_gradient(environment, policy, start, 2)
    numerical = finite_difference_gradient(environment, policy, start, 2)

    assert_allclose(
        analytic.detach().numpy(),
        numerical.numpy(),
        rtol=_GRADIENT_TOLERANCE,
        atol=_GRADIENT_TOLERANCE * float(np.linalg.norm(numerical.numpy())),
    )


@pytest.mark.smoke
def test_an_edge_naming_a_missing_node_is_refused() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
        PottsEnvironment.on_graph(COUPLING, FIELD, [(0, 3)], 3)
