"""The enumeration oracle for the coupled model, pinned two ways.

Against the factor graph's log-density assignment by assignment, and against
a second route to the evidence that shares no code with it: given the labels
the classes decouple and each is the forward recursion. Two implementations
agreeing prove nothing if both are wrong (`likelihood/CLAUDE.md`), which is
why the second route is a different algorithm and not a second enumeration.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.spatio_sequential import (
    conditional_state_posterior,
    enumerate_spatio_sequential,
    log_evidence_by_forward,
    log_joint_at,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.spatio_sequential import (
    SimulatedSpatioSequential,
    SpatioSequentialParams,
    coupled_factor_graph,
    simulate_spatio_sequential,
)


def _dataset(
    seed: int,
) -> tuple[SpatioSequentialParams, SimulatedSpatioSequential]:
    params = fixture("spatio_sequential", "ci").params
    return params, simulate_spatio_sequential(params, np.random.default_rng(seed))


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_enumerated_evidence_equals_the_per_class_forward_route(seed: int) -> None:
    params, data = _dataset(seed)

    exact = enumerate_spatio_sequential(params, data.observations)
    forward = log_evidence_by_forward(params, data.observations)

    assert abs(exact.log_evidence - forward) <= CROSS_DEVICE_RTOL_FLOAT64 * abs(forward)


@pytest.mark.oracle
def test_the_written_out_joint_is_the_factor_graph_log_density() -> None:
    params, data = _dataset(1)
    graph = coupled_factor_graph(params, data.observations)
    rng = np.random.default_rng(3)
    for _ in range(50):
        labels = rng.integers(0, params.n_classes, size=params.graph.n_nodes)
        states = rng.integers(
            0, params.n_states, size=(params.n_classes, params.n_positions)
        )
        assignment = {f"l{n}": int(labels[n]) for n in range(params.graph.n_nodes)}
        assignment.update(
            {
                f"k{s},{m}": int(states[m, s])
                for m in range(params.n_classes)
                for s in range(params.n_positions)
            }
        )

        expected = log_joint_at(params, data.observations, labels, states)
        assert abs(graph.log_density(assignment) - expected) < 1e-12 * abs(expected)


@pytest.mark.mathematical
def test_every_posterior_is_a_distribution() -> None:
    params, data = _dataset(2)

    exact = enumerate_spatio_sequential(params, data.observations)
    conditional = conditional_state_posterior(params, data.observations, data.labels)

    np.testing.assert_allclose(exact.label_posterior.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(exact.state_posterior.sum(axis=2), 1.0, rtol=1e-12)
    np.testing.assert_allclose(conditional.sum(axis=2), 1.0, rtol=1e-12)
    assert exact.log_evidence < 0.0  # discrete emissions: a probability


@pytest.mark.mathematical
def test_the_conditional_posterior_at_the_only_labelling_is_the_marginal_one() -> None:
    # With one class every node belongs to it, the labelling is unique, and
    # p(k | x) is Q(k | l, x): the two enumerations must agree exactly.
    params = fixture("spatio_sequential", "ci").params
    one_class = replace(
        params,
        n_classes=1,
        initial=params.initial[:1],
        emissions=params.emissions[:1],
    )
    data = simulate_spatio_sequential(one_class, np.random.default_rng(5))

    exact = enumerate_spatio_sequential(one_class, data.observations)
    conditional = conditional_state_posterior(one_class, data.observations, data.labels)

    np.testing.assert_allclose(exact.state_posterior, conditional, rtol=1e-12)


@pytest.mark.simulated_truth
def test_the_label_posterior_recovers_planted_labels_on_most_nodes() -> None:
    # 42 of 48 when measured; asserted at three quarters. The misses are the
    # nodes whose six observations happen to fit the other class.
    params = fixture("spatio_sequential", "ci").params
    planted = np.array([0, 0, 1, 1])
    hits, total = 0, 0
    for seed in range(12):
        data = simulate_spatio_sequential(
            params, np.random.default_rng(100 + seed), labels=planted
        )
        exact = enumerate_spatio_sequential(params, data.observations)
        hits += int((exact.label_posterior.argmax(axis=1) == planted).sum())
        total += planted.size

    assert hits >= 0.75 * total, (hits, total)


@pytest.mark.edge_case
def test_an_instance_past_the_enumeration_limit_is_refused() -> None:
    params = replace(fixture("spatio_sequential", "ci").params, n_positions=20)
    data = simulate_spatio_sequential(params, np.random.default_rng(0))
    with pytest.raises(ValueError, match="paths per class"):
        enumerate_spatio_sequential(params, data.observations)
