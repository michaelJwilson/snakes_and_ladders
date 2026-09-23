"""Relabelling a mixture's draws: six algorithms and the ordering baseline (issue #964).

Referees: each method's assignment step against brute force over ``K!`` on its
own objective; STEPHENS' divergence non-increasing and ECR-ITERATIVE-1's
agreement non-decreasing over passes, as the alternations guarantee; and
draws put in random labellings by the random permutation sampler's move,
whose permutations are known, put back in one labelling by every method.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.opt.mixture import responsibilities
from snakes_and_ladders.sample.relabel import (
    MAX_SJW_COMPONENTS,
    RelabelMethod,
    aic,
    all_permutations,
    allocation_draws,
    ecr,
    ecr_iterative_1,
    ecr_iterative_2,
    inverse,
    permute_allocations,
    permute_parameters,
    permute_probabilities,
    planted_recovery,
    pra,
    random_permutations,
    relabel,
    sjw,
    stephens,
)

K = 4
N_DRAWS = 60
N_OBS = 80
MEANS = np.array([-6.0, -2.0, 2.0, 6.0])
WEIGHTS = np.array([0.1, 0.2, 0.3, 0.4])


def _draws(seed: int) -> dict[str, np.ndarray]:
    """Draws of a separated 4-component Gaussian mixture, and their planted relabellings.

    Each draw's means and weights are the truth plus noise, its
    probabilities the responsibilities there, its allocations drawn from
    them; then every draw is put in a uniformly random labelling.
    """
    rng = np.random.default_rng([964, seed])
    labels = rng.choice(K, size=N_OBS, p=WEIGHTS)
    observations = rng.normal(MEANS[labels], 1.0)
    means = MEANS + rng.normal(0.0, 0.3, (N_DRAWS, K))
    weights = rng.dirichlet(200.0 * WEIGHTS, N_DRAWS)
    families = [GaussianEmission(row, np.ones(K), 1e-12) for row in means]
    drawn = allocation_draws(observations, np.log(weights), families, rng)
    planted = random_permutations(N_DRAWS, K, rng)
    parameters = np.stack([means, weights], axis=2)
    return {
        "observations": observations,
        "labels": labels,
        "planted": planted,
        "parameters": permute_parameters(parameters, planted),
        "probabilities": permute_probabilities(drawn.probabilities, planted),
        "allocations": permute_allocations(drawn.allocations, planted),
    }


@pytest.mark.critical
@pytest.mark.analytic
def test_the_permutation_helpers_are_one_convention() -> None:
    rng = np.random.default_rng(1)
    orders = random_permutations(50, 5, rng)
    assert np.array_equal(
        permute_parameters(inverse(orders), orders), np.tile(np.arange(5), (50, 1))
    )
    values = rng.normal(size=(50, 5))
    back = permute_parameters(permute_parameters(values, orders), inverse(orders))
    assert np.array_equal(back, values)
    # A draw allocating to j sees component j; relabelled, the same
    # observation is allocated to the label whose parameter is j's.
    allocations = rng.integers(0, 5, (50, 30))
    relabelled = permute_allocations(allocations, orders)
    moved = permute_parameters(values, orders)
    rows = np.arange(50)[:, None]
    assert np.array_equal(moved[rows, relabelled], values[rows, allocations])
    probabilities = rng.dirichlet(np.ones(5), (50, 30))
    moved_p = permute_probabilities(probabilities, orders)
    assert np.array_equal(
        moved_p[rows, np.arange(30)[None, :], relabelled],
        probabilities[rows, np.arange(30)[None, :], allocations],
    )


def _brute(score: np.ndarray) -> float:
    """The largest ``sum_k score[k, order[k]]`` over every order."""
    return max(
        float(score[np.arange(len(order)), list(order)].sum())
        for order in itertools.permutations(range(score.shape[0]))
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_ecr_and_pra_reach_the_enumerated_optimum() -> None:
    data = _draws(0)
    labels = data["allocations"]
    pivot = data["allocations"][0]
    result = ecr(labels, pivot, K)
    for t in range(N_DRAWS):
        agreement = np.zeros((K, K))
        np.add.at(agreement, (pivot, labels[t]), 1.0)
        achieved = agreement[np.arange(K), result.permutations[t]].sum()
        assert achieved == _brute(agreement)
    parameters = data["parameters"]
    centre = parameters[0]
    result = pra(parameters, centre)
    for t in range(N_DRAWS):
        score = centre @ parameters[t].T
        achieved = score[np.arange(K), result.permutations[t]].sum()
        assert abs(achieved - _brute(score)) <= 1e-12 * abs(achieved)


@pytest.mark.critical
@pytest.mark.oracle
def test_a_stephens_pass_reaches_the_enumerated_minimum() -> None:
    # One pass from the identity: q is the plain mean, and each draw's
    # permutation minimizes its own divergence to q over all 24 orders.
    data = _draws(1)
    p = np.clip(data["probabilities"], 1e-6, 1 - 1e-6)
    p /= p.sum(axis=2, keepdims=True)
    q = p.mean(axis=0)
    result = stephens(data["probabilities"], max_iterations=1)
    for t in range(N_DRAWS):
        divergence = {
            order: float(
                (
                    p[t][:, list(order)] * (np.log(p[t][:, list(order)]) - np.log(q))
                ).sum()
            )
            for order in itertools.permutations(range(K))
        }
        achieved = divergence[tuple(int(k) for k in result.permutations[t])]
        assert achieved <= min(divergence.values()) + 1e-9


@pytest.mark.critical
@pytest.mark.analytic
def test_the_alternations_are_monotone() -> None:
    # From the identity on randomly permuted draws: STEPHENS takes 3 passes
    # (5,421 to 48.7) and ECR-ITERATIVE-1 takes 2 (3,253 to 4,674 agreeing).
    data = _draws(2)
    result = stephens(data["probabilities"])
    assert result.converged
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(result.objective))
    iterative = ecr_iterative_1(data["allocations"], K)
    assert iterative.converged
    assert all(b >= a for a, b in itertools.pairwise(iterative.objective))


@pytest.mark.end2end
@pytest.mark.parametrize("method", list(RelabelMethod))
def test_every_method_undoes_the_planted_permutations(method: RelabelMethod) -> None:
    # Separated components and a noise of 0.3 against spacings of 4: every
    # draw can be put back, so every method must recover all of them.
    data = _draws(3)
    pivot_draw = 0
    scale = np.array([1.0, 10.0])

    def scores(estimate: np.ndarray) -> np.ndarray:
        family = GaussianEmission(estimate[:, 0], np.ones(K), 1e-12)
        weight = np.log(np.clip(estimate[:, 1], 1e-12, None))
        return (
            family.log_density(torch.as_tensor(data["observations"])).numpy() + weight
        )

    result = relabel(
        method,
        probabilities=data["probabilities"],
        allocations=data["allocations"],
        parameters=data["parameters"] * scale,
        pivot=(
            data["allocations"][pivot_draw]
            if method is RelabelMethod.ECR
            else (data["parameters"] * scale)[pivot_draw]
        ),
        scores=scores,
    )
    assert result.method is method
    assert planted_recovery(result.permutations, data["planted"]) == 1.0
    # And the relabelled means lie at the generating ones, in one order.
    means = permute_parameters(data["parameters"], result.permutations)[:, :, 0].mean(
        axis=0
    )
    np.testing.assert_allclose(np.sort(means), MEANS, atol=0.2)


@pytest.mark.oracle
def test_the_default_is_stephens_and_sjw_weights_are_a_distribution() -> None:
    data = _draws(4)
    default = relabel(probabilities=data["probabilities"])
    assert default.method is RelabelMethod.STEPHENS
    assert np.array_equal(
        default.permutations, stephens(data["probabilities"]).permutations
    )

    def scores(estimate: np.ndarray) -> np.ndarray:
        family = GaussianEmission(estimate[:, 0], np.ones(K), 1e-12)
        return family.log_density(
            torch.as_tensor(data["observations"])
        ).numpy() + np.log(np.clip(estimate[:, 1], 1e-12, None))

    result = sjw(data["parameters"], data["allocations"], scores)
    assert result.weights is not None
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, rtol=1e-12)
    assert np.array_equal(
        result.permutations, all_permutations(K)[result.weights.argmax(axis=1)]
    )


@pytest.mark.oracle
def test_allocation_probabilities_are_the_responsibilities() -> None:
    rng = np.random.default_rng(5)
    observations = rng.normal(size=40)
    family = GaussianEmission([-1.0, 0.5, 2.0], [1.0, 0.7, 1.2], 1e-12)
    log_weights = np.log(np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]))
    drawn = allocation_draws(observations, log_weights, [family, family], rng)
    for t in range(2):
        expected = responsibilities(
            torch.as_tensor(observations), torch.as_tensor(log_weights[t]), family
        ).numpy()
        np.testing.assert_allclose(drawn.probabilities[t], expected, rtol=1e-12)
    assert drawn.allocations.shape == (2, 40)


@pytest.mark.smoke
def test_what_a_method_cannot_run_on_is_refused() -> None:
    with pytest.raises(ValueError, match="needs probabilities"):
        relabel(allocations=np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="needs a pivot"):
        relabel(RelabelMethod.PRA, parameters=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="the limit is K"):
        sjw(
            np.zeros((2, MAX_SJW_COMPONENTS + 1)),
            np.zeros((2, 3), dtype=np.int64),
            lambda _: np.zeros((3, MAX_SJW_COMPONENTS + 1)),
        )
    with pytest.raises(ValueError, match="outside"):
        ecr(np.full((2, 3), 5), np.zeros(3, dtype=np.int64), 3)
    ordered = aic(np.array([[3.0, 1.0, 2.0]]))
    assert ordered.permutations.tolist() == [[1, 2, 0]]
    stalled = ecr_iterative_2(np.zeros((2, 3), dtype=np.int64), np.full((2, 3, 2), 0.5))
    assert stalled.converged
