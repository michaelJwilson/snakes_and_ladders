"""Regression tests for the discrete-HMM reference instance.

The forward recursion is checked against brute-force enumeration over every
hidden path -- the same independent-oracle pattern
``phylo.likelihood.brute_force`` establishes for pruning, and for the same
reason: a recursion checked only against itself is checked against nothing.
``opt/CLAUDE.md``'s finite-difference derivative check is here too.
"""

from __future__ import annotations

from itertools import permutations, product

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from phylo.opt.hmm import (
    HmmObjective,
    align_states,
    baum_welch,
    forward_log_likelihood,
)
from phylo.sim.hmm import load_hmm_params, simulate_sequences

from tests._fixtures import FIXTURES_DIR
from tests._objective_checks import assert_gradient_matches_finite_differences

FIXTURE = FIXTURES_DIR / "hmm_params.yaml"

# Relative throughout: the log-likelihood is a sum over sequences and sites
# (`DEV.md`, issue #111).
_RTOL_ORACLE = 1e-12
_RTOL_GRADIENT = 1e-6
_FINITE_DIFFERENCE_STEP = 1e-5

# Brute force is O(n_states ** length); 3 ** 7 = 2187 paths per sequence is
# the largest that stays a fast test, and the recursion is length-agnostic,
# so a longer chain would not exercise anything new.
_BRUTE_FORCE_SEQUENCES = 3
_BRUTE_FORCE_LENGTH = 7


def _brute_force_log_likelihood(
    observations: torch.Tensor,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> float:
    """Sum over every hidden path, with no message passing anywhere."""
    n_states, length = log_initial.shape[0], observations.shape[1]
    total = 0.0
    for row in observations:
        weights = [
            log_initial[path[0]]
            + sum(log_transition[path[t], path[t + 1]] for t in range(length - 1))
            + sum(log_emission[path[t], row[t]] for t in range(length))
            for path in product(range(n_states), repeat=length)
        ]
        total += float(torch.logsumexp(torch.stack(weights), dim=0))
    return total


def _log_truth(params: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(  # type: ignore[return-value]
        torch.log(torch.as_tensor(part, dtype=torch.float64))
        for part in (
            params.initial,  # type: ignore[attr-defined]
            params.transition,  # type: ignore[attr-defined]
            params.emission,  # type: ignore[attr-defined]
        )
    )


@pytest.mark.oracle
def test_forward_matches_brute_force_path_enumeration() -> None:
    params = load_hmm_params(FIXTURE)
    observations = torch.as_tensor(
        simulate_sequences(params).observations[
            :_BRUTE_FORCE_SEQUENCES, :_BRUTE_FORCE_LENGTH
        ]
    )
    log_initial, log_transition, log_emission = _log_truth(params)

    expected = _brute_force_log_likelihood(
        observations, log_initial, log_transition, log_emission
    )
    actual = forward_log_likelihood(
        observations, log_initial, log_transition, log_emission
    )
    assert_allclose(float(actual), expected, rtol=_RTOL_ORACLE)


@pytest.mark.mathematical
def test_the_likelihood_is_invariant_to_relabelling_the_hidden_states() -> None:
    # The identifiability caveat, asserted rather than only documented: a
    # recovery test that compared parameters without aligning the
    # permutation would fail on a correct fit.
    params = load_hmm_params(FIXTURE)
    observations = torch.as_tensor(simulate_sequences(params).observations[:20])
    log_initial, log_transition, log_emission = _log_truth(params)
    reference = float(
        forward_log_likelihood(observations, log_initial, log_transition, log_emission)
    )

    for order in permutations(range(params.n_states)):
        index = torch.as_tensor(order)
        permuted = float(
            forward_log_likelihood(
                observations,
                log_initial[index],
                log_transition[index][:, index],
                log_emission[index],
            )
        )
        assert_allclose(permuted, reference, rtol=_RTOL_ORACLE)


@pytest.mark.mathematical
@pytest.mark.parametrize("at_truth", [True, False])
def test_gradient_matches_central_finite_differences(at_truth: bool) -> None:
    params = load_hmm_params(FIXTURE)
    # A short slice: the finite-difference check costs two objective
    # evaluations per parameter, and the recursion it exercises is the same
    # at any length.
    objective = HmmObjective(
        simulate_sequences(params).observations[:40], params.n_states, params.n_symbols
    )
    theta = (
        objective.theta_from_truth(params.initial, params.transition, params.emission)
        if at_truth
        else objective.initial()
    )

    assert_gradient_matches_finite_differences(
        objective, theta, _FINITE_DIFFERENCE_STEP, _RTOL_GRADIENT
    )


@pytest.mark.oracle
def test_theta_round_trips_through_the_constraint_map() -> None:
    params = load_hmm_params(FIXTURE)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )
    constrained = objective.constrain(
        objective.theta_from_truth(params.initial, params.transition, params.emission)
    )
    assert_allclose(
        torch.exp(constrained["log_initial"]).numpy(), params.initial, rtol=1e-13
    )
    assert_allclose(
        torch.exp(constrained["log_transition"]).numpy(), params.transition, rtol=1e-13
    )
    assert_allclose(
        torch.exp(constrained["log_emission"]).numpy(), params.emission, rtol=1e-13
    )


@pytest.mark.structural
def test_theta_has_one_entry_per_free_probability() -> None:
    # 2 free initial + 3 rows x 2 free transition + 3 rows x 3 free emission.
    objective = HmmObjective(np.zeros((2, 5), dtype=np.int64), n_states=3, n_symbols=4)
    assert objective.n_parameters == 17
    assert objective.initial().shape == (17,)


@pytest.mark.structural
def test_the_initial_point_is_uninformative_but_not_symmetric() -> None:
    # Initial and transition start uniform; the emission rows are tilted
    # apart. See the stationary-point test below for why the tilt has to be
    # there.
    objective = HmmObjective(np.zeros((2, 5), dtype=np.int64), n_states=3, n_symbols=4)
    constrained = objective.constrain(objective.initial())
    assert_allclose(
        torch.exp(constrained["log_initial"]).numpy(), np.full(3, 1 / 3), rtol=1e-14
    )
    assert_allclose(
        torch.exp(constrained["log_transition"]).numpy(),
        np.full((3, 3), 1 / 3),
        rtol=1e-14,
    )
    emission = torch.exp(constrained["log_emission"])
    assert_allclose(emission.sum(dim=1).numpy(), np.ones(3), rtol=1e-14)
    # Every state favours a different symbol, so no two rows agree.
    assert emission[0].argmax() != emission[1].argmax()
    assert emission[1].argmax() != emission[2].argmax()


@pytest.mark.mathematical
def test_the_uniform_point_is_a_stationary_point_of_the_likelihood() -> None:
    # This is why `initial` breaks the symmetry, and it is a property of the
    # model rather than a quirk of the optimizer: with every hidden state
    # identical, no infinitesimal change to the initial or transition
    # parameters changes the likelihood at all. An optimizer started there
    # never moves those blocks, and the fit silently returns a model with one
    # effective state. Found by watching a fit do exactly that.
    params = load_hmm_params(FIXTURE)
    objective = HmmObjective(
        simulate_sequences(params).observations[:40], params.n_states, params.n_symbols
    )
    uniform = torch.zeros(objective.n_parameters, dtype=torch.float64)

    point = uniform.detach().clone().requires_grad_(True)
    gradient = torch.autograd.grad(objective(point), point)[0]

    n_free_initial = params.n_states - 1
    n_free_transition = params.n_states * (params.n_states - 1)
    blocked = gradient[: n_free_initial + n_free_transition]
    assert_allclose(blocked.numpy(), np.zeros(blocked.numel()), atol=1e-12)
    # The emission block is not flat, which is why the fit appears to make
    # progress while the states stay exchangeable.
    assert float(gradient[n_free_initial + n_free_transition :].abs().max()) > 1.0


@pytest.mark.mathematical
def test_baum_welch_increases_the_likelihood_monotonically() -> None:
    # An exact property of EM, not an empirical one: each iteration
    # maximizes a lower bound that is tight at the current parameters, so
    # the likelihood cannot decrease. A violation means the M step is wrong.
    params = load_hmm_params(FIXTURE)
    observations = simulate_sequences(params).observations[:60]
    objective = HmmObjective(observations, params.n_states, params.n_symbols)
    start = objective.constrain(objective.initial())

    previous = -float("inf")
    log_initial = start["log_initial"]
    log_transition = start["log_transition"]
    log_emission = start["log_emission"]
    for _ in range(8):
        log_initial, log_transition, log_emission, value = baum_welch(
            observations,
            log_initial,
            log_transition,
            log_emission,
            max_iterations=1,
        )
        assert value >= previous - 1e-9 * abs(value)
        previous = value


@pytest.mark.oracle
def test_align_states_recovers_a_known_permutation() -> None:
    params = load_hmm_params(FIXTURE)
    emission = torch.as_tensor(params.emission)
    order = (2, 0, 1)
    permuted = torch.log(emission[list(order)])
    # align_states returns the order that maps the permuted matrix back.
    recovered = align_states(permuted, emission)
    assert_allclose(
        torch.exp(permuted)[list(recovered)].numpy(), emission.numpy(), atol=1e-15
    )


@pytest.mark.structural
def test_baum_welch_stops_once_the_likelihood_stops_moving() -> None:
    # The convergence test is relative to the log-likelihood's magnitude, as
    # everywhere else. A loose tolerance must stop the iteration early, which
    # shows as a worse optimum than a tight one reaches from the same start.
    params = load_hmm_params(FIXTURE)
    observations = simulate_sequences(params).observations[:60]
    objective = HmmObjective(observations, params.n_states, params.n_symbols)
    start = objective.constrain(objective.initial())
    arguments = (
        observations,
        start["log_initial"],
        start["log_transition"],
        start["log_emission"],
    )

    *_, loose = baum_welch(*arguments, tolerance=1e-1)
    *_, tight = baum_welch(*arguments, tolerance=1e-14)
    assert loose < tight
