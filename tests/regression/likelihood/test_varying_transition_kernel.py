"""A chain whose transition kernel is a function of position (issue #653).

``forward_backward`` and ``forward_log_likelihood_from_density`` now take
either one ``(K, K)`` matrix for the whole chain or one per step. Three claims
are pinned here: the constant form is reproduced **bitwise** by its per-step
repeat, so admitting the second shape moved no committed number; the varying
form agrees with a path enumeration that shares no recursion with it; and a
planted two-regime kernel is recovered from the pairwise posteriors, which no
single matrix can express.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from snakes_and_ladders.likelihood.forward_backward import (
    forward_backward,
    sample_path,
    step_kernels,
)
from snakes_and_ladders.opt.hmm import forward_log_likelihood_from_density


def _chain(
    n_states: int, length: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Emission scores, log initial and one log transition, all from ``seed``."""
    rng = np.random.default_rng(seed)
    log_density = np.log(rng.dirichlet(np.ones(n_states), size=length))
    log_initial = np.log(rng.dirichlet(np.ones(n_states)))
    log_transition = np.log(rng.dirichlet(np.ones(n_states), size=n_states))
    return log_density, log_initial, log_transition


def _enumerate(
    log_density: np.ndarray, log_initial: np.ndarray, kernels: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evidence, posterior and pairwise by summing over all ``K ** T`` paths.

    The independent answer: no forward message, no backward message, every
    path's joint written out from the definition.
    """
    length, n_states = log_density.shape
    paths = list(itertools.product(range(n_states), repeat=length))
    joint = np.empty(len(paths))
    for index, path in enumerate(paths):
        total = log_initial[path[0]] + log_density[0, path[0]]
        for t in range(1, length):
            total += kernels[t - 1][path[t - 1], path[t]] + log_density[t, path[t]]
        joint[index] = total
    shift = joint.max()
    weights = np.exp(joint - shift)
    evidence = float(shift + np.log(weights.sum()))
    weights /= weights.sum()

    posterior = np.zeros((length, n_states))
    pairwise = np.zeros((length - 1, n_states, n_states))
    for weight, path in zip(weights, paths, strict=True):
        for t, state in enumerate(path):
            posterior[t, state] += weight
        for t in range(1, length):
            pairwise[t - 1, path[t - 1], path[t]] += weight
    return evidence, posterior, pairwise


@pytest.mark.structural
@pytest.mark.critical
@pytest.mark.parametrize(("n_states", "length", "seed"), [(2, 9, 1), (4, 12, 2)])
def test_a_repeated_kernel_reproduces_the_single_matrix_bitwise(
    n_states: int, length: int, seed: int
) -> None:
    """The second shape must cost the first nothing, and it costs it nothing.

    ``step_kernels`` hands the recursion a stride-zero view of the same
    ``(K, K)`` block, so the constant case sums the same terms in the same
    order. Equality here is exact, not a tolerance: were it a tolerance, the
    26 call sites that pass a matrix would have been re-refereed.
    """
    log_density, log_initial, log_transition = _chain(n_states, length, seed)
    repeated = np.repeat(log_transition[None], length - 1, axis=0)

    one = forward_backward(log_density, log_initial, log_transition)
    many = forward_backward(log_density, log_initial, repeated)

    assert one.log_evidence == many.log_evidence
    assert np.array_equal(one.posterior, many.posterior)
    assert np.array_equal(one.pairwise, many.pairwise)
    assert np.array_equal(
        sample_path(log_density, log_initial, log_transition, np.random.default_rng(5)),
        sample_path(log_density, log_initial, repeated, np.random.default_rng(5)),
    )


@pytest.mark.structural
@pytest.mark.critical
def test_the_torch_recursion_reproduces_its_matrix_form_bitwise() -> None:
    """The same claim for the differentiable path, which batches over sequences."""
    n_states, length = 3, 10
    generator = torch.Generator().manual_seed(11)
    log_density = torch.log(
        torch.rand(4, length, n_states, dtype=torch.float64, generator=generator)
    )
    log_initial = torch.log(
        torch.rand(n_states, dtype=torch.float64, generator=generator)
    )
    log_transition = torch.log(
        torch.rand(n_states, n_states, dtype=torch.float64, generator=generator)
    )
    repeated = log_transition.expand(length - 1, n_states, n_states).clone()

    one = forward_log_likelihood_from_density(log_density, log_initial, log_transition)
    many = forward_log_likelihood_from_density(log_density, log_initial, repeated)

    assert one.item() == many.item()


@pytest.mark.structural
def test_the_constant_form_is_a_view_and_not_a_copy() -> None:
    """The ``T``-fold memory is paid by the caller who asks for it, by nobody else.

    At ``T = 4096``, ``K = 5`` a copy would be 800 KiB against the matrix's
    200 B. The stride-zero view is neither, and the hoisted ``constant`` means
    the constant recursion does not even index it.
    """
    log_transition = np.zeros((5, 5))
    kernels, constant = step_kernels(log_transition, 4096, 5)

    assert kernels.shape == (4095, 5, 5)
    assert kernels.base is log_transition
    assert kernels.strides[0] == 0
    assert constant is log_transition

    varying = np.zeros((4095, 5, 5))
    kernels, constant = step_kernels(varying, 4096, 5)
    assert kernels is varying
    assert constant is None


@pytest.mark.oracle
@pytest.mark.parametrize(("n_states", "length", "seed"), [(2, 8, 3), (3, 6, 4)])
def test_a_varying_kernel_is_the_path_enumeration(
    n_states: int, length: int, seed: int
) -> None:
    """The varying recursion against every path written out from the definition."""
    rng = np.random.default_rng(seed)
    log_density = np.log(rng.dirichlet(np.ones(n_states), size=length))
    log_initial = np.log(rng.dirichlet(np.ones(n_states)))
    kernels = np.log(
        rng.dirichlet(np.ones(n_states), size=(length - 1, n_states))
    ).reshape(length - 1, n_states, n_states)

    run = forward_backward(log_density, log_initial, kernels)
    evidence, posterior, pairwise = _enumerate(log_density, log_initial, kernels)

    assert abs(run.log_evidence - evidence) < 1e-12 * abs(evidence)
    np.testing.assert_allclose(run.posterior, posterior, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(run.pairwise, pairwise, rtol=1e-11, atol=1e-13)


@pytest.mark.simulated_truth
def test_a_planted_two_regime_kernel_is_recovered_per_regime() -> None:
    """Recover a kernel that changes halfway, which no single matrix expresses.

    The chain is sticky over its first half and mixing over its second. The
    per-step pairwise posteriors, summed within each half and row-normalized,
    are the M-step estimate of that half's kernel; each recovers its plant.
    The pooled estimate --- the single matrix a constant kernel would fit ---
    recovers neither, which is the statement that the shape was missing.
    """
    n_states, half = 3, 1500
    length = 2 * half
    sticky = np.full((n_states, n_states), 0.02)
    np.fill_diagonal(sticky, 0.96)
    mixing = np.full((n_states, n_states), 0.34)
    np.fill_diagonal(mixing, 0.32)
    emission = np.full((n_states, n_states), 0.02)
    np.fill_diagonal(emission, 0.96)

    rng = np.random.default_rng(17)
    initial = np.full(n_states, 1.0 / n_states)
    states = np.empty(length, dtype=np.int64)
    states[0] = rng.choice(n_states, p=initial)
    for t in range(1, length):
        kernel = sticky if t <= half else mixing
        states[t] = rng.choice(n_states, p=kernel[states[t - 1]])
    observations = np.array([rng.choice(n_states, p=emission[s]) for s in states])

    kernels = np.log(
        np.concatenate(
            [
                np.repeat(sticky[None], half, axis=0),
                np.repeat(mixing[None], half - 1, axis=0),
            ]
        )
    )
    run = forward_backward(
        np.log(emission)[:, observations].T, np.log(initial), kernels
    )

    def _normalized(counts: np.ndarray) -> np.ndarray:
        rows: np.ndarray = counts / counts.sum(axis=1, keepdims=True)
        return rows

    first = _normalized(run.pairwise[:half].sum(axis=0))
    second = _normalized(run.pairwise[half:].sum(axis=0))
    pooled = _normalized(run.pairwise.sum(axis=0))

    np.testing.assert_allclose(first, sticky, atol=0.03)
    np.testing.assert_allclose(second, mixing, atol=0.05)
    assert np.abs(pooled - sticky).max() > 0.2
    assert np.abs(pooled - mixing).max() > 0.2


@pytest.mark.edge_case
def test_a_kernel_of_the_wrong_length_is_refused() -> None:
    """A step axis that is not ``T - 1`` is a model error, not a broadcast."""
    log_density, log_initial, log_transition = _chain(3, 7, 9)

    with pytest.raises(ValueError, match="is neither"):
        forward_backward(log_density, log_initial, np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="is neither"):
        forward_backward(log_density, log_initial, np.zeros((6, 2, 2)))
    with pytest.raises(ValueError, match="log_initial"):
        forward_backward(log_density, np.zeros(2), log_transition)

    density = torch.zeros(1, 7, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="is neither"):
        forward_log_likelihood_from_density(
            density, torch.zeros(3, dtype=torch.float64), torch.zeros(3, 3, 3)
        )
