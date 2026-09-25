"""A kernel that varies by position, in a fit rather than only an evaluator (#658).

#656 gave the evaluators a `(T - 1, K, K)` kernel; `baum_welch_family` and
`SpatioSequentialParams` took only one matrix or a scalar. A per-step kernel
is conditioned on, not fitted: it has `(T - 1) * K * (K - 1)` free values
against `T - 1` transitions per sequence. Told one, the fit holds it and fits
the rest, as with a covariate.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from sal.emissions import CategoricalEmission
from sal.opt.em import EM
from sal.opt.hmm import baum_welch_family
from sal.sim.spatio_sequential import canonical_spatio_sequential

_EMISSION = np.array([[0.85, 0.10, 0.05], [0.05, 0.10, 0.85]])
_INITIAL = np.array([0.5, 0.5])


def _chain(stay: np.ndarray, length: int, seed: int) -> np.ndarray:
    """One chain whose stickiness is ``stay[t]`` at step ``t``."""
    rng = np.random.default_rng(seed)
    states = np.empty(length, dtype=np.int64)
    states[0] = rng.integers(0, 2)
    for t in range(1, length):
        states[t] = states[t - 1] if rng.random() < stay[t - 1] else 1 - states[t - 1]
    return np.array([[rng.choice(3, p=_EMISSION[s]) for s in states]])


def _kernels(stay: np.ndarray) -> torch.Tensor:
    matrices = np.stack([np.array([[s, 1 - s], [1 - s, s]]) for s in stay])
    return torch.log(torch.as_tensor(matrices, dtype=torch.float64))


def _fit(
    observations: np.ndarray, log_transition: torch.Tensor
) -> tuple[torch.Tensor, float]:
    result = baum_welch_family(
        observations,
        torch.log(torch.as_tensor(_INITIAL)),
        log_transition,
        CategoricalEmission(np.array([[0.6, 0.2, 0.2], [0.2, 0.2, 0.6]])),
        config=replace(EM, max_iterations=100),
    )
    return result.log_transition, result.log_likelihood


@pytest.mark.smoke
@pytest.mark.critical
def test_a_repeated_kernel_fits_what_the_single_matrix_fits() -> None:
    """The constant case is unchanged where the kernel is also fitted.

    One repeated matrix is that matrix: the likelihoods agree, the transitions not.
    """
    stay = np.full(59, 0.9)
    observations = _chain(stay, 60, seed=3)

    _, single = _fit(observations, torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9]])))
    _, repeated = _fit(observations, _kernels(stay))

    assert single >= repeated - 1e-9


@pytest.mark.smoke
@pytest.mark.critical
def test_a_per_step_kernel_is_held_and_a_single_matrix_is_fitted() -> None:
    """The rule, asserted both ways so neither half can quietly change."""
    stay = np.linspace(0.95, 0.55, 59)
    observations = _chain(stay, 60, seed=5)
    given = _kernels(stay)

    held, _ = _fit(observations, given)
    start = torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64))
    fitted, _ = _fit(observations, start)

    assert torch.equal(held, given)
    assert not torch.allclose(fitted, start)


@pytest.mark.smoke
@pytest.mark.critical
def test_a_held_kernel_comes_back_bitwise_and_unnormalized() -> None:
    """ "Held" means the caller's values, not a copy that agrees to a tolerance.

    Rows sum to 1.4: renormalizing fails the sum, an `exp`/`log` trip `torch.equal`.
    """
    length = 40
    stay = np.full(length - 1, 0.9)
    observations = _chain(stay, length, seed=7)
    # Rows summing to 1.4, and deliberately not a probability kernel. Nothing
    # in the recursion requires one --- it is a log-domain weight --- so a fit
    # that holds this is the only one that returns it.
    given = torch.log(
        torch.as_tensor(
            np.stack([np.array([[0.9, 0.5], [0.5, 0.9]])] * (length - 1)),
            dtype=torch.float64,
        )
    )

    held, _ = _fit(observations, given)

    assert torch.equal(held, given)
    np.testing.assert_array_equal(
        torch.exp(held).sum(dim=2).numpy(), np.full((length - 1, 2), 1.4)
    )


@pytest.mark.end2end
def test_the_held_kernel_explains_the_data_better_than_a_constant_one() -> None:
    """And it is worth holding: the varying truth beats the best single matrix.

    0.95 to 0.55: -340.422 vs -345.477; margin +1.0, +5.1, +7.4 at 300, 400, 500.
    """
    stay = np.linspace(0.95, 0.55, 399)
    observations = _chain(stay, 400, seed=11)

    _, told = _fit(observations, _kernels(stay))
    _, fitted = _fit(observations, torch.log(torch.tensor([[0.7, 0.3], [0.3, 0.7]])))

    assert told > fitted + 1.0


@pytest.mark.smoke
def test_a_kernel_of_the_wrong_length_is_refused_by_the_fit() -> None:
    """The refusal `forward_backward` makes, made where a fit can hit it."""
    observations = _chain(np.full(9, 0.9), 10, seed=1)

    with pytest.raises(ValueError, match="is neither"):
        _fit(observations, _kernels(np.full(4, 0.9)))


@pytest.mark.smoke
def test_the_spatial_params_take_a_matrix_for_the_chain_or_one_per_transition() -> None:
    """`(K, K)` and `(S - 1, K, K)` are the forms; the scalar rate stays the default.

    A kernel assembled over a product space is no circulant, even when constant.
    """
    from dataclasses import replace

    params = canonical_spatio_sequential()
    steps, k = params.n_positions - 1, params.n_states
    assert params.transition.shape == (k, k)

    base = np.array([[0.8, 0.2], [0.3, 0.7]])
    switch = np.array([[0.9, 0.1], [0.1, 0.9]])
    constant = switch @ base
    # Genuinely not a circulant: reversing one row does not give the other.
    assert not np.allclose(constant[0], constant[1][::-1])

    one = replace(params, self_transition=constant)
    assert one.transition.shape == (k, k)
    np.testing.assert_array_equal(one.transition, constant)

    stack = np.stack(
        [
            np.array([[1 - s, s], [s, 1 - s]]) @ base
            for s in np.linspace(0.05, 0.4, steps)
        ]
    )
    many = replace(params, self_transition=stack)
    assert many.transition.shape == (steps, k, k)
    np.testing.assert_array_equal(many.transition, stack)


@pytest.mark.smoke
def test_a_transition_that_is_not_row_stochastic_is_refused() -> None:
    """Either matrix form is a kernel, so its rows are distributions."""
    from dataclasses import replace

    params = canonical_spatio_sequential()
    steps, k = params.n_positions - 1, params.n_states

    with pytest.raises(ValueError, match="must be a distribution"):
        replace(params, self_transition=np.full((k, k), 0.9))
    with pytest.raises(ValueError, match="must be a distribution"):
        replace(params, self_transition=np.full((steps, k, k), 0.9))
    with pytest.raises(ValueError, match="one matrix per transition"):
        replace(params, self_transition=np.full((steps + 3, k, k), 0.5))
