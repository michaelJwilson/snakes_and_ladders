"""A kernel that varies by position, in a fit rather than only an evaluator (#658).

#656 gave `forward_backward`, `sample_path` and
`forward_log_likelihood_from_density` a `(T - 1, K, K)` kernel. Nothing that
*fits* took one: `baum_welch_family`'s recursion took a single `(K, K)` and its
M step returned one, and `SpatioSequentialParams` carried a scalar
`self_transition` through `circulant_transition`, so the spatial model could
not express a varying kernel at all.

**A per-step kernel is conditioned on, not fitted**, and that is the claim
these pin. It carries `(T - 1) * K * (K - 1)` free values against `T - 1`
transitions per sequence, so an M step that re-estimated it would hand back the
posterior it was given. Told one, the fit holds it and fits everything else ---
the same standing a covariate has.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.emissions import CategoricalEmission
from snakes_and_ladders.opt.hmm import baum_welch_family
from snakes_and_ladders.sim.spatio_sequential import canonical_spatio_sequential

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
        max_iterations=100,
    )
    return result.log_transition, result.log_likelihood


@pytest.mark.structural
@pytest.mark.critical
def test_a_repeated_kernel_fits_what_the_single_matrix_fits() -> None:
    """The constant case is unchanged where the kernel is also fitted.

    A `(T - 1, K, K)` of one repeated matrix is the same model as that matrix,
    so the *likelihood* must agree. The fitted transitions cannot: told a
    per-step kernel the fit holds it, told one matrix it fits it, and that
    difference is the point of the test below.
    """
    stay = np.full(59, 0.9)
    observations = _chain(stay, 60, seed=3)

    _, single = _fit(observations, torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9]])))
    _, repeated = _fit(observations, _kernels(stay))

    assert single >= repeated - 1e-9


@pytest.mark.structural
@pytest.mark.critical
def test_a_per_step_kernel_is_held_and_a_single_matrix_is_fitted() -> None:
    """The rule, asserted both ways so neither half can quietly change.

    Told a per-step kernel the fit returns it unchanged; told a single matrix
    it returns something else, because it fitted it.
    """
    stay = np.linspace(0.95, 0.55, 59)
    observations = _chain(stay, 60, seed=5)
    given = _kernels(stay)

    held, _ = _fit(observations, given)
    start = torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64))
    fitted, _ = _fit(observations, start)

    assert torch.equal(held, given)
    assert not torch.allclose(fitted, start)


@pytest.mark.structural
@pytest.mark.critical
def test_a_held_kernel_comes_back_bitwise_and_unnormalized() -> None:
    """ "Held" means the caller's values, not a copy that agrees to a tolerance.

    `torch.equal` against a normalized kernel cannot tell holding from a
    round trip: an implementation that exponentiated, renormalized and took
    the log of a kernel whose rows already sum to one would return values that
    pass it. So the kernel handed in here **does not** normalize --- its rows
    sum to 1.4 --- and it must come back with that, to the bit.

    A fit that renormalizes returns rows summing to one and fails on the sum;
    a fit that round-trips through `exp` and `log` returns 1.4 to within a few
    ulps and fails on `torch.equal`. Both are the failure this names: a fit
    that holds a kernel and returns a drifted copy passes "it held it" and is
    still wrong, because the caller's next iteration is over different
    numbers than the one it asked for.
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


@pytest.mark.simulated_truth
def test_the_held_kernel_explains_the_data_better_than_a_constant_one() -> None:
    """And it is worth holding: the varying truth beats the best single matrix.

    The chain's stickiness falls from 0.95 to 0.55 across 400 positions. Told
    that, the fit reaches -340.422 against the -345.477 of one free to fit any
    single matrix --- the statement that the shape carries information a
    constant kernel cannot. The margin grows with the chain (+1.0 at 300
    positions, +5.1 at 400, +7.4 at 500); 400 is where it is comfortably
    outside the noise and the pair of fits is still inside the per-test
    duration cap (`DEV.md`).
    """
    stay = np.linspace(0.95, 0.55, 399)
    observations = _chain(stay, 400, seed=11)

    _, told = _fit(observations, _kernels(stay))
    _, fitted = _fit(observations, torch.log(torch.tensor([[0.7, 0.3], [0.3, 0.7]])))

    assert told > fitted + 1.0


@pytest.mark.edge_case
def test_a_kernel_of_the_wrong_length_is_refused_by_the_fit() -> None:
    """The refusal `forward_backward` makes, made where a fit can hit it."""
    observations = _chain(np.full(9, 0.9), 10, seed=1)

    with pytest.raises(ValueError, match="is neither"):
        _fit(observations, _kernels(np.full(4, 0.9)))


@pytest.mark.structural
def test_the_spatial_params_take_a_matrix_for_the_chain_or_one_per_transition() -> None:
    """`(K, K)` and `(S - 1, K, K)` are the forms; the scalar rate stays the default.

    The matrix forms are read as the kernel itself. A kernel assembled from
    parts --- a base over one latent and a kernel over another, combined into
    the product space --- is not a circulant at any rate, so the scalar could
    not express one **even when it does not vary along the chain**. That
    constant case is what `(K, K)` is for, and it is the one a caller reaches
    first.

    How a caller assembled a kernel is the caller's; this carries the result
    and does not reconstruct it.
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


@pytest.mark.edge_case
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
