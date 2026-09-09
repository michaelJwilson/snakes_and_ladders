"""``burn``'s taped gradient, against the taped gradient it would replace.

Route A of issue #449. ``pruning_torch`` is the oracle and stays; what is
checked is that a second reverse-mode tape, in Rust over
``Autodiff<NdArray<f64>>``, computes the same derivative. The tolerances and
the central-difference step are the ones
``tests/regression/likelihood/test_pruning_analytic.py`` derives, read from
there rather than re-derived: the sweep behind them measured all three routes
at once and found them equal to four significant figures.

The `f64` question the route was adopted on is settled by
``test_the_gradient_is_float64_throughout`` below and by the Rust unit tests
in ``src/pruning_burn.rs``: a tape that had narrowed to `f32` could not agree
with the taped `float64` gradient to 1e-13.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood import pruning_burn, pruning_torch
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.patterns import compress
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import EIGHT_TAXA, SMALL_SITES, load_fixture
from tests.regression.likelihood.test_pruning_analytic import (
    CENTRAL_DIFFERENCE_RTOL,
    CENTRAL_DIFFERENCE_STEP,
)


def _case(name: str, n_sites: int):  # type: ignore[no-untyped-def]
    params = load_fixture(name)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=n_sites,
    )
    return (
        params.tau,
        params.k,
        params.pi,
        dataset.alignment,
        pruning_torch.branch_lengths_from_tree(params.tau),
    )


def _gradient(evaluate, case, **kwargs) -> tuple[float, np.ndarray]:  # type: ignore[no-untyped-def]
    tau, k, pi, alignment, lengths = case
    branch_lengths = lengths.clone().requires_grad_(True)
    value = evaluate(tau, k, pi, alignment, branch_lengths, **kwargs)
    value.backward()
    assert branch_lengths.grad is not None
    return float(value.detach()), branch_lengths.grad.numpy().copy()


@pytest.mark.oracle
@pytest.mark.parametrize(("fixture_name", "n_sites"), [(SMALL_SITES, 2000), (EIGHT_TAXA, 2000)])
def test_value_is_the_taped_path_s(fixture_name: str, n_sites: int) -> None:
    """The forward value is ``pruning_torch``'s, which is the oracle."""
    tau, k, pi, alignment, lengths = _case(fixture_name, n_sites)
    expected = float(pruning_torch.log_likelihood(tau, k, pi, alignment, lengths))
    actual = float(pruning_burn.log_likelihood(tau, k, pi, alignment, lengths))
    assert_allclose(actual, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.oracle
@pytest.mark.parametrize(("fixture_name", "n_sites"), [(SMALL_SITES, 2000), (EIGHT_TAXA, 2000)])
def test_gradient_matches_the_taped_gradient(fixture_name: str, n_sites: int) -> None:
    """A second tape reaches the first tape's derivative, in ``float64``."""
    case = _case(fixture_name, n_sites)
    _, expected = _gradient(pruning_torch.log_likelihood, case)
    _, actual = _gradient(pruning_burn.log_likelihood, case)
    assert_allclose(actual, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.oracle
def test_gradient_matches_central_differences() -> None:
    """Against a quotient of the forward pass alone, at the derived tolerance."""
    tau, k, pi, alignment, lengths = _case(EIGHT_TAXA, 2000)
    _, gradient = _gradient(
        pruning_burn.log_likelihood, (tau, k, pi, alignment, lengths)
    )

    def value(at: torch.Tensor) -> float:
        return float(pruning_torch.log_likelihood(tau, k, pi, alignment, at))

    quotient = np.empty_like(gradient)
    for position in range(len(lengths)):
        step = torch.zeros_like(lengths)
        step[position] = CENTRAL_DIFFERENCE_STEP
        quotient[position] = (value(lengths + step) - value(lengths - step)) / (
            2.0 * CENTRAL_DIFFERENCE_STEP
        )
    assert_allclose(gradient, quotient, rtol=CENTRAL_DIFFERENCE_RTOL)


@pytest.mark.mathematical
def test_gradcheck_in_float64() -> None:
    """``torch.autograd.gradcheck`` over the Rust tape's ``backward``."""
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 200)
    assert torch.autograd.gradcheck(
        lambda at: pruning_burn.log_likelihood(tau, k, pi, alignment, at),
        (lengths.clone().requires_grad_(True),),
        eps=1e-6,
        atol=1e-7,
        rtol=1e-5,
    )


@pytest.mark.mathematical
def test_the_gradient_is_float64_throughout() -> None:
    """The axis the dependency was adopted on: a narrowed tape cannot pass this.

    `float32` carries about 7 decimal digits, so a tape that narrowed anywhere
    would disagree with the `float64` taped gradient at 1e-7 and not at the
    1e-11 this asserts.
    """
    case = _case(EIGHT_TAXA, 20_000)
    _, expected = _gradient(pruning_torch.log_likelihood, case)
    _, actual = _gradient(pruning_burn.log_likelihood, case)
    assert_allclose(actual, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.mathematical
def test_weighted_patterns_give_the_uncompressed_gradient() -> None:
    """The compressed alignment with its weights is the full alignment's gradient."""
    tau, k, pi, alignment, lengths = _case(EIGHT_TAXA, 2000)
    compressed = compress(alignment)
    case = (tau, k, pi, alignment, lengths)
    _, full = _gradient(pruning_burn.log_likelihood, case)
    _, weighted = _gradient(
        pruning_burn.log_likelihood,
        (tau, k, pi, compressed.alignment, lengths),
        weights=compressed.weights,
    )
    assert_allclose(weighted, full, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.edge_case
def test_a_general_rate_matrix_is_refused() -> None:
    """`burn` has no matrix exponential, and the route says so rather than ignoring."""
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 100)
    with pytest.raises(ValueError, match="Jukes-Cantor"):
        pruning_burn.log_likelihood(
            tau, k, pi, alignment, lengths, rate_matrix=torch.eye(k, dtype=torch.float64)
        )


@pytest.mark.edge_case
def test_branch_lengths_of_the_wrong_length_are_refused() -> None:
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 100)
    with pytest.raises(ValueError, match="branch_order"):
        pruning_burn.log_likelihood(tau, k, pi, alignment, lengths[:-1])


@pytest.mark.edge_case
def test_a_ragged_alignment_is_refused() -> None:
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 100)
    ragged = dict(alignment)
    first = next(iter(ragged))
    ragged[first] = ragged[first][:-1]
    with pytest.raises(ValueError, match="ragged"):
        pruning_burn.log_likelihood(tau, k, pi, ragged, lengths)
