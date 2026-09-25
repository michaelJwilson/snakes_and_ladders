"""The Baum-Welch E step in the compiled ragged kernel, and a held transition (issue #933, R5).

`baum_welch_family` walks each sequence in `src/ragged.rs` by default
instead of the padded torch recursion, which `backend=Backend.PYTHON` keeps as
the oracle; `fit_transition=False` holds a single ``(m, m)`` transition as a
per-step kernel is always held. Referees: the compiled route is the torch
route at the repository's float64 tolerance on ragged batches, a per-step
kernel keeps the torch recursion under either backend bitwise, and a held
transition is the same fit as that transition expanded per step, bitwise
under the torch recursion and at the tolerance under the compiled one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import NegativeBinomialEmission
from sal.opt.hmm import baum_welch_family
from sal.ragged import Ragged

from tests._rows import every_value

#: The float64 tolerance for a reordered recursion (DEV.md; issue #649).
TOLERANCE = 1e-11


def _batch(
    seed: int, n: int = 30, k: int = 4
) -> tuple[Ragged, NegativeBinomialEmission, torch.Tensor]:
    """Sticky negative-binomial chains of lengths 40-300, a start and a kernel."""
    rng = np.random.default_rng([933, 5, seed])
    lengths = rng.integers(40, 300, n)
    means = np.geomspace(2.0, 60.0, k)
    chains = []
    for length in lengths:
        states = np.zeros(length, dtype=np.int64)
        for t in range(1, length):
            states[t] = states[t - 1] if rng.random() < 0.97 else rng.integers(0, k)
        chains.append(rng.negative_binomial(8.0, 8.0 / (8.0 + means[states])))
    batch = Ragged(
        values=np.concatenate(chains).astype(float),
        lengths=tuple(int(x) for x in lengths),
    )
    kernel = np.full((k, k), 0.03 / (k - 1))
    np.fill_diagonal(kernel, 0.97)
    return (
        batch,
        NegativeBinomialEmission([3.0] * k, means * 1.3),
        torch.log(torch.as_tensor(kernel)),
    )


def _relative(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(((first - second).abs() / second.abs()).max())


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_the_compiled_e_step_is_the_torch_recursion() -> None:
    # Measured: the log-likelihood within a relative 7.7e-16 and every
    # parameter within 5.0e-13 after 10 iterations at this size; 3.7e-15 and
    # 2.0e-12 at 318,945 positions.
    def check(seed: int) -> None:
        batch, start, kernel = _batch(seed)
        initial = torch.full((4,), -math.log(4.0), dtype=torch.float64)
        fits = {
            backend: baum_welch_family(
                batch,
                initial,
                kernel,
                start,
                max_iterations=10,
                tolerance=0.0,
                backend=backend,
            )
            for backend in (Backend.PYTHON, Backend.RUST)
        }
        torch_fit, compiled = fits[Backend.PYTHON], fits[Backend.RUST]
        assert abs(
            torch_fit.log_likelihood - compiled.log_likelihood
        ) <= TOLERANCE * abs(torch_fit.log_likelihood)
        for name, value in torch_fit.emissions.named_parameters().items():
            assert (
                _relative(compiled.emissions.named_parameters()[name], value)
                < TOLERANCE
            )
        assert (
            float((torch_fit.log_transition - compiled.log_transition).abs().max())
            < TOLERANCE
        )
        assert (
            float((torch_fit.log_initial - compiled.log_initial).abs().max())
            < TOLERANCE
        )

    every_value(range(2), check)


@pytest.mark.critical
@pytest.mark.oracle
def test_a_held_transition_is_that_transition_expanded_per_step() -> None:
    batch, start, kernel = _batch(2, n=6)
    initial = torch.full((4,), -math.log(4.0), dtype=torch.float64)
    per_step = baum_welch_family(
        batch,
        initial,
        kernel.expand(max(batch.lengths) - 1, 4, 4).clone(),
        start,
        max_iterations=5,
    )
    # The torch recursion on both sides is bitwise; the default compiled E
    # step measured 1.3e-15 relative on the likelihood and 5.2e-13 on the
    # emission parameters, held to TOLERANCE.
    for backend in (Backend.PYTHON, Backend.RUST):
        held = baum_welch_family(
            batch,
            initial,
            kernel,
            start,
            max_iterations=5,
            fit_transition=False,
            backend=backend,
        )
        assert torch.equal(held.log_transition, kernel)
        fitted = held.emissions.named_parameters()
        if backend is Backend.PYTHON:
            assert held.log_likelihood == per_step.log_likelihood
            for name, value in fitted.items():
                assert torch.equal(value, per_step.emissions.named_parameters()[name])
        else:
            assert abs(
                held.log_likelihood - per_step.log_likelihood
            ) <= TOLERANCE * abs(per_step.log_likelihood)
            for name, value in fitted.items():
                torch.testing.assert_close(
                    value,
                    per_step.emissions.named_parameters()[name],
                    rtol=TOLERANCE,
                    atol=0.0,
                )


@pytest.mark.smoke
@pytest.mark.backend
def test_a_per_step_kernel_keeps_the_torch_recursion_under_either_backend() -> None:
    batch, start, kernel = _batch(3, n=4)
    initial = torch.full((4,), -math.log(4.0), dtype=torch.float64)
    per_step = kernel.expand(max(batch.lengths) - 1, 4, 4).clone()
    first, second = (
        baum_welch_family(batch, initial, per_step, start, max_iterations=3, backend=b)
        for b in (Backend.PYTHON, Backend.RUST)
    )
    assert first.log_likelihood == second.log_likelihood
    with pytest.raises(ValueError, match="E step"):
        baum_welch_family(
            batch, initial, kernel, start, max_iterations=1, backend=Backend.NUMBA
        )
