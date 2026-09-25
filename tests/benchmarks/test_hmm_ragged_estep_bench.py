"""Baum-Welch on a ragged count batch, torch recursion against the compiled ragged E step (issue #933, R5).

Thirty sticky negative-binomial chains of 40-300 positions, four states,
five iterations. `tests/regression/opt/test_opt_hmm_ragged_estep.py` pins the
two at the float64 tolerance; this measures the ratio. At 200 chains of
100-3,000 positions the ratio measured 19.4x (218 s against 11.2 s for ten
iterations), the stress size the port is kept for.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import NegativeBinomialEmission
from sal.opt.em import EmConfig
from sal.opt.hmm import baum_welch_family
from sal.ragged import Ragged


@pytest.fixture(scope="module")
def problem() -> tuple[Ragged, NegativeBinomialEmission, torch.Tensor]:
    rng = np.random.default_rng(933)
    lengths = rng.integers(40, 300, 30)
    means = np.geomspace(2.0, 60.0, 4)
    chains = []
    for length in lengths:
        states = np.zeros(length, dtype=np.int64)
        for t in range(1, length):
            states[t] = states[t - 1] if rng.random() < 0.97 else rng.integers(0, 4)
        chains.append(rng.negative_binomial(8.0, 8.0 / (8.0 + means[states])))
    kernel = np.full((4, 4), 0.01)
    np.fill_diagonal(kernel, 0.97)
    return (
        Ragged(
            values=np.concatenate(chains).astype(float),
            lengths=tuple(int(x) for x in lengths),
        ),
        NegativeBinomialEmission([3.0] * 4, means * 1.3),
        torch.log(torch.as_tensor(kernel)),
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST], ids=str)
def test_baum_welch_on_a_ragged_batch(
    benchmark: object,
    problem: tuple[Ragged, NegativeBinomialEmission, torch.Tensor],
    backend: Backend,
) -> None:
    batch, start, kernel = problem
    initial = torch.full((4,), -math.log(4.0), dtype=torch.float64)
    fit = benchmark(  # type: ignore[operator]
        lambda: baum_welch_family(
            batch,
            initial,
            kernel,
            start,
            backend=backend,
            config=EmConfig(max_iterations=5, tolerance=0.0),
        )
    )
    assert math.isfinite(fit.log_likelihood)
