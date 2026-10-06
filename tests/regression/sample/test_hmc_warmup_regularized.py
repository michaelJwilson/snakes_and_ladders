"""The warm-up variance, regularized as Stan does, on every route (issue #1207).

``chain.regularized_variance`` is ``(n / (n + 5)) var + 1e-3 (5 / (n + 5))``
over ``n`` recorded draws, pinned to that closed form; a coordinate whose
sample variance is exactly zero is reported on ``Adapted.flat`` rather than
refused. The torch (:data:`Backend.PYTHON`), Rust (``oxisal.HmcWalk``) and
JAX (``JaxWalk``) warm-ups share the rule, so a frozen warm-up gives the same
mass on all three, bitwise. On a Gaussian of known variances the adapted mass
approaches the precision as the warm-up grows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.sample import hmc
from sal.sample.chain import (
    SHRINKAGE_PROPOSALS,
    SHRINKAGE_VARIANCE,
    regularized_variance,
)
from sal.sample.hmc.jax import JaxWalk
from sal.validation.gaussian import GaussianTarget

#: A precision whose stability limit, 2 / sqrt(1e8) = 2e-4, sits below every
#: step of the downstream reproduction, so every warm-up proposal is rejected.
STIFF = np.array([1e8, 1e8])

#: The downstream reproduction's steps (issue #1204, item 2).
STEPS = (1e-3, 3e-3, 1e-2)


def _gaussian_energy(theta: Any, precision: Any) -> Any:
    """``theta' diag(precision) theta / 2``, traceable; module level so ``JaxWalk`` compiles it once."""
    return 0.5 * (precision * theta * theta).sum()


class _JaxGaussian:
    """A diagonal Gaussian declaring only a JAX energy, so ``hmc.sample`` runs ``JaxWalk``."""

    def __init__(self, precision: np.ndarray) -> None:
        self.precision = precision

    def initial(self) -> torch.Tensor:
        return torch.zeros(self.precision.size, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return named["x"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.from_numpy(self.precision) * theta * theta).sum()

    def jax_energy(self) -> tuple[Any, Any]:
        import jax.numpy as jnp

        return _gaussian_energy, jnp.asarray(self.precision)


def _target(route: str, precision: np.ndarray) -> tuple[Any, Backend]:
    """The objective and backend that put ``hmc.sample`` on ``route``."""
    if route == "jax":
        return _JaxGaussian(precision), Backend.RUST
    return GaussianTarget(precision), (
        Backend.RUST if route == "rust" else Backend.PYTHON
    )


@pytest.mark.oracle
def test_the_regularized_variance_is_stans_closed_form() -> None:
    # n = 3: the weight is 3/8 and the floor 1e-3 * 5/8; the zero entry is
    # the floor alone and is the one reported flat.
    variance = np.array([0.0, 2.0, 0.5])

    regularized, flat = regularized_variance(variance, 3)

    floor = 1e-3 * 5.0 / 8.0
    np.testing.assert_allclose(
        regularized, [floor, 0.75 + floor, 0.1875 + floor], rtol=1e-15, atol=0.0
    )
    assert flat == (0,)
    assert (SHRINKAGE_PROPOSALS, SHRINKAGE_VARIANCE) == (5.0, 1e-3)


@pytest.mark.smoke
def test_a_non_finite_variance_is_still_refused() -> None:
    # No shrinkage repairs a chain that reached a non-finite position.
    with pytest.raises(ValueError, match=r"not finite on coordinate\(s\) \[1\]"):
        regularized_variance(np.array([1.0, np.nan]), 4)


@pytest.mark.smoke
@pytest.mark.parametrize("route", ["torch", "rust", "jax"])
@pytest.mark.parametrize("step", STEPS)
def test_the_downstream_reproduction_runs_and_reports_flat(
    route: str, step: float
) -> None:
    # Issue #1204's reproduction: warmup = 8 records 2 draws and raised on 5
    # of 5 seeds. Every proposal is rejected on `STIFF`, so both coordinates
    # are flat, and the mass is the floor's over n = 2 on every route,
    # bitwise: 1 / (1e-3 * 5 / 7).
    objective, backend = _target(route, STIFF)
    floor_mass = 1.0 / (1e-3 * (5.0 / 7.0))
    for seed in range(5):
        chain = hmc.sample(
            objective,
            torch.Generator().manual_seed(seed),
            4,
            step_size=step,
            n_steps=4,
            adaptation=hmc.Adaptation(8, 0.65, 0.0),
            backend=backend,
        )
        assert chain.adapted is not None
        assert chain.adapted.flat == (0, 1)
        assert chain.adapted.mass_diagonal.tolist() == [floor_mass, floor_mass]


@pytest.mark.smoke
@pytest.mark.parametrize("route", ["torch", "rust", "jax"])
def test_a_chain_that_moves_reports_nothing_flat(route: str) -> None:
    objective, backend = _target(route, np.array([1.0, 4.0]))
    chain = hmc.sample(
        objective,
        torch.Generator().manual_seed(1207),
        4,
        step_size=0.3,
        n_steps=4,
        adaptation=hmc.Adaptation(40, 0.65, 0.0),
        backend=backend,
    )
    assert chain.adapted is not None
    assert chain.adapted.flat == ()
    assert bool(torch.isfinite(chain.adapted.mass_diagonal).all())


@pytest.mark.oracle
@pytest.mark.backend
def test_the_jax_and_torch_rules_are_one_function() -> None:
    # The JAX walk calls the torch route's `regularized_variance` on its
    # Welford variance, so the two scales agree to the variance estimators'
    # rounding: 1e-12 here, on 2 draws whose two-pass and Welford variances
    # are computed from the same pair.
    rng = np.random.default_rng(1207)
    pair = rng.normal(size=(2, 6))
    two_pass = torch.from_numpy(pair).var(dim=0, unbiased=True).numpy()
    mean, m2 = np.zeros(6), np.zeros(6)
    for count, row in enumerate(pair, start=1):
        delta = row - mean
        mean += delta / count
        m2 += delta * (row - mean)
    torch_scale = np.sqrt(regularized_variance(two_pass, 2)[0])
    jax_scale = np.sqrt(regularized_variance(m2 / 1.0, 2)[0])
    np.testing.assert_allclose(jax_scale, torch_scale, rtol=1e-12, atol=0.0)
    assert JaxWalk._warm_up.__code__.co_names.count("regularized_variance") == 1


@pytest.mark.oracle
def test_the_adapted_mass_approaches_the_known_precision() -> None:
    # The max over coordinates of |log(mass / precision)|, mean over 10
    # seeds, falls as the warm-up grows: measured 0.587, 0.195, 0.094, 0.059
    # at 800, 3,200, 12,800 and 51,200 proposals. At the last every seed is
    # inside 0.15 (measured 0.019 to 0.092): over 12,800 recorded draws the
    # shrinkage weight is 1 - 3.9e-4, below the sampling error it is judged
    # beside.
    precision = np.array([0.25, 1.0, 4.0, 100.0])
    means = []
    for warmup in (800, 3_200, 12_800, 51_200):
        errors = []
        for seed in range(10):
            chain = hmc.sample(
                GaussianTarget(precision),
                torch.Generator().manual_seed(seed),
                1,
                step_size=0.3,
                n_steps=5,
                adaptation=hmc.Adaptation(warmup, 0.65, 0.2),
            )
            assert chain.adapted is not None
            errors.append(
                float(
                    np.abs(
                        np.log(chain.adapted.mass_diagonal.numpy() / precision)
                    ).max()
                )
            )
        means.append(float(np.mean(errors)))
    assert means == sorted(means, reverse=True), means
    assert max(errors) < 0.15, errors


#: Mean ESS per gradient over seeds 0 to 9 at the base of issue #1207
#: (`253c84f`), and its standard error, on the benchmark cells of
#: `tests/benchmarks/test_hmc_declared_blackjax_bench.py`: 500 warm-up
#: proposals at 0.65, then 1,000 draws of 10 leapfrog steps.
BEFORE = {"rosenbrock": (1.0246e-03, 2.23e-04), "gaussian": (1.9966e-04, 5.62e-06)}


@pytest.mark.experiment
@pytest.mark.release
@pytest.mark.parametrize("family", sorted(BEFORE))
def test_the_regularized_warm_up_costs_no_effective_draws(family: str) -> None:
    # Measured after: 8.49e-4 (se 1.42e-4) and 2.026e-4 (se 6.7e-6), 0.67
    # and 0.34 standard errors of the difference from before. The bound is
    # before less three of its standard errors: no regression beyond noise.
    from sal.opt.testfunctions import Rosenbrock
    from sal.validation.gaussian import diagonal_precision

    objective: Any
    if family == "rosenbrock":
        objective, step = Rosenbrock(dimension=10), 0.01
    else:
        objective, step = (
            GaussianTarget(diagonal_precision(100)),
            0.9 / (2.0 * 100**0.25),
        )
    rates = []
    for seed in range(10):
        chain = hmc.sample(
            objective,
            torch.Generator().manual_seed(seed),
            1_000,
            step_size=step,
            n_steps=10,
            adaptation=hmc.Adaptation(500, 0.65, 0.0),
        )
        rates.append(float(hmc.effective_sample_size(chain.draws).min()) / chain.spent)
    mean, error = BEFORE[family]
    assert np.mean(rates) > mean - 3.0 * error, rates
