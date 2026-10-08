"""`EmissionHmmObjective` on a beta-binomial read by ``(rate, concentration)`` (issue #1347).

Referees: autograd through ``__call__`` (the ``TORCH`` route) against the
compiled count kernel the default takes; a Richardson central difference of
the value; the ``(alpha, beta)`` objective at the same distribution; and a
fit under :class:`~sal.opt.objective.Restricted` that holds the
concentration, read back bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    CountPairEmission,
    RateConcentrationBetaBinomialEmission,
)
from sal.opt.fit import fit
from sal.opt.hmm import EmissionHmmObjective, family_start
from sal.opt.objective import Restricted, coordinates
from sal.ragged import Ragged
from sal.sim.hmm import HmmParams, simulate_sequences

SEED = 1347
K = 3
LENGTHS = (120, 80, 200, 100)

#: Between routes, the value relative: #1332's bound.
ROUTE_TOLERANCE = 1e-13

#: Between routes, the gradient relative to its largest coordinate. #1332's
#: 1e-13 does not hold here: the kernel sums Fisher's identity over a count
#: histogram, autograd over positions, and the (alpha, beta) reading on the
#: same fixture differs by the same order (1.8e-12 of max(|g|, 1) per
#: coordinate against 3.5e-12 here). Measured: 4.4e-13.
GRADIENT_TOLERANCE = 1e-12


def _pair() -> tuple[Ragged, CountPairEmission]:
    """A joint count pair, three states, four segments of 80--200, simulated from a seeded truth."""
    rate = np.linspace(0.1, 0.9, K)
    family = CountPairEmission(
        np.full(K, 20.0),
        np.geomspace(10.0, 200.0, K),
        30.0 * rate,
        30.0 * (1.0 - rate),
        joint=True,
    )
    transition = np.full((K, K), 0.05 / (K - 1)) + np.eye(K) * (
        1.0 - 0.05 - 0.05 / (K - 1)
    )
    params = HmmParams(
        n_states=K,
        lengths=LENGTHS,
        initial=np.full(K, 1.0 / K),
        transition=transition,
        emissions=family,
        seed=SEED,
        tolerance=0.0,
    )
    values = np.asarray(simulate_sequences(params).observations, dtype=np.float64)
    return Ragged(values.reshape(-1, 2), LENGTHS), family


def _objective(backend: Backend = Backend.RUST) -> EmissionHmmObjective:
    data, family = _pair()
    return EmissionHmmObjective(data, family.rate_concentration(), backend=backend)


def _off_start(objective: EmissionHmmObjective) -> torch.Tensor:
    start = objective.initial()
    return start + torch.linspace(-0.3, 0.2, start.numel(), dtype=torch.float64)


@pytest.mark.oracle
def test_the_kernel_route_equals_autograd_within_the_route_bound() -> None:
    # The default takes `count_hmm` on `rate` and `concentration` slots;
    # TORCH is autograd through `__call__`. Measured: 6.2e-16 in the value
    # and 4.4e-13 of the largest gradient coordinate.
    objective, oracle = _objective(), _objective(Backend.TORCH)
    assert objective.supported_gradient() is not None
    theta = _off_start(objective)
    value, gradient = objective.value_and_gradient(theta)
    want_value, want_gradient = oracle.value_and_gradient(theta)
    assert abs(float(value - want_value)) <= ROUTE_TOLERANCE * abs(float(want_value))
    assert (gradient - want_gradient).abs().max() <= (
        GRADIENT_TOLERANCE * want_gradient.abs().max()
    )


@pytest.mark.oracle
def test_the_gradient_equals_a_richardson_central_difference() -> None:
    # Two central differences at h and h/2 of the TORCH value, combined to
    # cancel the h^2 term. Measured: 6.3e-9 of max(|g|, 1) at h = 1e-3.
    objective, oracle = _objective(), _objective(Backend.TORCH)
    theta = _off_start(objective)
    _, gradient = objective.value_and_gradient(theta)
    step = 1e-3
    difference = torch.empty_like(theta)
    for i in range(theta.numel()):
        unit = torch.zeros_like(theta)
        unit[i] = 1.0
        wide = (
            float(oracle(theta + step * unit)) - float(oracle(theta - step * unit))
        ) / (2.0 * step)
        narrow = (
            float(oracle(theta + step / 2 * unit))
            - float(oracle(theta - step / 2 * unit))
        ) / step
        difference[i] = (4.0 * narrow - wide) / 3.0
    assert ((gradient - difference).abs() / gradient.abs().clamp_min(1.0)).max() <= 1e-7


@pytest.mark.oracle
def test_the_alpha_beta_reading_scores_the_same_distribution() -> None:
    # The (alpha, beta) objective at the start the (p, tau) one derives from.
    # Measured: 0.0 relative on the kernel route (bitwise) and 1.1e-15 by
    # autograd through `__call__`.
    data, family = _pair()
    natural = EmissionHmmObjective(data, family)
    rate = _objective()
    value, _ = natural.value_and_gradient(natural.initial())
    want, _ = rate.value_and_gradient(rate.initial())
    assert torch.equal(value, want)
    assert abs(float(natural(natural.initial())) - float(rate(rate.initial()))) <= (
        1e-14 * abs(float(value))
    )


@pytest.mark.oracle
def test_family_start_in_rate_concentration_derives_the_alpha_beta_start() -> None:
    # Same rates at unit concentration: a = p, b = 1 - p against the (a, b)
    # start's exp(log p), exp(log1p(-p)). Measured: 1 ulp, and 4.4e-16 in
    # every log density.
    successes = np.random.default_rng([SEED, 0]).binomial(12, 0.3, size=(3, 40))
    trials = np.array([12, 12])
    natural = family_start(BetaBinomialEmission, successes, 2, trials=trials)
    rate = family_start(
        RateConcentrationBetaBinomialEmission, successes, 2, trials=trials
    )
    assert isinstance(rate, RateConcentrationBetaBinomialEmission)
    assert isinstance(natural, BetaBinomialEmission)
    assert torch.equal(rate.concentration, torch.ones(2, dtype=torch.float64))
    eps = torch.finfo(torch.float64).eps
    for got, want in ((rate.alpha, natural.alpha), (rate.beta, natural.beta)):
        assert ((got - want).abs() <= 2.0 * eps * want.abs()).all()
    values = torch.as_tensor(successes.reshape(-1), dtype=torch.float64)
    assert (rate.log_density(values) - natural.log_density(values)).abs().max() <= 1e-14
    objective = EmissionHmmObjective(successes, rate)
    assert list(objective.blocks)[-2:] == ["rate", "concentration"]


@pytest.mark.oracle
def test_restricted_holds_the_concentration_bitwise() -> None:
    # Restricted over every block but `concentration` keeps the compiled
    # route, bitwise; a fit moves the rates and returns tau exactly as held.
    objective = _objective()
    at = objective.initial()
    varied = coordinates(
        objective, [name for name in objective.blocks if name != "concentration"]
    )
    restricted = Restricted(objective, at, varied)
    theta = restricted.initial() + 0.05
    value, gradient = restricted.value_and_gradient(theta)
    want_value, want_gradient = objective.value_and_gradient(restricted.embed(theta))
    assert torch.equal(value, want_value)
    assert torch.equal(gradient, want_gradient[varied])

    result = fit(restricted)
    held = objective.constrain(at)["concentration"]
    fitted = restricted.constrain(result.theta)
    assert result.converged
    assert torch.equal(fitted["concentration"], held)
    assert not torch.equal(fitted["rate"], objective.constrain(at)["rate"])
