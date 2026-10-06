"""The beta-binomial read by rate and concentration (issue #1205).

Referees: the ``alpha, beta`` family at ``a = tau p``, ``b = tau (1 - p)``,
whose densities the reading reproduces bitwise; the chain rule through
``(a, b)``, which its gradient in ``(p, tau)`` equals; autograd through each
objective's ``__call__``, which the compiled routes are pinned to; and the
planted rate of a simulated HMM, recovered with the concentration held by
:class:`~sal.opt.objective.Restricted`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    CountPairEmission,
    Domain,
    EmissionFamily,
    ParameterDomainError,
    RateConcentrationBetaBinomialEmission,
    RateConcentrationCountPairEmission,
)
from sal.opt.emission_mixture import EmissionMixtureObjective
from sal.opt.fit import fit
from sal.opt.hmm import EmissionHmmObjective, align_families
from sal.opt.objective import (
    Restricted,
    autograd_value_and_gradient,
    coordinates,
    value_and_gradient,
)
from sal.ragged import Ragged
from sal.sim.count_pairs import IndependentCountPair
from sal.sim.hmm import HmmParams, simulate_sequences

SEED = 1205

#: Concentrations from the small-shape path to past the series threshold,
#: where ``a`` and ``b`` reach 1e16.
CONCENTRATIONS = (10.0, 1e2, 1e3, 1e6, 1e12, 1e16)
RATES = (0.03, 0.3, 0.5, 0.97)
TRIALS = 40


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Every rate at every concentration, one state each."""
    rate, tau = np.meshgrid(RATES, CONCENTRATIONS)
    return (
        torch.as_tensor(rate.reshape(-1), dtype=torch.float64),
        torch.as_tensor(tau.reshape(-1), dtype=torch.float64),
    )


def _counts() -> torch.Tensor:
    """Every count of the support, ``0..TRIALS``."""
    return torch.arange(TRIALS + 1, dtype=torch.float64)


@pytest.mark.oracle
def test_the_density_is_the_alpha_beta_family_s_bitwise() -> None:
    # At matched parameters, a = tau p and b = tau (1 - p), every count of
    # n = 40 under 24 states, tau from 10 to 1e16. Bitwise.
    rate, tau = _grid()
    trials = torch.full_like(rate, float(TRIALS))
    reading = RateConcentrationBetaBinomialEmission(trials, rate, tau)
    family = BetaBinomialEmission(trials, tau * rate, tau * (1.0 - rate))

    got = reading.log_density(_counts())
    want = family.log_density(_counts())

    assert bool(torch.isfinite(got).all())
    assert torch.equal(got, want)


@pytest.mark.oracle
@pytest.mark.parametrize("joint", [True, False], ids=["joint", "independent"])
def test_the_pair_s_density_is_the_alpha_beta_pair_s_bitwise(joint: bool) -> None:
    rate, tau = _grid()
    n = rate.numel()
    dispersion = torch.linspace(2.0, 30.0, n, dtype=torch.float64)
    mean = torch.linspace(5.0, 40.0, n, dtype=torch.float64)
    trials = None if joint else torch.full_like(rate, float(TRIALS))
    reading = RateConcentrationCountPairEmission(
        dispersion, mean, rate, tau, trials, joint=joint
    )
    family = CountPairEmission(
        dispersion, mean, tau * rate, tau * (1.0 - rate), trials, joint=joint
    )
    totals = torch.arange(TRIALS + 1, dtype=torch.float64)
    pairs = torch.stack(
        [totals, torch.minimum(totals, torch.tensor(float(TRIALS // 3)))], dim=-1
    )

    assert torch.equal(reading.log_density(pairs), family.log_density(pairs))


@pytest.mark.oracle
@pytest.mark.parametrize(
    "family",
    [
        RateConcentrationBetaBinomialEmission(
            [20, 20], [0.2, 0.7], [15.0, 3e5], tied=False
        ),
        RateConcentrationCountPairEmission(
            [4.0, 9.0], [10.0, 30.0], [0.2, 0.7], [15.0, 3e5], joint=True
        ),
    ],
    ids=["beta-binomial", "joint-pair"],
)
def test_the_with_parameters_round_trip_is_bitwise(family: EmissionFamily) -> None:
    named = family.named_parameters()
    rebuilt = family.with_parameters(named)
    values = torch.tensor([[12.0, 3.0], [20.0, 15.0], [0.0, 0.0]])
    observations = values[:, 1] if isinstance(family, BetaBinomialEmission) else values

    assert type(rebuilt) is type(family)
    for name, value in rebuilt.named_parameters().items():
        assert torch.equal(value, named[name]), name
    assert torch.equal(
        rebuilt.log_density(observations), family.log_density(observations)
    )


@pytest.mark.smoke
def test_the_reading_declares_rate_and_concentration_and_refuses_outside_them() -> None:
    family = BetaBinomialEmission([10, 10], [2.0, 6.0], [8.0, 2.0])
    pair = CountPairEmission([3.0, 3.0], [9.0, 9.0], [2.0, 6.0], [8.0, 2.0], joint=True)

    assert family.rate_concentration().parameter_domains() == {
        "rate": Domain.PROBABILITY,
        "concentration": Domain.POSITIVE,
    }
    assert list(pair.rate_concentration().named_parameters()) == [
        "dispersion",
        "mean",
        "rate",
        "concentration",
    ]
    with pytest.raises(ParameterDomainError, match="rate"):
        RateConcentrationBetaBinomialEmission([10], [1.0], [5.0])
    with pytest.raises(ParameterDomainError, match="concentration"):
        RateConcentrationBetaBinomialEmission([10], [0.5], [0.0])
    with pytest.raises(ValueError, match="parameterized by"):
        family.rate_concentration().with_parameters(family.named_parameters())


@pytest.mark.oracle
def test_rate_concentration_of_the_alpha_beta_family_scores_it() -> None:
    # p = a / (a + b), tau = a + b, then back: a and b to rounding, so the
    # densities to 1e-12 relative. Measured: 0.0, bitwise on this grid.
    family = BetaBinomialEmission([40, 40, 40], [2.0, 9.0, 3e5], [8.0, 3.0, 2e5])
    reading = family.rate_concentration()

    assert isinstance(reading, RateConcentrationBetaBinomialEmission)
    torch.testing.assert_close(
        reading.log_density(_counts()),
        family.log_density(_counts()),
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.analytic
def test_the_gradient_in_rate_and_concentration_is_the_chain_rule() -> None:
    # dL/dp = tau (dL/da - dL/db), dL/dtau = p dL/da + (1 - p) dL/db, with
    # dL/da and dL/db autograd of the alpha, beta family at a = tau p,
    # b = tau (1 - p); every count weighted so no state's sum vanishes.
    # Measured: 2.0e-16 relative at worst in p, 0.0 in tau.
    rate, tau = _grid()
    trials = torch.full_like(rate, float(TRIALS))
    weights = torch.linspace(0.5, 1.5, TRIALS + 1, dtype=torch.float64).unsqueeze(-1)
    p = rate.clone().requires_grad_(True)
    t = tau.clone().requires_grad_(True)
    reading = RateConcentrationBetaBinomialEmission(trials, p, t)
    d_rate, d_tau = torch.autograd.grad(
        (weights * reading.log_density(_counts())).sum(), (p, t)
    )
    a = (tau * rate).requires_grad_(True)
    b = (tau * (1.0 - rate)).requires_grad_(True)
    family = BetaBinomialEmission(trials, a, b)
    d_a, d_b = torch.autograd.grad(
        (weights * family.log_density(_counts())).sum(), (a, b)
    )

    torch.testing.assert_close(d_rate, tau * (d_a - d_b), rtol=1e-12, atol=0.0)
    torch.testing.assert_close(
        d_tau, rate * d_a + (1.0 - rate) * d_b, rtol=1e-12, atol=0.0
    )


@pytest.mark.oracle
def test_the_m_step_is_the_alpha_beta_family_s_read_by_rate() -> None:
    # The re-estimate of the alpha, beta family, read as p = a / (a + b) and
    # tau = a + b; an emptied state keeps its (p, tau) bitwise.
    rng = np.random.default_rng(SEED)
    truth = BetaBinomialEmission([30, 30, 30], [3.0, 12.0, 5.0], [9.0, 4.0, 5.0])
    states = rng.integers(0, 2, 2_000)
    observations = truth.sample(states, rng)
    posterior = np.eye(3)[states]
    start = BetaBinomialEmission([30, 30, 30], [2.0, 8.0, 4.0], [6.0, 3.0, 7.0])
    reading = start.rate_concentration()

    step = reading.reestimate(observations, posterior)
    want = start.reestimate(observations, posterior)

    assert isinstance(step.components, RateConcentrationBetaBinomialEmission)
    assert step.frozen == want.frozen == (2,)
    torch.testing.assert_close(
        step.components.alpha, want.components.alpha, rtol=1e-12, atol=0.0
    )
    torch.testing.assert_close(
        step.components.beta, want.components.beta, rtol=1e-12, atol=0.0
    )
    held = step.components.named_parameters()
    assert float(held["rate"][2]) == float(reading.named_parameters()["rate"][2])
    assert float(held["concentration"][2]) == float(
        reading.named_parameters()["concentration"][2]
    )


def _mixture_families() -> dict[
    str, tuple[EmissionFamily, Callable[[np.ndarray], np.ndarray]]
]:
    """Each family the compiled mixture gradient takes, in the rate--concentration reading."""
    joint = CountPairEmission(
        [6.0, 12.0, 3e5],
        [20.0, 60.0, 40.0],
        [2.0, 9.0, 3e5],
        [8.0, 3.0, 2e5],
        joint=True,
    ).rate_concentration()
    independent = CountPairEmission(
        [6.0, 12.0], [20.0, 60.0], [2.0, 9.0], [8.0, 3.0], [40.0, 40.0], joint=False
    ).rate_concentration()
    successes = independent.successes
    assert isinstance(successes, RateConcentrationBetaBinomialEmission)
    return {
        "joint": (joint, lambda pairs: pairs),
        "independent": (independent, lambda pairs: pairs),
        "IndependentCountPair": (
            IndependentCountPair(independent.total, successes),
            lambda pairs: pairs,
        ),
        "beta-binomial": (successes, lambda counts: counts),
    }


@pytest.mark.oracle
@pytest.mark.parametrize("name", list(_mixture_families()))
def test_the_compiled_mixture_gradient_is_autograds(name: str) -> None:
    # `oxisal.count_mixture_value_and_gradient` reads a = tau p, b = tau (1 - p)
    # and its gradient is carried back to (p, tau) by the chain rule; autograd
    # through `__call__` is the reference, at the tolerances of the alpha,
    # beta reading (`test_emission_mixture_robustness.py`). Measured: 3.3e-15
    # relative in the value, 2.9e-11 absolute in a gradient of magnitude up
    # to 2.9e3.
    family, read = _mixture_families()[name]
    rng = np.random.default_rng([SEED, len(name)])
    labels = rng.integers(0, family.n_states, 3_000)
    observations = read(np.asarray(family.sample(labels, rng), dtype=np.float64))
    objective = EmissionMixtureObjective(observations, family)
    theta = objective.initial() + 0.1 * torch.randn(
        objective.n_parameters,
        generator=torch.Generator().manual_seed(SEED),
        dtype=torch.float64,
    )

    value, gradient = objective.value_and_gradient(theta)
    want_value, want_gradient = autograd_value_and_gradient(objective, theta)

    assert objective._route is not None
    assert float(value) == pytest.approx(float(want_value), rel=1e-13)
    np.testing.assert_allclose(
        gradient.numpy(), want_gradient.numpy(), rtol=0, atol=1e-9
    )


def _hmm_pairs() -> tuple[Ragged, RateConcentrationCountPairEmission]:
    """Joint count pairs over three unequal segments, and the reading that drew them."""
    rng = np.random.default_rng([SEED, 1])
    family = CountPairEmission(
        [5.0, 12.0], [16.0, 30.0], [2.0, 9.0], [8.0, 3.0], joint=True
    ).rate_concentration()
    states = rng.integers(0, 2, 150)
    values = np.asarray(family.sample(states, rng), dtype=np.float64)
    return Ragged(values, (30, 50, 70)), family


@pytest.mark.oracle
@pytest.mark.parametrize("kind", ["pair", "beta-binomial"])
def test_the_compiled_hmm_gradient_is_autograds(kind: str) -> None:
    # Fisher's identity through `oxisal.ragged_posteriors`, against autograd
    # through `__call__`, at `test_opt_hmm_emission_objective.py`'s
    # tolerances. Measured: 0.0 in value, 6.0e-12 absolute in a gradient
    # of magnitude up to 75.
    batch, family = _hmm_pairs()
    start: EmissionFamily = family
    observations: Ragged = batch
    if kind == "beta-binomial":
        start = RateConcentrationBetaBinomialEmission(
            [40, 40], [0.2, 0.7], [10.0, 12.0]
        )
        observations = Ragged(np.minimum(batch.values[:, 1], 40.0), batch.lengths)
    objective = EmissionHmmObjective(observations, start)
    theta = objective.initial() + torch.linspace(
        -0.3, 0.2, objective.n_parameters, dtype=torch.float64
    )

    value, gradient = value_and_gradient(objective, theta)
    want_value, want_gradient = autograd_value_and_gradient(objective, theta)

    assert list(objective.blocks)[-2:] == ["rate", "concentration"]
    assert float(value) == pytest.approx(float(want_value), rel=1e-10, abs=0.0)
    np.testing.assert_allclose(
        gradient.numpy(), want_gradient.numpy(), rtol=0.0, atol=1e-10
    )


@pytest.mark.smoke
def test_the_jax_twin_refuses_the_reading_rather_than_mis_map_it() -> None:
    # The twin is written for the alpha, beta family by exact type; the
    # reading has none, and says so.
    start = RateConcentrationBetaBinomialEmission([40, 40], [0.2, 0.7], [10.0, 12.0])
    data = np.random.default_rng(SEED).integers(0, 41, (3, 40))

    with pytest.raises(ValueError, match="no JAX twin"):
        EmissionHmmObjective(data, start, backend=Backend.JAX)
    assert EmissionHmmObjective(data, start).jax_energy() is None


#: The recovery fixture: twenty segments of 50-249 positions, two states.
LENGTHS = tuple(
    int(length) for length in np.random.default_rng(SEED).integers(50, 250, 20)
)
INITIAL = np.array([0.5, 0.5])
TRANSITION = np.array([[0.95, 0.05], [0.08, 0.92]])
Z_BOUND = 4.0


@pytest.mark.end2end
def test_holding_the_concentration_recovers_the_rate() -> None:
    # A joint count-pair HMM drawn at rates (0.2, 0.75) and concentrations
    # (10, 30); the fit varies everything but the concentration, held at the
    # truth by `Restricted`. Each rate lies within `Z_BOUND` delta-method
    # standard errors of the truth, and the concentration comes back
    # bitwise. Measured worst |z|: 1.64.
    truth = RateConcentrationCountPairEmission(
        [6.0, 12.0], [10.0, 40.0], [0.2, 0.75], [10.0, 30.0], joint=True
    )
    batch = simulate_sequences(
        HmmParams(2, LENGTHS, INITIAL, TRANSITION, truth, seed=SEED, tolerance=0.0)
    ).batch
    start = RateConcentrationCountPairEmission(
        [2.0, 2.0], [8.0, 30.0], [0.4, 0.6], [10.0, 30.0], joint=True
    )
    objective = EmissionHmmObjective(batch, start)
    at = objective.initial()
    varied = coordinates(
        objective, [name for name in objective.blocks if name != "concentration"]
    )
    restricted = Restricted(objective, at, varied)

    result = fit(restricted, include_intervals=True)

    assert result.converged
    assert result.standard_errors is not None
    fitted = restricted.constrain(result.theta)
    assert torch.equal(
        fitted["concentration"], objective.constrain(at)["concentration"]
    )
    order = list(
        align_families(objective.components(restricted.embed(result.theta)), truth)
    )
    z = (
        (fitted["rate"].detach()[order] - truth.rate)
        / result.standard_errors["rate"][order]
    ).abs()
    assert float(z.max()) <= Z_BOUND, z
