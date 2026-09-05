"""An interval at a fit, whatever produced the fit.

The observed information is a property of an objective **at a point**, not of
the route that reached it. Until now only a gradient fit could ask for one,
because only a gradient fit had a ``theta``; expectation-maximization works in
the model's own parameters and never builds one, so half the fits here — the
half with an independent oracle — reported a point estimate and nothing else
(issue #268).

What is pinned here is the seam that closes that, and the refusals that make
an interval mean something: an interval from a Hessian is a statement about a
*maximum*, and this repository has three things that are not one.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from snakes_and_ladders.opt.fit import (
    constrained_standard_errors,
    covers,
    fit,
    fit_from,
    standard_errors_at,
)
from snakes_and_ladders.opt.hmm import (
    BetaBinomialHmmObjective,
    BinomialHmmObjective,
    GaussianHmmObjective,
    HmmObjective,
    NegativeBinomialHmmObjective,
    PoissonHmmObjective,
    align_states,
    baum_welch,
)
from snakes_and_ladders.opt.initialize import RandomRestart
from snakes_and_ladders.opt.mixture import GaussianMixtureObjective
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.potts import PottsObjective, PottsParams, simulate_chains
from snakes_and_ladders.sim.hmm import HmmParams, load_hmm_params, simulate_sequences
from snakes_and_ladders.sim.mixture import MixtureParams, simulate_mixture

from tests._fixtures import FIXTURES_DIR
from tests._objective_checks import AnalyticGaussian

HMM_FIXTURE = FIXTURES_DIR / "hmm_params.yaml"

INITIAL = np.array([0.5, 0.5])
TRANSITION = np.array([[0.75, 0.25], [0.25, 0.75]])
TRIALS = np.array([12, 12])


def _hmm_case() -> tuple[HmmObjective, torch.Tensor]:
    """The categorical HMM at its fixture, and a truth point."""
    params = load_hmm_params(HMM_FIXTURE)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )
    return objective, objective.theta_from_truth(
        params.initial, params.transition, params.emission
    )


def _two_state(emissions: object, seed: int = 5) -> np.ndarray:
    """Observations from a two-state HMM over one emission family."""
    params = HmmParams(
        n_states=2,
        sequence_length=8,
        n_sequences=40,
        initial=INITIAL,
        transition=TRANSITION,
        emissions=emissions,  # type: ignore[arg-type]
        seed=seed,
        tolerance=1e-12,
    )
    return simulate_sequences(params).observations


def _every_objective() -> list[tuple[Objective, torch.Tensor]]:
    """One instance of every objective here, each with a known truth point."""
    cases: list[tuple[Objective, torch.Tensor]] = [_hmm_case()]

    field = np.array([0.3, -0.2, -0.1])
    field = field - np.log(np.exp(field).sum())
    potts = PottsParams(
        n_states=3, chain_length=6, n_chains=40, coupling=0.5, field=field, seed=2
    )
    potts_objective = PottsObjective(simulate_chains(potts), 3)
    cases.append((potts_objective, potts_objective.theta_from_truth(0.5, field)))

    gaussian = GaussianEmission([-3.0, 3.0], [1.0, 1.5], 1e-12)
    observations = _two_state(gaussian)
    objective = GaussianHmmObjective(observations, 2)
    cases.append(
        (
            objective,
            objective.theta_from_truth(
                INITIAL, TRANSITION, gaussian.mean.numpy(), gaussian.scale.numpy()
            ),
        )
    )

    poisson = PoissonEmission([3.0, 15.0])
    poisson_objective = PoissonHmmObjective(_two_state(poisson), 2)
    cases.append(
        (
            poisson_objective,
            poisson_objective.theta_from_truth(
                INITIAL, TRANSITION, poisson.mean.numpy()
            ),
        )
    )

    binomial = BinomialEmission(TRIALS, [0.2, 0.75])
    binomial_objective = BinomialHmmObjective(_two_state(binomial), 2, TRIALS)
    cases.append(
        (
            binomial_objective,
            binomial_objective.theta_from_truth(
                INITIAL, TRANSITION, binomial.probability.numpy()
            ),
        )
    )

    beta_binomial = BetaBinomialEmission(TRIALS, [2.0, 8.0], [8.0, 2.0])
    beta_objective = BetaBinomialHmmObjective(_two_state(beta_binomial), 2, TRIALS)
    cases.append(
        (
            beta_objective,
            beta_objective.theta_from_truth(
                INITIAL,
                TRANSITION,
                beta_binomial.alpha.numpy(),
                beta_binomial.beta.numpy(),
            ),
        )
    )

    negative_binomial = NegativeBinomialEmission([2.0, 8.0], [3.0, 20.0])
    count_objective = NegativeBinomialHmmObjective(_two_state(negative_binomial), 2)
    cases.append(
        (
            count_objective,
            count_objective.theta_from_truth(
                INITIAL,
                TRANSITION,
                negative_binomial.dispersion.numpy(),
                negative_binomial.mean.numpy(),
            ),
        )
    )

    weights = np.array([0.35, 0.65])
    components = GaussianEmission([-3.0, 3.0], [1.0, 1.5], 1e-12)
    mixture = MixtureParams(
        weights=weights,
        components=components,
        n_samples=400,
        seed=7,
        tolerance=1e-12,
    )
    mixture_objective = GaussianMixtureObjective(
        simulate_mixture(mixture).observations, 2
    )
    cases.append(
        (
            mixture_objective,
            mixture_objective.theta_from_truth(
                weights, components.mean.numpy(), components.scale.numpy()
            ),
        )
    )
    return cases


EVERY_OBJECTIVE = _every_objective()
OBJECTIVE_IDS = [type(objective).__name__ for objective, _ in EVERY_OBJECTIVE]


@pytest.mark.parametrize(("objective", "theta"), EVERY_OBJECTIVE, ids=OBJECTIVE_IDS)
def test_every_objective_inverts_its_own_constraint_map(
    objective: Objective, theta: torch.Tensor
) -> None:
    # The seam the whole change rests on, and a partial one is worse than
    # none: a caller cannot tell which fits can be given intervals. Realized
    # deviations across the eight objectives are 0 to 4.4e-16 -- the round
    # trip is exact arithmetic, not a tolerance.
    recovered = objective.theta_from(objective.constrain(theta))

    assert_allclose(recovered.numpy(), theta.numpy(), atol=1e-14)


@pytest.mark.parametrize(("objective", "theta"), EVERY_OBJECTIVE, ids=OBJECTIVE_IDS)
def test_the_new_door_is_the_old_one(objective: Objective, theta: torch.Tensor) -> None:
    # `standard_errors_at` must be `constrained_standard_errors` reached
    # another way, not a second implementation of it. Where the information is
    # singular at this point both must refuse, which is as much of a match as
    # a match can be.
    try:
        expected = constrained_standard_errors(objective, theta)
    except ValueError:
        with pytest.raises(ValueError, match="not identifiable from this data"):
            standard_errors_at(objective, objective.constrain(theta))
        return

    realized = standard_errors_at(objective, objective.constrain(theta))

    assert realized.keys() == expected.keys()
    for name, value in expected.items():
        assert_allclose(realized[name].numpy(), value.numpy(), rtol=1e-10)


def test_an_em_fit_and_a_gradient_fit_agree_on_the_interval_at_their_optimum() -> None:
    # **The check this ticket is really for, and it costs nothing.** The two
    # algorithms share the model and nothing else -- no optimizer, no
    # parameterization, no constraint map -- and converge to the same optimum.
    # The Hessian is a property of the objective at a point, so the intervals
    # must agree, and a broken round trip fails this loudly.
    #
    # Realized: the log-likelihoods differ by 3.6e-9 relative, the points by
    # at most 2.4e-3 along the flat ridge EM approaches slowly, and the
    # standard errors by **0.31% relative**. The 1% bound below is the ridge's
    # width and not a tolerance chosen to pass; the 1e-8 on the likelihood is
    # how close two algorithms sharing only the model actually get.
    params = load_hmm_params(HMM_FIXTURE)
    observations = simulate_sequences(params).observations
    objective = HmmObjective(observations, params.n_states, params.n_symbols)

    gradient = fit(objective)
    gradient_errors = standard_errors_at(objective, objective.constrain(gradient.theta))

    start = objective.constrain(objective.initial())
    log_initial, log_transition, log_emission, log_likelihood = baum_welch(
        observations,
        start["log_initial"],
        start["log_transition"],
        start["log_emission"],
    )
    em_errors = standard_errors_at(
        objective,
        {
            "log_initial": log_initial,
            "log_transition": log_transition,
            "log_emission": log_emission,
        },
    )

    assert_allclose(log_likelihood, -float(gradient.value), rtol=1e-8)
    for name, value in gradient_errors.items():
        assert_allclose(em_errors[name].numpy(), value.numpy(), rtol=0.01)


def test_a_collapsing_component_is_refused_through_the_new_door_too() -> None:
    # The refusal is the point. A Gaussian emission's likelihood is unbounded
    # as a variance falls, so near a collapsing component there is no maximum
    # to expand around and the information is not positive definite. Paired
    # with the healthy point, because a guard that refused everything would
    # pass a refusal-only test.
    gaussian = GaussianEmission([-3.0, 3.0], [1.0, 1.0], 1e-12)
    observations = _two_state(gaussian, seed=3)
    objective = GaussianHmmObjective(observations, 2)
    healthy = {
        "log_initial": torch.log(torch.as_tensor(INITIAL)),
        "log_transition": torch.log(torch.as_tensor(TRANSITION)),
        "mean": torch.tensor([-3.0, 3.0], dtype=torch.float64),
        "scale": torch.tensor([1.0, 1.0], dtype=torch.float64),
    }

    errors = standard_errors_at(objective, healthy)
    assert all(bool(torch.isfinite(value).all()) for value in errors.values())

    collapsed = dict(healthy)
    collapsed["mean"] = torch.tensor(
        [float(observations.reshape(-1)[0]), 3.0], dtype=torch.float64
    )
    collapsed["scale"] = torch.tensor(
        [float(np.sqrt(objective.variance_floor)) * 1.01, 1.0], dtype=torch.float64
    )
    with pytest.raises(ValueError, match="not positive definite"):
        standard_errors_at(objective, collapsed)


def test_a_multi_start_interval_belongs_beside_the_spread_that_qualifies_it() -> None:
    # An interval at the best of several starts is conditional on *that mode*.
    # The spread across starts is what says whether that matters, so the two
    # are reported together and this pins that they can be: the interval is
    # taken at `best`, and `spread` is what a reader needs to know it is
    # conditional.
    params = load_hmm_params(HMM_FIXTURE)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )

    result = fit_from(
        objective,
        RandomRestart(3, scale=0.2, rng=np.random.default_rng(4)),
        include_intervals=True,
    )

    assert result.spread >= 0.0
    assert len(result.all_fits) == 3
    # One Hessian, on the fit a caller reads, and none on the starts it does
    # not: an interval per start would be the cost the flag exists to avoid.
    assert result.best.standard_errors is not None
    assert all(one.standard_errors is None for one in result.all_fits)
    expected = constrained_standard_errors(objective, result.best.theta)
    for name, value in expected.items():
        assert torch.equal(result.best.standard_errors[name], value)


def test_a_fit_asked_for_its_interval_gets_the_one_the_door_gives() -> None:
    # `include_intervals` is a convenience over `constrained_standard_errors`,
    # not a second implementation: bitwise the same numbers at the same
    # `theta`. (The `named` door is the same to 1e-10, not bitwise -- its
    # round trip is exact to 4e-16, and `test_the_new_door_is_the_old_one`
    # holds it to that.) And off is off: `None` and no Hessian, so a fit
    # inside a search loop costs what it cost before.
    params = load_hmm_params(HMM_FIXTURE)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )

    plain = fit(objective)
    with_interval = fit(objective, include_intervals=True)

    assert plain.standard_errors is None
    assert torch.equal(plain.theta, with_interval.theta)
    assert with_interval.standard_errors is not None
    expected = constrained_standard_errors(objective, plain.theta)
    assert with_interval.standard_errors.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(with_interval.standard_errors[name], value)


def test_an_unconverged_fit_is_refused_an_interval_but_not_a_result() -> None:
    # The route decides one thing: whether the point is a maximum. A fit the
    # optimizer left early is not, so asking for its interval raises; not
    # asking returns the unconverged fit for inspection exactly as before, so
    # the flag changes no existing behaviour.
    params = load_hmm_params(HMM_FIXTURE)
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )

    early = fit(objective, max_iterations=1)
    assert not early.converged
    assert early.standard_errors is None

    with pytest.raises(ValueError, match="did not converge in 1 iterations"):
        fit(objective, max_iterations=1, include_intervals=True)


def test_where_the_laplace_approximation_is_exact_the_chain_agrees_with_it() -> None:
    # The comparison on the one target where it has an exact answer. A
    # Gaussian's Hessian *is* its precision, so the Laplace interval equals
    # sqrt(diag(covariance)) to round-off and is asserted so; the chain then
    # has nothing to be approximate about, and its spread must match to Monte
    # Carlo error. Realized ratios 1.0004 and 0.9953 at 4000 draws; the 5%
    # bound is 2.7 times the largest deviation seen over three seeds, and the
    # Potts case below is where the two are *allowed* to differ.
    from snakes_and_ladders.opt.hmc import sample

    target = AnalyticGaussian([1.0, -2.0], [[2.0, 0.6], [0.6, 0.5]])
    exact = target.covariance.diagonal().sqrt()

    laplace = fit(target, include_intervals=True)
    assert laplace.standard_errors is not None
    assert_allclose(laplace.standard_errors["x"].numpy(), exact.numpy(), rtol=1e-8)

    chain = sample(
        target, seed=3, n_samples=4000, step_size=0.2, n_steps=10, burn_in=200
    )
    assert_allclose(chain.theta.std(0).numpy(), exact.numpy(), rtol=0.05)


def test_the_delta_method_interval_and_the_sampled_posterior_agree() -> None:
    # The comparison `hmc.py`'s docstring promises, in the regime where the
    # approximation is allowed to be one. A version already existed -- a raw
    # Hessian in *unconstrained* coordinates against grid quadrature, at
    # rtol=0.15. This is the missing half: the delta-method interval on the
    # parameters a person names, against a chain. The exact regime is the
    # analytic Gaussian above; here the deviation is the finding.
    #
    # Realized: the sampled spread is 1.057, 1.031 and 1.036 times the Laplace
    # one across the three parameters. The Laplace approximation is slightly
    # *optimistic* here, which is the expected direction for a mildly
    # non-Gaussian posterior and is reported rather than asserted away.
    from snakes_and_ladders.opt.hmc import WithGaussianPrior, sample

    field = np.array([0.3, -0.3])
    field = field - np.log(np.exp(field).sum())
    params = PottsParams(
        n_states=2, chain_length=8, n_chains=300, coupling=0.75, field=field, seed=7
    )
    objective = PottsObjective(simulate_chains(params), 2)

    laplace = standard_errors_at(objective, objective.constrain(fit(objective).theta))
    chain = sample(
        WithGaussianPrior(objective, scale=2.0),
        seed=11,
        n_samples=2000,
        step_size=0.05,
        n_steps=20,
        burn_in=400,
    )

    constrained = [objective.constrain(draw) for draw in chain.theta]
    coupling = torch.tensor([float(one["coupling"]) for one in constrained])
    sampled = torch.cat(
        [
            coupling.std().reshape(1),
            torch.stack([one["field"] for one in constrained]).std(0),
        ]
    )
    approximate = torch.cat([laplace["coupling"].reshape(1), laplace["field"]])

    ratio = (sampled / approximate).numpy()
    assert_allclose(ratio, np.ones(3), rtol=0.15)
    assert bool((ratio > 1.0).all())


@pytest.mark.release
def test_the_intervals_from_an_em_fit_cover_truth_at_the_nominal_rate() -> None:
    # An interval that exists and does not cover is worse than no interval, so
    # the EM path is held to exactly the standard the gradient path is.
    # Realized over 12 replicates: 243/264 = 0.920 cover, with 1 of 12
    # reaching the boundary of the parameter space and contributing none --
    # counted rather than dropped, since excluding them unannounced would
    # select for the well-behaved samples.
    base = load_hmm_params(HMM_FIXTURE)
    truth = {
        "log_initial": torch.log(torch.as_tensor(base.initial)),
        "log_transition": torch.log(torch.as_tensor(base.transition)),
        "log_emission": torch.log(torch.as_tensor(base.emission)),
    }
    covered = total = boundary = 0
    for replicate in range(12):
        params = replace(base, seed=base.seed + 7919 * replicate)
        observations = simulate_sequences(params).observations
        objective = HmmObjective(observations, params.n_states, params.n_symbols)
        start = objective.constrain(objective.initial())
        log_initial, log_transition, log_emission, _ = baum_welch(
            observations,
            start["log_initial"],
            start["log_transition"],
            start["log_emission"],
        )
        named = {
            "log_initial": log_initial,
            "log_transition": log_transition,
            "log_emission": log_emission,
        }
        try:
            errors = standard_errors_at(objective, named)
        except ValueError:
            boundary += 1
            continue
        order = list(align_states(log_emission, torch.as_tensor(params.emission)))
        for name, reference in truth.items():
            point, spread = named[name], errors[name]
            if name == "log_transition":
                point, spread = point[order][:, order], spread[order][:, order]
            else:
                point, spread = point[order], spread[order]
            hits = covers(point, spread, reference)
            covered += int(hits.sum())
            total += hits.numel()

    assert total > 0
    assert covered / total >= 0.85
    assert boundary <= 3


def test_the_categorical_family_is_still_what_the_fixture_declares() -> None:
    # Guards the fixture the module leans on: every case above assumes the
    # committed HMM fixture is categorical, and a fixture that changed family
    # would make eight round-trip checks silently test something else.
    params = load_hmm_params(HMM_FIXTURE)

    assert isinstance(params.emissions, CategoricalEmission)
