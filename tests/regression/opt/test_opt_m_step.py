"""The M-step hook on both EM drivers, and the L-BFGS M step that fills it (issue #1171).

Referees: on a closed-form family `LbfgsMStep` reaches the family's
`reestimate` within 1e-8, and on the negative binomial, whose `reestimate` is
its own solve, the two agree; an `MStep` never lowers the expected
complete-data log-likelihood it is handed; a parameter held is held, and the
rest reach their closed form; the family's own `reestimate` handed in as the
hook is each driver's default bitwise; and a fit on the L-BFGS M step reaches
the closed-form fit's maximum. `stages` defaults to the one stage a plain fit
runs, and a record that contradicts its termination or its cost is refused.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from sal.emissions import (
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
    Reestimate,
)
from sal.opt.em import EM, EMISSION_MIXTURE_EM
from sal.opt.emission_mixture import expectation_maximization
from sal.opt.hmm import EmFit, baum_welch_family
from sal.opt.m_step import LbfgsMStep, MStep, expected_complete_log_likelihood
from sal.opt.termination import Termination
from sal.ragged import Ragged

LENGTHS = (40, 25, 60, 15)


def _posterior(rng: np.random.Generator, k: int) -> torch.Tensor:
    """A padded HMM posterior: Dirichlet rows over the live positions, zero after each segment's end."""
    longest = max(LENGTHS)
    posterior = rng.dirichlet(np.ones(k), size=(len(LENGTHS), longest))
    for row, length in enumerate(LENGTHS):
        posterior[row, length:] = 0.0
    return torch.as_tensor(posterior)


def _cases() -> list[tuple[str, EmissionFamily, torch.Tensor]]:
    """A Gaussian, a Poisson and a categorical family, each with padded observations."""
    rng = np.random.default_rng(1171)
    shape = (len(LENGTHS), max(LENGTHS))
    return [
        (
            "gaussian",
            GaussianEmission(
                np.array([-1.0, 0.0, 2.0]), np.ones(3), variance_floor=1e-6
            ),
            torch.as_tensor(rng.normal(size=shape)),
        ),
        (
            "poisson",
            PoissonEmission(np.array([1.0, 3.0, 8.0])),
            torch.as_tensor(rng.poisson(4.0, size=shape), dtype=torch.float64),
        ),
        (
            "categorical",
            CategoricalEmission.from_log(
                torch.log(torch.as_tensor(rng.dirichlet(np.ones(4), size=3)))
            ),
            torch.as_tensor(rng.integers(0, 4, size=shape)),
        ),
    ]


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("case", range(3), ids=["gaussian", "poisson", "categorical"])
def test_the_lbfgs_m_step_reaches_the_closed_form_m_step(case: int) -> None:
    """Measured: within 4.9e-11 of every closed-form parameter (Gaussian 2.2e-16, categorical 1.4e-11)."""
    _, family, observations = _cases()[case]
    posterior = _posterior(np.random.default_rng(case), family.n_states)
    closed = family.reestimate(observations, posterior)
    solved = LbfgsMStep()(family, observations, posterior, None)
    assert solved.converged
    for name, value in closed.emissions.named_parameters().items():
        np.testing.assert_allclose(
            solved.emissions.named_parameters()[name].numpy(),
            value.numpy(),
            atol=1e-8,
            rtol=0.0,
            err_msg=name,
        )


@pytest.mark.oracle
def test_the_lbfgs_m_step_agrees_with_the_negative_binomial_solve() -> None:
    """Two solves of one weighted likelihood: the family's dispersion solve and L-BFGS over its domains."""
    rng = np.random.default_rng(7)
    shape = (len(LENGTHS), max(LENGTHS))
    observations = torch.as_tensor(
        rng.negative_binomial(5, 0.4, size=shape), dtype=torch.float64
    )
    family = NegativeBinomialEmission(np.array([2.0, 2.0]), np.array([3.0, 12.0]))
    posterior = _posterior(rng, 2)
    closed = family.reestimate(observations, posterior)
    solved = LbfgsMStep()(family, observations, posterior, None)
    assert closed.converged
    assert solved.converged
    q_closed = expected_complete_log_likelihood(
        closed.emissions, observations, posterior
    )
    q_solved = expected_complete_log_likelihood(
        solved.emissions, observations, posterior
    )
    assert q_solved >= q_closed - 1e-10 * abs(q_closed)
    for name, value in closed.emissions.named_parameters().items():
        np.testing.assert_allclose(
            solved.emissions.named_parameters()[name].numpy(),
            value.numpy(),
            rtol=1e-6,
            err_msg=name,
        )


@pytest.mark.critical
@pytest.mark.analytic
def test_an_lbfgs_m_step_never_lowers_the_expected_complete_log_likelihood() -> None:
    """Twenty random posteriors per family, from a family far from the maximum and from one at it."""
    rng = np.random.default_rng(11)
    step = LbfgsMStep()
    for _, family, observations in _cases():
        for _ in range(20):
            posterior = _posterior(rng, family.n_states)
            before = expected_complete_log_likelihood(family, observations, posterior)
            solved = step(family, observations, posterior, None).emissions
            after = expected_complete_log_likelihood(solved, observations, posterior)
            assert after >= before
            # From the maximum it does not move off it.
            again = step(solved, observations, posterior, None).emissions
            assert expected_complete_log_likelihood(
                again, observations, posterior
            ) >= after - 1e-12 * abs(after)


@pytest.mark.oracle
def test_a_held_scale_is_held_and_the_mean_reaches_its_closed_form() -> None:
    # The weighted mean maximizes Q at any scale, so holding the scale leaves
    # the mean's closed form unchanged.
    _, family, observations = _cases()[0]
    posterior = _posterior(np.random.default_rng(3), 3)
    held = LbfgsMStep(held=("scale",))(family, observations, posterior, None)
    assert isinstance(family, GaussianEmission)
    assert isinstance(held.emissions, GaussianEmission)
    assert torch.equal(held.emissions.scale, family.scale)
    closed = family.reestimate(observations, posterior).emissions
    assert isinstance(closed, GaussianEmission)
    np.testing.assert_allclose(held.emissions.mean, closed.mean, atol=1e-8, rtol=0)
    with pytest.raises(ValueError, match="not parameters"):
        LbfgsMStep(held=("variance",))(family, observations, posterior, None)
    everything = LbfgsMStep(held=("mean", "scale"))(
        family, observations, posterior, None
    )
    assert everything.emissions is family
    assert everything.iterations == 0


def _family_own(
    components: EmissionFamily,
    observations: torch.Tensor,
    posterior: torch.Tensor,
    covariate: torch.Tensor | None,
) -> Reestimate[EmissionFamily]:
    """The family's own M step, through the hook."""
    if covariate is None:
        return components.reestimate(observations, posterior)
    return components.reestimate(observations, posterior, covariate=covariate)


@pytest.mark.critical
@pytest.mark.smoke
@pytest.mark.patch
def test_the_family_m_step_as_a_hook_is_each_default_bitwise() -> None:
    rng = np.random.default_rng(5)
    hook: MStep = _family_own
    # Baum-Welch on a ragged batch, which takes the general route either way.
    observations = Ragged(rng.normal(size=sum(LENGTHS)), LENGTHS)
    family = GaussianEmission(
        np.array([-1.0, 0.0, 1.0]), np.ones(3), variance_floor=1e-6
    )
    log_initial = torch.log(torch.full((3,), 1.0 / 3.0, dtype=torch.float64))
    log_transition = torch.log(torch.as_tensor(rng.dirichlet(np.full(3, 5.0), size=3)))
    config = replace(EM, max_iterations=30)
    default = baum_welch_family(
        observations, log_initial, log_transition, family, config
    )
    hooked = baum_welch_family(
        observations, log_initial, log_transition, family, config, m_step=hook
    )
    assert default.log_likelihood == hooked.log_likelihood
    assert torch.equal(default.log_transition, hooked.log_transition)
    assert torch.equal(default.components.mean, hooked.components.mean)  # type: ignore[attr-defined]
    assert default.termination == hooked.termination
    assert default.stages == hooked.stages == (default.termination,)
    # The mixture on float counts, the per-observation route either way.
    counts = rng.poisson(np.repeat([2.0, 9.0], 150)).astype(float)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    start = PoissonEmission(np.array([1.0, 5.0]))
    plain = expectation_maximization(counts, weights, start)
    through = expectation_maximization(counts, weights, start, m_step=hook)
    assert plain.log_likelihood == through.log_likelihood
    assert torch.equal(plain.weights, through.weights)
    assert torch.equal(plain.responsibilities, through.responsibilities)
    assert plain.spent == through.spent


@pytest.mark.end2end
def test_a_fit_on_the_lbfgs_m_step_reaches_the_closed_form_fit() -> None:
    """Baum-Welch and the mixture, 40 iterations each: one maximum, to 1e-9 relative."""
    rng = np.random.default_rng(13)
    observations = Ragged(
        np.concatenate([rng.normal(-1.5, 0.7, 60), rng.normal(1.5, 0.7, 80)]),
        (60, 80),
    )
    family = GaussianEmission(np.array([-0.5, 0.5]), np.ones(2), variance_floor=1e-6)
    log_initial = torch.log(torch.full((2,), 0.5, dtype=torch.float64))
    log_transition = torch.log(
        torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)
    )
    config = replace(EM, max_iterations=40)
    closed = baum_welch_family(
        observations, log_initial, log_transition, family, config
    )
    solved = baum_welch_family(
        observations, log_initial, log_transition, family, config, m_step=LbfgsMStep()
    )
    assert solved.log_likelihood == pytest.approx(closed.log_likelihood, rel=1e-9)
    counts = rng.poisson(np.repeat([2.0, 9.0], 150)).astype(float)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    start = PoissonEmission(np.array([1.0, 5.0]))
    budget = replace(EMISSION_MIXTURE_EM, max_iterations=40)
    plain = expectation_maximization(counts, weights, start, budget)
    through = expectation_maximization(
        counts, weights, start, budget, m_step=LbfgsMStep()
    )
    assert through.log_likelihood == pytest.approx(plain.log_likelihood, rel=1e-9)


def _fit(
    termination: Termination, spent: int, stages: tuple[Termination, ...] = ()
) -> EmFit:
    """An `EmFit` whose parameters are placeholders: only its accounting is read."""
    return EmFit(
        torch.zeros(2),
        torch.zeros(2, 2),
        PoissonEmission(np.array([1.0, 2.0])),
        -1.0,
        termination=termination,
        spent=spent,
        stages=stages,
    )


@pytest.mark.critical
@pytest.mark.smoke
def test_stages_default_to_the_termination_and_a_contradiction_is_refused() -> None:
    done = Termination.after(3, converged=True)
    assert _fit(done, 3).stages == (done,)
    tempered = Termination.after(1, converged=False)
    assert _fit(done, 5, (tempered, tempered, done)).stages[-1] is done
    with pytest.raises(ValueError, match="is not the termination"):
        _fit(done, 4, (done, tempered))
    with pytest.raises(ValueError, match="spent 4"):
        _fit(done, 4)
