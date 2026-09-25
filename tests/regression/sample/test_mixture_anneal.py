"""Simulated annealing over the count-pair mixture's assignments (issue #901).

`sample.mixture_anneal.anneal_assignments` draws every observation's component
from its tempered responsibilities and re-estimates the components at the
draw. The referees are the heat bath's two limits, read on the draw itself:
at `T -> 0` a sweep is the argmax of the responsibilities, and at `T = 1` its
frequencies are the responsibilities. Then the run on `emission_mixture/ci`:
its best state is at least where it started, EM from it never lowers the
likelihood, and one seed reproduces it bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.emissions import CountPairEmission
from sal.opt.emission_mixture import (
    CountPairSeeding,
    expectation_maximization,
    uniform_start,
)
from sal.opt.mixture import mixture_log_likelihood, responsibilities
from sal.sample.mixture_anneal import anneal_assignments, draw_assignments
from sal.search.mixture_starts import (
    STARTS,
    gibbs_schedule,
    instance_from,
)
from sal.sim.emission_mixture import simulate_emission_mixture
from sal.sim.fixtures import fixture
from scipy.stats import norm

#: Draws the frequency test counts.
DRAWS = 2_000

#: The frequency bound, in binomial standard errors: 2,700 (pair, component)
#: cells each at 3 would exceed it about 7 times by chance, so the bound is
#: the Bonferroni quantile that holds all of them jointly at 1%.
CELLS = 900 * 3
SIGMAS = float(norm.ppf(1.0 - 0.01 / (2 * CELLS)))


def _draw() -> tuple[np.ndarray, torch.Tensor, CountPairSeeding, CountPairEmission]:
    """The ci draw, its generating joint log-density, the seam and its data start."""
    params = fixture("emission_mixture", "ci").params
    truth = params.components
    assert isinstance(truth, CountPairEmission)
    observations = simulate_emission_mixture(params).observations.astype(float)
    at = CountPairSeeding(
        float(truth.total.dispersion.mean()),
        float(truth.concentration.mean()),
        joint=True,
    )
    joint = torch.log(
        torch.as_tensor(params.weights, dtype=torch.float64)
    ) + truth.log_density(torch.as_tensor(observations, dtype=torch.float64))
    start = uniform_start(observations, 3, at, np.random.default_rng(0))
    assert isinstance(start, CountPairEmission)
    return observations, joint, at, start


@pytest.mark.analytic
def test_a_cold_sweep_is_the_argmax_of_the_responsibilities() -> None:
    _, joint, _, _ = _draw()
    drawn = draw_assignments(joint, 1e-12, np.random.default_rng(0))
    assert np.array_equal(drawn, joint.argmax(dim=1).numpy())


@pytest.mark.oracle
def test_a_sweep_at_one_draws_each_pair_from_its_responsibilities() -> None:
    # The referee is the posterior itself, computed by the one normalization
    # the E step uses and not by the sweep's inverse-CDF draw.
    observations, joint, _, _ = _draw()
    rng = np.random.default_rng(1)
    counts = np.zeros(tuple(joint.shape))
    rows = np.arange(joint.shape[0])
    for _ in range(DRAWS):
        np.add.at(counts, (rows, draw_assignments(joint, 1.0, rng)), 1.0)
    frequencies = counts / DRAWS
    posterior = torch.softmax(joint, dim=-1).numpy()
    error = np.sqrt(posterior * (1.0 - posterior) / DRAWS)
    # A responsibility of exactly 0 or 1 has no spread: the draw must match.
    exact = error == 0.0
    assert np.array_equal(frequencies[exact], posterior[exact])
    within = np.abs(frequencies - posterior) <= SIGMAS * np.where(exact, 1.0, error)
    assert bool(within.all()), int((~within).sum())


@pytest.mark.end2end
def test_the_anneal_keeps_its_best_state_and_em_from_it_never_falls() -> None:
    observations, _, _, start = _draw()
    values = torch.as_tensor(observations, dtype=torch.float64)
    weights = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
    begun = float(mixture_log_likelihood(values, torch.log(weights), start))
    run = anneal_assignments(
        observations, weights, start, gibbs_schedule(), np.random.default_rng(0)
    )
    assert run.temperatures == gibbs_schedule()
    assert len(run.log_likelihoods) == len(run.path) == len(run.temperatures)
    # The best state is the best of the trace, and at least the start.
    assert run.log_likelihood == max(run.log_likelihoods)
    assert run.log_likelihood == run.log_likelihoods[run.best_step]
    assert run.log_likelihood >= begun
    assert run.log_likelihood == pytest.approx(
        float(mixture_log_likelihood(values, torch.log(run.weights), run.components)),
        rel=1e-12,
    )
    polished = expectation_maximization(
        observations, run.weights, run.components, tolerance=1e-8
    )
    assert polished.log_likelihood >= run.log_likelihood
    # And each E step's posterior sums to one, as EM's does.
    posterior = responsibilities(values, torch.log(run.weights), run.components)
    assert float((posterior.sum(dim=1) - 1.0).abs().max()) < 1e-12


@pytest.mark.smoke
def test_the_start_reads_its_generator_and_reproduces_on_a_seed() -> None:
    params = fixture("emission_mixture", "ci").params
    truth = params.components
    assert isinstance(truth, CountPairEmission)
    instance = instance_from(
        simulate_emission_mixture(params),
        CountPairSeeding(
            float(truth.total.dispersion.mean()),
            float(truth.concentration.mean()),
            joint=True,
        ),
    )
    first = STARTS["gibbs-anneal"](instance, np.random.default_rng(3))
    again = STARTS["gibbs-anneal"](instance, np.random.default_rng(3))
    other = STARTS["gibbs-anneal"](instance, np.random.default_rng(4))
    same = [
        torch.equal(a, b)
        for a, b in zip(
            first.components.named_parameters().values(),
            again.components.named_parameters().values(),
            strict=True,
        )
    ]
    assert all(same)
    assert not all(
        torch.equal(a, b)
        for a, b in zip(
            first.components.named_parameters().values(),
            other.components.named_parameters().values(),
            strict=True,
        )
    )
    assert first.passes == len(gibbs_schedule()) + 1
    assert [step for step, _ in first.path] == list(range(len(gibbs_schedule())))


@pytest.mark.smoke
def test_an_anneal_refuses_an_empty_or_nonpositive_schedule() -> None:
    observations, _, _, start = _draw()
    weights = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least one"):
        anneal_assignments(observations, weights, start, [], np.random.default_rng(0))
    with pytest.raises(ValueError, match="positive and finite"):
        anneal_assignments(
            observations, weights, start, [1.0, 0.0], np.random.default_rng(0)
        )
