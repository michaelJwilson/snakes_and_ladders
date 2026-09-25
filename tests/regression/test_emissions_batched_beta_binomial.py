"""The batched beta-binomial M step against the per-component solve it replaced (issue #892).

`mstep.solve_beta_binomial_batched` runs every component's alternating
bisection at once, on each channel's distinct values weighted by the
responsibility summed there. `_solve_beta_binomial`, one component at a time
over every observation, is kept as the oracle. The sums are reordered, so
bitwise is the target and #648's 2e-06 the declared floor (that issue measured a
reordered sum moving a fitted `alpha` by 1.0e-06 on a flat likelihood). Every
draw below is measured bitwise. Both call paths are pinned: the joint
count pair, whose trial count is each observation's depth, and a family with
one trial count per component, the projection's success channel.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.emissions import BetaBinomialEmission, CountPairEmission, mstep
from sal.opt.mixture import responsibilities
from sal.search.projection import flatten, project
from sal.sim.count_pairs import binned_model
from sal.sim.emission_mixture import simulate_emission_mixture
from sal.sim.fixtures import fixture


def _oracle(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: torch.Tensor | list[float],
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> list[mstep.SolvedBetaBinomial]:
    """The per-component solve, component by component, as the M step ran before #892."""
    solved = []
    for state in range(weights.shape[1]):
        total = float(alpha[state] + beta[state])
        solved.append(
            mstep.solve_beta_binomial(
                values,
                weights[:, state],
                trials if isinstance(trials, torch.Tensor) else trials[state],
                float(alpha[state]) / total,
                total,
            )
        )
    return solved


def _batched(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: torch.Tensor | list[float],
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> list[mstep.SolvedBetaBinomial]:
    totals = [float(a + b) for a, b in zip(alpha, beta, strict=True)]
    return mstep.solve_beta_binomial_batched(
        values,
        weights,
        trials,
        [float(a) / t for a, t in zip(alpha, totals, strict=True)],
        totals,
    )


def _joint(tier: str) -> tuple[torch.Tensor, torch.Tensor, CountPairEmission]:
    """A draw of the emission mixture and its responsibilities at the generating parameters."""
    params = fixture("emission_mixture", tier).params
    truth = params.components
    assert isinstance(truth, CountPairEmission)
    values = torch.as_tensor(
        simulate_emission_mixture(params).observations, dtype=torch.float64
    )
    posterior = responsibilities(
        values, torch.log(torch.as_tensor(params.weights, dtype=torch.float64)), truth
    )
    return values, posterior, truth


@pytest.mark.smoke
@pytest.mark.backend
@pytest.mark.parametrize("tier", ["ci", "stress"])
def test_the_joint_pairs_m_step_is_the_per_component_solve_bitwise(tier: str) -> None:
    # K = 3 over 900 pairs and K = 10 over 3,000, each observation's depth its
    # trial count. The components stop at different outer iterations on the
    # stress draw (6 to 8), so a component carried unchanged while the others
    # move is part of what is pinned.
    values, posterior, truth = _joint(tier)
    alpha, beta = truth.named_parameters()["alpha"], truth.named_parameters()["beta"]
    successes, totals = values[:, 1], values[:, 0]
    oracle = _oracle(successes, posterior, totals, alpha, beta)
    batched = _batched(successes, posterior, totals, alpha, beta)
    assert batched == oracle
    if tier == "stress":
        assert len({one.iterations for one in oracle}) > 1
    # And through the family's own M step, which is what EM calls.
    fitted = truth.reestimate(values, posterior)
    assert fitted.converged
    assert torch.equal(
        fitted.emissions.named_parameters()["alpha"],
        torch.tensor([one.alpha for one in oracle], dtype=torch.float64),
    )


def _projected(tier: str) -> tuple[torch.Tensor, torch.Tensor, BetaBinomialEmission]:
    """The projection's success channel: one trial count per component."""
    params = binned_model(fixture("spatio_sequential_counts", tier).params.model, 1)
    instance = project(params, 400, np.random.default_rng([892, 1]))
    truth = flatten(params)
    values = torch.as_tensor(instance.observations, dtype=torch.float64)
    k = truth.n_states
    posterior = responsibilities(
        values, torch.full((k,), -float(np.log(k)), dtype=torch.float64), truth
    )
    channel = truth._successes
    assert isinstance(channel, BetaBinomialEmission)
    return values[:, 1], posterior, channel


@pytest.mark.smoke
@pytest.mark.backend
def test_a_trial_count_per_component_is_the_per_component_solve_bitwise() -> None:
    values, posterior, channel = _projected("ci")
    trials = [float(t) for t in channel.trials]
    alpha = channel.named_parameters()["alpha"]
    beta = channel.named_parameters()["beta"]
    oracle = _oracle(values, posterior, trials, alpha, beta)
    assert _batched(values, posterior, trials, alpha, beta) == oracle
    fitted = channel.reestimate(values, posterior).emissions
    assert torch.equal(
        fitted.named_parameters()["beta"],
        torch.tensor([one.beta for one in oracle], dtype=torch.float64),
    )


@pytest.mark.smoke
def test_a_histogram_is_the_weights_summed_at_each_distinct_value() -> None:
    values = torch.tensor([3.0, 1.0, 3.0, 0.0, 1.0, 3.0], dtype=torch.float64)
    columns = torch.arange(12, dtype=torch.float64).reshape(2, 6)
    distinct, summed = mstep.weighted_histogram(values, columns)
    assert distinct.tolist() == [0.0, 1.0, 3.0]
    assert summed.tolist() == [
        [3.0, 1.0 + 4.0, 0.0 + 2.0 + 5.0],
        [9.0, 7.0 + 10.0, 6.0 + 8.0 + 11.0],
    ]
