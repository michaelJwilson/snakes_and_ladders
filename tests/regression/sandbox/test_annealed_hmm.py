"""Deterministic annealing Baum-Welch, conserved in the sandbox (issue #1171).

One E and M step per temperature on forward-backward over ``log_density /
T``, ``log_initial / T`` and ``log_transition / T``, then plain EM. Referees:
the tempered E step is the tempered joint's posterior by enumeration over
every path, and at ``T = 1`` it is `ragged.posteriors` bitwise; an empty
schedule is `baum_welch_family` bitwise; the log tempered evidence does not
fall at a fixed temperature (Ueda & Nakano, 1998). The release experiment is
the gate #1171 set for shipping a schedule in `opt`, and records why it does
not ship.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest
import torch
from sal.emissions import GaussianEmission
from sal.likelihood.ragged import posteriors
from sal.opt.em import EM, EmConfig
from sal.opt.hmm import EmFit, baum_welch_family
from sal.ragged import Ragged
from sal.sample.schedule import ExponentialTempSchedule
from sal.sandbox.annealed_em import TemperedEStep, annealed_baum_welch
from sal.sim.hmm import HmmParams, simulate_sequences

LENGTHS = (5, 3, 4)


def _problem() -> tuple[Ragged, torch.Tensor, torch.Tensor, GaussianEmission]:
    """Three short segments over three Gaussian states, small enough to enumerate."""
    rng = np.random.default_rng(1171)
    observations = Ragged(rng.normal(size=sum(LENGTHS)), LENGTHS)
    log_initial = torch.log(torch.as_tensor(rng.dirichlet(np.ones(3))))
    log_transition = torch.log(torch.as_tensor(rng.dirichlet(np.full(3, 2.0), size=3)))
    family = GaussianEmission(
        np.array([-1.0, 0.0, 1.0]), np.full(3, 0.8), variance_floor=1e-6
    )
    return observations, log_initial, log_transition, family


def _log_density(observations: Ragged, family: GaussianEmission) -> Ragged:
    """The family's scores per position, as the E step is handed them."""
    values = torch.as_tensor(observations.values, dtype=torch.float64)
    return Ragged(family.log_density(values).numpy(), observations.lengths)


def _same(first: EmFit, second: EmFit) -> bool:
    """Every returned tensor and scalar equal bitwise."""
    return (
        torch.equal(first.log_initial, second.log_initial)
        and torch.equal(first.log_transition, second.log_transition)
        and all(
            torch.equal(a, b)
            for a, b in zip(
                first.components.named_parameters().values(),
                second.components.named_parameters().values(),
                strict=True,
            )
        )
        and first.log_likelihood == second.log_likelihood
        and first.spent == second.spent
        and first.stages == second.stages
    )


@pytest.mark.critical
@pytest.mark.oracle
def test_the_tempered_e_step_is_the_tempered_joint_by_enumeration() -> None:
    """Every path of every segment scored at ``p(x, z)^(1/T)``: marginals, pair counts and evidence."""
    observations, log_initial, log_transition, family = _problem()
    scores = _log_density(observations, family)
    initial, transition = log_initial.numpy(), log_transition.numpy()
    for temperature in (0.5, 1.0, 3.0):
        tempered = TemperedEStep(temperature)(scores, initial, transition)
        gamma = np.zeros_like(scores.values)
        counts = np.zeros((3, 3))
        evidence = np.zeros(len(LENGTHS))
        offset = 0
        for segment, length in enumerate(LENGTHS):
            rows = scores.values[offset : offset + length]
            paths = list(itertools.product(range(3), repeat=length))
            joint = np.array(
                [
                    (
                        initial[path[0]]
                        + sum(transition[a, b] for a, b in itertools.pairwise(path))
                        + sum(rows[t, s] for t, s in enumerate(path))
                    )
                    / temperature
                    for path in paths
                ]
            )
            evidence[segment] = np.logaddexp.reduce(joint)
            weight = np.exp(joint - evidence[segment])
            for path, w in zip(paths, weight, strict=True):
                for t, s in enumerate(path):
                    gamma[offset + t, s] += w
                for a, b in itertools.pairwise(path):
                    counts[a, b] += w
            offset += length
        np.testing.assert_allclose(np.exp(tempered.log_posterior), gamma, atol=1e-12)
        np.testing.assert_allclose(np.exp(tempered.log_counts), counts, atol=1e-12)
        np.testing.assert_allclose(tempered.log_evidence, evidence, rtol=1e-12)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_the_tempered_e_step_at_one_is_exact_forward_backward_bitwise() -> None:
    observations, log_initial, log_transition, family = _problem()
    scores = _log_density(observations, family)
    initial, transition = log_initial.numpy(), log_transition.numpy()
    tempered = TemperedEStep(1.0)(scores, initial, transition)
    exact = posteriors(scores, initial, transition)
    np.testing.assert_array_equal(tempered.log_posterior, exact.log_posterior)
    np.testing.assert_array_equal(tempered.log_counts, exact.log_counts)
    np.testing.assert_array_equal(tempered.log_evidence, exact.log_evidence)


@pytest.mark.critical
@pytest.mark.smoke
@pytest.mark.patch
def test_no_schedule_is_baum_welch_bitwise_and_a_schedule_is_staged() -> None:
    observations, log_initial, log_transition, family = _problem()
    config = replace(EM, max_iterations=60)
    plain = baum_welch_family(observations, log_initial, log_transition, family, config)
    unset = annealed_baum_welch(
        observations, log_initial, log_transition, family, [], config
    )
    assert _same(plain, unset.fit)
    assert unset.fit.stages == (plain.termination,)
    staged = annealed_baum_welch(
        observations, log_initial, log_transition, family, [2.0, 1.5, 1.0], config
    ).fit
    # One stage per tempered step, of one iteration ending on its count, then
    # plain EM's; the budget is shared and `spent` is the total.
    assert len(staged.stages) == 4
    assert all(
        stage.iterations == 1 and not stage.converged for stage in staged.stages[:3]
    )
    assert staged.termination == staged.stages[-1]
    assert staged.spent == 3 + staged.termination.iterations <= 60


@pytest.mark.analytic
def test_the_log_tempered_evidence_does_not_fall_at_a_fixed_temperature() -> None:
    """Ten steps at each of 3, 2 and 1.5: ``T log Z_T`` non-decreasing within each block."""
    observations, log_initial, log_transition, family = _problem()
    steps = [3.0] * 10 + [2.0] * 10 + [1.5] * 10
    run = annealed_baum_welch(
        observations,
        log_initial,
        log_transition,
        family,
        steps,
        replace(EM, max_iterations=len(steps)),
    )
    assert run.temperatures == tuple(steps)
    values = np.asarray(run.log_tempered_evidences)
    for block in range(3):
        block_values = values[10 * block : 10 * (block + 1)]
        rises = np.diff(block_values)
        assert bool((rises >= -1e-12 * np.abs(block_values[1:])).all()), (block, rises)


@pytest.mark.smoke
def test_a_temperature_is_positive_and_the_schedule_fits_the_budget() -> None:
    observations, log_initial, log_transition, family = _problem()
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive and finite"):
            annealed_baum_welch(
                observations, log_initial, log_transition, family, [bad]
            )
    with pytest.raises(ValueError, match="do not fit"):
        annealed_baum_welch(
            observations,
            log_initial,
            log_transition,
            family,
            [2.0, 1.0],
            replace(EM, max_iterations=1),
        )


#: The gate's fixture: six Gaussian states one unit apart at scale 0.5, sticky
#: transitions, 40 chains of 250. Plain EM reaches several maxima from random
#: starts, which is the multimodality the gate asks a schedule to resolve.
N_STATES = 6

#: The schedules read: the mixture's (#903), and a mild one from 1.2.
SCHEDULES = {
    "from 8, 20 steps": ExponentialTempSchedule(8.0, 1.0, 20),
    "from 1.2, 30 steps": ExponentialTempSchedule(1.2, 1.0, 30),
}


def _multimodal() -> np.ndarray:
    transition = np.full((N_STATES, N_STATES), 0.1 / (N_STATES - 1))
    np.fill_diagonal(transition, 0.9)
    params = HmmParams(
        n_states=N_STATES,
        lengths=(250,) * 40,
        initial=np.full(N_STATES, 1.0 / N_STATES),
        transition=transition,
        emissions=GaussianEmission(
            np.arange(N_STATES, dtype=float),
            np.full(N_STATES, 0.5),
            variance_floor=1e-6,
        ),
        seed=1171,
        tolerance=0.0,
    )
    observations: np.ndarray = simulate_sequences(params).observations
    return observations


@pytest.mark.release
@pytest.mark.experiment
def test_a_schedule_does_not_reach_further_than_plain_em_over_ten_starts() -> None:
    """The #1171 gate: annealed against plain EM from ten random starts, same budget.

    Measured, final log-likelihood annealed minus plain, mean +- SE over the
    ten starts, wins and losses beyond 1e-8 relative:

    - from 8, 20 steps: -8,350.9 +- 111.6, 0 wins, 10 losses. Every state
      merges into one at T = 8, a symmetric point EM never leaves
      (`opt/CLAUDE.md`), and the fit ends at -20,034.064.
    - from 1.2, 30 steps: +134.3 +- 90.3, 2 wins, 2 losses: 1.5 SE, and as
      many starts lost as won.

    Plain EM reaches the best maximum, -11,411.632, from 5 of 10 starts. So
    no schedule clears the gate --- a higher maximum than plain EM, beyond
    two standard errors and on more starts than it loses --- and `schedule`
    stays here rather than in `baum_welch_family`. Asserted: neither clears
    it.
    """
    observations = _multimodal()
    config = EmConfig(max_iterations=3000, tolerance=1e-10)
    plain: list[float] = []
    starts = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        means = np.sort(rng.uniform(observations.min(), observations.max(), N_STATES))
        family = GaussianEmission(
            means, np.full(N_STATES, observations.std()), variance_floor=1e-6
        )
        log_initial = torch.log(
            torch.full((N_STATES,), 1.0 / N_STATES, dtype=torch.float64)
        )
        log_transition = torch.log(
            torch.as_tensor(rng.dirichlet(np.full(N_STATES, 5.0), size=N_STATES))
        )
        starts.append((log_initial, log_transition, family))
        plain.append(
            baum_welch_family(
                observations, log_initial, log_transition, family, config
            ).log_likelihood
        )
    for name, schedule in SCHEDULES.items():
        annealed = [
            annealed_baum_welch(
                observations, *start, schedule, config
            ).fit.log_likelihood
            for start in starts
        ]
        gain = np.asarray(annealed) - np.asarray(plain)
        margin = 1e-8 * np.abs(plain)
        wins, losses = int((gain > margin).sum()), int((gain < -margin).sum())
        mean, error = gain.mean(), gain.std(ddof=1) / np.sqrt(gain.size)
        print(f"{name}: {mean:+.1f} +- {error:.1f}, {wins} wins, {losses} losses")
        assert not (mean > 2.0 * error and wins > losses), name
