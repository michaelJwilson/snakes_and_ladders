"""The E-step hook on `baum_welch_family` (issue #1170).

Referees: `ragged.posteriors` handed in as the hook is the default route
bitwise, every field of the fit; the hook is handed the live rows as one
segment per sequence and the current parameters; and a kernel per step,
which the hook cannot take, is refused rather than read as one matrix. A
release experiment records what the Monte-Carlo E step,
`ragged.sampled_posteriors`, buys against exact EM at stress size.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch
from sal.emissions import GaussianEmission
from sal.likelihood.ragged import posteriors, sampled_posteriors
from sal.opt.em import EM
from sal.opt.hmm import EStep, Posteriors, baum_welch_family
from sal.ragged import Ragged
from sal.sim.hmm import HmmParams, simulate_sequences

LENGTHS = (50, 120, 7, 300)


def _problem() -> tuple[Ragged, torch.Tensor, torch.Tensor, GaussianEmission]:
    """Four segments of unequal length over three Gaussian states."""
    rng = np.random.default_rng(1170)
    observations = Ragged(rng.normal(size=sum(LENGTHS)), LENGTHS)
    log_initial = torch.log(torch.full((3,), 1.0 / 3.0, dtype=torch.float64))
    log_transition = torch.log(torch.as_tensor(rng.dirichlet(np.full(3, 5.0), size=3)))
    family = GaussianEmission(
        np.array([-1.0, 0.0, 1.0]), np.ones(3), variance_floor=1e-6
    )
    return observations, log_initial, log_transition, family


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.backend
def test_the_compiled_e_step_as_a_hook_is_the_default_bitwise() -> None:
    """Ragged batch, 40 iterations: every field of the two fits equal."""
    observations, log_initial, log_transition, family = _problem()
    config = replace(EM, max_iterations=40)
    default = baum_welch_family(
        observations, log_initial, log_transition, family, config
    )
    hooked = baum_welch_family(
        observations, log_initial, log_transition, family, config, e_step=posteriors
    )
    assert default.log_likelihood == hooked.log_likelihood
    assert torch.equal(default.log_initial, hooked.log_initial)
    assert torch.equal(default.log_transition, hooked.log_transition)
    for name in ("mean", "scale"):
        np.testing.assert_array_equal(
            np.asarray(getattr(default.components, name)),
            np.asarray(getattr(hooked.components, name)),
        )
    assert default.spent == hooked.spent
    assert default.termination == hooked.termination


@pytest.mark.critical
@pytest.mark.smoke
def test_the_hook_is_handed_the_live_rows_and_the_current_parameters() -> None:
    """One call per iteration, segments the sequences', parameters the M step's."""
    observations, start_initial, start_transition, family = _problem()
    seen: list[tuple[tuple[int, ...], np.ndarray, np.ndarray]] = []

    def recording(
        log_density: Ragged, log_initial: np.ndarray, log_transition: np.ndarray
    ) -> Posteriors:
        seen.append((log_density.lengths, log_initial.copy(), log_transition.copy()))
        return posteriors(log_density, log_initial, log_transition)

    hook: EStep = recording
    fit = baum_welch_family(
        observations,
        start_initial,
        start_transition,
        family,
        replace(EM, max_iterations=3),
        e_step=hook,
    )
    assert len(seen) == fit.spent == 3
    assert all(lengths == LENGTHS for lengths, _, _ in seen)
    np.testing.assert_array_equal(seen[0][1], start_initial.numpy())
    np.testing.assert_array_equal(seen[0][2], start_transition.numpy())
    # Each call after the first reads the transition the M step before wrote:
    # the rows normalize, and it moved from the start.
    np.testing.assert_allclose(np.exp(seen[1][2]).sum(axis=1), 1.0, rtol=1e-12)
    assert not np.array_equal(seen[1][2], seen[0][2])


@pytest.mark.critical
@pytest.mark.analytic
def test_a_kernel_per_step_is_refused_with_a_hook() -> None:
    observations, log_initial, log_transition, family = _problem()
    per_step = log_transition.expand(max(LENGTHS) - 1, 3, 3).clone()
    with pytest.raises(ValueError, match="e_step takes one"):
        baum_welch_family(
            observations, log_initial, per_step, family, e_step=posteriors
        )


@pytest.mark.release
@pytest.mark.experiment
def test_monte_carlo_em_reaches_further_than_exact_em_at_stress_size() -> None:
    """2e5 positions over ten overlapping Gaussian states, 120 iterations each.

    Exact EM converges slowly here: after 500 iterations (598 s) it still
    gains 0.009 nats an iteration. The Monte-Carlo E step at four paths
    (`ragged.sampled_posteriors`) reaches -3.30 nats from that 500-iteration
    value at iteration 120 where exact EM reaches -4.76, and within 1 nat of
    it at 107 s (iteration 173) against exact EM's 440 s (iteration 365), at
    0.60 s an iteration against 1.20 s on four threads. Recorded, not
    asserted: the wall times, and the serial cost, 2.17 s an iteration, under
    which the same 173 iterations take 375 s. Asserted: the ordering at 120.
    """
    n = 10
    transition = np.full((n, n), 0.05 / (n - 1))
    np.fill_diagonal(transition, 0.95)
    lengths = (1000,) * 200
    params = HmmParams(
        n_states=n,
        lengths=lengths,
        initial=np.full(n, 1.0 / n),
        transition=transition,
        emissions=GaussianEmission(
            np.arange(n, dtype=float), np.full(n, 0.6), variance_floor=1e-6
        ),
        seed=1170,
        tolerance=0.0,
    )
    data = simulate_sequences(params)
    observations = Ragged(
        np.asarray(data.observations, dtype=float).reshape(-1), lengths
    )
    rng = np.random.default_rng(3)
    start = GaussianEmission(
        np.sort(rng.uniform(0, n - 1, n)), np.ones(n), variance_floor=1e-6
    )
    log_initial = torch.log(torch.full((n,), 1.0 / n, dtype=torch.float64))
    stay = np.full((n, n), 0.1 / (n - 1)) + np.eye(n) * (0.9 - 0.1 / (n - 1))
    log_transition = torch.log(torch.as_tensor(stay))
    # A tolerance no step meets, so both spend the whole budget.
    config = replace(EM, max_iterations=120, tolerance=1e-300)
    exact = baum_welch_family(observations, log_initial, log_transition, start, config)
    sampled = baum_welch_family(
        observations,
        log_initial,
        log_transition,
        start,
        config,
        e_step=partial(sampled_posteriors, rng=np.random.default_rng(7), n_paths=4),
    )
    assert exact.spent == sampled.spent == 120
    # Each reports the exact evidence at the parameters its last E step read.
    assert sampled.log_likelihood - exact.log_likelihood > 1.0, (
        sampled.log_likelihood,
        exact.log_likelihood,
    )
