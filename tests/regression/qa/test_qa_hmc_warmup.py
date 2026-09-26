"""Regression tests for sal.qa.hmc_warmup.

Refereed by closed forms. The target is ``tests._objective_checks``'s
``AnalyticGaussian``, the sampler tests' own; both chains are correct
samplers (a stuck chain reports an effective size equal to its length); and
the caption's claim, that the warm-up repays its discarded draws in effective
draws per gradient, is asserted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sal.qa.hmc_warmup import (
    COVARIANCE,
    MEAN,
    N_SAMPLES,
    SETTLED,
    TRACKED,
    WARMUP,
    _ess_per_gradient,
    build_figure,
    chains,
    main,
    running_mean,
    settling_draw,
    target,
)
from sal.sample.hmc import effective_sample_size

from tests._objective_checks import AnalyticGaussian

#: One pair of chains, drawn once: the sampling is the whole cost of this
#: module, and every test below reads the same two chains the figure draws.
ADAPTED, FIXED = chains()


@pytest.mark.oracle
def test_the_target_is_the_gaussian_the_samplers_tests_use() -> None:
    # The failure this prevents: the figure quietly checking a different
    # Gaussian from the one `AnalyticGaussian` pins the sampler against, so
    # that its agreement means nothing.
    figure_target = target()
    oracle = AnalyticGaussian(list(MEAN), [list(row) for row in COVARIANCE])

    torch.testing.assert_close(figure_target.mean, oracle.mean)
    torch.testing.assert_close(figure_target.covariance, oracle.covariance)
    torch.testing.assert_close(figure_target.precision, oracle._precision)
    torch.testing.assert_close(figure_target.initial(), oracle.initial())
    for point in (
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([3.0, 1.0], dtype=torch.float64),
        torch.tensor([-7.5, 2.25], dtype=torch.float64),
    ):
        torch.testing.assert_close(figure_target(point), oracle(point))


@pytest.mark.analytic
def test_the_targets_constraint_map_inverts_itself() -> None:
    figure_target = target()
    theta = torch.tensor([1.5, -0.25], dtype=torch.float64)

    named = figure_target.constrain(theta)

    torch.testing.assert_close(figure_target.theta_from(named), theta)


@pytest.mark.analytic
def test_both_chains_are_correct_samplers_and_not_one_stuck_one() -> None:
    # The failure this prevents, and the one the first draft of this figure
    # had: a fixed step above the stability limit, so the comparison is
    # against a chain that never moved. Such a chain accepts nothing, and its
    # effective sample size is its length, which flatters it.
    for chain in (ADAPTED, FIXED):
        assert 0.2 < chain.acceptance_rate < 1.0
        moves = torch.count_nonzero(chain.draws[1:] - chain.draws[:-1])
        assert moves > N_SAMPLES // 2


@pytest.mark.oracle
def test_both_chains_recover_the_exact_mean_within_monte_carlo_error() -> None:
    # Both are kernels with the right stationary distribution, so neither may
    # miss the mean: the figure's point is cost, not correctness. The bound is
    # the standard error of the mean at the chain's own effective sample size,
    # widened to four of them.
    gaussian = target()
    exact_sd = torch.sqrt(torch.diagonal(gaussian.covariance))
    for chain in (ADAPTED, FIXED):
        standard_error = exact_sd / torch.sqrt(effective_sample_size(chain.draws))
        deviation = torch.abs(chain.draws.mean(dim=0) - gaussian.mean)

        assert bool(torch.all(deviation < 4.0 * standard_error)), (
            f"{deviation} against {4.0 * standard_error}"
        )


@pytest.mark.analytic
def test_the_warm_up_repays_its_discarded_draws() -> None:
    # The caption's claim, asserted: the adapted chain buys more effective
    # draws per gradient than the fixed one *after* being charged for the
    # gradients its warm-up spent.
    assert ADAPTED.spent > FIXED.spent

    assert _ess_per_gradient(ADAPTED) > _ess_per_gradient(FIXED)


@pytest.mark.analytic
def test_the_adapted_chain_settles_first() -> None:
    gaussian = target()
    exact_mean = float(gaussian.mean[TRACKED])
    exact_sd = float(torch.sqrt(gaussian.covariance[TRACKED, TRACKED]))

    adapted_draw = settling_draw(running_mean(ADAPTED, TRACKED), exact_mean, exact_sd)
    fixed_draw = settling_draw(running_mean(FIXED, TRACKED), exact_mean, exact_sd)

    assert adapted_draw is not None
    assert fixed_draw is not None
    assert adapted_draw < fixed_draw


@pytest.mark.smoke
def test_settling_takes_the_last_excursion_and_not_the_first_entry() -> None:
    # A running mean that enters the band, leaves it, and returns is credited
    # with the return: the statistic is where it stays, not where it arrived.
    trace = np.array([1.0, 0.0, 1.0, 0.0, 0.0])

    assert settling_draw(trace, 0.0, 1.0) == 4
    assert settling_draw(np.zeros(4), 0.0, 1.0) == 1
    assert settling_draw(np.array([0.0, 0.0, 1.0]), 0.0, 1.0) is None
    # The band is closed: a trace exactly on it has arrived.
    assert settling_draw(np.array([SETTLED]), 0.0, 1.0) == 1


@pytest.mark.infra
def test_the_running_mean_is_a_running_mean() -> None:
    trace = running_mean(ADAPTED, TRACKED)
    draws = ADAPTED.draws[:, TRACKED].detach().numpy()

    assert trace.shape == (N_SAMPLES,)
    assert trace[0] == pytest.approx(draws[0])
    assert trace[-1] == pytest.approx(draws.mean())


@pytest.mark.infra
def test_the_caption_reports_the_constants_the_figure_was_drawn_from(
    tmp_path: Path,
) -> None:
    # A caption that names no seed or no sizes cannot be checked against a
    # re-render, which is the whole contract a QA caption carries.
    fig, caption = build_figure(ADAPTED, FIXED)
    try:
        assert str(N_SAMPLES) in caption
        assert str(WARMUP) in caption
        assert "20260914" in caption
        assert "effective draws per gradient" in caption
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    written = main(["--output-dir", str(tmp_path)])

    assert written.figure_path.exists()
    assert written.caption_path.exists()
