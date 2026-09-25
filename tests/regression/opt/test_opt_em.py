"""The shared expectation-maximization loop, against the contract its callers read.

`opt.em.em_loop` carries the alternation the three entry points ran a copy of
each (issue #859), so what has to hold is what those copies did: the
log-likelihood is the one the step was handed, the stopping rule is relative,
and the iteration count is what `MixtureFit` and `EmissionMixtureFit` report.
A scripted step states each of those without a model in the way --- the fits
themselves are refereed by their own oracles, which is a different claim and
is made elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import pytest
from sal.opt.em import EM, EMISSION_MIXTURE_EM, EmConfig, em_loop
from sal.opt.termination import Stop


def _scripted(values: list[float]) -> Callable[[int], tuple[int, float]]:
    """A step returning the declared log-likelihoods in order, counting its calls."""

    def step(state: int) -> tuple[int, float]:
        return state + 1, values[state]

    return step


@pytest.mark.smoke
def test_the_loop_stops_where_the_relative_change_falls_to_the_tolerance() -> None:
    # -100, -10, -1, -1: the third step moves by 9 of 1, the fourth by 0.
    calls, log_likelihood, termination = em_loop(
        _scripted([-100.0, -10.0, -1.0, -1.0, -999.0]),
        0,
        config=EmConfig(max_iterations=10, tolerance=1e-12),
    )

    assert (calls, termination.iterations) == (4, 4)
    assert termination.reason is Stop.CONVERGED
    assert log_likelihood == -1.0


@pytest.mark.smoke
def test_an_exhausted_budget_returns_the_last_state_rather_than_raising() -> None:
    # The loop does not raise on exhaustion: `baum_welch_family` returns an
    # `EmFit` either way, and the mixture fits report the count they reached.
    # Which branch ended it is the `Termination` (#860).
    calls, log_likelihood, termination = em_loop(
        _scripted([-8.0, -4.0, -2.0, -1.0]),
        0,
        config=EmConfig(max_iterations=3, tolerance=1e-12),
    )

    assert (calls, termination.iterations) == (3, 3)
    assert termination.reason is Stop.BUDGET
    assert log_likelihood == -2.0


@pytest.mark.smoke
def test_an_empty_budget_runs_no_step_and_reports_no_likelihood() -> None:
    calls, log_likelihood, termination = em_loop(
        _scripted([-1.0]),
        0,
        config=EmConfig(max_iterations=0, tolerance=1e-12),
    )

    assert (calls, termination.iterations) == (0, 0)
    assert termination.reason is Stop.BUDGET
    assert log_likelihood == -float("inf")


@pytest.mark.smoke
def test_replacing_a_field_leaves_the_shared_config_unchanged() -> None:
    # Issue #1059: `EM` is one frozen default every entry point shares, so a
    # caller's `replace` makes a new config and no other caller sees it.
    changed = replace(EM, tolerance=1e-10)
    assert changed.tolerance == 1e-10
    assert EmConfig(max_iterations=500, tolerance=1e-12) == EM
    assert EmConfig(max_iterations=200, tolerance=1e-10) == EMISSION_MIXTURE_EM
    with pytest.raises(FrozenInstanceError):
        EM.tolerance = 0.0  # type: ignore[misc]
