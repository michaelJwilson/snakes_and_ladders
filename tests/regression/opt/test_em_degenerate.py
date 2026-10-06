"""An EM fit whose emission M step does not settle ends with `Stop.DEGENERATE`, not a raise (issue #1235).

- **The analytic property:** the step that degenerates is discarded whole, so
  the fit it returns is the fit a run capped one iteration earlier returns,
  bitwise --- parameters, log-likelihood and responsibilities. Read on every
  EM route that re-estimates through a family: Baum-Welch streamed and on the
  torch recursion, the count mixture on its distinct counts and per
  observation, the Gaussian mixture's tensor route and the annealed mixture.
- **The end2end check:** a best-of over restarts in which one restart seeds a
  component on a lone outlier, whose M step then cannot settle on the one
  observation it holds, still returns the planted means from the others.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import (
    EmissionFamily,
    GaussianEmission,
    PoissonEmission,
    Reestimate,
)
from sal.opt import emission_mixture, mixture, split_merge
from sal.opt.em import EmConfig, Unsettled
from sal.opt.hmm.estimation import baum_welch_family
from sal.opt.termination import Stop, Termination
from sal.sandbox.annealed_em import annealed_expectation_maximization

#: The EM iteration whose M step is made to report no settlement.
AT = 3

#: A budget no fit below reaches: each ends at `AT`, or at `AT - 1` capped.
LONG = EmConfig(max_iterations=50, tolerance=0.0)

#: The run the degenerate one is pinned against: one iteration short of `AT`.
CAPPED = EmConfig(max_iterations=AT - 1, tolerance=0.0)

#: What the planted M step reports when it does not settle.
REPORT = Unsettled(iterations=7, residual=0.5)


def _unsettle_at(
    monkeypatch: pytest.MonkeyPatch, family: type[EmissionFamily], call: int
) -> None:
    """Make ``family.reestimate`` report no settlement on its ``call``-th call, and only then."""
    original: Callable[..., Reestimate[Any]] = family.reestimate
    calls = 0

    def reestimate(self: Any, *args: Any, **kwargs: Any) -> Reestimate[Any]:
        nonlocal calls
        calls += 1
        step = original(self, *args, **kwargs)
        if calls != call:
            return step
        return replace(
            step,
            converged=False,
            iterations=REPORT.iterations,
            residual=REPORT.residual,
        )

    monkeypatch.setattr(family, "reestimate", reestimate)


def _counts() -> np.ndarray:
    """Two Poisson states, means 4 and 30, as 8 chains of 50."""
    rng = np.random.default_rng(1235)
    labels = rng.integers(0, 2, (8, 50))
    return rng.poisson(np.where(labels == 0, 4.0, 30.0))


def _hmm(backend: Backend) -> Callable[[EmConfig], Any]:
    observations = _counts()
    log_initial = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    log_transition = torch.log(torch.tensor([[0.8, 0.2], [0.3, 0.7]]).double())

    def run(config: EmConfig) -> Any:
        return baum_welch_family(
            observations,
            log_initial,
            log_transition,
            PoissonEmission([2.0, 20.0]),
            config,
            backend=backend,
        )

    return run


def _count_mixture(dtype: type) -> Callable[[EmConfig], Any]:
    # Integer counts take the distinct-count route; floats take the
    # per-observation route, its oracle (issue #997).
    observations: np.ndarray = _counts().reshape(-1).astype(dtype)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)

    def run(config: EmConfig) -> Any:
        return emission_mixture.expectation_maximization(
            observations, weights, PoissonEmission([2.0, 20.0]), config
        )

    return run


def _gaussian_mixture() -> Callable[[EmConfig], Any]:
    rng = np.random.default_rng(1235)
    observations = np.concatenate(
        [rng.normal(-2.0, 1.0, 150), rng.normal(3.0, 1.0, 150)]
    )
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)

    def run(config: EmConfig) -> Any:
        return mixture.expectation_maximization(
            observations,
            weights,
            GaussianEmission([-1.0, 1.0], [1.0, 1.0], 1e-6),
            config,
            backend=Backend.PYTHON,
        )

    return run


#: Each route, the family whose M step is unsettled, and its parameter fields.
ROUTES: dict[
    str,
    tuple[
        Callable[[], Callable[[EmConfig], Any]], type[EmissionFamily], tuple[str, ...]
    ],
] = {
    "baum_welch_streamed": (
        lambda: _hmm(Backend.RUST),
        PoissonEmission,
        ("log_initial", "log_transition"),
    ),
    "baum_welch_torch": (
        lambda: _hmm(Backend.PYTHON),
        PoissonEmission,
        ("log_initial", "log_transition"),
    ),
    "emission_mixture_cells": (
        lambda: _count_mixture(np.int64),
        PoissonEmission,
        ("weights", "responsibilities"),
    ),
    "emission_mixture_rows": (
        lambda: _count_mixture(np.float64),
        PoissonEmission,
        ("weights", "responsibilities"),
    ),
    "gaussian_mixture": (_gaussian_mixture, GaussianEmission, ("weights",)),
}


def _same(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Bitwise equality of two tensors of one shape."""
    return bool(torch.equal(torch.as_tensor(left), torch.as_tensor(right)))


@pytest.mark.analytic
@pytest.mark.mixture
@pytest.mark.parametrize("route", ROUTES)
def test_a_degenerate_fit_returns_the_previous_iteration_bitwise(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    build, family, fields = ROUTES[route]
    run = build()
    capped = run(CAPPED)
    _unsettle_at(monkeypatch, family, AT)

    degenerate = run(LONG)

    assert degenerate.termination == Termination(
        converged=False, iterations=AT, reason=Stop.DEGENERATE
    )
    assert degenerate.spent == AT
    assert degenerate.unsettled == REPORT
    assert capped.unsettled is None
    assert degenerate.log_likelihood == capped.log_likelihood
    for name in fields:
        assert _same(getattr(degenerate, name), getattr(capped, name)), name
    for name, value in capped.components.named_parameters().items():
        assert _same(degenerate.components.named_parameters()[name], value), name


@pytest.mark.analytic
@pytest.mark.mixture
def test_a_degenerate_first_iteration_returns_the_start() -> None:
    # No iteration completed: the start comes back with the log-likelihood a
    # loop reports before its first step, as `max_iterations=0` reports it.
    start = PoissonEmission([2.0, 20.0])
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    observations = _counts().reshape(-1)
    with pytest.MonkeyPatch.context() as patch:
        _unsettle_at(patch, PoissonEmission, 1)
        fit = emission_mixture.expectation_maximization(
            observations, weights, start, LONG
        )

    assert fit.termination.reason is Stop.DEGENERATE
    assert fit.termination.iterations == fit.spent == 1
    assert fit.log_likelihood == -math.inf
    assert fit.components is start
    assert fit.weights is weights


@pytest.mark.analytic
@pytest.mark.mixture
def test_an_annealed_fit_ends_degenerate_at_a_tempered_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The second of three tempered steps degenerates: one stage completed and
    # the degenerate one, two iterations spent, one evidence recorded.
    observations = _counts().reshape(-1).astype(np.float64)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    start = PoissonEmission([2.0, 20.0])
    first = annealed_expectation_maximization(
        observations, weights, start, [4.0], config=EmConfig(1, 0.0)
    )
    _unsettle_at(monkeypatch, PoissonEmission, 2)

    annealed = annealed_expectation_maximization(
        observations, weights, start, [4.0, 2.0, 1.5], config=LONG
    )

    fit = annealed.fit
    assert fit.termination.reason is Stop.DEGENERATE
    assert fit.spent == 2
    assert [stage.reason for stage in fit.stages] == [Stop.BUDGET, Stop.DEGENERATE]
    assert fit.unsettled == REPORT
    assert annealed.log_tempered_evidences == first.log_tempered_evidences
    assert _same(fit.weights, first.fit.weights)
    for name, value in first.fit.components.named_parameters().items():
        assert _same(fit.components.named_parameters()[name], value), name


class _OneObservationPoisson(PoissonEmission):
    """A Poisson family whose M step cannot settle on a component holding about one observation.

    The planted degeneracy of issue #1235: a rate read off one count is the
    count, and the family stands for one whose inner solve has nothing to
    converge on there.
    """

    def reestimate(
        self,
        observations: Any,
        posterior: Any,
        covariate: Any = None,
    ) -> Reestimate[PoissonEmission]:
        step = super().reestimate(observations, posterior, covariate)
        if float(torch.as_tensor(posterior).sum(dim=0).min()) < 1.5:
            return replace(step, converged=False, iterations=1, residual=1.0)
        return step


#: The planted means, and the lone outlier one restart seeds a component on.
MEANS = (5.0, 40.0)
OUTLIER = 1000.0


@pytest.mark.end2end
@pytest.mark.mixture
def test_a_best_of_with_one_degenerate_restart_recovers_the_planted_means() -> None:
    rng = np.random.default_rng(12350)
    labels = rng.integers(0, 2, 600)
    observations = np.append(rng.poisson(np.asarray(MEANS)[labels]), OUTLIER).astype(
        np.float64
    )

    def at(rows: np.ndarray) -> PoissonEmission:
        return _OneObservationPoisson(np.asarray(rows, dtype=np.float64).reshape(-1))

    restarts = 6
    # Each restart as `split_merge.fit` runs it, on its own stream, to name
    # which degenerate: the claim needs at least one of each.
    fits = [
        emission_mixture.expectation_maximization(
            observations,
            *emission_mixture.seed(observations, 2, at, method="plus_plus", rng=stream),
        )
        for stream in np.random.default_rng(1235).spawn(restarts)
    ]
    reasons = [one.termination.reason for one in fits]
    assert Stop.DEGENERATE in reasons
    assert Stop.CONVERGED in reasons

    best = split_merge.fit(
        observations,
        2,
        at,
        method="plus_plus",
        rng=np.random.default_rng(1235),
        restarts=restarts,
    )

    assert best.termination.reason is Stop.CONVERGED
    assert best.log_likelihood == max(one.log_likelihood for one in fits)
    # The labelled means, the outlier with the upper component: what a fit
    # that assigns every count to its generating component reads.
    counts = observations[:-1]
    labelled = (
        counts[labels == 0].mean(),
        np.append(counts[labels == 1], OUTLIER).mean(),
    )
    fitted = np.sort(best.components.named_parameters()["mean"].detach().numpy())
    np.testing.assert_allclose(fitted, labelled, rtol=0.02)
    np.testing.assert_allclose(fitted, MEANS, rtol=0.1)
