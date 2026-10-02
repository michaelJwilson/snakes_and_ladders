"""Every EM result carries its cost and its termination, and an HMM fit names the states it froze (issue #1165).

- **The guard** reads the four EM result types' declared fields: ``spent``,
  ``unit`` and ``termination``, with ``unit`` defaulting to
  :attr:`Cost.ITERATIONS`, and runs each entry point once to read
  ``spent == termination.iterations``.
- **The oracle** is the #1136 planted instance under HMM emissions: two
  generating count-pair states and a third seeded at a mean of 1e12, whose
  posterior the first E step empties. ``EmFit.frozen`` names that state, on
  the compiled ragged E step and on the torch recursion alike, and the two
  states with data recover their generating means.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.cost import Cost
from sal.emissions import (
    CategoricalEmission,
    CountPairEmission,
    GaussianEmission,
    NegativeBinomialEmission,
    Reestimate,
)
from sal.opt import emission_mixture, mixture
from sal.opt.em import EmConfig, inner_termination
from sal.opt.emission_mixture import EmissionMixtureFit
from sal.opt.hmm.estimation import (
    CategoricalFit,
    EmFit,
    baum_welch,
    baum_welch_family,
)
from sal.opt.mixture import MixtureFit
from sal.opt.termination import Stop

#: The four results the EM loop's three callers build (issue #1165).
RESULTS = (EmFit, CategoricalFit, EmissionMixtureFit, MixtureFit)

#: The #1136 planted means; the third state is seeded at 1e12.
MEANS = (30.0, 200.0)

#: Three iterations: enough to step, short of every tolerance below.
SHORT = EmConfig(max_iterations=3, tolerance=0.0)


@pytest.mark.smoke
@pytest.mark.mixture
@pytest.mark.parametrize("result", RESULTS, ids=lambda result: result.__name__)
def test_every_em_result_declares_spent_unit_and_termination(result: type) -> None:
    fields = {field.name: field for field in dataclasses.fields(result)}
    assert {"spent", "unit", "termination"} <= set(fields)
    assert fields["unit"].default is Cost.ITERATIONS
    # Required, so no constructor reports a cost it did not count.
    assert fields["spent"].default is dataclasses.MISSING
    assert fields["spent"].default_factory is dataclasses.MISSING


@pytest.mark.smoke
@pytest.mark.mixture
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST])
def test_every_em_entry_point_spends_the_iterations_it_ran(backend: Backend) -> None:
    rng = np.random.default_rng(1165)
    log_initial = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    log_transition = torch.log(torch.tensor([[0.9, 0.1], [0.2, 0.8]]).double())
    symbols = rng.integers(0, 3, (4, 30))
    log_emission = torch.log(
        torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float64)
    )
    values = rng.normal(0.0, 1.0, 200)
    gaussian = GaussianEmission([-1.0, 1.0], [1.0, 1.0], 1e-6)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    counts = rng.poisson(10.0, 200).astype(np.float64)
    fits = (
        baum_welch(
            symbols, log_initial, log_transition, log_emission, SHORT, backend=backend
        ),
        baum_welch_family(
            values.reshape(4, 50),
            log_initial,
            log_transition,
            gaussian,
            SHORT,
            backend=backend,
        ),
        mixture.expectation_maximization(
            values, weights, gaussian, SHORT, backend=backend
        ),
        emission_mixture.expectation_maximization(
            counts, weights, NegativeBinomialEmission([5.0, 5.0], [8.0, 12.0]), SHORT
        ),
    )
    for fit in fits:
        assert fit.termination.iterations == SHORT.max_iterations
        assert fit.spent == fit.termination.iterations
        assert fit.unit is Cost.ITERATIONS


@pytest.mark.smoke
def test_baum_welch_takes_its_covariate_and_backend_by_name() -> None:
    log_initial = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    log_transition = torch.log(torch.tensor([[0.9, 0.1], [0.2, 0.8]]).double())
    family = CategoricalEmission.from_log(
        torch.log(torch.tensor([[0.6, 0.4], [0.4, 0.6]], dtype=torch.float64))
    )
    symbols = np.zeros((2, 5), dtype=np.int64)
    with pytest.raises(TypeError):
        baum_welch_family(  # type: ignore[call-arg]
            symbols, log_initial, log_transition, family, SHORT, None
        )
    with pytest.raises(TypeError):
        baum_welch(  # type: ignore[call-arg]
            symbols,
            log_initial,
            log_transition,
            family.log_matrix,
            SHORT,
            Backend.PYTHON,
        )


@pytest.mark.analytic
def test_an_inner_solve_reads_as_a_termination() -> None:
    family = NegativeBinomialEmission([5.0], [8.0])
    closed = inner_termination(Reestimate(family))
    assert closed.converged
    assert closed.iterations == 0
    assert closed.reason is Stop.CONVERGED
    capped = inner_termination(Reestimate(family, converged=False, iterations=50))
    assert not capped.converged
    assert capped.iterations == 50
    assert capped.reason is Stop.BUDGET


def _planted() -> tuple[np.ndarray, np.ndarray, CountPairEmission]:
    """The #1136 pairs as 40 chains of 100, their labels, and a three-state start.

    The labels are drawn independently, so the generating chain is an HMM
    whose transition rows are the mixing weights.
    """
    rng = np.random.default_rng(1136)
    truth = CountPairEmission(
        [5.0, 20.0], list(MEANS), [2.0, 8.0], [8.0, 2.0], [40.0, 40.0], joint=False
    )
    labels = rng.integers(0, 2, 4_000)
    start = CountPairEmission(
        [5.0, 20.0, 5.0],
        [25.0, 180.0, 1e12],
        [2.0, 8.0, 5.0],
        [8.0, 2.0, 5.0],
        [40.0, 40.0, 40.0],
        joint=False,
    )
    pairs = truth.sample(labels, rng).astype(np.float64)
    return pairs.reshape(40, 100, 2), labels.reshape(40, 100), start


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.RUST])
def test_an_emptied_hmm_state_is_held_and_named(backend: Backend) -> None:
    observations, labels, start = _planted()
    uniform = -math.log(3.0)
    log_initial = torch.full((3,), uniform, dtype=torch.float64)
    log_transition = torch.full((3, 3), uniform, dtype=torch.float64)

    fit = baum_welch_family(
        observations, log_initial, log_transition, start, backend=backend
    )

    fitted = fit.components
    assert isinstance(fitted, CountPairEmission)
    assert fit.frozen == (2,)
    assert fit.termination.converged
    assert float(fitted.total.mean[2]) == 1e12
    for k, planted in enumerate(MEANS):
        observed = observations[labels == k, 0].mean()
        assert float(fitted.total.mean[k]) == pytest.approx(observed, rel=0.02)
        assert float(fitted.total.mean[k]) == pytest.approx(planted, rel=0.05)


@pytest.mark.oracle
def test_a_fit_with_no_emptied_state_freezes_none() -> None:
    observations, _, _ = _planted()
    two = CountPairEmission(
        [5.0, 20.0], [25.0, 180.0], [2.0, 8.0], [8.0, 2.0], [40.0, 40.0], joint=False
    )
    log_initial = torch.full((2,), -math.log(2.0), dtype=torch.float64)
    log_transition = torch.full((2, 2), -math.log(2.0), dtype=torch.float64)

    fit = baum_welch_family(observations, log_initial, log_transition, two)

    assert fit.frozen == ()
