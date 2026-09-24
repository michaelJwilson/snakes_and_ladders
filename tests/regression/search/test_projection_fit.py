"""The projected fit read to its end, against the truth the draw was made from (issue #729).

Three things `test_projection_seeding.py` never reached: the convergence test,
the log-likelihood a `Fitted` reports, and `SeededFit`, the `opt.budget`
method the release sweep uses. Referee: the planted draw's truth (each
observation's component, each negative-binomial mean). 100 observations,
since the fit runs to convergence: 134 iterations in 4 s, against 266 in
11 s at 400.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget, compare
from snakes_and_ladders.search.projection import (
    DETERMINISTIC,
    LOCATION_SEEDINGS,
    SEEDINGS,
    CountPairAt,
    ProjectedCounts,
    ProjectedTrial,
    SeededFit,
    TimedFit,
    fit_projection,
    flatten,
    project,
)
from snakes_and_ladders.sim.count_pairs import binned_model
from snakes_and_ladders.sim.fixtures import fixture

PROBLEM = "spatio_sequential_counts"

#: Observations the draw carries: four components, so chance recovery is 0.25.
SAMPLES = 100

#: Iterations the fit may run. Above the 134 it converges in, so the run ends
#: on its own convergence test and not on the budget --- which is the branch
#: this module exists to reach.
CONVERGED_BUDGET = Budget(Cost.PASSES, 150)

#: The short budget `SeededFit` is read at, where what is pinned is that the
#: method reports the fit rather than how far the fit got.
SHORT_BUDGET = Budget(Cost.PASSES, 6)

#: What the fit must leave, measured on this draw: recovery 0.690 against a
#: chance of 0.25, and a largest relative error of 0.246 in a component's
#: negative-binomial mean. The bounds are around those rather than at them,
#: because one draw of 100 observations is a noisy estimate of either.
RECOVERY = 0.60
MEAN_ERROR = 0.35


def rng() -> np.random.Generator:
    """The one seed every route below is run from."""
    return np.random.default_rng([541, 9])


def _instance() -> ProjectedCounts:
    """One projected draw of the declared CI model, at a bin factor of one."""
    params = binned_model(fixture(PROBLEM, "ci").params.model, 1)
    return project(params, SAMPLES, np.random.default_rng([541, 2]))


def _seam() -> CountPairAt:
    """The `ComponentsAt` every candidate ends at, its shapes the model's own."""
    truth = flatten(binned_model(fixture(PROBLEM, "ci").params.model, 1))
    return CountPairAt(
        float(truth.total.dispersion.mean()),
        float(truth.successes.concentration.mean()),
        float(truth.successes.trials[0]),
    )


@pytest.mark.end2end
def test_a_converged_projected_fit_recovers_the_components_it_was_drawn_from() -> None:
    # To convergence against the planted truth (`scipy` linear assignment):
    # 134 of 150 iterations, recovery 0.690 (chance 0.25), worst mean error
    # 0.246, log-likelihood never falling over 135 entries.
    fitted = fit_projection(_instance(), "kmeans++", _seam(), CONVERGED_BUDGET, rng())

    assert fitted.iterations < CONVERGED_BUDGET.size, fitted.iterations
    assert fitted.recovery > RECOVERY, fitted.recovery
    assert fitted.mean_error < MEAN_ERROR, fitted.mean_error
    # The reported log-likelihood is the curve's last entry, and the curve is
    # the non-decreasing one `Fitted` declares: expectation-maximization
    # cannot lower the likelihood, so a decrease is a defect and not noise.
    assert fitted.log_likelihood == float(fitted.log_likelihoods[-1])
    assert fitted.log_likelihoods.shape == (fitted.iterations + 1,)
    assert bool((np.diff(fitted.log_likelihoods) >= 0.0).all())


@pytest.mark.end2end
def test_the_budget_method_reports_the_fit_it_runs() -> None:
    # The method's `Outcome` is the fit's negated log-likelihood and iteration
    # count, bitwise: 809.350605, 6 iterations, recovery 0.660 (chance 0.25).
    instance, at = _instance(), _seam()

    outcome = SeededFit("kmeans++", at)(instance, SHORT_BUDGET, rng())
    fitted = fit_projection(instance, "kmeans++", at, SHORT_BUDGET, rng())

    assert outcome.value == -fitted.log_likelihood
    assert outcome.spent == fitted.iterations == SHORT_BUDGET.size
    assert fitted.recovery > 0.25, fitted.recovery


@pytest.mark.smoke
@pytest.mark.patch
@pytest.mark.analytic
def test_the_last_value_is_the_one_a_further_iteration_would_report() -> None:
    # #891: the final value and posterior are the E step alone (the discarded
    # M step cost 4.6 of 5.0 s); curves agree bitwise on shared entries.
    instance, at = _instance(), _seam()
    short = fit_projection(instance, "kmeans++", at, SHORT_BUDGET, rng())
    longer = fit_projection(
        instance, "kmeans++", at, Budget(Cost.PASSES, SHORT_BUDGET.size + 1), rng()
    )

    assert short.iterations == SHORT_BUDGET.size
    np.testing.assert_array_equal(
        short.log_likelihoods, longer.log_likelihoods[: SHORT_BUDGET.size + 1]
    )
    assert short.components is not None
    assert short.components.n_states == instance.n_components


@pytest.mark.end2end
def test_the_timed_method_reports_the_fit_and_the_seconds_it_ran() -> None:
    # The seconds method (#891): the fit's value bitwise, spend the wall clock
    # rounded up under `compare`'s ceiling, its trial judged against the truth.
    instance, at = _instance(), _seam()
    method = TimedFit("kmeans++", at, SEEDINGS["kmeans++"], SHORT_BUDGET)
    comparison = compare(
        {"kmeans++": method},
        [instance],
        Budget(Cost.SECONDS, 600),
        [0, 1],
        workers=1,
    )
    fitted = fit_projection(
        instance, "kmeans++", at, SHORT_BUDGET, np.random.default_rng([0, 0])
    )

    first = comparison.outcomes[0]
    assert isinstance(first.detail, ProjectedTrial)
    assert first.value == -fitted.log_likelihood
    assert first.detail.fitted.recovery == fitted.recovery > 0.25
    assert 0.0 < first.detail.seconds <= first.spent <= 600
    assert first.detail.curve == tuple(float(v) for v in fitted.log_likelihoods)
    assert comparison.budget.unit is Cost.SECONDS


@pytest.mark.smoke
def test_a_deterministic_location_start_reads_no_generator() -> None:
    # `DETERMINISTIC` is what lets a caller run those starts once: two
    # generators give one seeding.
    instance, at = _instance(), _seam()
    for name in DETERMINISTIC:
        first = LOCATION_SEEDINGS[name](instance, at, np.random.default_rng(0))
        second = LOCATION_SEEDINGS[name](instance, at, np.random.default_rng(1))
        for key, value in first.components.named_parameters().items():
            assert bool((value == second.components.named_parameters()[key]).all())
