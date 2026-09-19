"""The projected fit read to its end, against the truth the draw was made from (issue #729).

`test_projection_seeding.py` compares eight seedings at a fixed six-iteration
budget, so three things in `search/projection.py` were entered by no test at
#729's measurement: the convergence test that ends a fit, the log-likelihood
a `Fitted` reports, and `SeededFit`, the `opt.budget` method the release
sweep runs every candidate through.

All three are read here on one draw whose components are planted, so the
referee is that draw's own truth --- which observation each component
generated, and the negative-binomial mean each was drawn from --- as
`Fitted`'s own docstring states. The draw is 100 observations rather than the
sibling module's 4,000 because the fit has to run to convergence and the cost
is one pass per iteration: 134 iterations in 4 s here against 266 in 11 s at
400.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.search.projection import (
    CountPairAt,
    ProjectedCounts,
    SeededFit,
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
CONVERGED_BUDGET = Budget("passes", 150)

#: The short budget `SeededFit` is read at, where what is pinned is that the
#: method reports the fit rather than how far the fit got.
SHORT_BUDGET = Budget("passes", 6)

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
    # The fit driven to its own convergence test rather than to a budget, and
    # judged against the planted truth: the generating component of every
    # observation, and the generating negative-binomial mean of each
    # component under the assignment `scipy`'s linear assignment gives.
    # Realized: 134 iterations of a 150 budget, recovery 0.690 against a
    # chance of 0.25, largest relative mean error 0.246, and a log-likelihood
    # curve that never decreases over its 135 entries.
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
    # `opt.budget.compare` ranks candidates by an `Outcome`, so what the
    # release sweep compares is this method's value and not the fit's. It is
    # the fit's negated log-likelihood and its iteration count, bitwise, on
    # the same draw and the same seed; the fit beside it is judged against
    # the planted truth, so a method reporting something else is reporting
    # something unrefereed. Realized at the six-iteration budget: value
    # 809.350605, 6 iterations, recovery 0.660 against a chance of 0.25.
    instance, at = _instance(), _seam()

    outcome = SeededFit("kmeans++", at)(instance, SHORT_BUDGET, rng())
    fitted = fit_projection(instance, "kmeans++", at, SHORT_BUDGET, rng())

    assert outcome.value == -fitted.log_likelihood
    assert outcome.spent == fitted.iterations == SHORT_BUDGET.size
    assert fitted.recovery > 0.25, fitted.recovery
