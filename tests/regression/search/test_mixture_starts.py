"""The emission-mixture starts, polished by EM, against the draw's own truth (issue #891).

`search.mixture_starts` starts the joint count-pair mixture every way the
package can and polishes each start by
`opt.emission_mixture.expectation_maximization` at one budget. The referee is
the simulated truth of the `emission_mixture/ci` draw: the generating
component of every pair. The timed record is held to the run it was read
from: its curve is the polish's own values, in the order `track` sampled
them, with the start's path before the handover.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget, compare
from snakes_and_ladders.opt.emission_mixture import CountPairSeeding
from snakes_and_ladders.search.mixture_starts import (
    DETERMINISTIC,
    STARTS,
    MixtureInstance,
    StartRow,
    TimedStart,
    Trial,
    instance_from,
)
from snakes_and_ladders.sim.emission_mixture import simulate_emission_mixture
from snakes_and_ladders.sim.fixtures import fixture

#: The polish's budget: experiment 009's six passes.
PASSES = Budget(Cost.PASSES, 6)

#: The initialization's ceiling, which no start on the ci draw approaches.
CEILING = Budget(Cost.SECONDS, 120)

#: What every start that reads the pairs leaves at six passes, measured on
#: this draw: 0.911 (the uniform draw) to 0.953, against the 0.50 of the
#: largest component. The prior draw reads no pair and reaches 0.218, which is
#: what the control is for, so it is held only to the curve.
RECOVERY = 0.90


def _instance() -> MixtureInstance:
    """The ci draw, and the seam at its pooled dispersion and concentration."""
    params = fixture("emission_mixture", "ci").params
    return instance_from(
        simulate_emission_mixture(params),
        CountPairSeeding(
            float(params.components.total.dispersion.mean()),
            float(params.components.concentration.mean()),
            joint=True,
        ),
    )


@pytest.fixture(scope="module")
def trials() -> dict[str, Trial]:
    """Every start run once through `compare`, serially, at seed 0."""
    comparison = compare(
        {name: TimedStart(name, PASSES) for name in STARTS},
        [_instance()],
        CEILING,
        [0],
        workers=1,
    )
    found: dict[str, Trial] = {}
    for name, outcome in zip(comparison.methods, comparison.outcomes, strict=True):
        assert isinstance(outcome.detail, Trial)
        found[name] = outcome.detail
    return found


@pytest.mark.end2end
def test_every_start_that_reads_the_pairs_recovers_the_generating_components(
    trials: dict[str, Trial],
) -> None:
    for name, trial in trials.items():
        values = trial.polished.log_likelihoods
        # EM cannot lower the likelihood, from any start.
        assert bool((np.diff(values) >= 0.0).all()), name
        if name != "prior":
            assert trial.recovery > RECOVERY, (name, trial.recovery)


@pytest.mark.analytic
def test_the_curve_is_the_path_then_the_polish_in_the_order_it_was_sampled(
    trials: dict[str, Trial],
) -> None:
    for name, trial in trials.items():
        times = [seconds for seconds, _ in trial.curve]
        assert times == sorted(times), name
        assert trial.handover == len(trial.curve) - len(trial.polished.log_likelihoods)
        polish = [value for _, value in trial.curve[trial.handover :]]
        assert polish == [float(v) for v in trial.polished.log_likelihoods], name
        assert 0.0 <= trial.curve[trial.handover][0] <= trial.seconds
    # The starts that iterate carry their path; the others hand over at once.
    assert {name for name, trial in trials.items() if trial.handover > 0} == {
        "gaussian-em",
        "burn-in",
        "hmc",
    }


@pytest.mark.smoke
def test_a_deterministic_start_reads_no_generator() -> None:
    instance = _instance()
    for name in DETERMINISTIC:
        first = STARTS[name](instance, np.random.default_rng(0)).components
        second = STARTS[name](instance, np.random.default_rng(1)).components
        for key, value in first.named_parameters().items():
            assert bool((value == second.named_parameters()[key]).all()), name


@pytest.mark.analytic
def test_a_row_reads_its_trials_against_the_reference(
    trials: dict[str, Trial],
) -> None:
    # The gap is the reference less the reached value, entry by entry, and
    # the seeding's seconds are the handover's.
    reference = _instance().reference
    for name, trial in trials.items():
        row = StartRow.from_trials(name, [trial], reference)
        assert row.trials == 1
        assert row.gap == (reference - float(trial.polished.log_likelihoods[-1]),)
        assert row.seeding_seconds == (trial.curve[trial.handover][0],)
        assert row.deterministic == (name in DETERMINISTIC)
