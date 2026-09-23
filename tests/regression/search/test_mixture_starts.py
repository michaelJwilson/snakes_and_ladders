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

import math

import numpy as np
import pytest
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget, compare
from snakes_and_ladders.opt.emission_mixture import CountPairSeeding
from snakes_and_ladders.search.mixture_starts import (
    DETERMINISTIC,
    POLISH_TOLERANCE,
    STARTS,
    MixtureInstance,
    StartRow,
    TimedStart,
    Trial,
    gap_band,
    instance_from,
    polish,
)
from snakes_and_ladders.sim.emission_mixture import simulate_emission_mixture
from snakes_and_ladders.sim.fixtures import fixture

#: The polish's budget: experiment 009's six passes.
PASSES = Budget(Cost.PASSES, 6)

#: The initialization's ceiling, which no start on the ci draw approaches;
#: under the one budget, the whole cell's (issue #898).
CEILING = Budget(Cost.SECONDS, 120)

#: Starts the one-budget test runs: two rules over the pairs, each measured
#: converging in 13 to 23 iterations, 2 s or less, on the ci draw.
CONVERGING = ("data", "emission++")

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
        "gibbs-anneal",
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
    # The gaps are the reference less the handover's and the reached value,
    # entry by entry, and the seeding's seconds are the handover's.
    reference = _instance().reference
    for name, trial in trials.items():
        row = StartRow.from_trials(name, [trial], reference)
        assert row.trials == 1
        assert row.gap == (reference - float(trial.polished.log_likelihoods[-1]),)
        assert row.seeded_gap == (reference - float(trial.polished.log_likelihoods[0]),)
        assert row.iterations == (trial.polished.iterations,)
        assert row.converged == (trial.polished.converged,)
        assert row.seeding_seconds == (trial.curve[trial.handover][0],)
        assert row.deterministic == (name in DETERMINISTIC)


@pytest.mark.analytic
def test_the_one_budget_polishes_to_the_tolerance_and_charges_the_whole_cell() -> None:
    # Issue #898: one budget in seconds covers the start and its polish. The
    # referee is the stopping rule itself, read off the trace: the polish
    # stops at the *first* iteration whose relative change in the
    # log-likelihood is at most the tolerance, EM never lowers the
    # likelihood, and the spend compare checks is the cell's whole seconds.
    instance = _instance()
    comparison = compare(
        {name: TimedStart(name) for name in CONVERGING},
        [instance],
        CEILING,
        [0],
        workers=1,
    )
    for name, outcome in zip(comparison.methods, comparison.outcomes, strict=True):
        trial = outcome.detail
        assert isinstance(trial, Trial)
        assert trial.polished.converged, name
        # The trace's last entry is the closing E step at the components the
        # last iteration left; the change the rule read is between the two
        # entries before it.
        visited = trial.polished.log_likelihoods[:-1]
        change = np.abs(np.diff(visited)) / np.abs(visited[:-1])
        assert change[-1] <= POLISH_TOLERANCE, name
        assert bool((change[:-1] > POLISH_TOLERANCE).all()), name
        assert bool((np.diff(trial.polished.log_likelihoods) >= 0.0).all()), name
        assert outcome.spent == math.ceil(trial.seconds) <= CEILING.size, name


@pytest.mark.smoke
def test_a_polish_stops_at_one_of_its_two_stops_and_a_timed_start_is_in_seconds() -> (
    None
):
    instance = _instance()
    seeded = STARTS["data"](instance, np.random.default_rng(0)).components
    # No seconds left: no iteration, the handover's E step alone.
    spent = polish(instance, seeded, seconds=0.0)
    assert spent.iterations == 0
    assert not spent.converged
    assert polish(instance, seeded, passes=2).iterations == 2
    with pytest.raises(ValueError, match="exactly one"):
        polish(instance, seeded)
    with pytest.raises(ValueError, match="exactly one"):
        polish(instance, seeded, passes=2, seconds=1.0)
    with pytest.raises(ValueError, match="budgeted in seconds"):
        TimedStart("data")(instance, PASSES, np.random.default_rng(0))


@pytest.mark.analytic
def test_a_band_holds_each_trial_between_its_samples_and_averages_across_trials(
    trials: dict[str, Trial],
) -> None:
    # Issue #898's figure: the referee is the curve itself. At a sample's own
    # time the band is that sample's gap; between samples it is the earlier
    # one's; before a trial's first sample the band is undefined; at one
    # trial there is no spread; over two copies of one trial the mean is the
    # trial and the spread zero.
    reference = _instance().reference
    trial = trials["hmc"]
    times = np.asarray([point[0] for point in trial.curve])
    gaps = reference - np.asarray([point[1] for point in trial.curve])
    grid = np.concatenate([[times[0] / 2.0], times, [times[-1] * 2.0]])
    band = gap_band([trial], reference, grid)
    assert math.isnan(band.mean[0])
    # Samples recorded at one clock reading: the band reads the last of them.
    last = dict(zip(times, gaps, strict=True))
    assert band.mean[1:-1].tolist() == [last[t] for t in times]
    assert band.mean[-1] == gaps[-1]
    assert bool(np.isnan(band.std).all())
    assert band.handover == (times[trial.handover], gaps[trial.handover])
    twice = gap_band([trial, trial], reference, grid)
    assert np.array_equal(twice.mean[1:], band.mean[1:])
    assert bool((twice.std[1:] == 0.0).all())
    with pytest.raises(ValueError, match="at least one"):
        gap_band([], reference, grid)


@pytest.mark.analytic
def test_a_polish_stops_where_em_empties_a_component() -> None:
    # Issue #898: EM can drive a weight to underflow while the likelihood
    # rises, and the M step then refuses a component the E step leaves no
    # responsibility on. A component seeded at (1e5, 5e4), far past every
    # pair of the ci draw, keeps a weight of 6.5e-276 after eight
    # iterations and is empty at the ninth E step. The polish under its
    # seconds stops there with the eighth iteration's fit; EM never lowered
    # the likelihood on the way; a polish of fixed passes has no such stop
    # and raises the refusal, as it did before the stop existed.
    instance = _instance()
    rows = np.array([[30.0, 5.0], [200.0, 160.0], [1e5, 5e4]])
    polished = polish(instance, instance.at(rows), seconds=CEILING.size)
    assert polished.emptied
    assert not polished.converged
    assert polished.iterations == 8
    assert float(polished.weights.min()) < 1e-200
    assert bool((np.diff(polished.log_likelihoods) >= 0.0).all())
    with pytest.raises(ValueError, match="must be positive"):
        polish(instance, instance.at(rows), passes=polished.iterations + 1)
    # A component far from the pairs that EM does not empty is no stop.
    recovered = polish(
        instance,
        instance.at(np.array([*rows[:2].tolist(), [2000.0, 1000.0]])),
        seconds=60.0,
    )
    assert not recovered.emptied
    assert recovered.converged
