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
import torch
from sal.cost import Cost
from sal.emissions import EmissionFamily
from sal.opt.budget import Budget, compare
from sal.opt.emission_mixture import CountPairSeeding
from sal.opt.mixture import mixture_log_likelihood
from sal.opt.starts import Polished, Trial
from sal.opt.termination import Stop
from sal.search.mixture_starts import (
    BEST_OF,
    BEST_OF_EM_STARTS,
    BEST_OF_STARTS,
    DETERMINISTIC,
    POLISH_TOLERANCE,
    STARTS,
    MixtureInstance,
    MixtureTrial,
    Selection,
    StartRow,
    TimedStart,
    best_of,
    gap_band,
    instance_from,
    polish,
)
from sal.sim.emission_mixture import simulate_emission_mixture
from sal.sim.fixtures import fixture

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
def trials() -> dict[str, MixtureTrial]:
    """Every start run once through `compare`, serially, at seed 0."""
    comparison = compare(
        {name: TimedStart(name, PASSES) for name in STARTS},
        [_instance()],
        CEILING,
        [0],
        workers=1,
    )
    found: dict[str, MixtureTrial] = {}
    for name, outcome in zip(comparison.methods, comparison.outcomes, strict=True):
        assert isinstance(outcome.detail, MixtureTrial)
        found[name] = outcome.detail
    return found


@pytest.mark.end2end
def test_every_start_that_reads_the_pairs_recovers_the_generating_components(
    trials: dict[str, MixtureTrial],
) -> None:
    for name, trial in trials.items():
        values = trial.polished.log_likelihoods
        # EM cannot lower the likelihood, from any start.
        assert bool((np.diff(values) >= 0.0).all()), name
        if name != "prior":
            assert trial.recovery > RECOVERY, (name, trial.recovery)


@pytest.mark.analytic
def test_the_curve_is_the_path_then_the_polish_in_the_order_it_was_sampled(
    trials: dict[str, MixtureTrial],
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
    trials: dict[str, MixtureTrial],
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
    # #898: one budget in seconds covers start and polish; the polish stops at
    # the first iteration within tolerance, EM never falls, spend is the cell's.
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
        assert isinstance(trial, MixtureTrial)
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
    trials: dict[str, MixtureTrial],
) -> None:
    # #898's curve: the band at a sample is its gap, between samples the
    # earlier; undefined before the first; one trial, no spread.
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
    # #898: a component seeded at (1e5, 5e4) keeps weight 6.5e-276 after eight
    # iterations and is empty at the ninth E step; the seconds polish stops
    # with the eighth fit, EM never lowering; a fixed-pass polish raises.
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


def _equal(first: EmissionFamily, second: EmissionFamily) -> bool:
    """Every parameter tensor equal bitwise."""
    return all(
        torch.equal(a, b)
        for a, b in zip(
            first.named_parameters().values(),
            second.named_parameters().values(),
            strict=True,
        )
    )


@pytest.mark.analytic
def test_best_of_one_is_the_start_and_best_of_five_is_the_best_of_its_seedings() -> (
    None
):
    # #905, #912: referee the start on the cell's spawned generators: best-of-one
    # is the first child's, bitwise; best-of-five the highest of five, never below.
    instance = _instance()
    values = torch.as_tensor(instance.observations, dtype=torch.float64)
    uniform = torch.full((instance.n_components,), 1.0 / instance.n_components)
    uniform = torch.log(uniform.to(torch.float64))
    for name in ("data", "emission++", "burn-in"):
        (child,) = np.random.default_rng(7).spawn(1)
        alone = STARTS[name](instance, child)
        one = best_of(name, 1)(instance, np.random.default_rng(7))
        assert _equal(one.components, alone.components), name
        assert one.passes == alone.passes + 1.0
        seedings = [
            STARTS[name](instance, child) for child in np.random.default_rng(7).spawn(5)
        ]
        scores = [
            float(mixture_log_likelihood(values, uniform, s.components))
            for s in seedings
        ]
        five = best_of(name, 5)(instance, np.random.default_rng(7))
        assert [step for step, _ in five.path] == [0, 1, 2, 3, 4]
        for (_, family), seeded in zip(five.path, seedings, strict=True):
            assert _equal(family, seeded.components), name
        handed = float(mixture_log_likelihood(values, uniform, five.components))
        assert handed == max(scores), name
        assert five.passes == sum(s.passes for s in seedings) + 5.0
        again = best_of(name, 5)(instance, np.random.default_rng(7))
        assert _equal(again.components, five.components), name


@pytest.mark.smoke
@pytest.mark.backend
def test_a_parallel_best_of_is_the_serial_one_bitwise() -> None:
    # Issue #912: every seeding draws from its own spawned generator and
    # records into a run private to its thread, so four threads, and four
    # processes, hand over exactly what one worker does, polished or not.
    instance = _instance()
    serial = best_of("emission++", 5)(instance, np.random.default_rng(11))
    for pool in ("threads", "processes"):
        parallel = best_of("emission++", 5, workers=4, pool=pool)(
            instance, np.random.default_rng(11)
        )
        assert _equal(parallel.components, serial.components), pool
        assert parallel.passes == serial.passes, pool
        for (_, a), (_, b) in zip(parallel.path, serial.path, strict=True):
            assert _equal(a, b), pool
    one, fit_one = best_of("data", 5, Selection.POLISHED).polished(
        instance, np.random.default_rng(12), passes=PASSES.size
    )
    four, fit_four = best_of("data", 5, Selection.POLISHED, workers=4).polished(
        instance, np.random.default_rng(12), passes=PASSES.size
    )
    assert _equal(one.components, four.components)
    assert np.array_equal(fit_one.log_likelihoods, fit_four.log_likelihoods)


@pytest.mark.analytic
def test_a_timed_best_of_charges_its_seedings_and_draws_their_scores() -> None:
    # Through TimedStart: the curve before the handover is the five scored
    # seedings in order, and the seconds to the handover cover all five.
    instance = _instance()
    trial = TimedStart(BEST_OF_STARTS["data" + f"x{BEST_OF}"].key, PASSES)(
        instance, CEILING, np.random.default_rng(0)
    ).detail
    assert isinstance(trial, MixtureTrial)
    assert trial.handover == BEST_OF
    scores = [value for _, value in trial.curve[: trial.handover]]
    assert trial.curve[trial.handover][1] == max(scores)
    assert trial.curve[trial.handover - 1][0] <= trial.curve[trial.handover][0]


@pytest.mark.smoke
def test_best_of_refuses_a_deterministic_start_or_no_seeding() -> None:
    assert set(BEST_OF_STARTS) == {
        f"{name}x{BEST_OF}" for name in STARTS if name not in DETERMINISTIC
    }
    with pytest.raises(ValueError, match="stochastic"):
        best_of("quantile", 5)
    with pytest.raises(ValueError, match="stochastic"):
        best_of("nothing", 5)
    with pytest.raises(ValueError, match="at least one seeding"):
        best_of("data", 0)
    with pytest.raises(ValueError, match="at least one worker"):
        best_of("data", 2, workers=0)
    with pytest.raises(ValueError, match="threads or processes"):
        best_of("data", 2, pool="serial")


@pytest.mark.analytic
def test_a_polished_best_of_keeps_the_best_of_its_polished_seedings() -> None:
    # Issue #912. The referee polishes the same five seedings directly, on
    # the same spawned generators and at the same fixed passes: the polished best-of
    # hands over the seeding whose fit ends highest, with that fit, and
    # charges the seedings' passes and every polish's iterations.
    instance = _instance()
    for name in ("data", "emission++"):
        seedings = [
            STARTS[name](instance, child) for child in np.random.default_rng(3).spawn(5)
        ]
        fits = [polish(instance, s.components, passes=PASSES.size) for s in seedings]
        finals = [float(fit.log_likelihoods[-1]) for fit in fits]
        start = best_of(name, 5, Selection.POLISHED)
        seeded, chosen = start.polished(
            instance, np.random.default_rng(3), passes=PASSES.size
        )
        best = int(np.argmax(finals))
        assert _equal(seeded.components, seedings[best].components), name
        assert float(chosen.log_likelihoods[-1]) == max(finals), name
        assert seeded.passes == sum(s.passes for s in seedings) + 5 * PASSES.size
        for (_, family), drawn in zip(seeded.path, seedings, strict=True):
            assert _equal(family, drawn.components), name


@pytest.mark.analytic
def test_a_timed_polished_best_of_hands_over_its_chosen_fit() -> None:
    # Through TimedStart under the one seconds budget: the trial's fit is
    # the chosen one, reached at the handover, and the cell stays inside it.
    instance = _instance()
    key = f"datax{BEST_OF}+em"
    outcome = TimedStart(key)(instance, CEILING, np.random.default_rng(0))
    trial = outcome.detail
    assert isinstance(trial, MixtureTrial)
    assert trial.handover == BEST_OF
    assert trial.curve[-1][1] == float(trial.polished.log_likelihoods[-1])
    assert outcome.spent <= CEILING.size
    assert -outcome.value == float(trial.polished.log_likelihoods[-1])


@pytest.mark.smoke
def test_a_polished_best_of_is_keyed_and_refuses_a_call_without_its_budget() -> None:
    assert set(BEST_OF_EM_STARTS) == {
        f"{name}x{BEST_OF}+em" for name in STARTS if name not in DETERMINISTIC
    }
    start = best_of("data", 2, Selection.POLISHED)
    assert start.key == "datax2+em"
    with pytest.raises(ValueError, match="call polished"):
        start(_instance(), np.random.default_rng(0))
    with pytest.raises(ValueError, match="exactly one"):
        start.polished(_instance(), np.random.default_rng(0))


@pytest.mark.smoke
def test_a_mixture_polish_and_trial_are_the_shared_types() -> None:
    # Issue #926: one Polished and one Trial for every polish and timed fit;
    # the mixture's value is the negative log-likelihood where it stopped and
    # its termination counts the iterations its trace holds.
    instance = _instance()
    seeded = STARTS["data"](instance, np.random.default_rng(0))
    fixed = polish(instance, seeded.components, passes=PASSES.size)
    assert isinstance(fixed, Polished)
    assert fixed.value == -float(fixed.log_likelihoods[-1])
    assert fixed.iterations == PASSES.size == len(fixed.log_likelihoods) - 1
    assert fixed.termination.reason is Stop.BUDGET
    assert not fixed.converged
    trial = TimedStart("data", PASSES)(
        instance, CEILING, np.random.default_rng(0)
    ).detail
    assert isinstance(trial, Trial)
    assert isinstance(trial, MixtureTrial)
    assert isinstance(trial.polished, Polished)
