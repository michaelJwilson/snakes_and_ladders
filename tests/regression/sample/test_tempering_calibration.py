"""The tempering start's ladder, calibrated by its own exchanges (issue #902).

`sample.initialize.FromTempering` ran the ladder it was handed. A
`LadderCalibration` measures candidate ladders by short runs of
`hmc.parallel_tempering` and revises them by `schedule.adapt_ladder` or
`schedule.adapt_ladder_by_round_trips`, before the run the start is taken
from. The target is the enumerable two-component Gaussian mixture of
`tests._posteriors`, whose weight posterior is integrated by quadrature over
all 4,096 assignments, so the cold replica is judged against a number no
sampler produced; the calibrated ladder is judged against the band it was
asked for, measured on the ladder it settled on and again on the run.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.opt.constrain import log_simplex
from snakes_and_ladders.opt.emission_mixture import CountPairSeeding
from snakes_and_ladders.opt.mixture import (
    GaussianMixtureObjective,
    mixture_log_likelihood,
)
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.sample.initialize import (
    FromTempering,
    LadderCalibration,
    LadderRule,
)
from snakes_and_ladders.sample.schedule import AdaptedLadder, FeedbackLadder
from snakes_and_ladders.search import mixture_starts
from snakes_and_ladders.sim.emission_mixture import simulate_emission_mixture
from snakes_and_ladders.sim.fixtures import fixture

from tests._posteriors import enumerated_quadrature, weight_posterior

#: A ladder with a gap: the two rungs 64 apart exchange at 0.067 over 150
#: rounds at seed 0, below the band.
FIXED = (1.0, 64.0)
BAND = (0.2, 0.6)
STEP = 0.9
TRAJECTORY = 5
#: Rounds of the run the start is taken from, and those discarded before the
#: cold replica is read.
ROUNDS = 400
BURN_IN = 50
#: Three standard errors of the cold replica's mean weight, each at the
#: draws' own effective sample size.
SIGMAS = 3.0
SEED = 0


def _calibration() -> LadderCalibration:
    """150 rounds a measurement, so an acceptance resolves to about 0.04; at most 4."""
    return LadderCalibration(
        LadderRule.ACCEPTANCE, rounds=150, max_rounds=4, band=BAND, max_replicas=6
    )


def _start(
    seed: int,
    calibration: LadderCalibration | None,
    ladder: tuple[float, ...] = FIXED,
    rounds: int | Budget = ROUNDS,
) -> FromTempering:
    return FromTempering(
        ladder,
        rounds,
        STEP,
        torch.Generator().manual_seed(seed),
        n_steps=TRAJECTORY,
        calibration=calibration,
    )


@pytest.fixture(scope="module")
def runs() -> dict[str, tuple[FromTempering, hmc.Tempered]]:
    """The fixed ladder and the calibrated one, each run once at seed 0."""
    target, _, _ = weight_posterior()
    found = {}
    for name, calibration in (("fixed", None), ("calibrated", _calibration())):
        start = _start(SEED, calibration)
        found[name] = (start, start.run(target))
    return found


def _cold_sigmas(run: hmc.Tempered, reference: float) -> float:
    """The cold replica's mean weight against ``reference``, in its own standard errors."""
    weight = torch.exp(log_simplex(run.positions[BURN_IN:, 0]))[:, :1]
    error = float(weight.std()) / math.sqrt(float(hmc.effective_sample_size(weight)[0]))
    return abs(float(weight.mean()) - reference) / error


@pytest.mark.analytic
def test_the_calibrated_ladder_exchanges_inside_the_band_where_the_fixed_one_does_not(
    runs: dict[str, tuple[FromTempering, hmc.Tempered]],
) -> None:
    """The band, measured where the ladder settled and again on the run it served.

    At seed 0 the fixed ladder ``(1, 64)`` exchanges at **0.067** over 150
    rounds, below the band ``(0.2, 0.6)``. The calibration bisects the gap
    geometrically once and stops: ``(1, 8, 64)``, measured at **0.373** and
    **0.327**, within the band after two measurements and 750 transitions
    (1.0 to 1.1 s at a 1-minute load of 3.4 to 5.6 on the 4-core host). The 400-round run on
    it exchanges at **0.305** and **0.285**, inside the band on a measurement
    the calibration did not make; the fixed ladder's own 400-round run
    exchanges at **0.072**. The first measurement *is* the fixed ladder's: a
    run of 150 rounds from a generator seeded alike reproduces it bitwise.

    Where it stops: on seeds 0 to 5 the calibration settles inside the band
    on 5; seed 5 inserts a rung at 4.76, measures 0.747 on ``(4.76, 8)`` and
    spends its four measurements, which ``settled`` reports as ``False``.
    """
    start, run = runs["calibrated"]
    _, fixed = runs["fixed"]
    assert start.spent is not None
    calibrated = start.spent.calibration
    assert calibrated is not None
    assert isinstance(calibrated.placement, AdaptedLadder)
    low, high = BAND

    first = hmc.parallel_tempering(
        weight_posterior()[0],
        FIXED,
        torch.Generator().manual_seed(SEED),
        150,
        step_size=STEP,
        n_steps=TRAJECTORY,
    )
    assert float(first.swap_acceptance[0]) < low, first.swap_acceptance
    assert float(fixed.swap_acceptance[0]) < low, fixed.swap_acceptance

    assert calibrated.ladder == (1.0, 8.0, 64.0)
    assert calibrated.settled
    assert all(low <= value <= high for value in calibrated.placement.acceptance)
    assert bool(((run.swap_acceptance >= low) & (run.swap_acceptance <= high)).all())
    # Two measurements, on the ladder of two and then of three rungs.
    assert calibrated.placement.rounds == 2
    assert calibrated.rounds_run == 2 * 150
    assert calibrated.transitions == 150 * (2 + 3) == 750
    assert calibrated.force_evaluations == 750 * hmc.leapfrog.force_evaluations(
        TRAJECTORY
    )
    assert start.spent.transitions == ROUNDS * 3
    assert start.spent.total_transitions == 750 + ROUNDS * 3


@pytest.mark.oracle
def test_the_cold_replica_matches_quadrature_on_either_ladder(
    runs: dict[str, tuple[FromTempering, hmc.Tempered]],
) -> None:
    """The calibration moves the hot rungs and leaves the cold marginal where it was.

    The posterior mean weight integrated over the enumerated evidence is
    0.4068. After 50 rounds discarded, the cold replica's mean over 350 is
    **0.13** standard errors from it on the fixed ladder and **0.41** on the
    calibrated one, each at the draws' own effective sample size (198 and
    118), against the three declared. The calibrated run starts at the
    lowest-valued point its warm-up visited, and the pin is the same.
    """
    _, observations, components = weight_posterior()
    reference, _ = enumerated_quadrature(observations, components)
    for name, (_, run) in runs.items():
        assert _cold_sigmas(run, reference) < SIGMAS, name


@pytest.mark.smoke
def test_one_seed_reproduces_the_calibrated_ladder_and_the_start_bitwise() -> None:
    """Every measurement draws from the start's generator, so one seed is the whole start.

    And without a calibration the start is the fixed ladder's
    ``parallel_tempering`` run, bitwise, as it was before the calibration
    existed.
    """
    target, _, _ = weight_posterior()
    calibration = LadderCalibration(
        LadderRule.ACCEPTANCE, rounds=40, max_rounds=3, band=BAND, max_replicas=6
    )
    first, second = (_start(3, calibration, rounds=20) for _ in range(2))
    one, two = first.run(target), second.run(target)
    assert first.spent is not None
    assert second.spent is not None
    assert first.spent.calibration is not None
    assert second.spent.calibration is not None
    assert first.spent.calibration.ladder == second.spent.calibration.ladder
    assert first.spent.calibration.placement == second.spent.calibration.placement
    assert torch.equal(one.positions, two.positions)
    assert torch.equal(one.theta, two.theta)
    assert torch.equal(one.swap_acceptance, two.swap_acceptance)

    plain = _start(3, None, rounds=20)
    direct = hmc.parallel_tempering(
        target,
        FIXED,
        torch.Generator().manual_seed(3),
        20,
        step_size=STEP,
        n_steps=TRAJECTORY,
    )
    started = plain.starts(target)[0]
    assert torch.equal(started, direct.theta)
    assert plain.spent is not None
    assert plain.spent.calibration is None
    assert plain.spent.total_transitions == 20 * len(FIXED)


class _Slow:
    """An objective that sleeps on every call, so a warm-up outlasts a budget."""

    def __init__(self, inner: Objective, seconds: float) -> None:
        self.inner = inner
        self.seconds = seconds

    def initial(self) -> torch.Tensor:
        return self.inner.initial()

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.inner.constrain(theta)

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.inner.theta_from(named)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        time.sleep(self.seconds)
        return self.inner(theta)


@pytest.mark.smoke
def test_a_budget_in_seconds_stops_inside_it_and_another_unit_is_refused() -> None:
    """A round runs only if the longest so far ends inside the budget; the clock includes the warm-up.

    Measured at a 1-minute load of 5.6 on the 4-core host: 165 rounds of
    ``(1, 8, 64)`` in 0.989 s of a 1 s budget, and with a 0.31 s calibration
    369 rounds in 1.991 s of 2 s.
    """
    target, _, _ = weight_posterior()
    alone = _start(0, None, (1.0, 8.0, 64.0), Budget(Cost.SECONDS, 1))
    run = alone.run(target)
    assert alone.spent is not None
    assert alone.spent.budget == Budget(Cost.SECONDS, 1)
    assert 1 < alone.spent.rounds == run.positions.shape[0]
    assert alone.spent.seconds <= 1.0, alone.spent.seconds
    assert run.force_evaluations == alone.spent.rounds * 3 * (
        hmc.leapfrog.force_evaluations(TRAJECTORY)
    )

    calibration = LadderCalibration(
        LadderRule.ACCEPTANCE, rounds=50, max_rounds=3, band=BAND, max_replicas=6
    )
    warmed = _start(0, calibration, FIXED, Budget(Cost.SECONDS, 2))
    warmed.run(target)
    assert warmed.spent is not None
    assert warmed.spent.calibration is not None
    assert warmed.spent.calibration.seconds < warmed.spent.seconds <= 2.0

    with pytest.raises(ValueError, match="'seconds'"):
        _start(0, None, FIXED, Budget(Cost.GRADIENTS, 100))
    # Two transitions of nine calls each at 60 ms: over 1 s before a round.
    slow = _start(
        0,
        LadderCalibration(
            LadderRule.ACCEPTANCE, rounds=1, max_rounds=1, band=BAND, max_replicas=2
        ),
        (1.0, 2.0),
        Budget(Cost.SECONDS, 1),
    )
    with pytest.raises(ValueError, match="leaving no round of the run"):
        slow.run(_Slow(target, 0.06))


@pytest.mark.smoke
def test_the_round_trip_rule_keeps_the_length_and_the_endpoints() -> None:
    """The feedback placement redistributes rungs and buys none; its cost is its measurements'.

    At seed 0, 100 rounds a measurement on ``(1, 8, 64)`` read an up-fraction
    of 0.061 at the middle rung, which the placement moves to 16.58; two
    measurements do not converge at a tolerance of 0.05, and ``settled``
    says so.
    """
    target, _, _ = weight_posterior()
    calibration = LadderCalibration(
        LadderRule.ROUND_TRIPS, rounds=100, max_rounds=2, tolerance=0.05
    )
    start = _start(0, calibration, (1.0, 8.0, 64.0), rounds=5)
    start.run(target)
    assert start.spent is not None
    calibrated = start.spent.calibration
    assert calibrated is not None
    assert isinstance(calibrated.placement, FeedbackLadder)
    assert len(calibrated.ladder) == 3
    assert (calibrated.ladder[0], calibrated.ladder[-1]) == (1.0, 64.0)
    assert calibrated.settled is calibrated.placement.converged
    assert calibrated.transitions == calibrated.placement.replicas_measured * 100


@pytest.mark.smoke
def test_a_calibration_refuses_a_setting_off_its_rule() -> None:
    with pytest.raises(ValueError, match="needs a band and max_replicas"):
        LadderCalibration(LadderRule.ACCEPTANCE, rounds=10, max_rounds=2)
    with pytest.raises(ValueError, match="round-trip rule's"):
        LadderCalibration(
            LadderRule.ACCEPTANCE,
            rounds=10,
            max_rounds=2,
            band=BAND,
            max_replicas=4,
            tolerance=0.1,
        )
    with pytest.raises(ValueError, match="needs a tolerance"):
        LadderCalibration(LadderRule.ROUND_TRIPS, rounds=10, max_rounds=2)
    with pytest.raises(ValueError, match="keeps the ladder's length"):
        LadderCalibration(
            LadderRule.ROUND_TRIPS, rounds=10, max_rounds=2, band=BAND, tolerance=0.1
        )
    with pytest.raises(ValueError, match="at least 1"):
        LadderCalibration(
            LadderRule.ACCEPTANCE, rounds=0, max_rounds=2, band=BAND, max_replicas=4
        )
    with pytest.raises(ValueError, match="is not a valid LadderRule"):
        LadderCalibration("bisect", rounds=10, max_rounds=2)  # type: ignore[arg-type]
    assert (
        LadderCalibration("round_trips", rounds=1, max_rounds=1, tolerance=0.1).rule  # type: ignore[arg-type]
        is LadderRule.ROUND_TRIPS
    )


#: The warm-up the ticket costs on the mixture's surrogate: measurements of 20
#: rounds, at most 3, on 4 to 8 replicas.
MIXTURE_CALIBRATION = LadderCalibration(
    LadderRule.ACCEPTANCE, rounds=20, max_rounds=3, band=BAND, max_replicas=8
)


def _handover(
    instance: mixture_starts.MixtureInstance,
    objective: GaussianMixtureObjective,
    theta: torch.Tensor,
) -> float:
    """The count-pair log-likelihood at the components a surrogate point seeds, as `TimedStart` hands over."""
    seeded = mixture_starts.at_locations(instance, objective.components(theta).mean)
    # The polish's first recorded value: the E step at equal weights, read
    # here without the polish, whose signature is #898's to change.
    k = instance.n_components
    return float(
        mixture_log_likelihood(
            torch.as_tensor(instance.observations, dtype=torch.float64),
            torch.log(torch.full((k,), 1.0 / k, dtype=torch.float64)),
            seeded,
        )
    )


@pytest.mark.experiment
@pytest.mark.snapshot
@pytest.mark.warning
def test_on_the_mixture_the_calibration_buys_no_handover_at_equal_transitions() -> None:
    """The calibrated start against the fixed ladder given the same transitions, on `emission_mixture/ci`.

    The fixed ladder is `search.projection.TEMPERATURES`, ``(1, 2, 4, 8)``, at
    the step and trajectory `search.mixture_starts` runs it at; the
    calibrated start runs `TEMPERING_ROUNDS = 2` on the ladder its warm-up
    settled, and the fixed one runs ``ceil(total / 4)`` rounds, so it spends
    at least the calibrated start's transitions, warm-up included.

    **The ticket's expectation does not hold.** Over seeds 0 to 7 the
    calibrated start's handover is above the fixed ladder's on 2 (+4.0,
    +7.4), equal on 2 and below on 4 (-1.0 to -10.5), a mean of -1.25 in
    log-likelihood. The warm-up settled inside the band on 4 of the 8, three
    of them on ``(1, 2, 4, 8)`` itself, and spent its three measurements on
    the other 4 with a pair outside it. Both
    are above today's two-round start on 8 of 8, by 13.5 to 64.2, which is
    the extra rounds and not the ladder. The calibrated start spent 88 to 374
    transitions, warm-up included, in 0.5 to 2.9 s at a 1-minute load of 6.8
    on the 4-core host, 6 to 7 ms a transition.

    Pinned, one seed on each side: seed 4, where the warm-up inserts 5.66
    and the calibrated start hands over at -8000.7 against the fixed
    ladder's -8004.7, and seed 5, where it keeps the ladder and hands over at
    -7990.9 against -7989.9. A warning because the default stays the fixed
    ladder and the notebook opting in should know that on this fixture the
    calibration's cost buys no better start at equal transitions.
    """
    params = fixture("emission_mixture", "ci").params
    instance = mixture_starts.instance_from(
        simulate_emission_mixture(params),
        CountPairSeeding(
            float(params.components.total.dispersion.mean()),
            float(params.components.concentration.mean()),
            joint=True,
        ),
    )
    objective = mixture_starts.surrogate(instance)
    handed: dict[int, tuple[float, float]] = {}
    for seed in (4, 5):
        # The start `tempered_seeding` makes, read for its settings and its
        # stream; the two below differ from it in one argument each.
        today = mixture_starts.tempering_initializer(np.random.default_rng(seed))
        calibrated = FromTempering(
            today.temperatures,
            today.n_rounds,
            today.step_size,
            today.generator,
            n_steps=today.n_steps,
            calibration=MIXTURE_CALIBRATION,
        )
        warm = calibrated.run(objective)
        assert calibrated.spent is not None
        total = calibrated.spent.total_transitions
        fixed = mixture_starts.tempering_initializer(np.random.default_rng(seed))
        fixed.n_rounds = math.ceil(total / len(fixed.temperatures))
        cold = fixed.run(objective)
        assert fixed.spent is not None
        assert fixed.spent.total_transitions >= total
        handed[seed] = (
            _handover(instance, objective, warm.theta),
            _handover(instance, objective, cold.theta),
        )
    assert handed[4][0] > handed[4][1], handed
    assert handed[5][0] < handed[5][1], handed
