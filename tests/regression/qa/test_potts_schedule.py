"""The committed schedule search is the search the script runs, and it read no reported seed (issues #1038, #1046).

`docs/nb/potts_starts.ipynb` reads `docs/nb/data/potts_schedule.json` rather
than re-running the search, so what is pinned here is that the file is what
`qa.potts_schedule` computes: a recorded evaluation re-run on its tuning seed
gives the recorded energy bitwise, the chosen schedule is the lowest mean
recorded, and every search began at the current schedule and stayed inside
its cap. The #1046 machinery is held to problems whose answer is known: the
stopping rule to a quadratic's minimum, the step-count match to a linear and
a non-monotone spend, and the rounds to spends whose fixed point is known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from snakes_and_ladders.qa.potts_schedule import (
    CAPPED,
    CONVERGE_CAP,
    CONVERGED,
    COUNT_FIXED,
    EVALUATIONS,
    MATCH_TOLERANCE,
    MOVES,
    REPORTED_SEEDS,
    ROUNDS_SPENT,
    TUNING_SEEDS,
    VERTEX_TOLERANCE,
    WALL,
    evaluate,
    handover,
    joint_rounds,
    load,
    match_steps,
    nelder_mead,
    params_of,
)
from snakes_and_ladders.sample.potts_mcmc import PottsMove
from snakes_and_ladders.sample.schedule import ScheduleParams, ScheduleShape
from snakes_and_ladders.search.ground_state import ANNEAL_SCHEDULE


@pytest.mark.smoke
@pytest.mark.snapshot
def test_a_recorded_evaluation_reruns_bitwise() -> None:
    tuned = load()
    entry = tuned.raw["moves"]["swendsen-wang"]["chosen"]
    params = ScheduleParams(
        ScheduleShape(entry["shape"]), entry["t_start"], entry["t_end"], entry["hold"]
    )

    energy, spent = handover(
        PottsMove.SWENDSEN_WANG,
        params,
        None,
        np.random.default_rng([TUNING_SEEDS[0], 0]),
    )

    assert energy == entry["energies"][0]
    assert spent == entry["spent"][0]


@pytest.mark.smoke
def test_the_chosen_schedule_is_the_lowest_mean_recorded() -> None:
    tuned = load()
    for move in MOVES:
        shapes = tuned.raw["moves"][str(move)]["shapes"]
        lowest = min(
            (entry for trace in shapes.values() for entry in trace),
            key=lambda entry: entry["mean"],
        )
        assert tuned.chosen[str(move)] == ScheduleParams(
            ScheduleShape(lowest["shape"]),
            lowest["t_start"],
            lowest["t_end"],
            lowest["hold"],
        )


@pytest.mark.smoke
def test_every_search_starts_at_the_current_schedule_inside_its_cap() -> None:
    tuned = load()
    assert tuned.raw["tuning_seeds"] == list(TUNING_SEEDS)
    for move in MOVES:
        for shape, trace in tuned.raw["moves"][str(move)]["shapes"].items():
            assert 1 <= len(trace) <= EVALUATIONS
            assert math.isclose(trace[0]["t_start"], ANNEAL_SCHEDULE.t_start)
            assert math.isclose(trace[0]["t_end"], ANNEAL_SCHEDULE.t_end)
            assert trace[0]["hold"] == 0.0
            assert trace[0]["shape"] == shape


@pytest.mark.smoke
def test_a_reported_seed_is_refused_for_tuning() -> None:
    with pytest.raises(ValueError, match="reported"):
        evaluate(PottsMove.WOLFF, ANNEAL_SCHEDULE, seeds=(REPORTED_SEEDS[0],))


@pytest.mark.smoke
def test_a_search_point_is_its_schedule() -> None:
    point = (math.log(2.0), math.log(0.05), 0.25)

    params = params_of(point, ScheduleShape.COSINE)

    assert params.shape is ScheduleShape.COSINE
    assert math.isclose(params.t_start, 2.0)
    assert math.isclose(params.t_end, 0.05)
    assert params.hold == 0.25


# --- Issue #1046: the stopping rule, the step-count match and the rounds.


@pytest.mark.analytic
def test_the_search_stops_converged_at_a_quadratics_minimum() -> None:
    # A quadratic's minimum is known, and a standard error of 1e-6 is met once
    # the vertices are within 1e-3 of it: the stop is the rule's, not the cap's.
    # The search starts on the hold's lower bound, where SciPy's clipped
    # Nelder-Mead ends at hold 0 against the minimum's 0.4.
    centre = np.array([0.3, -1.2, 0.4])

    def objective(point: np.ndarray) -> tuple[float, float]:
        return float(np.sum((point - centre) ** 2)), 1e-6

    start = np.zeros(3)
    result = nelder_mead(
        objective,
        np.vstack([start, start + np.diag([0.5, -0.5, 0.2])]),
        [(-5.0, 5.0), (-5.0, 5.0), (0.0, 0.9)],
    )

    assert result.stop == CONVERGED
    assert result.evaluations < CONVERGE_CAP
    np.testing.assert_allclose(result.point, centre, atol=VERTEX_TOLERANCE)
    assert np.max(np.abs(result.vertices - result.vertices[0])) <= VERTEX_TOLERANCE


@pytest.mark.analytic
def test_a_minimum_on_a_bound_is_found_on_the_bound() -> None:
    # The hold's minimum at -0.2 is outside [0, 0.9]: the clipped search ends at 0.
    def objective(point: np.ndarray) -> tuple[float, float]:
        return float((point[0] - 1.0) ** 2 + (point[1] + 0.2) ** 2), 1e-6

    start = np.array([0.0, 0.5])
    result = nelder_mead(
        objective,
        np.vstack([start, start + np.diag([0.5, 0.2])]),
        [(-5.0, 5.0), (0.0, 0.9)],
    )

    assert result.stop == CONVERGED
    np.testing.assert_allclose(result.point, [1.0, 0.0], atol=VERTEX_TOLERANCE)


@pytest.mark.smoke
def test_noise_above_the_range_stops_the_search_at_the_cap() -> None:
    # A standard error of zero is never above the range: only the cap stops it.
    def objective(point: np.ndarray) -> tuple[float, float]:
        return float(np.sum(point**2)), 0.0

    start = np.ones(2)
    result = nelder_mead(
        objective, np.vstack([start, start + np.eye(2)]), [(-5, 5), (-5, 5)], cap=25
    )

    assert result.stop == CAPPED
    assert result.evaluations == 25


@pytest.mark.smoke
def test_the_wall_clock_stops_the_search_and_says_so() -> None:
    ticks = iter(range(1000))

    def objective(point: np.ndarray) -> tuple[float, float]:
        return float(np.sum(point**2)), 0.0

    start = np.ones(2)
    result = nelder_mead(
        objective,
        np.vstack([start, start + np.eye(2)]),
        [(-5, 5), (-5, 5)],
        deadline=10.0,
        clock=lambda: float(next(ticks)),
    )

    assert result.stop == WALL
    assert result.evaluations == 10


@pytest.mark.analytic
def test_the_match_meets_a_linear_spend_within_its_tolerance_from_either_side() -> None:
    # Spend 700 visits a step against a budget of 34.7 million: the root is
    # 49,601 steps, reached growing from below and shrinking from above.
    budget = 34_721_000.0
    for start in (1000, 400_000):
        steps, probes = match_steps(lambda steps: 700.0 * steps, budget, start)

        assert abs(700.0 * steps - budget) <= MATCH_TOLERANCE * budget
        assert probes[0][0] == start
        assert (steps, 700.0 * steps) in probes


@pytest.mark.analytic
def test_a_non_monotone_spend_returns_the_nearest_probe() -> None:
    # #1041's shape: spend rises with the count but not monotonically, so a
    # count past the root can spend less than one before it.
    budget = 34_721_000.0

    def spend(steps: int) -> float:
        return 700.0 * steps * (1.0 + 0.3 * math.sin(steps / 500.0))

    steps, probes = match_steps(spend, budget, 41_250)

    nearest = min(probes, key=lambda probe: abs(probe[1] - budget))
    assert steps == nearest[0]
    assert len(probes) < 40
    assert [count for count, _ in probes].count(steps) >= 1


@pytest.mark.analytic
def test_the_rounds_stop_when_the_count_is_fixed() -> None:
    # Spend `steps * (1 + p)` matches at `budget / (1 + p)`; a search that
    # quarters `p` from 0.08 moves the count by 5.9%, 1.5% and 0.4%, so the
    # third search's count is the first to change by less than 1%.
    budget = 10_000.0

    def match(point: np.ndarray, _count: int) -> int:
        return round(budget / (1.0 + float(point[0])))

    def search(point: np.ndarray, _count: int) -> tuple[np.ndarray, str]:
        return point / 4.0, CONVERGED

    point, count, stop, history = joint_rounds(
        match, search, np.array([0.08]), 5000, rounds=5
    )

    assert stop == COUNT_FIXED
    assert [entry["count"] for entry in history] == [9259, 9804, 9950, 9988]
    assert point[0] == 0.08 / 64
    assert count == 9988


@pytest.mark.analytic
def test_the_rounds_stop_after_their_cap_when_the_count_keeps_moving() -> None:
    # `p` doubles every search, so the count halves and never settles.
    def match(point: np.ndarray, count: int) -> int:
        return match_steps(lambda steps: steps * point[0], 10_000.0, count)[0]

    def search(point: np.ndarray, _count: int) -> tuple[np.ndarray, str]:
        return point * 2.0, CONVERGED

    _, _, stop, history = joint_rounds(match, search, np.array([1.0]), 100, rounds=3)

    assert stop == ROUNDS_SPENT
    assert len(history) == 4
    assert [entry.get("search") for entry in history] == [CONVERGED] * 3 + [None]


@pytest.mark.smoke
def test_a_search_stopped_by_the_wall_stops_the_rounds() -> None:
    def search(point: np.ndarray, _count: int) -> tuple[np.ndarray, str]:
        return point, WALL

    _, count, stop, history = joint_rounds(
        lambda _point, _count: 7, search, np.array([1.0]), 100
    )

    assert (count, stop, len(history)) == (7, WALL, 1)
