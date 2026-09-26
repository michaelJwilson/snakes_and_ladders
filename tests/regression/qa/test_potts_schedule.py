"""The committed schedule search is the search the script runs, and it read no reported seed (issue #1038).

`docs/nb/potts_starts.ipynb` reads `docs/nb/data/potts_schedule.json` rather
than re-running the search, so what is pinned here is that the file is what
`qa.potts_schedule` computes: a recorded evaluation re-run on its tuning seed
gives the recorded energy bitwise, the chosen schedule is the lowest mean
recorded, and every search began at the current schedule and stayed inside
its cap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sal.qa.potts_schedule import (
    EVALUATIONS,
    MOVES,
    REPORTED_SEEDS,
    TUNING_SEEDS,
    evaluate,
    handover,
    load,
    params_of,
)
from sal.sample.potts_mcmc import PottsMove
from sal.sample.schedule import ScheduleParams, ScheduleShape
from sal.search.ground_state import ANNEAL_SCHEDULE

#: Relative agreement of a rerun energy with its record: a sum of ~5,000
#: terms reordered (#1045) moves the last few bits, never 1e-12.
RERUN_TOLERANCE = 1e-12


@pytest.mark.smoke
@pytest.mark.snapshot
def test_a_recorded_evaluation_reruns_within_the_declared_tolerance() -> None:
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

    # Bitwise until #1045 moved the edge term outside BLAS, which reorders the
    # energy's sum: the rerun is -10142.328645965481 against the recorded
    # -10142.328645965412, 6.8e-15 relative. The declared tolerance is the
    # floor, well inside the 1e-12 a reordered sum of this length needs; the
    # spend is an integer count and stays exact.
    assert energy == pytest.approx(entry["energies"][0], rel=RERUN_TOLERANCE)
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
