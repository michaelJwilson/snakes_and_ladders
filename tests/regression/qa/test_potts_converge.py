"""The committed searches to convergence are the searches the script runs (issue #1046).

`docs/nb/potts_starts.ipynb` reads `docs/nb/data/potts_converge.json` rather
than re-running the searches, so what is pinned here is that the file is what
`qa.potts_converge` computes: the chosen evaluation re-run on a tuning seed
gives the recorded energy bitwise, every search began at its plan's warm
start and recorded the stop it met, and the first step-count match of Wolff
and of Niedermayer reproduces #1038's and #1041's probe at the same count.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from snakes_and_ladders.qa.potts_converge import (
    CLUSTERS,
    ORDER,
    load,
    plans,
    schedule_of,
)
from snakes_and_ladders.qa.potts_schedule import (
    CAPPED,
    CONVERGE_CAP,
    CONVERGED,
    COUNT_FIXED,
    ROUNDS_SPENT,
    TUNING_SEEDS,
    WALL,
    handover,
    params_of,
)
from snakes_and_ladders.qa.potts_schedule import load as load_tuned
from snakes_and_ladders.sample.schedule import ScheduleShape


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_chosen_swendsen_wang_evaluation_reruns_its_chain() -> None:
    entry = load()["moves"]["swendsen-wang"]
    chosen = entry["chosen"]

    energy, spent = handover(
        plans()["swendsen-wang"].move,
        schedule_of(chosen),
        entry["count"],
        np.random.default_rng([TUNING_SEEDS[0], 0]),
    )

    # Bitwise where the evaluation ran after #1044's edge sum; a replayed
    # record from before it is held to the 1e-13 relative tolerance #1038's
    # rerun declares (`test_potts_schedule.py`).
    np.testing.assert_allclose(energy, chosen["energies"][0], rtol=1e-13, atol=0)
    assert spent == chosen["spent"][0]


@pytest.mark.smoke
def test_every_search_starts_at_its_warm_start_and_records_its_stop() -> None:
    result = load()
    assert result["tuning_seeds"] == list(TUNING_SEEDS)
    for name in ORDER:
        entry, plan = result["moves"][name], plans()[name]
        first = entry["matches"][0]["probes"][0] if plan.joint else None
        first = first or entry["searches"][0]["trace"][0]
        assert schedule_of(first) == params_of(plan.start, plan.shape)
        assert first.get("threshold") == (0.0 if plan.thresholds else None)
        assert entry["stop"] in (CONVERGED, CAPPED, WALL, COUNT_FIXED, ROUNDS_SPENT)
        assert entry["evaluations"] <= CONVERGE_CAP * len(entry["searches"]) + sum(
            len(match["probes"]) for match in entry["matches"]
        )
        for search in entry["searches"]:
            assert search["stop"] in (CONVERGED, CAPPED, WALL)


@pytest.mark.smoke
def test_the_chosen_evaluation_is_one_the_search_recorded() -> None:
    for entry in load()["moves"].values():
        recorded = [
            record for search in entry["searches"] for record in search["trace"]
        ] + [record for match in entry["matches"] for record in match["probes"]]

        assert entry["chosen"] in recorded
        assert entry["chosen"]["steps"] == entry["count"]
        assert ScheduleShape(entry["chosen"]["shape"]) is ScheduleShape(entry["shape"])


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_first_matches_reproduce_the_earlier_probes_at_their_count() -> None:
    # Wolff's first probe is #1038's at 41,250 steps on the current schedule,
    # Niedermayer's is #1041's at 46,083 on Swendsen-Wang's tuned one: the
    # same runs, so the same energies bitwise.
    result = load()
    earlier = {
        "wolff": load_tuned().raw["matched_wolff"]["probes"],
        "niedermayer": json.loads(CLUSTERS.read_text())["matched_niedermayer"][
            "probes"
        ],
    }
    for name, probes in earlier.items():
        first = result["moves"][name]["matches"][0]["probes"][0]
        match = next(probe for probe in probes if probe["steps"] == first["steps"])

        assert first["energies"] == match["energies"]
        assert first["spent"] == match["spent"]
