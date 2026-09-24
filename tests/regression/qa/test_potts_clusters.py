"""The committed cluster diagnosis and arms are what `qa.potts_clusters` computes (issue #1041).

`docs/nb/potts_starts.ipynb` reads `docs/nb/data/potts_clusters.json` rather
than running the arms, so what is pinned here is that the file is the
script's: a recorded arm rerun at its seed gives the recorded energies
bitwise, Niedermayer's matched count is the probe nearest the budget and its
search started at Wolff's, and every diagnosis row series has the declared
length.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.opt.starts import StartsBenchmark
from snakes_and_ladders.qa.potts_clusters import (
    DIAGNOSIS_ROWS,
    POLISH,
    arms,
    load,
)
from snakes_and_ladders.qa.potts_schedule import TUNING_SEEDS, release_rung
from snakes_and_ladders.qa.potts_schedule import load as load_tuned
from snakes_and_ladders.search.potts_starts import PottsObjective, polish_by_icm


@pytest.mark.smoke
@pytest.mark.snapshot
def test_a_recorded_arm_reruns_bitwise() -> None:
    recorded = load()["arms"]
    name = "label-directed"
    result = StartsBenchmark(
        PottsObjective(release_rung()),
        {name: arms()[name]},
        polish_by_icm,
        seeding_budget=Budget(Cost.EVALUATIONS, 1),
        polish_budget=POLISH,
        seeds=[0],
        workers=1,
        reference=[recorded["reference"]],
    ).run()
    trial = result.trials(name)[0]

    assert trial.seeded_value == recorded["arms"][name]["handed"][0]
    assert trial.value == recorded["arms"][name]["polished"][0]


@pytest.mark.smoke
def test_the_matched_count_is_the_probe_nearest_the_budget() -> None:
    file = load()
    probes = file["matched_niedermayer"]["probes"]
    nearest = min(probes, key=lambda probe: abs(probe["mean_spent"] - file["budget"]))

    assert file["matched_niedermayer"]["steps"] == nearest["steps"]
    assert probes[0]["steps"] == load_tuned().matched_steps
    for probe in probes:
        assert len(probe["spent"]) == len(TUNING_SEEDS)
        assert probe["mean_spent"] == pytest.approx(float(np.mean(probe["spent"])))


@pytest.mark.smoke
def test_every_diagnosis_series_has_the_declared_rows() -> None:
    diagnosis = load()["diagnosis"]

    assert diagnosis["tuning_seeds"] == list(TUNING_SEEDS)
    for entry in diagnosis["moves"].values():
        for series in entry["per_row"].values():
            assert len(series) == DIAGNOSIS_ROWS
