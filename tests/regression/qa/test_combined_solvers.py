"""The combined solver study's table on the seeded fixtures, and its two seams (issue #1414).

`qa.combined_solvers` runs on a downstream package's stream where it is
installed (`docs/experiments/041`), and on the `potts_labelling` and
`count_hmm_reference` CI cells where it is not. This pins the untimed table
on those cells; that the anneal's arm polished outside the loop is
`Polish.ICM_MERGE`'s labelling; and that the stream adapter refuses before
starting a process where the downstream package is absent.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sal.cost import Cost
from sal.external import runner, stream
from sal.opt.budget import Budget
from sal.qa import combined_solvers as study
from sal.sample.potts_mcmc import PottsMove
from sal.sample.schedule import ExponentialTempSchedule, Polish

#: The untimed table at `study.CI` on the CI cells, as measured.
TABLE = """| panel | arm | stage | n | gap, median [IQR] | missed %, median [IQR] |
| --- | --- | --- | --- | --- | --- |
| a | argmax | before | 1 | 148.13 [148.13, 148.13] | 29.3 [29.3, 29.3] |
| a | argmax | after | 1 | 6.33 [6.33, 6.33] | 15.3 [15.3, 15.3] |
| a | icm | before | 1 | 7.04 [7.04, 7.04] | 16.3 [16.3, 16.3] |
| a | icm | after | 1 | 7.04 [7.04, 7.04] | 16.3 [16.3, 16.3] |
| a | downstream icm+floor+merge | before | 1 | 7.04 [7.04, 7.04] | 16.3 [16.3, 16.3] |
| a | downstream icm+floor+merge | after | 1 | 7.04 [7.04, 7.04] | 16.3 [16.3, 16.3] |
| a | icm+floor | before | 1 | 7.04 [7.04, 7.04] | 16.3 [16.3, 16.3] |
| a | icm+floor | after | 1 | 7.04 [7.04, 7.04] | 16.3 [16.3, 16.3] |
| a | ae+icm | before | 1 | 0.00 [0.00, 0.00] | 14.0 [14.0, 14.0] |
| a | ae+icm | after | 1 | 0.00 [0.00, 0.00] | 14.0 [14.0, 14.0] |
| a | trws | before | 1 | 0.00 [0.00, 0.00] | 14.0 [14.0, 14.0] |
| a | trws | after | 1 | 0.00 [0.00, 0.00] | 14.0 [14.0, 14.0] |
| a | anneal heat bath | before | 1 | 0.30 [0.30, 0.30] | 14.8 [14.8, 14.8] |
| a | anneal heat bath | after | 1 | 0.30 [0.30, 0.30] | 14.8 [14.8, 14.8] |
| a | anneal wolff+gibbs | before | 1 | 0.02 [0.02, 0.02] | 14.3 [14.3, 14.3] |
| a | anneal wolff+gibbs | after | 1 | 0.02 [0.02, 0.02] | 14.3 [14.3, 14.3] |
| a | anneal sw+gibbs | before | 1 | 0.02 [0.02, 0.02] | 14.3 [14.3, 14.3] |
| a | anneal sw+gibbs | after | 1 | 0.02 [0.02, 0.02] | 14.3 [14.3, 14.3] |
| a | merge | before | 1 | 279.18 [279.18, 279.18] | 38.5 [38.5, 38.5] |
| a | merge | after | 1 | 13.67 [13.67, 13.67] | 17.5 [17.5, 17.5] |
| b | run start | start | 1 | 620.73 [620.73, 620.73] | 50.0 [50.0, 50.0] |
| b | run start | downstream fit | 1 | 31.21 [31.21, 31.21] | 38.0 [38.0, 38.0] |
| b | run start | converged | 1 | 27.43 [27.43, 27.43] | 44.0 [44.0, 44.0] |
| b | restart | start | 1 | 476.84 [476.84, 476.84] | 16.0 [16.0, 16.0] |
| b | restart | downstream fit | 1 | 21.66 [21.66, 21.66] | 62.0 [62.0, 62.0] |
| b | restart | converged | 1 | 0.00 [0.00, 0.00] | 58.0 [58.0, 58.0] |
| b | k-means++ | start | 1 | 605.04 [605.04, 605.04] | 30.0 [30.0, 30.0] |
| b | k-means++ | downstream fit | 1 | 86.54 [86.54, 86.54] | 42.0 [42.0, 42.0] |
| b | k-means++ | converged | 1 | 47.68 [47.68, 47.68] | 38.0 [38.0, 38.0] |
| b | quantile | start | 1 | 462.76 [462.76, 462.76] | 40.0 [40.0, 40.0] |
| b | quantile | downstream fit | 1 | 28.58 [28.58, 28.58] | 44.0 [44.0, 44.0] |
| b | quantile | converged | 1 | 12.01 [12.01, 12.01] | 62.0 [62.0, 62.0] |
"""


@pytest.fixture(autouse=True)
def _single_thread() -> Iterator[None]:
    # The suite's one thread per process; restored after.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.smoke
@pytest.mark.snapshot
def test_the_untimed_table_on_the_ci_cells_is_conserved() -> None:
    found = study.from_fixtures("ci")
    runs = study.study(found, 0, study.CI)
    assert study.table(study.summarize(runs), timed=False) == TABLE


@pytest.mark.smoke
@pytest.mark.patch
@pytest.mark.parametrize(
    "move", [PottsMove.SINGLE_SITE, PottsMove.WOLFF, PottsMove.SWENDSEN_WANG], ids=str
)
def test_the_polish_after_the_anneal_is_the_loops_icm_merge_bitwise(
    move: PottsMove,
) -> None:
    # A short hot anneal, so the polish has something to change.
    found = study.from_fixtures("ci")
    work = study._Held(found.labelling)
    schedule = ExponentialTempSchedule(3.0 * work.t0, work.t0, 5)
    budget = Budget(Cost.SITE_VISITS, 5 * work.visits)
    for seed in range(3):
        start = np.random.default_rng([seed, 0]).integers(
            0, found.labelling.n_labels, found.labelling.n_sites
        )
        looped = work.held.anneal(
            schedule, np.random.default_rng(seed), start=start, move=move,
            budget=budget, polish=Polish.ICM_MERGE,
        )  # fmt: skip
        raw = work.held.anneal(
            schedule, np.random.default_rng(seed), start=start, move=move,
            budget=budget,
        )  # fmt: skip
        composed = study.polish(
            work, np.asarray(raw.labelling), np.random.default_rng(0)
        )
        assert work.held.energy(composed) < work.held.energy(raw.labelling)
        np.testing.assert_array_equal(np.asarray(looped.labelling), composed)


@pytest.mark.smoke
def test_an_absent_downstream_package_is_refused_before_any_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> None:
        started.append((args, kwargs))
        message = "no subprocess may start"
        raise AssertionError(message)

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(runner, "installed", lambda _module: False)
    assert not stream.available()
    with pytest.raises(ModuleNotFoundError, match="installed by hand"):
        stream.load_or_capture(tmp_path, stream.MANIFEST, 5)
    assert started == []
    assert list(tmp_path.iterdir()) == []
