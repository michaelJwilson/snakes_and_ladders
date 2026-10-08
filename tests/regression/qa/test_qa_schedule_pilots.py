"""Schedule pilots on ``potts_reference/ci``: what they cost and how far their ranking transfers (issue #1390, questions 2 and 3).

The study's own functions at the ``ci`` tier with fewer seeds, serially:
the full-length reference on 8 held-out seeds, question 2's pilots on 4
tuning seeds and question 3's four combinations on 8. What is pinned is the
finding of ``docs/experiments/033-schedule-pilot-transfer.md`` where it is
measured at this size: the pilots' share of cost, racing's spend at an equal
budget, and a pilot ranking that agrees with the full-length one at tau
below 0.5 at every pilot length.
"""

from __future__ import annotations

import pytest
from sal.qa.schedule_pilots import (
    COMBINATIONS,
    references,
    selections,
    transfer,
)
from sal.sample.tune import SCHEDULE_GRID

CI = ("ci", None)
#: Pilot length over run length: the shortest and the run's own.
FRACTIONS = (1 / 16, 1.0)


@pytest.mark.experiment
def test_on_the_ci_cell_pilots_cost_most_of_the_tuned_run_and_rank_it_weakly() -> None:
    refs = references([CI], seeds=range(8), workers=1)
    moved = transfer([CI], refs, fractions=FRACTIONS, seeds=range(4), workers=1)
    # The pilots' share is len(grid) * f over that plus one run: 3/7 and 12/13.
    grid = len(SCHEDULE_GRID)
    for fraction in FRACTIONS:
        share = moved[CI, fraction].share
        assert share == pytest.approx(grid * fraction / (grid * fraction + 1))
        # Measured 0.24 at 1/16 and 0.11 at 1: the full-length grid spans
        # 0.78 nats after the polish, so its ranking is mostly seed noise.
        assert moved[CI, fraction].tau < 0.5
    # At one budget racing runs 12 x s/4 + 6 x s/2 + 3 x s = 9 s of the
    # grid's 12 s sweeps, whatever common does.
    chosen = selections([CI], refs, fractions=(1 / 4,), seeds=range(8), workers=1)
    base = chosen[CI, 1 / 4, False, False].spent
    for racing, common in COMBINATIONS:
        spent = chosen[CI, 1 / 4, racing, common].spent
        assert spent == (0.75 * base if racing else base)
