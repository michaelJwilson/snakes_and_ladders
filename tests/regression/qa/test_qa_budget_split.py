"""One long anneal against restarts and tempering at one budget, and Wolff with and without Gibbs, on the ci cell (issue #1390, question 5).

`sal.qa.budget_split` finds, on `potts_reference`'s stress cell and its
variants, one long anneal at or below 16 restarts and cluster tempering at
x4 alpha-expansion+ICM's visits, and heat-bath Wolff alone far behind
Wolff composed with Gibbs
(``docs/experiments/035-one-long-anneal-restarts-or-tempering-at-one-budget.md``).
Here both are pinned on the 300-site ``ci`` cell.
"""

from __future__ import annotations

import statistics

import pytest
from sal.qa import budget_split


@pytest.mark.experiment
def test_one_long_anneal_beats_sixteen_restarts_and_tempering_at_x4() -> None:
    # The mean polished gap to the TRW-S bound over 8 seeds, the polish
    # counted in the budget: the long run below restart-16 and tempering.
    _, _, rows = budget_split.split_study(
        names=("ci",), tier="ci", levels=(4,), seeds=range(100, 108)
    )
    gap = {row.arm: statistics.fmean(o.gap for o in row.counted) for row in rows}

    assert gap["long"] >= -1e-9
    assert gap["long"] < gap["restart-16"]
    assert gap["long"] < gap["tempering"]


@pytest.mark.experiment
def test_heat_bath_wolff_alone_trails_wolff_with_gibbs() -> None:
    # 200 sweeps' visits, unpolished: the sequence (WOLFF_HEAT_BATH,) relabels
    # domains and never moves a boundary, so it ends far above the bare WOLFF
    # that composes a Gibbs sweep.
    _, rows = budget_split.move_study(tier="ci", sweeps=(200,), seeds=range(4))
    raw = {row.moves: statistics.fmean(row.raw) for row in rows}

    assert raw["wolff-hb alone"] > 10.0 * max(raw["wolff + gibbs"], 1.0)
