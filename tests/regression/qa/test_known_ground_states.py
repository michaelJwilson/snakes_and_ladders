"""Which method reached further against the exact cut at the CI size (issue #1378, Part A).

`qa.known_ground_states` runs the whole grid as a study; this pins one cell of
it, the seeded 12 x 12 two-label rung in a field at the ladder's top level on
seed 0, where alpha-expansion+ICM reaches the graph cut's optimum and the
tuned anneal and restart-ICM do not. `docs/experiments/029-*.md` reports the
grid.
"""

from __future__ import annotations

import pytest
from sal.qa.known_ground_states import LEVELS, cut_instance, measure_instance

#: The gaps as measured on seed 0: a sum of ~400 terms on one path, so they
#: are pinned to the last bits a reordered reduction could move, not bitwise.
GAP_TOLERANCE = 1e-9
#: ``(E - E*) / |E*|`` of the tuned anneal and of restart-ICM, measured.
ANNEAL_GAP = 0.008845881460462541
RESTART_GAP = 0.0244235151578825


@pytest.mark.experiment
def test_expansion_reaches_the_cut_where_the_tuned_anneal_and_restarts_stop_short() -> (
    None
):
    cells = {
        c.method: c
        for c in measure_instance(cut_instance(12), seeds=(0,), levels=(LEVELS[-1],))
    }

    assert cells["ae+icm"].exact == 1.0
    assert cells["anneal-sw"].exact == 0.0
    assert cells["restart-icm"].exact == 0.0
    assert cells["anneal-sw"].median_gap == pytest.approx(ANNEAL_GAP, abs=GAP_TOLERANCE)
    assert cells["restart-icm"].median_gap == pytest.approx(
        RESTART_GAP, abs=GAP_TOLERANCE
    )
    assert cells["ae+icm"].spent * LEVELS[-1] == cells["restart-icm"].spent
    assert [name for name, _ in cells["anneal-sw"].stages] == [
        "init",
        "polish",
        "merge",
    ]
