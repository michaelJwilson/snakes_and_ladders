"""Which arm reaches the TRW-S bound on the labelling cell at the CI size (issue #1413).

`qa.potts_labelling` runs every arm from two starts at ``stress`` and its
variants; `docs/experiments/040-*.md` reports it. This pins the CI cell from
the field's argmax: alpha-expansion+ICM, TRW-S's decode and the anneal reach
the bound, ICM stops 6.33 nats above it with or without the floor or the
merge, and the descent crosses into Rust once per solve.
"""

from __future__ import annotations

import pytest
from sal.qa.potts_labelling import ARMS, FLOOR_SHARE, Instance, measure
from sal.search.trws import trws
from sal.sim.fixtures import fixture

#: A sum of ~1,200 terms on one path: pinned to the last bits a reordered
#: reduction could move, not bitwise.
ENERGY_TOLERANCE = 1e-9
#: The TRW-S bound, and the argmax's and ICM's energies above it, as measured.
BOUND = -1405.3453831388447
ARGMAX_GAP = 148.13090678780918
ICM_GAP = 6.330127668826663


@pytest.mark.experiment
def test_expansion_trws_and_the_anneal_reach_the_bound_where_icm_stops_short() -> None:
    cell = fixture("potts_labelling", "ci").params.instance()
    floor = round(FLOOR_SHARE * cell.graph.n_nodes)
    work = Instance(cell, floor, float(trws(cell.graph, cell.field).bound), 0.0)
    rows = {arm: measure(work, arm, "argmax") for arm in ARMS}
    scale = ENERGY_TOLERANCE * abs(BOUND)

    assert work.bound == pytest.approx(BOUND, abs=scale)
    assert rows["argmax"].gap == pytest.approx(ARGMAX_GAP, abs=scale)
    for arm in ("ae+icm", "trws", "anneal+icm_merge"):
        assert abs(rows[arm].gap) < scale, arm
        assert rows[arm].unlike == 42, arm
    for arm in ("icm", "icm-floor", "icm+floor-smallest", "icm+merge"):
        assert rows[arm].gap == pytest.approx(ICM_GAP, abs=scale), arm
    assert rows["icm"].crossings == 1
    assert rows["argmax"].crossings == 0
