"""ICM and the label merge alternated to a joint fixed point, on the ci cell (issue #1390, adaptation).

`sal.qa.icm_merge_alternation` finds that a second ICM pass after
``Polish.ICM_MERGE`` recovers no energy on `potts_reference`'s stress cell and
five variants: the merge fires only where it collapses the labelling to the
one-label optimum (``docs/experiments/036-alternating-icm-and-the-label-merge.md``).
Here both behaviours are pinned on the 300-site ``ci`` cell.
"""

from __future__ import annotations

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.qa import icm_merge_alternation as alternation
from sal.sample.potts_mcmc import anneal_potts
from sal.sim.fixtures import fixture
from sal.sim.potts_cell import PottsReferenceParams


def _margin_one_tenth() -> PottsReferenceParams:
    # The ci cell with stress's margin_0.1 variant applied: ci declares no variants.
    params: PottsReferenceParams = fixture("potts_reference", "ci").params
    changed = {**params.declared, "margin_ratio": 0.1, "planted_digest": ""}
    return type(params).from_declared(changed, params.path)


@pytest.mark.experiment
def test_one_pass_is_the_joint_fixed_point_on_the_ci_cell() -> None:
    # 8 seeds per start; study() asserts round 1 equals Polish.ICM_MERGE bitwise.
    ci = alternation.reference_cells("ci", ("ci",))
    rows = alternation.study(ci, seeds=range(8))

    assert all(run.gain == 0.0 for row in rows for run in row.runs)
    assert all(len(run.rounds) == 1 for row in rows for run in row.runs)


@pytest.mark.experiment
def test_a_collapse_reaches_the_bound_and_the_second_icm_keeps_it() -> None:
    # At margin 0.1, from 8 random and 8 x1-anneal starts, 8 merges leave one
    # label 25.6 to 54.9 nats below ICM; the next ICM sweep is then clean.
    rows = alternation.study(
        [("margin_0.1", _margin_one_tenth())], levels=(1,), seeds=range(8)
    )
    runs = [run for row in rows for run in row.runs]
    collapsed = [run for run in runs if run.collapsed]

    assert len(collapsed) >= 4
    assert all(len(run.rounds) == 2 for run in collapsed)
    assert all(run.gain == 0.0 for run in runs)
    assert all(abs(run.final - rows[0].bound) < 1e-6 for run in collapsed)


@pytest.mark.experiment
def test_the_init_stage_is_the_unpolished_best() -> None:
    # The study's start, Polish.ICM_MERGE's init stage, equals polish=None's best.
    each = alternation.cell(fixture("potts_reference", "ci").params, "ci")
    init, _, _, _ = alternation.anneal_start(each, 4, 0)
    budget = 4 * each.base
    bare = anneal_potts(
        each.problem.graph,
        each.problem.field,
        each.schedule().build(max(2, budget // each.per_sweep)),
        np.random.default_rng([0, 4]),
        budget=Budget(Cost.SITE_VISITS, budget),
    )

    np.testing.assert_array_equal(init, np.asarray(bare.best))
