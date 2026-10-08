"""The count-pair HMM reference cell rebuilds from its seed, and its baselines recompute (issue #1390).

`count_hmm_reference` declares an instance drawn from a seed, never committed:
each tier's file pins the draw by a digest, and each record beside it is
recomputed by `infra/baselines.py` here. The variants are held to the one
factor each declares.
"""

from __future__ import annotations

import dataclasses

import baselines as baseline_script
import numpy as np
import pytest
from sal.cost import Cost
from sal.fixtures import Scale
from sal.opt.budget import Budget
from sal.opt.hmm import EmissionHmmObjective
from sal.opt.starts import polish_by_baum_welch
from sal.ragged import Ragged
from sal.sim.count_hmm_cell import (
    CountHmm,
    CountHmmReferenceParams,
    observations_digest,
)
from sal.sim.fixtures import baseline, fixture, read_baseline

#: The tiers the per-pull-request tests rebuild; `release` is 1e5 and rebuilt
#: by the release gate.
FAST = (Scale.CI, Scale.STRESS)


def _tiers() -> list[object]:
    return [
        *FAST,
        pytest.param(Scale.RELEASE, marks=pytest.mark.release),
    ]


@pytest.mark.analytic
@pytest.mark.parametrize("tier", _tiers())
def test_the_count_hmm_cell_rebuilds_from_its_seed_bitwise(tier: Scale) -> None:
    # The digest pins the paths, the pairs and the covariate; the chain is
    # stochastic with stationary law `shares`.
    params: CountHmmReferenceParams = fixture("count_hmm_reference", tier).params
    cell = params.instance()

    assert observations_digest(cell) == params.observations_digest
    np.testing.assert_allclose(cell.transition.sum(axis=1), 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        params.shares @ cell.transition, params.shares, rtol=0, atol=1e-15
    )
    assert np.all(cell.observations[:, 1] <= cell.covariate[:, 1])


@pytest.mark.smoke
def test_every_variant_changes_only_the_factor_it_declares() -> None:
    # A variant is one factor from `stress`: every field it does not name is
    # the stress cell's, and every variant builds.
    stress = fixture("count_hmm_reference", Scale.STRESS).params
    held = {"variants", "declared", "planted_digest", "observations_digest"}
    assert stress.variants
    for name, changed in stress.variants.items():
        variant = stress.variant(name)
        for entry in dataclasses.fields(stress):
            if entry.name in held or entry.name in changed:
                continue
            ours, theirs = getattr(variant, entry.name), getattr(stress, entry.name)
            assert np.array_equal(ours, theirs), f"{name} moved {entry.name}"
        variant.instance()


@pytest.mark.oracle
@pytest.mark.parametrize("tier", FAST)
def test_the_reference_baselines_recompute(tier: Scale) -> None:
    # The record against a fresh run of `infra/baselines.py`'s reference
    # algorithms, at the tolerance each measurement declares.
    (spec,) = baseline_script.selected([f"count_hmm_reference/{tier}"])
    computed = baseline_script.compute(spec)

    assert baseline_script.differences(computed, read_baseline(computed.path)) == []


@pytest.mark.end2end
def test_baum_welch_from_the_generating_parameters_does_not_lose_likelihood() -> None:
    # The generating parameters are a point of the likelihood, not its
    # maximum: twenty Baum-Welch iterations from them climb, and never below
    # the recorded log-likelihood at the truth.
    params: CountHmmReferenceParams = fixture("count_hmm_reference", Scale.CI).params
    cell: CountHmm = params.instance()
    objective = EmissionHmmObjective(
        Ragged(cell.observations, cell.lengths),
        cell.components,
        covariate=cell.covariate,
    )
    theta = objective.theta_from_truth(
        cell.initial, cell.transition, **cell.components.named_parameters()
    )
    at_truth = baseline("count_hmm_reference", Scale.CI).value(
        "generating_log_likelihood"
    )
    polished = polish_by_baum_welch(objective, theta, Budget(Cost.ITERATIONS, 20))

    assert -float(objective(theta)) == pytest.approx(at_truth, rel=1e-12)
    assert -float(objective(polished.theta)) >= at_truth - 1e-9 * abs(at_truth)
