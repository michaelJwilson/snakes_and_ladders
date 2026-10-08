"""The Potts reference cell rebuilds from its seed, and its baselines recompute (issue #1390).

`potts_reference` declares an instance drawn from a seed, never committed:
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
from sal.sample.potts_mcmc import PottsMove, Recolour, anneal_potts
from sal.sample.schedule import Polish
from sal.search.ground_state import ANNEAL_SCHEDULE
from sal.sim.fixtures import baseline, fixture, read_baseline
from sal.sim.potts_cell import PlantedPotts, PottsReferenceParams, planted_digest

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
def test_the_potts_cell_rebuilds_from_its_seed_bitwise(tier: Scale) -> None:
    # The digest pins the planted labels and the field; the shares are the
    # quantile cuts', so each label's count is its share to within one site.
    params: PottsReferenceParams = fixture("potts_reference", tier).params
    cell = params.instance()

    assert planted_digest(cell) == params.planted_digest
    assert planted_digest(params.instance()) == params.planted_digest
    counts = np.bincount(cell.planted, minlength=params.n_states)
    np.testing.assert_allclose(counts, np.array(params.shares) * params.n_nodes, atol=1)
    assert cell.margin == pytest.approx(3.0)


@pytest.mark.smoke
def test_every_variant_changes_only_the_factor_it_declares() -> None:
    # A variant is one factor from `stress`: every field it does not name is
    # the stress cell's, and every variant builds.
    stress = fixture("potts_reference", Scale.STRESS).params
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
    (spec,) = baseline_script.selected([f"potts_reference/{tier}"])
    computed = baseline_script.compute(spec)

    assert baseline_script.differences(computed, read_baseline(computed.path)) == []


@pytest.mark.analytic
@pytest.mark.parametrize("tier", FAST)
def test_the_potts_bracket_holds(tier: Scale) -> None:
    # TRW-S's bound is below every labelling's energy, the planted and its
    # own decoding among them.
    record = baseline("potts_reference", tier)
    bound = record.value("trws_bound")

    assert bound <= record.value("trws_energy") + 1e-9 * abs(bound)
    assert bound <= record.value("planted_energy")


@pytest.mark.end2end
def test_an_anneal_on_the_potts_cell_lands_inside_the_bracket() -> None:
    # The downstream run's arm on the ci cell, judged against the truth that
    # built it: no labelling scores below the TRW-S bound, and 200 sweeps of
    # Swendsen-Wang then ICM and the merge reach the planted energy or below.
    params: PottsReferenceParams = fixture("potts_reference", Scale.CI).params
    cell: PlantedPotts = params.instance()
    record = baseline("potts_reference", Scale.CI)
    visits = cell.graph.n_nodes + 2 * len(cell.graph.edges)
    run = anneal_potts(
        cell.graph,
        cell.field,
        ANNEAL_SCHEDULE.build(200),
        np.random.default_rng(0),
        move=PottsMove.SWENDSEN_WANG,
        recolour=Recolour.HEAT_BATH,
        budget=Budget(Cost.SITE_VISITS, 200 * visits),
        polish=Polish.ICM_MERGE,
    )
    bound = record.value("trws_bound")

    assert run.energy >= bound - 1e-9 * abs(bound)
    assert run.energy <= cell.planted_energy
