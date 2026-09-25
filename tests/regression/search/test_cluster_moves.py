"""The two-part cluster arms and the two-end reduction behind them, checked against enumeration and against their own parts (issue #1041).

`search.ground_state.ends_labelling` is the argument that a `spatio_only`
rung's optimum uses only the ladder's two end classes; the referee is the
enumeration of every labelling of `spatio_only/ci`'s nine sites at three
classes, 19,683 of them. The hybrids and the cluster tempering are held to
what they are charged and to the energy of the part they start from.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pytest
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.sample.potts_mcmc import PottsMove
from sal.sample.schedule import ScheduleParams, ScheduleShape
from sal.search.cluster_moves import run_cluster_tempering
from sal.search.ground_state import (
    EXPANSION_RESERVE_CYCLES,
    Rung,
    ends_labelling,
    expansion_then_swendsen_wang,
    run_alpha_expansion,
    run_annealed,
    swendsen_wang_then_expansion,
)
from sal.search.potts_starts import spatio_rung
from sal.sim.fixtures import fixture
from sal.sim.potts import energies, energy

#: Sweeps' worth of site visits for the release-size arms: the notebook's
#: 1,000 cut to keep each run near a second.
SWEEPS = 40
SCHEDULE = ScheduleParams(ScheduleShape.LINEAR, 0.7836, 0.3236)


@functools.cache
def _ci() -> Rung:
    return spatio_rung(fixture("spatio_only", "ci").params, "ci-q3")


@functools.cache
def _release() -> Rung:
    return spatio_rung(fixture("spatio_only", "release").params, "release-q10")


def _budget(rung: Rung) -> Budget:
    return Budget(Cost.SITE_VISITS, SWEEPS * rung.visits_per_sweep)


@pytest.mark.oracle
def test_the_optimum_over_every_class_is_the_optimum_over_the_two_ends() -> None:
    rung = _ci()
    every = np.array(
        list(itertools.product(range(rung.n_states), repeat=rung.n_nodes)),
        dtype=np.int64,
    )
    values = energies(rung.graph, rung.field, every)
    ends = {int(np.argmin(rung.alpha)), int(np.argmax(rung.alpha))}
    on_ends = np.isin(every, list(ends)).all(axis=1)

    assert values.min() == values[on_ends].min()


@pytest.mark.analytic
def test_the_reduction_never_raises_an_energy() -> None:
    # Every labelling of the nine sites, mapped to the two ends.
    rung = _ci()
    every = np.array(
        list(itertools.product(range(rung.n_states), repeat=rung.n_nodes)),
        dtype=np.int64,
    )
    moved = np.array([ends_labelling(rung, labelling) for labelling in every])
    before = energies(rung.graph, rung.field, every)
    after = energies(rung.graph, rung.field, moved)

    assert (after <= before + 1e-12).all()
    assert set(np.unique(moved)) <= {
        int(np.argmin(rung.alpha)),
        int(np.argmax(rung.alpha)),
    }


@pytest.mark.analytic
def test_the_reduction_lowers_a_ten_class_anneal_to_two_classes() -> None:
    rung = _release()
    run = run_annealed(
        rung,
        _budget(rung),
        np.random.default_rng(0),
        PottsMove.SWENDSEN_WANG,
        schedule=SCHEDULE,
    )
    moved = ends_labelling(rung, run.labelling)

    assert energy(rung.graph, rung.field, moved) <= run.energy + 1e-9
    assert np.unique(moved).size <= 2


@pytest.mark.smoke
def test_a_field_not_linear_in_the_ladder_is_refused() -> None:
    rung = _ci()
    bent = Rung(
        name="bent",
        graph=rung.graph,
        field=rung.field + np.eye(rung.n_nodes, rung.n_states),
        alpha=rung.alpha,
        sizes=rung.sizes,
        n_states=rung.n_states,
        optimum=None,
    )
    with pytest.raises(ValueError, match="alpha times"):
        ends_labelling(bent, np.zeros(rung.n_nodes, dtype=np.int64))


@pytest.mark.analytic
@pytest.mark.parametrize("seed", [0, 1])
def test_expansion_then_swendsen_wang_hands_over_no_more_than_the_expansion(
    seed: int,
) -> None:
    rung = _release()
    budget = _budget(rung)
    expansion = run_alpha_expansion(rung, budget, np.random.default_rng(seed))
    run = expansion_then_swendsen_wang(SCHEDULE)(
        rung, budget, np.random.default_rng(seed)
    )

    assert run.energy <= expansion.energy
    assert expansion.spent < run.spent <= budget.size


@pytest.mark.analytic
def test_swendsen_wang_then_expansion_is_charged_both_parts() -> None:
    # The reserve's ten cycles and forty sweeps of anneal.
    rung = _release()
    per_cycle = rung.n_states * rung.visits_per_sweep
    budget = Budget(
        Cost.SITE_VISITS,
        EXPANSION_RESERVE_CYCLES * per_cycle + SWEEPS * rung.visits_per_sweep,
    )
    run = swendsen_wang_then_expansion(SCHEDULE)(rung, budget, np.random.default_rng(3))
    anneal = run_annealed(
        rung,
        _budget(rung),
        np.random.default_rng(3),
        PottsMove.SWENDSEN_WANG,
        schedule=SCHEDULE,
    )

    assert run.energy <= anneal.energy
    assert anneal.spent < run.spent <= budget.size
    assert (run.spent - anneal.spent) % per_cycle == 0


@pytest.mark.analytic
def test_cluster_tempering_spends_within_its_budget() -> None:
    rung = _release()
    budget = _budget(rung)
    run = run_cluster_tempering(
        rung, budget, np.random.default_rng(5), t_hot=0.7836, t_cold=0.3236
    )

    assert run.spent <= budget.size
    assert run.energy == pytest.approx(
        energy(rung.graph, rung.field, run.labelling), abs=1e-9
    )
    assert run.trace[0].proposals > 0
