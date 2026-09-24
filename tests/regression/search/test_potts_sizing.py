"""Where the Potts baseline first fails, and which instrument says so.

Issue #596, first arm. In zero field the optimum is ``-J * n_edges``, three
ways to 2.7e-16; restart-ICM and Wolff reach it from every start (64 sites
here, 1,024 in the sweep), so nothing can outperform it. In a field at two
labels alpha-expansion is exact, bitwise. At three labels the baseline fails,
and by how much depends on the instrument: at 144 sites exact hit reads 0.00
for every method while the runs sit 0.04-9.06% above the expansion and
annealing reaches 0.38 within 1%. One test is `release` (25x the budget); the
rest take 2.0 s.
"""

from __future__ import annotations

import numpy as np
import pytest
from snakes_and_ladders.cost import Cost
from snakes_and_ladders.learn.failure import probe_failure
from snakes_and_ladders.opt.budget import Budget, restarts
from snakes_and_ladders.search.ground_state import (
    Entry,
    lattice_rung,
    uniform_ground_energy,
)
from snakes_and_ladders.search.maxflow import ising_ground_state
from snakes_and_ladders.sim.potts import energy

from tests._rows import every_value

#: Sweeps of budget every probe gets, converted to site visits per rung. The
#: unit is the module's own (`search/ground_state.py`): a Wolff step flips one
#: cluster while a heat-bath sweep touches every site, so equal sweeps would
#: hand the cluster moves a free lattice per move.
SWEEPS = 200
#: Starts per probe in the fast tier. Eight is enough to separate a fraction of
#: 1.00 from one of 0.00, which is the claim here; the release tier runs twelve.
STARTS = 8
#: Sweeps a single descent is billed for inside the restart baseline. ICM
#: settles well inside this on these instances, so the restart count is the
#: budget's own arithmetic and not a tuned number.
DESCENT = 8


def _budget(rung: object) -> Budget:
    """The matched budget at this rung, in site visits."""
    return Budget(size=SWEEPS * rung.visits_per_sweep, unit=Cost.SITE_VISITS)  # type: ignore[attr-defined]


@pytest.mark.oracle
def test_the_zero_field_optimum_is_a_closed_form_three_ways() -> None:
    """``-J * n_edges``, the exact cut, and a uniform labelling agree.

    1.7e-16, 2.7e-16 and 1.4e-16 relative: summation order, not disagreement.
    """

    def check(side: int) -> None:
        rung = lattice_rung(side, 2)
        closed = uniform_ground_energy(rung)
        _, cut = ising_ground_state(rung.graph, rung.field)
        uniform = energy(rung.graph, rung.field, np.zeros(rung.n_nodes, dtype=np.int64))

        assert closed == pytest.approx(cut, rel=1e-12)
        assert closed == pytest.approx(uniform, rel=1e-12)
        assert rung.optimum == closed

    every_value([4, 6, 8], check)


@pytest.mark.smoke
def test_the_closed_form_is_refused_in_a_field() -> None:
    """A field breaks the derivation, so the closed form refuses rather than lies.

    A wrong target reads as a baseline that never succeeds: #596's finding, falsely.
    """
    with pytest.raises(ValueError, match="zero field only"):
        uniform_ground_energy(lattice_rung(6, 3, seed=6))


@pytest.mark.oracle
def test_the_zero_field_baseline_does_not_fail() -> None:
    """Restart-ICM and Wolff reach the closed-form optimum from every start.

    So zero field is excluded from the gate: the baseline never fails there.
    """
    rung = lattice_rung(8, 3)
    target = uniform_ground_energy(rung)
    budget = _budget(rung)

    for method in (
        restarts(Entry("icm"), cost=DESCENT * rung.visits_per_sweep),
        Entry("wolff"),
    ):
        probe = probe_failure(
            method,
            rung,
            target=target,
            budget=budget,
            starts=STARTS,
            rng=np.random.default_rng(596),
            size=rung.n_nodes,
        )
        assert probe.fraction == 1.0
        assert not probe.truncated


@pytest.mark.oracle
def test_alpha_expansion_is_exact_at_two_labels() -> None:
    """In a field at ``q = 2`` the expansion equals the graph cut, gap 0.0.

    Bitwise here and at 16, 24, 32 in the sweep: excluded, as #596 says of exact baselines.
    """

    def check(side: int) -> None:
        rung = lattice_rung(side, 2, seed=side)
        _, exact = ising_ground_state(rung.graph, rung.field)

        expansion = Entry("alpha-expansion")(
            rung, _budget(rung), np.random.default_rng(596)
        )

        assert expansion.value == exact

    every_value([8, 12], check)


@pytest.mark.analytic
def test_the_two_instruments_disagree_on_the_same_runs() -> None:
    """Exact hit reads 0.00 where the relative reading reads the failure.

    144 sites, three labels: ICM 5.31% off; annealing within 0.04%, 0.38 of
    starts within 1%; exact hit reads 0.00. Hence `probe_failure`'s `relative`.
    """
    rung = lattice_rung(12, 3, seed=12)
    budget = _budget(rung)
    target = Entry("alpha-expansion")(rung, budget, np.random.default_rng(596)).value

    readings = {}
    for name in ("icm", "anneal"):
        readings[name] = tuple(
            probe_failure(
                Entry(name),
                rung,
                target=target,
                budget=budget,
                starts=STARTS,
                rng=np.random.default_rng(596),
                size=rung.n_nodes,
                relative=relative,
            )
            for relative in (None, 0.01)
        )

    exact_icm, loose_icm = readings["icm"]
    exact_anneal, loose_anneal = readings["anneal"]

    # ICM fails under both readings, and by a margin no allowance of this size
    # closes: measured 5.31% above the target.
    assert exact_icm.fraction == 0.0
    assert loose_icm.fraction == 0.0
    assert (exact_icm.best - target) / abs(target) > 0.02

    # Annealing is where the instruments part: exact hit cannot see it at all
    # and a 1% allowance puts it on 0.38 of starts.
    assert loose_anneal.fraction > exact_anneal.fraction
    assert loose_anneal.fraction > 0.0


@pytest.mark.smoke
def test_a_negative_relative_allowance_is_refused() -> None:
    """An allowance below zero would count nothing as reaching anything."""
    rung = lattice_rung(4, 2)
    with pytest.raises(ValueError, match="relative allowance is non-negative"):
        probe_failure(
            Entry("icm"),
            rung,
            target=uniform_ground_energy(rung),
            budget=_budget(rung),
            starts=1,
            rng=np.random.default_rng(0),
            size=rung.n_nodes,
            relative=-0.1,
        )


@pytest.mark.smoke
def test_beating_the_target_counts_as_reaching_it() -> None:
    """A start better than the target has not failed, and the reading is one-sided.

    #596's target may be the best known value, not an optimum.
    """
    rung = lattice_rung(6, 2)
    optimum = uniform_ground_energy(rung)

    probe = probe_failure(
        Entry("wolff"),
        rung,
        # A target deliberately worse than the optimum: every start beats it.
        target=optimum + 10.0,
        budget=_budget(rung),
        starts=4,
        rng=np.random.default_rng(596),
        size=rung.n_nodes,
    )

    assert probe.best < probe.target
    assert probe.fraction == 1.0


@pytest.mark.analytic
def test_the_field_baseline_fails_and_wolff_fails_hardest() -> None:
    """At 576 sites every local method sits 3--34% above alpha-expansion.

    Wolff degrades in a field (3.4% at 144 sites, 34.4% at 576); annealing is
    the baseline (0.17 exact hits, all within 1%). The gate is argued against
    annealing and the expansion; the ordering is asserted. Eight starts: at
    twelve ICM reads 8.06% rather than 10.59%.
    """
    rung = lattice_rung(24, 3, seed=24)
    budget = _budget(rung)
    target = Entry("alpha-expansion")(rung, budget, np.random.default_rng(596)).value

    excess = {}
    for name in ("icm", "wolff", "anneal"):
        probe = probe_failure(
            Entry(name),
            rung,
            target=target,
            budget=budget,
            starts=STARTS,
            rng=np.random.default_rng(596),
            size=rung.n_nodes,
        )
        excess[name] = (probe.best - target) / abs(target)

    assert excess["anneal"] < excess["icm"] < excess["wolff"]
    assert excess["anneal"] == pytest.approx(0.0, abs=1e-9)
    assert excess["wolff"] > 0.2


@pytest.mark.release
@pytest.mark.analytic
def test_icm_s_shortfall_is_flat_in_the_budget() -> None:
    """A single descent gains nothing from 25x the budget; its restarts do.

    ICM 5.31% above at 200 and at 5,000 sweeps; restarts close from 3.06% to
    1.24%, the comparison #596 insists on. `release`, 1.1 s.
    """
    rung = lattice_rung(12, 3, seed=12)
    reference = _budget(rung)
    target = Entry("alpha-expansion")(rung, reference, np.random.default_rng(596)).value

    shortfall = {}
    for sweeps in (SWEEPS, 25 * SWEEPS):
        budget = Budget(size=sweeps * rung.visits_per_sweep, unit=Cost.SITE_VISITS)
        for name, method in (
            ("icm", Entry("icm")),
            (
                "restart-icm",
                restarts(Entry("icm"), cost=DESCENT * rung.visits_per_sweep),
            ),
        ):
            probe = probe_failure(
                method,
                rung,
                target=target,
                budget=budget,
                starts=STARTS,
                rng=np.random.default_rng(596),
                size=rung.n_nodes,
            )
            shortfall[name, sweeps] = (probe.best - target) / abs(target)

    assert shortfall["icm", SWEEPS] == pytest.approx(
        shortfall["icm", 25 * SWEEPS], rel=1e-12
    )
    assert (
        shortfall["restart-icm", 25 * SWEEPS] < 0.5 * shortfall["restart-icm", SWEEPS]
    )
