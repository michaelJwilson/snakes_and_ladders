"""Where the Potts baseline first fails, and which instrument says so.

Issue #596, first per-problem arm. The ticket asks for the size at which the
classical baseline *measurably fails*, because a gate cannot be argued on a
fixture where it does not. Two answers came out of the sweep and they are kept
apart, since each excludes something different:

* **In zero field the optimum is a closed form**, ``-J * n_edges``, checked
  three ways --- against the exact two-label graph cut and against a uniform
  labelling's own energy, agreeing to 2.7e-16 relative. Restart-ICM and Wolff
  reach it from every start, here at 64 sites and to 1,024 in the experiment
  file's sweep, so the instance cannot host the gate: nothing outperforms a
  closed form, which is the argument #596 makes about Viterbi.
* **In a field at two labels alpha-expansion is exact**, equalling the graph
  cut bitwise at every size measured. Excluded for the same reason.
* **In a field at three labels the baseline does fail**, and by how much
  depends on the instrument. Against alpha-expansion's energy at 144 sites
  every method reads a `0.00` fraction by exact hit, while the same eight runs
  sit 0.04--9.06% above it and annealing reaches 0.38 under a 1% allowance. A
  curve that does not say which reading it used has not reported a failure
  size.

One test carries `release` --- the same claim at 25x the budget --- and the
rest run in 2.0 s together, so the arm's headline reading gates a merge rather
than waiting for one.
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
@pytest.mark.parametrize("side", [4, 6, 8])
def test_the_zero_field_optimum_is_a_closed_form_three_ways(side: int) -> None:
    """``-J * n_edges``, the exact cut, and a uniform labelling agree.

    Three routes to one number: the closed form, the two-label graph cut which
    is exact for a ferromagnet, and the energy of the all-zero labelling read
    by `maxflow.energy`. They agree to 1.7e-16, 2.7e-16 and 1.4e-16 relative at
    the three sides, which is a reduction's ordering and not a disagreement ---
    the closed form sums the couplings in a different order from either of the
    others.
    """
    rung = lattice_rung(side, 2)
    closed = uniform_ground_energy(rung)
    _, cut = ising_ground_state(rung.graph, rung.field)
    uniform = energy(rung.graph, rung.field, np.zeros(rung.n_nodes, dtype=np.int64))

    assert closed == pytest.approx(cut, rel=1e-12)
    assert closed == pytest.approx(uniform, rel=1e-12)
    assert rung.optimum == closed


@pytest.mark.smoke
def test_the_closed_form_is_refused_in_a_field() -> None:
    """A field breaks the derivation, so the closed form refuses rather than lies.

    The wrong target under a failure curve reads as a baseline that never
    succeeds, which is the most expensive kind of wrong answer this harness can
    give: it looks exactly like the finding #596 is looking for.
    """
    with pytest.raises(ValueError, match="zero field only"):
        uniform_ground_energy(lattice_rung(6, 3, seed=6))


@pytest.mark.oracle
def test_the_zero_field_baseline_does_not_fail() -> None:
    """Restart-ICM and Wolff reach the closed-form optimum from every start.

    So this instance is excluded from the gate, and the exclusion is the
    result: the ticket asks for the size where the baseline fails, and in zero
    field it does not fail at any size measured here --- nor could a policy
    beat it, since the answer is a closed form.
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
@pytest.mark.parametrize("side", [8, 12])
def test_alpha_expansion_is_exact_at_two_labels(side: int) -> None:
    """In a field at ``q = 2`` the expansion equals the graph cut, gap 0.0.

    Bitwise, not to a tolerance, at both sides here and at 16, 24 and 32 in the
    experiment file's sweep: one expansion cycle over two labels *is* the cut
    the exact method solves. So two labels are excluded from the gate for the reason #596
    gives about any exact baseline, and the number is worth pinning because it
    is what makes the three-label measurement below a measurement of the
    *labels* rather than of the implementation.
    """
    rung = lattice_rung(side, 2, seed=side)
    _, exact = ising_ground_state(rung.graph, rung.field)

    expansion = Entry("alpha-expansion")(
        rung, _budget(rung), np.random.default_rng(596)
    )

    assert expansion.value == exact


@pytest.mark.analytic
def test_the_two_instruments_disagree_on_the_same_runs() -> None:
    """Exact hit reads 0.00 where the relative reading reads the failure.

    At 144 sites and three labels, against alpha-expansion's energy: ICM misses
    by 5.31% and both instruments call that a failure, while annealing comes
    within 0.04% and lands inside a 1% allowance on 0.38 of starts. Exact hit
    reports annealing at 0.00 and the relative reading at 0.38 --- the same
    eight runs, two different answers to "where does it first fail".

    This is why `probe_failure` takes `relative` and why a curve states which
    it used. A continuous energy is approached, not hit, and the instrument
    that suits an enumerated optimum does not suit this.
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

    The target may be the best value *known* at a size rather than an optimum
    --- which is what #596's sweep uses where no exact method reaches the size
    --- and a two-sided comparison would score the run that improved on it as a
    miss. Against an exact optimum the two readings agree, since nothing goes
    below it by more than a reduction's ordering.
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

    The table this arm reports, at the ticket's budget rather than the per-PR
    one. Two results in it are not what the plan expected:

    * **Wolff degrades with size in a field** --- 3.4% above the target at 144
      sites and 34.4% at 576 --- the opposite of its zero-field behaviour,
      because a cluster flip is blind to the unary term it moves sites against.
    * **annealing is the baseline**, not ICM: it is the only method that
      reaches alpha-expansion's energy at all, on 0.17 of starts by exact hit
      and every start within 1%.

    So the gate on this problem is argued against annealing and the expansion,
    and a policy over single-site flips is competing on the wrong axis. Asserted
    as an ordering rather than as the digits, which are in the experiment file.

    Eight starts, stated because the number is a *best of*: at twelve starts
    ICM reads 8.06% here rather than 10.59%, and a shortfall quoted without its
    start count is not a measurement.
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

    The arm's sharpest claim, and the one that says the failure is structural
    rather than a budget the sweep was too mean with. ICM sits **5.31%** above
    alpha-expansion's energy at 144 sites at 200 sweeps and at **5.31%** at
    5,000 --- identical, because a descent converges and further sweeps buy no
    moves. The restart baseline over the same budget closes from 3.06% to
    1.24%, which is what a budget is supposed to buy and is why the comparison
    #596 insists on is against restarts.

    The flatness is the method's and not the accounting's: `run_icm` bills the
    whole budget whether or not the descent used it, so 25x the sweeps is 25x
    the charge for the same answer --- which understates ICM rather than
    flattering it.

    `release` because it is the test above's claim at 25x the budget, so it is
    not needed to gate a merge; it costs 1.1 s.
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
