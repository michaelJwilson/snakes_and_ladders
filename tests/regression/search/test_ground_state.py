"""Ground-state recovery on `potts_spots`, and the two referees it is scored by.

Issue #551's rungs, each pinned where an exact answer reaches: enumeration at
nine sites, a graph cut at 5,041 sites and two states. The structural referee
--- the size tilt, the null class, the per-class occupancy --- is read off the
parameters that built the field, so it is exact at both sizes and is asserted
beside the energy rather than instead of it.

Two things here exist to fail loudly rather than quietly.

**The cluster accept step.** A cluster move running without it looks like a
fast winner, so `test_the_field_accept_step_rejects` asserts the rejections
are there before any timing is believed.

**The bracket's shift.** Boykov, Veksler & Zabih bound a *non-negative*
energy, and this repository's energy is negative; applied unshifted the
factor-2 statement is false. `test_the_bracket_contains_the_known_optimum`
checks it where the optimum is enumerable, which is the only place the claim
can be checked at all.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from snakes_and_ladders.opt.budget import Budget
from snakes_and_ladders.opt.schedule import Exponential
from snakes_and_ladders.search import ground_state
from snakes_and_ladders.search.alpha_expansion import (
    alpha_beta_swap,
    alpha_expansion,
    energy,
    swap,
)
from snakes_and_ladders.search.backend import Backend
from snakes_and_ladders.search.maxflow import ising_ground_state
from snakes_and_ladders.search.maxflow_rust import (
    ising_ground_state as rust_ground_state,
)
from snakes_and_ladders.search.potts_mcmc import (
    PottsMove,
    anneal_potts,
    sample_potts,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.potts import PottsSpotsParams

#: Two exact routes sum the same weights in a different order, so they agree
#: to the last bits of a float64 reduction rather than bitwise.
_EXACT = 1e-9

CI: PottsSpotsParams = fixture("potts_spots", "ci").params
RELEASE: PottsSpotsParams = fixture("potts_spots", "release").params

#: The exact ground state of `potts_spots/release` at q = 2, measured by graph
#: cut on this branch: energy, the field-only labelling's agreement with it,
#: and its size tilt. The tilt is **below** the fixture's thermal 0.4935, and
#: that is the physics rather than a defect --- a ferromagnet orders as the
#: temperature falls and the majority class takes sites whose own field points
#: elsewhere. Pinned so a change to the cut, the field or the sizes moves it.
RELEASE_Q2_ENERGY = -10454.1562900565
RELEASE_Q2_GREEDY_AGREEMENT = 0.5376
RELEASE_Q2_TILT = 0.2151


def _rung(params: PottsSpotsParams, n_states: int) -> ground_state.Rung:
    field, alpha = ground_state.rung_field(params, n_states)
    return ground_state.Rung(
        name=f"q{n_states}",
        graph=params.graph,
        field=field,
        alpha=alpha,
        sizes=params.sizes,
        n_states=n_states,
        optimum=None,
    )


def _enumerated(rung: ground_state.Rung) -> tuple[np.ndarray, float]:
    """The exact minimum by exhaustive enumeration, which nine sites affords."""
    configurations = np.array(
        list(itertools.product(range(rung.n_states), repeat=rung.n_nodes)),
        dtype=np.int64,
    )
    values = np.array([energy(rung.graph, rung.field, row) for row in configurations])
    best = int(np.argmin(values))
    return configurations[best], float(values[best])


# --- rung 1: nine sites, enumeration ----------------------------------------


@pytest.mark.oracle
@pytest.mark.critical
def test_the_graph_cut_is_the_enumerated_ground_state_at_two_states() -> None:
    # The claim rung 2 rests on, checked at the only size where it can be:
    # a ferromagnet in an arbitrary per-site field is submodular, so the cut
    # is exact and not merely good. Both implementations, since the Rust one
    # is what the 5,041-site rung runs.
    rung = _rung(CI, 2)
    labelling, exact = _enumerated(rung)

    python_state, python_energy = ising_ground_state(rung.graph, rung.field)
    rust_state, rust_energy = rust_ground_state(rung.graph, rung.field)

    assert python_energy == pytest.approx(exact, abs=_EXACT)
    assert rust_energy == pytest.approx(exact, abs=_EXACT)
    assert np.array_equal(python_state, labelling)
    assert np.array_equal(rust_state, labelling)


@pytest.mark.oracle
@pytest.mark.parametrize("n_states", [2, 3])
def test_both_cut_move_sets_reach_the_enumerated_optimum(n_states: int) -> None:
    # Alpha expansion carries a bound and the swap carries none, so this is
    # the only place the swap's answer is refereed rather than compared.
    rung = _rung(CI, n_states)
    _, exact = _enumerated(rung)

    expansion = alpha_expansion(rung.graph, rung.field, n_states)
    swapped = alpha_beta_swap(rung.graph, rung.field, n_states)

    assert expansion.energy == pytest.approx(exact, abs=_EXACT)
    assert swapped.energy == pytest.approx(exact, abs=_EXACT)


@pytest.mark.mathematical
def test_the_swap_never_raises_the_energy_from_any_start() -> None:
    # Monotonicity over a finite state space is what makes the loop terminate,
    # and a sign error in the capacities breaks it immediately.
    rung = _rung(CI, 3)
    rng = np.random.default_rng(551)

    for _ in range(12):
        start = rng.integers(0, 3, size=rung.n_nodes)
        run = alpha_beta_swap(rung.graph, rung.field, 3, start=start)
        assert run.energy <= energy(rung.graph, rung.field, start) + _EXACT


@pytest.mark.simulated_truth
def test_the_enumerated_ground_state_recovers_the_generating_structure() -> None:
    # The second referee, at the size where the labelling it scores is exact.
    # The null class carries no field at any site, so its occupancy cannot
    # tilt with size unless the method put it there.
    rung = _rung(CI, 3)
    labelling, _ = _enumerated(rung)

    recovered = ground_state.structure(rung.alpha, rung.sizes, labelling)

    assert recovered.tilt > 0.0
    assert recovered.monotone
    assert recovered.null_tilt == pytest.approx(0.0, abs=_EXACT)
    assert sum(recovered.occupancy) == rung.n_nodes


@pytest.mark.mathematical
def test_the_bracket_contains_the_known_optimum() -> None:
    # Boykov, Veksler & Zabih bound a non-negative energy; this one is
    # negative, so the bound is applied to the shifted form and carried back.
    # Unshifted, the lower end lands above the optimum and this fails.
    rung = _rung(CI, 3)
    _, exact = _enumerated(rung)
    expansion = alpha_expansion(rung.graph, rung.field, 3)

    lower, upper = ground_state.expansion_bracket(rung, expansion.energy)

    assert lower <= exact + _EXACT
    assert exact <= upper + _EXACT


# --- the guards against a quiet failure --------------------------------------


@pytest.mark.structural
@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG, PottsMove.WOLFF])
def test_the_field_accept_step_rejects(move: PottsMove) -> None:
    # A cluster move silently running without its accept step would look like
    # a fast winner, so the rejections are asserted before any timing is read.
    # `potts_spots/ci`'s field is strong enough that some recolouring must be
    # refused; a run with no rejection at all is the failure this catches.
    rung = _rung(CI, 3)

    run = anneal_potts(
        rung.graph,
        rung.field,
        Exponential(2.0, 0.05, 40),
        np.random.default_rng(551),
        move=move,
    )

    proposals = sum(counter.proposals for counter in run.trace)
    accepts = sum(counter.accepts for counter in run.trace)
    assert proposals > 0
    assert accepts < proposals


@pytest.mark.mathematical
@pytest.mark.parametrize("move", list(PottsMove))
def test_a_field_repeated_per_site_is_the_shared_field(move: PottsMove) -> None:
    # The per-site widening (issue #551) must be the identity where every row
    # is equal. A broadcast applied to the wrong axis passes every shape check
    # and changes the chain, which is what this pins.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.6)
    shared = np.array([0.4, -0.2, 0.1])
    rows = np.tile(shared, (graph.n_nodes, 1))

    one = sample_potts(graph, shared, move, np.random.default_rng(4), 20)
    two = sample_potts(graph, rows, move, np.random.default_rng(4), 20)

    assert np.array_equal(one.states, two.states)


@pytest.mark.mathematical
def test_no_entry_spends_more_than_its_budget() -> None:
    # The unit is site visits, not sweeps: matched in sweeps, a Wolff step
    # would be handed a free lattice per move. Every entry reports what it
    # spent and `opt.budget.compare` refuses an overspend, so this asserts the
    # reports themselves are inside the ceiling.
    rung = _rung(CI, 3)
    # 60 sweep-equivalents, which is the comparison's own budget and enough
    # for flooding to settle at nine sites -- so the converging branch of
    # max-product is exercised here and the refusing one in the test below.
    budget = Budget("site-visits", 60 * rung.visits_per_sweep)

    for name, method in ground_state.METHODS.items():
        run = method(rung, budget, np.random.default_rng(9))
        assert run.spent <= budget.size, name


@pytest.mark.structural
def test_gibbs_at_zero_temperature_is_the_descent_update() -> None:
    # Reported on one axis for this reason: at T -> 0 the heat bath is the
    # argmin over each site's conditional, which is the descent's update. A
    # comparison listing them as independent entries double-counts.
    rung = _rung(CI, 3)
    rng = np.random.default_rng(12)
    state = rng.integers(0, 3, size=rung.n_nodes)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(rung.n_nodes)]
    for (first, second), coupling in rung.graph.weighted_edges():
        neighbours[first].append((second, coupling))
        neighbours[second].append((first, coupling))

    cold = state.copy()
    from snakes_and_ladders.search.potts_mcmc import _single_site_sweep

    _single_site_sweep(cold, rung.field, neighbours, np.random.default_rng(3), 1e6)

    descent = state.copy()
    for node in range(rung.n_nodes):
        local = -rung.field[node].copy()
        for neighbour, coupling in neighbours[node]:
            local[descent[neighbour]] -= coupling
        descent[node] = int(np.argmin(local))

    assert np.array_equal(cold, descent)


# --- refusals ----------------------------------------------------------------


@pytest.mark.edge_case
def test_a_rung_is_refused_at_a_label_count_no_fixture_declares() -> None:
    with pytest.raises(ValueError, match="a rung is built at 2 states"):
        ground_state.rung_field(CI, 5)


@pytest.mark.edge_case
def test_a_swap_of_a_label_with_itself_is_refused() -> None:
    # The identity move. A caller asking for it has a loop bound wrong, and
    # returning the input unchanged would hide that.
    rung = _rung(CI, 3)
    labelling = np.zeros(rung.n_nodes, dtype=np.int64)

    with pytest.raises(ValueError, match="two distinct labels"):
        swap(rung.graph, rung.field, labelling, 1, 1)


@pytest.mark.edge_case
def test_the_rust_sweep_refuses_a_field_that_varies_by_site() -> None:
    # The kernel takes one row shared by every site. Handed a varying field it
    # would sample the wrong model rather than fail, so it is refused.
    rung = _rung(CI, 3)

    with pytest.raises(ValueError, match="one field row shared by every site"):
        anneal_potts(
            rung.graph,
            rung.field,
            Exponential(2.0, 0.05, 2),
            np.random.default_rng(1),
            backend=Backend.RUST,
        )


# --- rung 2: 5,041 sites, the graph cut --------------------------------------


@pytest.mark.oracle
@pytest.mark.release
def test_the_exact_ground_state_at_five_thousand_sites() -> None:
    # The rung the experiment turns on: the only place a 5,041-site gap is a
    # measured gap rather than a gap to the best thing anyone found. The two
    # implementations must report the same energy exactly, a ground state
    # being a combinatorial minimum rather than a tolerance.
    rung = _rung(RELEASE, 2)

    python_state, python_energy = ising_ground_state(rung.graph, rung.field)
    rust_state, rust_energy = rust_ground_state(rung.graph, rung.field)

    assert python_energy == pytest.approx(RELEASE_Q2_ENERGY, abs=1e-6)
    assert rust_energy == python_energy
    assert np.array_equal(python_state, rust_state)


@pytest.mark.simulated_truth
@pytest.mark.release
def test_the_exact_ground_state_at_five_thousand_sites_tilts_with_size() -> None:
    # Both arms of issue #551's Step 1 gate. The greedy agreement is the
    # trap check: above ~95% the instance would be field-dominated and the
    # comparison would measure nothing. The tilt is positive and **below**
    # the fixture's thermal 0.4935, which is the ordering a ferromagnet shows
    # as the temperature falls, not a defect --- the same cut reproduces the
    # enumerated ground state exactly at nine sites.
    rung = _rung(RELEASE, 2)
    labelling, _ = rust_ground_state(rung.graph, rung.field)

    recovered = ground_state.structure(rung.alpha, rung.sizes, labelling)
    agreement = float((rung.field.argmax(axis=1) == labelling).mean())

    assert agreement == pytest.approx(RELEASE_Q2_GREEDY_AGREEMENT, abs=1e-3)
    assert agreement < 0.95
    assert recovered.tilt == pytest.approx(RELEASE_Q2_TILT, abs=1e-3)
    assert 0.0 < recovered.tilt < 0.4935


@pytest.mark.structural
def test_the_comparison_records_the_labelling_each_entry_returned() -> None:
    # `opt.budget.compare` returns an energy and a spend; the structural
    # referee needs the labelling, and running every method twice to get it
    # would double the experiment. The entries memo what they already
    # computed, and this asserts the memo covers every cell rather than
    # silently missing one -- which would show up as a structural column
    # quietly read from the wrong run.
    from snakes_and_ladders.opt.budget import compare

    rung = _rung(CI, 3)
    budget = Budget("site-visits", 20 * rung.visits_per_sweep)
    ground_state.forget()

    comparison = compare(ground_state.entries(), [rung] * 2, budget, [551], workers=1)
    runs = ground_state.recorded()

    assert len(runs) == len(ground_state.METHODS) * 2
    assert {name for name, _, _ in runs} == set(ground_state.METHODS)
    for name, _, run in runs:
        assert ground_state.outcome(run).value == run.energy, name
    assert set(comparison.methods) == set(ground_state.METHODS)
    ground_state.forget()
    assert ground_state.recorded() == ()


@pytest.mark.edge_case
def test_max_product_that_does_not_settle_reports_no_answer() -> None:
    # Flooding is refused rather than truncated, and the refusal is carried
    # through: an infinite energy and `converged=False`, never the field-only
    # labelling's numbers entered under max-product's name.
    rung = _rung(CI, 3)

    run = ground_state.run_max_product(
        rung, Budget("site-visits", 1), np.random.default_rng(1)
    )

    assert not run.converged
    assert run.energy == float("inf")


@pytest.mark.edge_case
def test_the_swap_refuses_a_backend_it_has_no_cut_for() -> None:
    rung = _rung(CI, 3)
    labelling = np.zeros(rung.n_nodes, dtype=np.int64)

    with pytest.raises(ValueError, match="no numba minimum-cut backend"):
        swap(rung.graph, rung.field, labelling, 0, 1, backend=Backend.NUMBA)


@pytest.mark.edge_case
def test_a_swap_of_two_labels_no_site_carries_is_the_identity() -> None:
    # Not an error: a cycle over every pair reaches pairs the labelling does
    # not use, and the move is genuinely empty there.
    rung = _rung(CI, 3)
    labelling = np.zeros(rung.n_nodes, dtype=np.int64)

    moved, value = swap(rung.graph, rung.field, labelling, 1, 2)

    assert np.array_equal(moved, labelling)
    assert value == pytest.approx(energy(rung.graph, rung.field, labelling))


@pytest.mark.edge_case
def test_the_swap_refuses_a_negative_coupling() -> None:
    # The binary sub-problem is submodular only for a non-negative coupling,
    # so this is the boundary rather than a slow case.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, -0.5)

    with pytest.raises(ValueError, match="submodular only then"):
        alpha_beta_swap(graph, np.zeros((graph.n_nodes, 3)), 3)


@pytest.mark.edge_case
def test_the_swap_refuses_rather_than_looping_past_its_cycle_cap() -> None:
    # Monotonicity over a finite state space makes reaching the cap
    # impossible on a correct implementation, so it is a defect report and
    # not a budget -- the same contract `alpha_expansion` states.
    rung = _rung(CI, 3)
    start = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="did not settle in 1 cycles"):
        alpha_beta_swap(rung.graph, rung.field, 3, start=start, max_cycles=1)
