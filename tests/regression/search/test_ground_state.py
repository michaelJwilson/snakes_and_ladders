"""Ground-state recovery on `spatio_only`, and the two referees it is scored by.

Issue #551: enumeration at nine sites, a graph cut at 5,041 sites and two
states; the structural referee (size tilt, null class, occupancy) is read off
the parameters that built the field, beside the energy.
`test_the_field_accept_step_rejects` shows the cluster accept step is present
before any timing is believed. Boykov, Veksler & Zabih bound a non-negative
energy, so the bracket is shifted, checked where the optimum is enumerable
(`test_the_bracket_contains_the_known_optimum`).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from sal.backend import Backend
from sal.cost import Cost
from sal.likelihood.message_passing import (
    MessageScheduleName,
    max_product,
)
from sal.opt.budget import Budget
from sal.opt.termination import Stop, Termination
from sal.sample.potts_mcmc import (
    PottsMove,
    anneal_potts,
    parallel_tempering,
    sample_potts,
)
from sal.sample.schedule import ExponentialTempSchedule
from sal.search import ground_state
from sal.search.alpha_expansion import (
    alpha_beta_swap,
    alpha_expansion,
    swap,
)
from sal.search.maxflow import ising_ground_state
from sal.search.maxflow.rust import (
    ising_ground_state as rust_ground_state,
)
from sal.sim.factor_graph import from_potts
from sal.sim.fixtures import fixture
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import SpatioOnlyParams, energy

from tests._rows import every_value

#: Two exact routes sum the same weights in a different order, so they agree
#: to the last bits of a float64 reduction rather than bitwise.
_EXACT = 1e-9

CI: SpatioOnlyParams = fixture("spatio_only", "ci").params
RELEASE: SpatioOnlyParams = fixture("spatio_only", "release").params

#: `spatio_only/release` at q = 2 by graph cut: energy, field-only agreement,
#: and a tilt below the thermal 0.4935, as a ferromagnet orders when cooled.
RELEASE_Q2_ENERGY = -10454.1562900565
RELEASE_Q2_GREEDY_AGREEMENT = 0.5376
RELEASE_Q2_TILT = 0.2151


def _rung(params: SpatioOnlyParams, n_states: int) -> ground_state.Rung:
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


def _enumerated_minimizers(
    rung: ground_state.Rung, exact: float
) -> set[tuple[int, ...]]:
    """Every labelling at the enumerated minimum, to :data:`_EXACT`."""
    return {
        configuration
        for configuration in itertools.product(
            range(rung.n_states), repeat=rung.n_nodes
        )
        if abs(energy(rung.graph, rung.field, np.asarray(configuration)) - exact)
        <= _EXACT
    }


# --- rung 1: nine sites, enumeration ----------------------------------------


@pytest.mark.oracle
@pytest.mark.critical
def test_the_graph_cut_is_the_enumerated_ground_state_at_two_states() -> None:
    # A ferromagnet in any per-site field is submodular: the cut is exact, both
    # implementations. All-0 and all-1 tie at -7.2 bitwise, so the claim is
    # energy and membership of the minimizers (#935).
    rung = _rung(CI, 2)
    _, exact = _enumerated(rung)
    minimizers = _enumerated_minimizers(rung, exact)

    python_state, python_energy = ising_ground_state(rung.graph, rung.field)
    rust_state, rust_energy = rust_ground_state(rung.graph, rung.field)

    assert python_energy == pytest.approx(exact, abs=_EXACT)
    assert rust_energy == pytest.approx(exact, abs=_EXACT)
    assert tuple(python_state.tolist()) in minimizers
    assert tuple(rust_state.tolist()) in minimizers
    assert np.array_equal(python_state, rust_state)


@pytest.mark.oracle
@pytest.mark.backend
def test_the_backend_seam_reaches_the_rust_cut_bitwise() -> None:
    # `ising_ground_state(..., backend=Backend.RUST)` is the one way a caller
    # names the kernel (#813); it must hand back the twin's own answer and not
    # a third one, so the comparison is equality, state and energy both.
    rung = _rung(CI, 2)
    seam_state, seam_energy = ising_ground_state(
        rung.graph, rung.field, backend=Backend.RUST
    )
    rust_state, rust_energy = rust_ground_state(rung.graph, rung.field)

    assert np.array_equal(seam_state, rust_state)
    assert seam_energy == rust_energy
    with pytest.raises(ValueError, match="runs on"):
        ising_ground_state(rung.graph, rung.field, backend=Backend.NUMBA)


@pytest.mark.oracle
def test_both_cut_move_sets_reach_the_enumerated_optimum() -> None:
    # Alpha expansion carries a bound and the swap carries none, so this is
    # the only place the swap's answer is refereed rather than compared.
    def check(n_states: int) -> None:
        rung = _rung(CI, n_states)
        _, exact = _enumerated(rung)

        expansion = alpha_expansion(rung.graph, rung.field, n_states)
        swapped = alpha_beta_swap(rung.graph, rung.field, n_states)

        assert expansion.energy == pytest.approx(exact, abs=_EXACT)
        assert swapped.energy == pytest.approx(exact, abs=_EXACT)

    every_value([2, 3], check)


@pytest.mark.analytic
def test_the_swap_never_raises_the_energy_from_any_start() -> None:
    # Monotonicity over a finite state space is what makes the loop terminate,
    # and a sign error in the capacities breaks it immediately.
    rung = _rung(CI, 3)
    rng = np.random.default_rng(551)

    for _ in range(12):
        start = rng.integers(0, 3, size=rung.n_nodes)
        run = alpha_beta_swap(rung.graph, rung.field, 3, start=start)
        assert run.energy <= energy(rung.graph, rung.field, start) + _EXACT


@pytest.mark.end2end
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


@pytest.mark.analytic
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


@pytest.mark.smoke
@pytest.mark.parametrize("move", [PottsMove.SWENDSEN_WANG, PottsMove.WOLFF])
def test_the_field_accept_step_rejects(move: PottsMove) -> None:
    # A cluster move silently running without its accept step would look like
    # a fast winner, so the rejections are asserted before any timing is read.
    # `spatio_only/ci`'s field is strong enough that some recolouring must be
    # refused; a run with no rejection at all is the failure this catches.
    rung = _rung(CI, 3)

    run = anneal_potts(
        rung.graph,
        rung.field,
        ExponentialTempSchedule(2.0, 0.05, 40),
        np.random.default_rng(551),
        move=move,
    )

    proposals = sum(counter.proposals for counter in run.trace)
    accepts = sum(counter.accepts for counter in run.trace)
    assert proposals > 0
    assert accepts < proposals


@pytest.mark.analytic
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


@pytest.mark.analytic
def test_no_entry_spends_more_than_its_budget() -> None:
    # The unit is site visits, not sweeps: matched in sweeps, a Wolff step
    # would be handed a free lattice per move. Every entry reports what it
    # spent and `opt.budget.compare` refuses an overspend, so this asserts the
    # reports themselves are inside the ceiling.
    rung = _rung(CI, 3)
    # 60 sweep-equivalents, which is the comparison's own budget and enough
    # for flooding to settle at nine sites -- so the converging branch of
    # max-product is exercised here and the refusing one in the test below.
    budget = Budget(Cost.SITE_VISITS, 60 * rung.visits_per_sweep)

    for name, method in ground_state.METHODS.items():
        run = method(rung, budget, np.random.default_rng(9))
        assert run.spent <= budget.size, name


@pytest.mark.smoke
def test_gibbs_at_zero_temperature_is_the_descent_update() -> None:
    # Reported on one axis for this reason: at T -> 0 the heat bath is the
    # argmin over each site's conditional, which is the descent's update. A
    # comparison listing them as independent entries double-counts.
    rung = _rung(CI, 3)
    rng = np.random.default_rng(12)
    state = rng.integers(0, 3, size=rung.n_nodes)
    offsets, neighbours, couplings = rung.graph.compressed_adjacency()

    cold = state.copy()
    from sal.sample.potts_mcmc.sweeps import single_site_sweep

    single_site_sweep(
        cold, rung.field, offsets, neighbours, couplings, np.random.default_rng(3), 1e6
    )

    descent = state.copy()
    for node in range(rung.n_nodes):
        local = -rung.field[node].copy()
        for position in range(int(offsets[node]), int(offsets[node + 1])):
            local[descent[neighbours[position]]] -= couplings[position]
        descent[node] = int(np.argmin(local))

    assert np.array_equal(cold, descent)


# --- refusals ----------------------------------------------------------------


@pytest.mark.smoke
def test_a_rung_is_refused_at_a_label_count_no_fixture_declares() -> None:
    with pytest.raises(ValueError, match="a rung is built at 2 states"):
        ground_state.rung_field(CI, 5)


@pytest.mark.smoke
def test_a_swap_of_a_label_with_itself_is_refused() -> None:
    # The identity move. A caller asking for it has a loop bound wrong, and
    # returning the input unchanged would hide that.
    rung = _rung(CI, 3)
    labelling = np.zeros(rung.n_nodes, dtype=np.int64)

    with pytest.raises(ValueError, match="two distinct labels"):
        swap(rung.graph, rung.field, labelling, 1, 1)


@pytest.mark.oracle
def test_the_rust_sweep_runs_the_per_site_field_and_matches_the_oracle() -> None:
    # The kernel once refused the per-site field (#571); it now reproduces the
    # Python sweep on the same stream.
    rung = _rung(CI, 3)
    assert not bool(np.all(rung.field == rung.field[0])), (
        "this fixture is the per-site case; a shared field tests nothing here"
    )

    runs = [
        anneal_potts(
            rung.graph,
            rung.field,
            ExponentialTempSchedule(2.0, 0.05, 8),
            np.random.default_rng(1),
            backend=backend,
        )
        for backend in (Backend.PYTHON, Backend.RUST)
    ]

    np.testing.assert_array_equal(runs[0].labelling, runs[1].labelling)
    np.testing.assert_array_equal(runs[0].final, runs[1].final)
    assert runs[0].energy == runs[1].energy


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


@pytest.mark.end2end
@pytest.mark.release
def test_the_exact_ground_state_at_five_thousand_sites_tilts_with_size() -> None:
    # #551's Step 1 gate: field_argmax agreement under ~95% (not field-dominated);
    # tilt positive and below 0.4935, the cut exact at nine sites.
    rung = _rung(RELEASE, 2)
    labelling, _ = rust_ground_state(rung.graph, rung.field)

    recovered = ground_state.structure(rung.alpha, rung.sizes, labelling)
    agreement = float((rung.field.argmax(axis=1) == labelling).mean())

    assert agreement == pytest.approx(RELEASE_Q2_GREEDY_AGREEMENT, abs=1e-3)
    assert agreement < 0.95
    assert recovered.tilt == pytest.approx(RELEASE_Q2_TILT, abs=1e-3)
    assert 0.0 < recovered.tilt < 0.4935


@pytest.mark.oracle
@pytest.mark.critical
def test_the_runners_record_the_energy_their_kernels_return() -> None:
    # The rung below (#734): a runner records its kernel's number and labelling
    # on the same rung, seed and budget. `run_tempering`, `run_field_argmax` and
    # `run_max_product` against `parallel_tempering`, the field argmax and
    # flooding `max_product`: difference 0.0, exact equality declared.
    rung = _rung(CI, 3)
    budget = Budget(Cost.SITE_VISITS, 60 * rung.visits_per_sweep)
    seed = 11

    tempering = ground_state.run_tempering(rung, budget, np.random.default_rng(seed))
    per_replica = max(
        1, budget.size // (ground_state.N_REPLICAS * rung.visits_per_sweep)
    )
    ladder = tuple(
        float(value)
        for value in np.geomspace(
            ground_state.ANNEAL_START, ground_state.ANNEAL_END, ground_state.N_REPLICAS
        )
    )
    kernel = parallel_tempering(
        rung.graph, rung.field, ladder, np.random.default_rng(seed), per_replica
    )

    assert tempering.energy == kernel.best_energy
    assert np.array_equal(tempering.labelling, kernel.best)
    assert (
        tempering.spent == ground_state.N_REPLICAS * per_replica * rung.visits_per_sweep
    )

    argmax = ground_state.run_field_argmax(rung, budget, np.random.default_rng(seed))
    field_only = rung.field.argmax(axis=1).astype(np.int64)

    assert argmax.energy == energy(rung.graph, rung.field, field_only)
    assert np.array_equal(argmax.labelling, field_only)
    assert argmax.spent == rung.n_nodes

    product = ground_state.run_max_product(rung, budget, np.random.default_rng(seed))
    iterations = max(1, budget.size // rung.visits_per_sweep)
    assignment, _ = max_product(
        from_potts(rung.graph, rung.field),
        schedule=MessageScheduleName.FLOODING,
        max_iterations=iterations,
    )
    decoded = np.array(
        [assignment[f"s{node}"] for node in range(rung.n_nodes)], dtype=np.int64
    )

    assert product.converged
    assert product.energy == energy(rung.graph, rung.field, decoded)
    assert np.array_equal(product.labelling, decoded)
    assert product.spent == iterations * rung.visits_per_sweep


@pytest.mark.smoke
def test_the_comparison_records_the_labelling_each_entry_returned() -> None:
    # Each entry carries its run's labelling; the records must cover every
    # cell, or a structural column is read from the wrong run.
    from sal.opt.budget import compare

    rung = _rung(CI, 3)
    budget = Budget(Cost.SITE_VISITS, 20 * rung.visits_per_sweep)

    comparison = compare(ground_state.entries(), [rung] * 2, budget, [551], workers=1)
    records = ground_state.recorded(comparison)

    assert len(records) == len(ground_state.METHODS) * 2
    assert {record.method for record in records} == set(ground_state.METHODS)
    for record in records:
        value = ground_state.outcome(record.run).value
        assert value == record.run.energy, record.method
    assert set(comparison.methods) == set(ground_state.METHODS)


@pytest.mark.smoke
@pytest.mark.bug
def test_every_cell_is_recorded_on_a_worker_pool() -> None:
    # The memo was a module-level list, so at `workers > 1` the cells ran in
    # other processes and the parent read an empty record (issue #856). The
    # record returns on the outcome, and the pooled run is the serial run:
    # each cell seeds itself, so the energies and the labellings match.
    from sal.opt.budget import compare

    rung = _rung(CI, 3)
    budget = Budget(Cost.SITE_VISITS, 20 * rung.visits_per_sweep)

    serial = compare(ground_state.entries(), [rung] * 2, budget, [551], workers=1)
    pooled = compare(ground_state.entries(), [rung] * 2, budget, [551], workers=2)

    records = ground_state.recorded(pooled)
    assert len(records) == len(ground_state.METHODS) * 2
    assert np.array_equal(pooled.best, serial.best)
    for record, expected in zip(records, ground_state.recorded(serial), strict=True):
        assert record.method == expected.method
        assert record.rung == expected.rung
        assert np.array_equal(record.run.labelling, expected.run.labelling)


@pytest.mark.smoke
def test_max_product_that_does_not_settle_reports_no_answer() -> None:
    # Flooding is refused rather than truncated, and the refusal is carried
    # through: an infinite energy and `converged=False`, never the field-only
    # labelling's numbers entered under max-product's name.
    rung = _rung(CI, 3)

    run = ground_state.run_max_product(
        rung, Budget(Cost.SITE_VISITS, 1), np.random.default_rng(1)
    )

    assert not run.converged
    assert run.energy == float("inf")


@pytest.mark.smoke
def test_the_swap_refuses_a_backend_it_has_no_cut_for() -> None:
    rung = _rung(CI, 3)
    labelling = np.zeros(rung.n_nodes, dtype=np.int64)

    with pytest.raises(
        ValueError, match="minimum cut runs on python or rust, not numba"
    ):
        swap(rung.graph, rung.field, labelling, 0, 1, backend=Backend.NUMBA)


@pytest.mark.smoke
def test_a_swap_of_two_labels_no_site_carries_is_the_identity() -> None:
    # Not an error: a cycle over every pair reaches pairs the labelling does
    # not use, and the move is genuinely empty there.
    rung = _rung(CI, 3)
    labelling = np.zeros(rung.n_nodes, dtype=np.int64)

    moved, value, *_ = swap(rung.graph, rung.field, labelling, 1, 2)

    assert np.array_equal(moved, labelling)
    assert value == pytest.approx(energy(rung.graph, rung.field, labelling))


@pytest.mark.smoke
def test_the_swap_refuses_a_negative_coupling() -> None:
    # The binary sub-problem is submodular only for a non-negative coupling,
    # so this is the boundary rather than a slow case.
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, -0.5)

    with pytest.raises(ValueError, match="submodular only then"):
        alpha_beta_swap(graph, np.zeros((graph.n_nodes, 3)), 3)


@pytest.mark.smoke
def test_the_swap_warns_and_returns_what_it_holds_at_its_cycle_cap() -> None:
    # Issue #1059: a caller's cap is a budget (`ground_state` derives one from
    # site visits), so reaching it warns and returns the labelling held, its
    # energy in full, and a termination recording the cap rather than raising.
    rung = _rung(CI, 3)
    start = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)

    with pytest.warns(UserWarning, match="did not settle in max_cycles=1"):
        capped = alpha_beta_swap(rung.graph, rung.field, 3, start=start, max_cycles=1)
    assert capped.cycles == 1
    assert capped.termination == Termination(False, 1, Stop.BUDGET)
    assert capped.energy == energy(rung.graph, rung.field, capped.labelling)
    settled = alpha_beta_swap(rung.graph, rung.field, 3, start=start)
    assert settled.energy <= capped.energy


@pytest.mark.smoke
def test_the_compiled_swendsen_wang_anneal_keeps_no_counter() -> None:
    # Issue #923: the comparison anneals Swendsen-Wang on the compiled pass,
    # a chain of the same law on another order of draws (its law is pinned in
    # test_potts_mcmc_cluster_rust.py); it reads no cluster, so the trace is
    # empty, and the Python pass still records every accept step.
    rung = _rung(CI, 3)
    schedule = ExponentialTempSchedule(2.0, 0.05, 40)
    compiled = anneal_potts(
        rung.graph,
        rung.field,
        schedule,
        np.random.default_rng(923),
        move=PottsMove.SWENDSEN_WANG,
        cluster_backend=Backend.RUST,
    )
    oracle = anneal_potts(
        rung.graph,
        rung.field,
        schedule,
        np.random.default_rng(923),
        move=PottsMove.SWENDSEN_WANG,
    )
    assert compiled.trace == ()
    assert len(oracle.trace) == schedule.n_steps
    assert compiled.energy == energy(rung.graph, rung.field, compiled.labelling)
