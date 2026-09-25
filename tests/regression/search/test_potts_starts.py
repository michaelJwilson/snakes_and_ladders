"""The ground-state solvers as starts of the `opt.starts` seam, polished by ICM (issue #906).

Referees, from the strongest: at two states the graph cut is the exact
ground state, and alpha-expansion equals it; a polished labelling is a local
minimum, checked by trying every single-site change; ICM never raises the
energy it starts from, and no solver ends below the expansion bracket's lower
end. The declared fixture is the instance `ground_state.lattice_rung` builds,
bitwise.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.objective import Objective
from sal.opt.starts import StartsBenchmark
from sal.search.ground_state import (
    METHODS,
    Rung,
    expansion_bracket,
    lattice_rung,
    run_alpha_expansion,
    rung_field,
)
from sal.search.icm import iterated_conditional_modes
from sal.search.potts_starts import (
    PottsObjective,
    SolverStart,
    binary_sibling,
    describe,
    polish_by_icm,
    rung_of,
    spatio_rung,
)
from sal.sim.fixtures import fixture
from sal.sim.potts import PottsLatticeParams, SpatioOnlyParams, energy

#: A small rung of the same family: 8x8 at three states in the size tilt.
SIDE = 8
#: Sweeps each solver is given, in site visits per rung.
SWEEPS = 20
#: The ICM polish's cap, in sweeps; it settles well inside it here.
POLISH = Budget(Cost.SWEEPS, 100)


def _small() -> Rung:
    return lattice_rung(SIDE, 3, seed=906)


def _is_local_minimum(rung: Rung, labelling: np.ndarray) -> bool:
    """No single site's change of state lowers the energy: every change tried."""
    base = energy(rung.graph, rung.field, labelling)
    for site in range(rung.n_nodes):
        for state in range(rung.n_states):
            if state == labelling[site]:
                continue
            moved = labelling.copy()
            moved[site] = state
            if energy(rung.graph, rung.field, moved) < base - 1e-12:
                return False
    return True


@pytest.mark.smoke
def test_the_release_fixture_is_the_sizing_familys_rung_bitwise() -> None:
    params = fixture("potts_lattice", "release").params
    assert isinstance(params, PottsLatticeParams)
    rung = rung_of(params, "release")
    built = lattice_rung(50, 3, seed=906)
    assert np.array_equal(rung.field, built.field)
    assert np.array_equal(rung.sizes, built.sizes)
    assert np.array_equal(rung.alpha, built.alpha)
    assert rung.graph.edges == built.graph.edges
    assert np.array_equal(
        np.asarray(rung.graph.edge_coupling), np.asarray(built.graph.edge_coupling)
    )


@pytest.mark.oracle
def test_the_binary_siblings_optimum_is_the_cut_and_expansion_reaches_it() -> None:
    # At two states the minimum cut is exact; alpha-expansion is exact there
    # too (`test_potts_sizing`), so the two agree, and the sibling carries the
    # three-state rung's sizes and lattice.
    rung = _small()
    sibling = binary_sibling(rung, "small-q2")
    assert sibling.n_states == 2
    assert sibling.name == "small-q2"
    assert np.array_equal(sibling.sizes, rung.sizes)
    assert sibling.optimum is not None
    expanded = run_alpha_expansion(
        sibling,
        Budget(Cost.SITE_VISITS, SWEEPS * sibling.visits_per_sweep),
        np.random.default_rng(0),
    )
    assert expanded.energy == pytest.approx(sibling.optimum, rel=1e-12)


@pytest.mark.analytic
def test_the_icm_polish_ends_at_a_local_minimum_never_above_its_start() -> None:
    rung = _small()
    objective = PottsObjective(rung)
    assert isinstance(objective, Objective)
    start = np.random.default_rng(1).integers(0, 3, size=rung.n_nodes)
    theta = torch.as_tensor(start)
    assert float(objective(theta)) == energy(rung.graph, rung.field, start)
    polished = polish_by_icm(objective, theta, POLISH)
    labelling = polished.theta.numpy()
    assert polished.termination.converged
    assert polished.value <= energy(rung.graph, rung.field, start)
    assert polished.value == energy(rung.graph, rung.field, labelling)
    assert _is_local_minimum(rung, labelling)
    # One sweep at a time is the descent in one call, bitwise.
    whole = iterated_conditional_modes(
        rung.graph,
        rung.field,
        3,
        np.random.default_rng(0),
        start=start,
        max_sweeps=POLISH.size,
    )
    assert np.array_equal(whole.labelling, labelling)


@pytest.mark.end2end
def test_every_solver_through_the_seam_is_polished_inside_the_bracket() -> None:
    # Every entry of `ground_state.METHODS` as a start on the small rung,
    # polished by ICM through `StartsBenchmark`. ICM cannot raise an energy,
    # nothing may end below the bracket's lower end, and one seed reproduces
    # the whole comparison bitwise.
    rung = _small()
    objective = PottsObjective(rung)
    budget = Budget(Cost.SITE_VISITS, SWEEPS * rung.visits_per_sweep)
    expansion = run_alpha_expansion(rung, budget, np.random.default_rng(0)).energy
    bracket = expansion_bracket(rung, expansion)
    starts = {name: functools.partial(SolverStart, name, budget) for name in METHODS}

    def run() -> tuple[tuple[float, ...], ...]:
        result = StartsBenchmark(
            objective,
            starts,
            polish_by_icm,
            seeding_budget=Budget(Cost.EVALUATIONS, 1),
            polish_budget=POLISH,
            seeds=[0, 1],
            workers=1,
            reference=[expansion],
        ).run()
        values = []
        for name in result.names:
            for trial in result.trials(name):
                assert trial.value <= trial.seeded_value, name
                assert trial.value >= bracket.lower, name
                values.append((trial.seeded_value, trial.value))
        return tuple(values)

    assert run() == run()


@pytest.mark.smoke
def test_an_annealed_entry_is_described_by_its_function_and_move() -> None:
    assert (
        describe("icm")
        == "Iterated conditional modes: single-site descent in index order."
    )
    assert describe("wolff").startswith("One annealed run, its step count fixed")
    assert describe("wolff").endswith("Move set: wolff.")


@pytest.mark.smoke
def test_the_adapter_refuses_what_it_cannot_read(tmp_path: Path) -> None:
    from sal.opt.mixture import GaussianMixtureObjective

    rung = _small()
    objective = PottsObjective(rung)
    theta = objective.initial()
    with pytest.raises(ValueError, match="in sweeps"):
        polish_by_icm(objective, theta, Budget(Cost.SITE_VISITS, 10))
    other = GaussianMixtureObjective(np.arange(4.0), 2)
    with pytest.raises(ValueError, match="not a Potts energy"):
        polish_by_icm(other, theta, POLISH)
    text = (
        Path(__file__).parents[1] / "fixtures/potts_lattice/release.yaml"
    ).read_text()
    broken = tmp_path / "release.yaml"
    broken.write_text(text.replace("  size_seed: 906\n", ""))
    with pytest.raises(ValueError, match="size-tilted field declares exactly"):
        PottsLatticeParams.from_declared(yaml.safe_load(broken.read_text()), broken)


@pytest.mark.smoke
def test_the_q10_rung_is_the_spatio_only_release_fixture_bitwise() -> None:
    # Issue #927: the notebook's instance is the declared spatio_only/release,
    # at its own ten classes, read through the one field builder the ground
    # state tests use.
    params = fixture("spatio_only", "release").params
    assert isinstance(params, SpatioOnlyParams)
    rung = spatio_rung(params, "release-q10")
    field, alpha = rung_field(params, params.n_classes)
    assert rung.n_states == params.n_classes == 10
    assert rung.n_nodes == 71 * 71
    assert np.array_equal(rung.field, field)
    assert np.array_equal(rung.alpha, alpha)
    assert np.array_equal(rung.sizes, params.sizes)
    assert rung.graph is params.graph
    assert rung.optimum is None


@pytest.mark.end2end
def test_at_ten_states_the_polish_never_rises_and_nothing_ends_below_the_bracket() -> (
    None
):
    # The q = 10 notebook's claims on its own rung at a short budget: ICM
    # never raises the solver's energy, and no labelling ends below the
    # expansion bracket's lower end.
    rung = spatio_rung(fixture("spatio_only", "release").params, "release-q10")
    budget = Budget(Cost.SITE_VISITS, 50 * rung.visits_per_sweep)
    expansion = run_alpha_expansion(rung, budget, np.random.default_rng(0)).energy
    bracket = expansion_bracket(rung, expansion)
    objective = PottsObjective(rung)
    for name in ("greedy", "anneal", "alpha-expansion"):
        (seeded,) = SolverStart(name, budget, np.random.default_rng(1)).starts(
            objective
        )
        start = float(objective(seeded))
        polished = polish_by_icm(objective, seeded, POLISH)
        assert polished.value <= start, name
        assert polished.value >= bracket.lower, name
