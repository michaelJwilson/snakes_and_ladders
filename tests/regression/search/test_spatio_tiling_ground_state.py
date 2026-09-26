"""The ground state of a tiling field, against enumeration and against the planted states (issue #1050).

At `spatio_tiling/ci` every one of the 3**12 labellings is scored, so the
optimum is known exactly; the graph cuts, the bracket and the two annealed
handovers of `search.cluster_moves` are held to what is exact about them and
nothing else. At `spatio_tiling/release` no optimum is known, and the referee
is the planted truth through `search.potts_starts.recovery_bound`: a tile
whose strength clears its bound is labelled its state in full at every
graph-cut local minimum.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pytest
from sal.backend import Backend
from sal.cost import Cost
from sal.likelihood.potts import enumerate_potts
from sal.opt.budget import Budget
from sal.search.alpha_expansion import alpha_beta_swap, alpha_expansion
from sal.search.ground_state import (
    ANNEAL_SCHEDULE,
    EXPANSION_RESERVE_CYCLES,
    ExpansionReserve,
    MethodRun,
    chain,
    expansion_bracket,
    part,
    run_alpha_beta_swap,
    run_alpha_expansion,
)
from sal.search.potts_starts import (
    TilingRung,
    recovery,
    recovery_bound,
    tiling_rung,
)
from sal.sim.fixtures import fixture
from sal.sim.graph import PottsGraph
from sal.sim.potts import energies

#: Heat-bath sweeps' worth of site visits per run, the notebook's budget.
SWEEPS = 1000
#: The two exact routes to one energy sum the same terms in another order.
EXACT = 1e-12


@functools.cache
def _ci() -> TilingRung:
    return tiling_rung(fixture("spatio_tiling", "ci").params, "ci-tiling")


@functools.cache
def _release() -> TilingRung:
    return tiling_rung(fixture("spatio_tiling", "release").params, "release-tiling")


@functools.cache
def _enumerated() -> tuple[np.ndarray, np.ndarray]:
    """Every labelling of the ci rung and its energy."""
    rung = _ci()
    every = np.array(
        list(itertools.product(range(rung.n_states), repeat=rung.n_nodes)),
        dtype=np.int64,
    )
    return every, energies(rung.graph, rung.field, every)


def _budget(rung: TilingRung) -> Budget:
    return Budget(Cost.SITE_VISITS, SWEEPS * rung.visits_per_sweep)


def _held(rung: TilingRung) -> np.ndarray:
    """The tiles whose strength clears the recovery bound."""
    return np.asarray(rung.strengths > recovery_bound(rung))


def _graph_cuts(rung: TilingRung) -> dict[str, MethodRun]:
    rng = np.random.default_rng(0)
    return {
        "alpha-expansion": run_alpha_expansion(rung, _budget(rung), rng),
        "alpha-beta-swap": run_alpha_beta_swap(rung, _budget(rung), rng),
    }


@pytest.mark.oracle
def test_the_ci_optimum_is_unique_and_uses_all_three_states() -> None:
    # The margin the fixture declares, and the non-linearity it is for: the
    # best labelling on any two states sits above the optimum, which the
    # two-end reduction of `spatio_only` would forbid.
    rung = _ci()
    every, values = _enumerated()
    levels = np.unique(np.round(values, 9))
    on_two = min(
        values[np.isin(every, pair).all(axis=1)].min()
        for pair in itertools.combinations(range(rung.n_states), 2)
    )

    assert levels[1] - levels[0] == pytest.approx(0.6, abs=1e-9)
    assert on_two - levels[0] == pytest.approx(1.4, abs=1e-9)
    assert np.unique(every[np.argmin(values)]).size == 3

    # The same optimum from `likelihood.potts`'s own enumeration, which scores
    # through `log_weights` and shares no code with `sim.potts.energies`: at
    # beta = 40 the 0.6 margin puts every other labelling below e**-24 of the
    # optimum's weight, so each site's marginal mode is the optimum's state.
    beta = 40.0
    cold = PottsGraph(
        n_nodes=rung.n_nodes,
        edges=rung.graph.edges,
        coupling=tuple(beta * np.asarray(rung.graph.edge_coupling)),
    )
    marginals = enumerate_potts(
        cold, beta * rung.field, max_configurations=len(every)
    ).single_site

    assert (marginals.argmax(axis=1) == every[np.argmin(values)]).all()
    assert marginals.max(axis=1).min() > 1.0 - 1e-6


@pytest.mark.oracle
def test_the_ci_optimum_and_every_graph_cut_minimum_hold_the_bounded_tiles() -> None:
    # Tile 0 at 0.6 is under its bound of 1.4; tiles 1 and 2 clear
    # theirs. The bound is exact, so it holds at the enumerated optimum and at
    # every graph-cut run that ended on a cycle lowering nothing.
    rung = _ci()
    every, values = _enumerated()
    held = _held(rung)
    optimum = every[np.argmin(values)]

    assert held.tolist() == [False, True, True]
    assert (recovery(rung, optimum)[held] == 1.0).all()
    for run in _graph_cuts(rung).values():
        assert run.termination is not None
        assert run.termination.converged
        assert (recovery(rung, run.labelling)[held] == 1.0).all()
        assert run.energy >= values.min() - EXACT


@pytest.mark.oracle
def test_the_expansion_bracket_holds_the_enumerated_optimum() -> None:
    rung = _ci()
    _, values = _enumerated()
    expansion = _graph_cuts(rung)["alpha-expansion"]
    lower, upper = expansion_bracket(rung, expansion.energy)

    assert lower <= values.min() <= upper + EXACT


@pytest.mark.oracle
def test_the_annealed_handovers_end_no_lower_than_the_optimum_and_no_higher_than_their_start() -> (
    None
):
    # Swendsen-Wang then the expansion ends on a converged expansion, so the
    # bound holds there; the expansion then Swendsen-Wang returns its start or
    # lower. Both are at or above the enumerated minimum.
    rung = _ci()
    _, values = _enumerated()
    held = _held(rung)
    expansion = _graph_cuts(rung)["alpha-expansion"]
    for seed in range(3):
        then_cut = chain(
            part(
                "swendsen-wang",
                schedule=ANNEAL_SCHEDULE,
                reserve=ExpansionReserve(EXPANSION_RESERVE_CYCLES),
            ),
            "alpha-expansion",
        )(rung, _budget(rung), np.random.default_rng(seed))
        then_anneal = chain(
            "alpha-expansion", part("swendsen-wang", schedule=ANNEAL_SCHEDULE)
        )(rung, _budget(rung), np.random.default_rng(seed))

        assert then_cut.termination is not None
        assert then_cut.termination.converged
        assert (recovery(rung, then_cut.labelling)[held] == 1.0).all()
        assert then_cut.energy >= values.min() - EXACT
        assert values.min() - EXACT <= then_anneal.energy <= expansion.energy


@pytest.mark.end2end
def test_the_graph_cuts_recover_every_bounded_tile_at_five_thousand_sites() -> None:
    # The release fixture through its rung to the two graph-cut moves, judged
    # against the planted states: the four strongest tiles, 2.79 to 7.0
    # against bounds of 2.1 and 2.8, are labelled their state at every site.
    rung = _release()
    held = _held(rung)
    runs = [
        move(rung.graph, rung.field, max_iterations=50, backend=Backend.RUST)
        for move in (alpha_expansion, alpha_beta_swap)
    ]

    assert np.flatnonzero(held).tolist() == [12, 13, 14, 15]
    for run in runs:
        assert run.termination is not None
        assert run.termination.converged
        assert (recovery(rung, run.labelling)[held] == 1.0).all()
