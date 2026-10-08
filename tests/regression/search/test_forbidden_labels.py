"""A label the field forbids, ``-inf``, is never returned (issues #1081, #1139).

Referees: enumeration over every allowed labelling of a 3x3 lattice with 3
states, and of a 2x4 lattice with 2 and 3 states. Every ground-state method,
TRW-S and the dual bound return an allowed labelling with a finite energy;
the exact ones reach the enumerated constrained minimum; the dual's bound
stays below it. The same holds from a forbidden start for every method that
takes one, for the expansion and the swap on both backends (the swap at two
states is one cut over every site, so exact), and for the floor of
``merge_small_labels``, which ends ``Stop.INFEASIBLE`` where only a forbidden
label meets it. And ``forbid``'s mask is the ``-inf`` field, bitwise, so the
two spellings are one problem. On a 2x4, four-state instance every method
scores no worse than its own run on ``penalized``'s field, and bifurcation,
which rounded onto a forbidden label there, now leaves it (#1324).
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest
from sal.backend import Backend
from sal.cost import Cost
from sal.opt.budget import Budget
from sal.opt.termination import Stop
from sal.search.alpha_expansion import alpha_beta_swap, alpha_expansion
from sal.search.bifurcation import simulated_bifurcation
from sal.search.ground_state import METHODS, ground_state
from sal.search.icm import iterated_conditional_modes, merge_small_labels
from sal.search.tightening import dual_bound
from sal.search.trws import trws
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import energies, forbid, penalized

GRAPH = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.8)
FINITE = np.random.default_rng(1081).normal(0.0, 1.0, (GRAPH.n_nodes, 3))
ALLOWED = np.ones_like(FINITE, dtype=bool)
ALLOWED[:, 0] = False  # the first label nowhere
ALLOWED[::2, 2] = False  # the third on every other site
FIELD = forbid(FINITE, ALLOWED)
BUDGET = Budget(Cost.SITE_VISITS, 400 * GRAPH.n_nodes * 5)
#: Every site at the first label, which ``ALLOWED`` forbids everywhere.
FORBIDDEN_START = np.zeros(GRAPH.n_nodes, dtype=np.int64)
#: The methods `ground_state` refuses a start for, each by name.
STARTLESS = {
    "bifurcation",
    "field_argmax",
    "max-product",
    "tempering",
    "tempering-mixed",
}

#: Eight sites, the size the issue's exhaustive referee is stated at.
SMALL = lattice_graph((2, 4), BoundaryCondition.OPEN, 0.7)


def _constrained_minimum() -> float:
    every = np.array(list(itertools.product(range(3), repeat=GRAPH.n_nodes)))
    keep = ALLOWED[np.arange(GRAPH.n_nodes), every].all(axis=1)
    return float(energies(GRAPH, FINITE, every[keep]).min())


def _allowed(labelling: np.ndarray) -> bool:
    return bool(ALLOWED[np.arange(GRAPH.n_nodes), labelling].all())


@pytest.mark.oracle
@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_returns_an_allowed_labelling(method: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = ground_state(GRAPH, FIELD, method, BUDGET, np.random.default_rng(3))

    assert _allowed(run.labelling)
    assert np.isfinite(run.energy)
    assert run.energy >= _constrained_minimum() - 1e-9


@pytest.mark.oracle
def test_the_bounded_solvers_bound_the_constrained_minimum() -> None:
    minimum = _constrained_minimum()
    for bounded in (trws(GRAPH, FIELD), dual_bound(GRAPH, FIELD)):
        assert _allowed(bounded.labelling)
        assert np.isfinite(bounded.energy)
        assert np.isfinite(bounded.bound)
        assert bounded.bound <= minimum + 1e-9 <= bounded.energy + 2e-9
    relaxed = simulated_bifurcation(
        GRAPH, FIELD, n_states=3, rng=np.random.default_rng(0)
    )
    assert _allowed(relaxed.labelling)


@pytest.mark.smoke
def test_a_mask_is_the_negative_infinite_field_and_a_site_allowing_nothing_is_refused() -> (
    None
):
    written = FINITE.copy()
    written[~ALLOWED] = -np.inf
    assert np.array_equal(forbid(FINITE, ALLOWED), written)
    nothing = ALLOWED.copy()
    nothing[4] = False
    with pytest.raises(ValueError, match="allows no label"):
        forbid(FINITE, nothing)


def _small_problem(seed: int, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    """A finite field on ``SMALL`` and a mask allowing about half its labels, one per site at least."""
    rng = np.random.default_rng(seed)
    finite = rng.normal(0.0, 1.0, (SMALL.n_nodes, n_states))
    allowed = rng.random((SMALL.n_nodes, n_states)) < 0.5
    allowed[np.arange(SMALL.n_nodes), rng.integers(n_states, size=SMALL.n_nodes)] = True
    return finite, allowed


def _small_minimum(finite: np.ndarray, allowed: np.ndarray) -> float:
    """The constrained minimum on ``SMALL``, by enumerating every allowed labelling."""
    n_states = finite.shape[1]
    every = np.array(list(itertools.product(range(n_states), repeat=SMALL.n_nodes)))
    keep = allowed[np.arange(SMALL.n_nodes), every].all(axis=1)
    return float(energies(SMALL, finite, every[keep]).min())


def _forbidden_start(allowed: np.ndarray) -> np.ndarray:
    """Each site at its first forbidden label, or its first label where it forbids none."""
    return np.argmin(allowed, axis=1).astype(np.int64)


@pytest.mark.oracle
@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_taking_a_start_leaves_a_forbidden_one(method: str) -> None:
    if method in STARTLESS:
        with pytest.raises(ValueError, match="takes no start"):
            ground_state(
                GRAPH,
                FIELD,
                method,
                BUDGET,
                np.random.default_rng(3),
                start=FORBIDDEN_START,
            )
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = ground_state(
            GRAPH,
            FIELD,
            method,
            BUDGET,
            np.random.default_rng(3),
            start=FORBIDDEN_START,
        )

    assert _allowed(run.labelling)
    assert np.isfinite(run.energy)
    assert run.energy >= _constrained_minimum() - 1e-9


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.PYTHON])
@pytest.mark.parametrize("move", [alpha_expansion, alpha_beta_swap])
@pytest.mark.parametrize("n_states", [2, 3])
@pytest.mark.parametrize("seed", range(6))
def test_the_cut_moves_leave_a_forbidden_start_for_an_allowed_local_minimum(
    move: object, backend: Backend, n_states: int, seed: int
) -> None:
    finite, allowed = _small_problem(seed, n_states)
    start = _forbidden_start(allowed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        run = move(SMALL, forbid(finite, allowed), start=start, backend=backend)  # type: ignore[operator]

    minimum = _small_minimum(finite, allowed)
    assert allowed[np.arange(SMALL.n_nodes), run.labelling].all()
    assert run.termination.converged
    assert np.isfinite(run.energy)
    assert run.energy == energies(SMALL, finite, run.labelling[None])[0]
    assert run.energy >= minimum - 1e-9
    if move is alpha_beta_swap and n_states == 2:
        # One swap cuts every site over both labels: the exact minimum.
        assert run.energy == pytest.approx(minimum, abs=1e-9)


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("move", [alpha_expansion, alpha_beta_swap])
def test_the_cut_moves_agree_across_backends_from_a_forbidden_start(
    move: object,
) -> None:
    finite, allowed = _small_problem(7, 3)
    field, start = forbid(finite, allowed), _forbidden_start(allowed)
    rust = move(SMALL, field, start=start, backend=Backend.RUST)  # type: ignore[operator]
    python = move(SMALL, field, start=start, backend=Backend.PYTHON)  # type: ignore[operator]

    assert np.array_equal(rust.labelling, python.labelling)
    assert rust.energy == python.energy


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.NUMBA, Backend.PYTHON])
@pytest.mark.parametrize("min_sites", [2, 3])
@pytest.mark.parametrize("seed", range(6))
def test_the_floor_dissolves_only_into_allowed_labels(
    seed: int, min_sites: int, backend: Backend
) -> None:
    finite, allowed = _small_problem(seed, 3)
    field = forbid(finite, allowed)
    merged = merge_small_labels(
        SMALL,
        field,
        _forbidden_start(allowed),
        np.random.default_rng(seed),
        min_sites=min_sites,
        backend=backend,
    )

    rows = np.arange(SMALL.n_nodes)
    assert allowed[rows, merged.labelling].all()
    assert np.isfinite(merged.energy)
    assert merged.energy >= _small_minimum(finite, allowed) - 1e-9
    counts = np.bincount(merged.labelling, minlength=3)
    below = (counts > 0) & (counts < min_sites)
    stuck = below[merged.labelling] & ~allowed[:, counts >= min_sites].any(axis=1)
    # Infeasible exactly where a site below the floor allows no state at it.
    assert (merged.termination.reason is Stop.INFEASIBLE) == bool(stuck.any())
    if merged.termination.converged:
        assert not below.any()


@pytest.mark.oracle
@pytest.mark.parametrize("backend", [Backend.NUMBA, Backend.PYTHON])
def test_a_floor_only_a_forbidden_label_meets_is_infeasible_and_left(
    backend: Backend,
) -> None:
    # Site 0 allows only the second label and every other site only the
    # first, so one labelling is allowed and a floor of two cannot hold it.
    allowed = np.zeros((GRAPH.n_nodes, 2), dtype=bool)
    allowed[0, 1] = True
    allowed[1:, 0] = True
    field = forbid(FINITE[:, :2], allowed)
    only = allowed.argmax(axis=1)

    merged = merge_small_labels(
        GRAPH, field, only, np.random.default_rng(0), min_sites=2, backend=backend
    )
    descended = iterated_conditional_modes(
        GRAPH, field, np.random.default_rng(0), min_sites=2, backend=backend
    )

    for run in (merged, descended):
        assert np.array_equal(run.labelling, only)
        assert run.energy == energies(GRAPH, FINITE[:, :2], only[None])[0]
        assert run.termination.reason is Stop.INFEASIBLE
        assert not run.termination.converged


def _rounded_onto_forbidden() -> tuple[np.ndarray, np.ndarray]:
    """The 2x4, four-state instance on which bifurcation rounded onto a forbidden label (#1324)."""
    rng = np.random.default_rng(3)
    finite = rng.normal(0.0, 1.0, (SMALL.n_nodes, 4))
    allowed = np.ones_like(finite, dtype=bool)
    allowed[:, 0] = False
    allowed[rng.random(allowed.shape) < 0.3] = False
    allowed[np.arange(SMALL.n_nodes), rng.integers(1, 4, SMALL.n_nodes)] = True
    return finite, allowed


@pytest.mark.oracle
def test_bifurcation_leaves_a_forbidden_rounding_for_an_allowed_label() -> None:
    # Before #1324 the rounding of rng 7's replica held a forbidden label:
    # penalized keeps every optimum allowed, not every rounding. The run on
    # the penalized field still rounds there, which is the regression's witness.
    finite, allowed = _rounded_onto_forbidden()
    field = forbid(finite, allowed)
    budget = Budget(Cost.SITE_VISITS, 400 * SMALL.n_nodes * 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = ground_state(
            SMALL, field, "bifurcation", budget, np.random.default_rng(7)
        )
        on_penalty = ground_state(
            SMALL,
            penalized(SMALL, field),
            "bifurcation",
            budget,
            np.random.default_rng(7),
        )
    nodes = np.arange(SMALL.n_nodes)
    assert allowed[nodes, run.labelling].all()
    assert not allowed[nodes, on_penalty.labelling].all()
    assert run.energy == energies(SMALL, finite, run.labelling[None])[0]
    assert run.energy < on_penalty.energy


@pytest.mark.oracle
@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_scores_no_worse_than_on_the_penalized_field(method: str) -> None:
    # No entry is exact on a loopy lattice, so each is held to its own run on
    # penalized's finite stand-in: allowed, and no higher in energy (#1324).
    finite, allowed = _rounded_onto_forbidden()
    field = forbid(finite, allowed)
    budget = Budget(Cost.SITE_VISITS, 400 * SMALL.n_nodes * 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run = ground_state(SMALL, field, method, budget, np.random.default_rng(7))
        on_penalty = ground_state(
            SMALL, penalized(SMALL, field), method, budget, np.random.default_rng(7)
        )
    assert allowed[np.arange(SMALL.n_nodes), run.labelling].all()
    assert run.energy <= on_penalty.energy
