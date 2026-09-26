"""The five ways a result said "did it finish", each read off one `Termination`.

Issue #860 (`docs/reviews/2026-09-20-design.md`): a `converged` bool, a raise
on exhaustion, a caught exception, an iteration count against the cap, and an
`optimal` property; one test each holds the old and new forms equal. `smoke`:
the science is refereed in `test_opt_fit.py`, `test_opt_mixture.py`,
`test_ground_state.py` and `test_tightening.py`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from sal.cost import Cost
from sal.emissions import GaussianEmission
from sal.opt.budget import Budget
from sal.opt.em import EM
from sal.opt.fit import fit
from sal.opt.mixture import expectation_maximization
from sal.opt.termination import Stop, Termination
from sal.opt.testfunctions import Rosenbrock
from sal.search import ground_state
from sal.search.tightening import dual_bound
from sal.sim.fixtures import fixture
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.potts import SpatioOnlyParams

#: Components of the one-dimensional mixture the fourth encoding is read on,
#: well separated so a handful of iterations settles it.
_WEIGHTS = torch.tensor([0.5, 0.5], dtype=torch.float64)
_VARIANCE_FLOOR = 1e-6


def _mixture_draws() -> np.ndarray:
    rng = np.random.default_rng([860, 1])
    return np.concatenate([rng.normal(-3.0, 0.5, 80), rng.normal(3.0, 0.5, 80)])


def _components() -> GaussianEmission:
    return GaussianEmission(
        torch.tensor([-2.0, 2.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        _VARIANCE_FLOOR,
    )


@pytest.mark.smoke
def test_the_converged_flag_and_the_termination_are_one_statement() -> None:
    # Encoding 1: a `converged: bool` beside an `iterations`. Both branches,
    # since a mapping that holds only where a fit converged is half a mapping.
    settled = fit(Rosenbrock(dimension=2), max_iterations=200)
    stopped = fit(Rosenbrock(dimension=2), max_iterations=1)

    assert settled.termination == Termination(
        converged=settled.converged,
        iterations=settled.iterations,
        reason=Stop.CONVERGED,
    )
    assert settled.converged
    assert stopped.termination == Termination(
        converged=False, iterations=stopped.iterations, reason=Stop.BUDGET
    )
    assert not stopped.converged


@pytest.mark.smoke
def test_the_fit_a_raise_refuses_an_interval_on_is_the_one_reading_budget() -> None:
    # Encoding 2: the refusal stays --- an interval at a point the optimizer
    # left early is not a statement about a maximum (`opt/CLAUDE.md`) --- and
    # the result the caller inspects instead names the branch it left by.
    objective = Rosenbrock(dimension=2)

    with pytest.raises(ValueError, match="did not converge"):
        fit(objective, max_iterations=1, include_intervals=True)

    inspected = fit(objective, max_iterations=1)
    assert inspected.termination is not None
    assert inspected.termination.reason is Stop.BUDGET


@pytest.mark.smoke
def test_a_caught_refusal_reads_as_refused_rather_than_as_a_budget() -> None:
    # Encoding 3: `run_max_product` catches the flooding schedule's
    # `ConvergenceError` and reports `converged=False`. That is not the same
    # answer as a run that spent its budget, and `Stop` keeps the two apart.
    params: SpatioOnlyParams = fixture("spatio_only", "ci").params
    field, alpha = ground_state.rung_field(params, 3)
    rung = ground_state.Rung(
        name="q3",
        graph=params.graph,
        field=field,
        alpha=alpha,
        sizes=params.sizes,
        n_states=3,
        optimum=None,
    )

    run = ground_state.run_max_product(
        rung, Budget(Cost.SITE_VISITS, 1), np.random.default_rng(1)
    )

    assert not run.converged
    assert run.termination is not None
    assert run.termination.reason is Stop.REFUSED
    assert not run.termination.converged


@pytest.mark.smoke
def test_the_iteration_count_a_caller_compared_is_the_reason_already() -> None:
    # Encoding 4: `iterations == max_iterations` was how a caller told a
    # settled fit from a capped one, which reads a converged run that took
    # every iteration as a failure. The reason says which happened.
    draws = _mixture_draws()
    capped = expectation_maximization(
        draws, _WEIGHTS, _components(), config=replace(EM, max_iterations=1)
    )
    settled = expectation_maximization(
        draws, _WEIGHTS, _components(), config=replace(EM, max_iterations=200)
    )

    assert capped.iterations == 1
    assert capped.termination == Termination(
        converged=False, iterations=1, reason=Stop.BUDGET
    )
    assert settled.termination is not None
    assert settled.termination.converged
    assert settled.termination.iterations == settled.iterations


@pytest.mark.smoke
def test_the_optimal_property_answers_about_the_gap_and_not_about_the_loop() -> None:
    # Encoding 5: `Certificate.optimal` was read as "did it finish" and is a
    # statement about the instance --- a run whose sweeps settled can leave a
    # gap open. One sweep cannot settle, so the two answers separate here.
    graph = lattice_graph((4, 4), BoundaryCondition.PERIODIC, coupling=-1.0)
    # A field of its own, because a flat one leaves the dual where the first
    # sweep found it and both runs then settle on sweep one.
    field = np.random.default_rng([860, 3]).normal(size=(graph.n_nodes, 3))

    capped = dual_bound(graph, field, max_iterations=1)
    settled = dual_bound(graph, field, max_iterations=400)

    assert capped.termination == Termination(
        converged=False, iterations=1, reason=Stop.BUDGET
    )
    assert settled.termination is not None
    assert settled.termination.converged
    assert settled.termination.iterations == settled.iterations


@pytest.mark.smoke
def test_a_termination_that_contradicts_itself_is_refused() -> None:
    # The one reading is what the type is for: a `converged` that disagrees
    # with its reason is the drift the five encodings allowed.
    with pytest.raises(ValueError, match="disagree"):
        Termination(converged=True, iterations=3, reason=Stop.BUDGET)
    with pytest.raises(ValueError, match="disagree"):
        Termination(converged=False, iterations=3, reason=Stop.CONVERGED)
    with pytest.raises(ValueError, match="at least no iterations"):
        Termination(converged=True, iterations=-1, reason=Stop.CONVERGED)
