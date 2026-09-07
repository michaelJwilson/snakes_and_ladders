"""Benchmarks for the phylogenetic objective, at the size ROADMAP.md names.

See tests/regression/test_likelihood_objective.py for correctness.

The number that matters here is ROADMAP.md's performance requirement --
sub-second gradient updates at n = 100 -- which until now was asserted rather
than measured. One update means one objective evaluation plus one reverse
pass, which is what an optimizer step costs; the line search L-BFGS runs on
top of that is a separate multiplier, and is why the fit benchmark in
tests/benchmarks/test_opt_fit_bench.py is not this number times the iteration
count.

The 100-taxon topology is generated here rather than added as a fixture:
nothing is simulated from it that a test asserts against, so it carries no
ground truth and `sim/CLAUDE.md`'s yaml rule does not apply. It is a size,
not a model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from snakes_and_ladders.sim.gtr import gtr_rate_matrix
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

# ROADMAP.md's stated size, and a site count in the range the same document
# assumes for the memory bound.
_TAXA = 100
_SITES = 1000
_SEED = 20260904
_BRANCH_LENGTH = 0.1


def _balanced(names: list[str]) -> Node:
    """A balanced binary subtree over ``names``, with uniform branch lengths."""
    if len(names) == 1:
        return Node(name=names[0], branch_length=_BRANCH_LENGTH)
    middle = len(names) // 2
    return Node(
        name=f"internal_{names[0]}_{names[-1]}",
        branch_length=_BRANCH_LENGTH,
        children=(_balanced(names[:middle]), _balanced(names[middle:])),
    )


def _topology(n_taxa: int) -> Node:
    """An ``n_taxa``-leaf topology in the trifurcating-root convention.

    Trifurcating rather than rooted binary so every branch is separately
    estimable -- see the confounding measured in the regression module.
    """
    names = [f"t{index:03d}" for index in range(n_taxa)]
    third = n_taxa // 3
    return Node(
        name="root",
        branch_length=None,
        children=(
            _balanced(names[:third]),
            _balanced(names[third : 2 * third]),
            _balanced(names[2 * third :]),
        ),
    )


@pytest.fixture(scope="module")
def objective() -> BranchLengthObjective:
    tau = _topology(_TAXA)
    pi = np.full(4, 0.25)
    dataset = simulate_alignment(tau=tau, k=4, pi=pi, seed=_SEED, n_sites=_SITES)
    return BranchLengthObjective(tau, 4, pi, dict(dataset.alignment))


def test_gradient_update_at_roadmap_scale(
    benchmark: BenchmarkFixture, objective: BranchLengthObjective
) -> None:
    """One objective evaluation plus one reverse pass at n = 100."""
    theta = objective.initial()

    def _update() -> float:
        point = theta.detach().clone().requires_grad_(True)
        value = objective(point)
        torch.autograd.grad(value, point)
        return float(value.detach())

    result = benchmark(_update)

    # Benchmarks assert finiteness only; correctness is pinned in
    # tests/regression/test_likelihood_objective.py.
    assert math.isfinite(result)


def test_forward_pass_at_roadmap_scale(
    benchmark: BenchmarkFixture, objective: BranchLengthObjective
) -> None:
    """The objective alone, so the reverse pass's share is visible."""
    theta = objective.initial()
    assert math.isfinite(float(benchmark(objective, theta)))


def test_general_model_gradient_update_at_roadmap_scale(
    benchmark: BenchmarkFixture,
) -> None:
    """The same update under the general model, so the extra cost is visible.

    GTR routes through ``torch.linalg.matrix_exp`` per branch where
    Jukes-Cantor has a closed form, and adds a symmetric eigen-style
    construction per evaluation. The ratio to the JC number above is the
    price of having rate parameters to fit at all.
    """
    tau = _topology(_TAXA)
    pi = np.full(4, 0.25)
    rate = gtr_rate_matrix(np.array([1.6, 0.4, 0.9, 0.7, 2.1, 1.0]), pi)
    dataset = simulate_alignment(
        tau=tau, k=4, pi=pi, seed=_SEED, n_sites=_SITES, rate_matrix=rate
    )
    objective = SubstitutionModelObjective(tau, 4, dict(dataset.alignment))
    theta = objective.initial()

    def _update() -> float:
        point = theta.detach().clone().requires_grad_(True)
        value = objective(point)
        torch.autograd.grad(value, point)
        return float(value.detach())

    assert math.isfinite(benchmark(_update))
