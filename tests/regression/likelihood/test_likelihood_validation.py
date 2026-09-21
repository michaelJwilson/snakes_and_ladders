"""Validation-error paths for ``snakes_and_ladders.likelihood``, route by route.

Separated from `test_likelihood_pruning.py`, which pins scientific
correctness; these pin the guardrails around malformed inputs, mirroring
`tests/regression/sim/test_jc_validation.py`'s split for the simulator.

**One body per claim, parametrised over the routes** (issue #863). The same
three refusals were written three times --- here for the NumPy oracle and
brute force, in `test_pruning_rust.py` for the compiled route, and in
`test_pruning_torch_validation.py` for the taped one --- each with its own
copy of the two-leaf tree. Three copies is three chances to add a route and
not the fourth test, which is what `pruning_analytic` was: it shares
`pruning_common`'s checks with the rest and had a test for neither the
alignment nor ``pi``. A route is added to :data:`ROUTES` now and is asked
every question the others are.

The adapters exist because the differentiable routes take their branch
lengths as a tensor rather than off the tree. Each one builds that tensor
from the tree it is handed, so the tree without a branch length is refused
where that route refuses it --- in `branch_lengths_from_tree` rather than in
`log_likelihood` --- with the message `pruning_common` gives every route.
`test_pruning_torch_validation.py` keeps the claim that is torch's alone: a
branch-length tensor of the wrong shape, which no other route can be given.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from snakes_and_ladders.likelihood import pruning_analytic, pruning_rust, pruning_torch
from snakes_and_ladders.likelihood.brute_force import brute_force_log_likelihood
from snakes_and_ladders.likelihood.pruning import log_likelihood
from snakes_and_ladders.sim.tree import Node

#: A route from a tree, an alphabet, a root distribution and an alignment to
#: a log-likelihood. What every route is, once its branch lengths are read
#: off the tree they belong to.
Route = Callable[[Node, int, np.ndarray, dict[str, np.ndarray]], object]


def _torch(
    tau: Node, k: int, pi: np.ndarray, alignment: dict[str, np.ndarray]
) -> object:
    """`pruning_torch`, with its branch lengths read off ``tau``."""
    return pruning_torch.log_likelihood(
        tau, k, pi, alignment, pruning_torch.branch_lengths_from_tree(tau)
    )


def _analytic(
    tau: Node, k: int, pi: np.ndarray, alignment: dict[str, np.ndarray]
) -> object:
    """`pruning_analytic`, with its branch lengths read off ``tau``."""
    return pruning_analytic.log_likelihood(
        tau, k, pi, alignment, pruning_torch.branch_lengths_from_tree(tau)
    )


#: Every route that computes this likelihood, by the name a failure reports.
ROUTES: dict[str, Route] = {
    "pruning": log_likelihood,
    "brute_force": brute_force_log_likelihood,
    "pruning_rust": pruning_rust.log_likelihood,
    "pruning_torch": _torch,
    "pruning_analytic": _analytic,
}

_TAU = Node(
    name="root",
    branch_length=None,
    children=(
        Node(name="A", branch_length=0.1),
        Node(name="B", branch_length=0.2),
    ),
)
_ALIGNMENT = {
    "A": np.zeros(5, dtype=np.int64),
    "B": np.zeros(5, dtype=np.int64),
}


@pytest.mark.smoke
@pytest.mark.parametrize("route", ROUTES.values(), ids=list(ROUTES))
def test_rejects_mismatched_pi_shape(route: Route) -> None:
    with pytest.raises(ValueError, match="pi has shape"):
        route(_TAU, 4, np.full(3, 1.0 / 3), _ALIGNMENT)


@pytest.mark.smoke
@pytest.mark.parametrize("route", ROUTES.values(), ids=list(ROUTES))
def test_rejects_alignment_missing_a_leaf(route: Route) -> None:
    with pytest.raises(ValueError, match="alignment is missing leaf"):
        route(_TAU, 4, np.full(4, 0.25), {"A": _ALIGNMENT["A"]})


@pytest.mark.smoke
@pytest.mark.parametrize("route", ROUTES.values(), ids=list(ROUTES))
def test_rejects_non_root_node_without_branch_length(route: Route) -> None:
    tau = Node(
        name="root",
        branch_length=None,
        children=(
            Node(name="A", branch_length=None),
            Node(name="B", branch_length=0.2),
        ),
    )
    with pytest.raises(ValueError, match="has no branch_length"):
        route(tau, 4, np.full(4, 0.25), _ALIGNMENT)
