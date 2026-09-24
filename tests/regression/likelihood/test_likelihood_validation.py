"""Validation-error paths for ``snakes_and_ladders.likelihood``, route by route.

Separated from `test_likelihood_pruning.py`, which pins scientific
correctness; these pin the guardrails around malformed inputs, mirroring
`tests/regression/sim/test_jc_validation.py`'s split for the simulator.

**One body per claim, parametrised over the routes** (issue #863). The same
three refusals were written three times, each with its own copy of the
two-leaf tree; a route is added to `ROUTES` (`conftest.py`) now and is asked
every question the others are.

The adapters in `ROUTES` build the differentiable routes' branch-length
tensor from the tree they are handed, so the tree without a branch length is
refused where that route refuses it --- in `branch_lengths_from_tree` --- with
the message `pruning_common` gives every route. The last test is torch's
alone: a branch-length tensor of the wrong shape, which no other route can be
given (merged from `test_pruning_torch_validation.py`, issue #982).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.likelihood import pruning_torch
from snakes_and_ladders.sim.tree import Node

from tests.regression.likelihood.conftest import ROUTES, Route

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


@pytest.mark.smoke
def test_rejects_mismatched_branch_lengths_shape() -> None:
    with pytest.raises(ValueError, match="branch_lengths has shape"):
        pruning_torch.log_likelihood(
            _TAU, 4, np.full(4, 0.25), _ALIGNMENT, torch.zeros(3, dtype=torch.float64)
        )
