"""Validation-error paths for ``sal.likelihood``, route by route.

The guardrails around malformed inputs, one body per claim parametrised over
`ROUTES` (issue #863); refusals come from `branch_lengths_from_tree` with the
message `pruning_common` gives every route. The last test is torch's alone, a
branch-length tensor of the wrong shape (issue #982).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sal.likelihood.pruning import torch as pruning_torch
from sal.sim.tree import Node

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
