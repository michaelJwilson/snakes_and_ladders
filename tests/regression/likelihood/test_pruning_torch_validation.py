"""The refusals that are this route's alone (issue #863).

The three every route makes --- ``pi``, the alignment, a non-root node
without a branch length --- are one body each in
`test_likelihood_validation.py`, parametrised over the routes and asked of
this one there. What is left here is what only a route taking its branch
lengths as a tensor can be asked: a tensor of the wrong shape, and the
refusal `branch_lengths_from_tree` makes when it builds one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.likelihood.pruning_torch import (
    branch_lengths_from_tree,
    log_likelihood,
)
from snakes_and_ladders.sim.tree import Node

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
_BRANCH_LENGTHS = branch_lengths_from_tree(_TAU)


@pytest.mark.smoke
def test_rejects_mismatched_branch_lengths_shape() -> None:
    with pytest.raises(ValueError, match="branch_lengths has shape"):
        log_likelihood(
            _TAU, 4, np.full(4, 0.25), _ALIGNMENT, torch.zeros(3, dtype=torch.float64)
        )


@pytest.mark.smoke
def test_branch_lengths_from_tree_rejects_missing_branch_length() -> None:
    tau = Node(
        name="root",
        branch_length=None,
        children=(
            Node(name="A", branch_length=None),
            Node(name="B", branch_length=0.2),
        ),
    )
    with pytest.raises(ValueError, match="has no branch_length"):
        branch_lengths_from_tree(tau)
