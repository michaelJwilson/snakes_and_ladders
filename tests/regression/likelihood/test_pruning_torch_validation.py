"""Validation-error paths for ``phylo.likelihood.pruning_torch``.

Separated from tests/regression/test_pruning_torch.py, which pins
scientific correctness; these pin the guardrails around malformed inputs,
mirroring tests/regression/test_likelihood_validation.py's split for the
NumPy oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from phylo.likelihood.pruning_torch import branch_lengths_from_tree, log_likelihood
from phylo.sim.tree import Node

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


@pytest.mark.edge_case
def test_rejects_mismatched_pi_shape() -> None:
    with pytest.raises(ValueError, match="pi has shape"):
        log_likelihood(_TAU, 4, np.full(3, 1.0 / 3), _ALIGNMENT, _BRANCH_LENGTHS)


@pytest.mark.edge_case
def test_rejects_mismatched_branch_lengths_shape() -> None:
    with pytest.raises(ValueError, match="branch_lengths has shape"):
        log_likelihood(
            _TAU, 4, np.full(4, 0.25), _ALIGNMENT, torch.zeros(3, dtype=torch.float64)
        )


@pytest.mark.edge_case
def test_rejects_alignment_missing_a_leaf() -> None:
    with pytest.raises(ValueError, match="alignment is missing leaf"):
        log_likelihood(
            _TAU, 4, np.full(4, 0.25), {"A": _ALIGNMENT["A"]}, _BRANCH_LENGTHS
        )


@pytest.mark.edge_case
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
