"""Validation-error paths for ``phylo.likelihood``.

Separated from tests/regression/test_likelihood_pruning.py, which pins
scientific correctness; these pin the guardrails around malformed inputs,
mirroring tests/regression/test_jc_validation.py's split for the simulator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from phylo.likelihood.brute_force import brute_force_log_likelihood
from phylo.likelihood.pruning import log_likelihood
from phylo.sim.tree import Node

_LikelihoodFunc = Callable[[Node, int, np.ndarray, dict[str, np.ndarray]], float]

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


@pytest.mark.edge_case
@pytest.mark.parametrize("func", [log_likelihood, brute_force_log_likelihood])
def test_rejects_mismatched_pi_shape(func: _LikelihoodFunc) -> None:
    with pytest.raises(ValueError, match="pi has shape"):
        func(_TAU, 4, np.full(3, 1.0 / 3), _ALIGNMENT)


@pytest.mark.edge_case
@pytest.mark.parametrize("func", [log_likelihood, brute_force_log_likelihood])
def test_rejects_alignment_missing_a_leaf(func: _LikelihoodFunc) -> None:
    with pytest.raises(ValueError, match="alignment is missing leaf"):
        func(_TAU, 4, np.full(4, 0.25), {"A": _ALIGNMENT["A"]})


@pytest.mark.edge_case
@pytest.mark.parametrize("func", [log_likelihood, brute_force_log_likelihood])
def test_rejects_non_root_node_without_branch_length(func: _LikelihoodFunc) -> None:
    tau = Node(
        name="root",
        branch_length=None,
        children=(
            Node(name="A", branch_length=None),
            Node(name="B", branch_length=0.2),
        ),
    )
    alignment = {"A": _ALIGNMENT["A"], "B": _ALIGNMENT["B"]}
    with pytest.raises(ValueError, match="has no branch_length"):
        func(tau, 4, np.full(4, 0.25), alignment)
