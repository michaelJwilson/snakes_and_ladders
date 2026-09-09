"""Regression tests for ``snakes_and_ladders.sandbox.pruning_problem`` (issue #444, PR 1).

The handle is a declined implementation --- the measurement that refused it is
in ``STATUS.md`` and in the module's own docstring --- and
``sandbox/CLAUDE.md`` keeps a declined implementation's tests running with it.

What they pin is the claim the measurement rests on: the handle and
``likelihood.pruning_rust.log_likelihood`` return the same ``float``
**bitwise**, so the two wall times compared in ``STATUS.md`` are two ways of
computing one number and not two numbers. Agreement with the NumPy oracle is
checked directly as well, at the same ``float64`` bound
``test_pruning_rust.py`` uses, so neither backend's correctness rests on the
other.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood import pruning, pruning_rust
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.sandbox.pruning_problem import PruningProblem
from snakes_and_ladders.search.rl import with_uniform_branch_lengths
from snakes_and_ladders.search.topology import random_topology
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node


def _tree_n6() -> Node:
    """6-taxon tree, mirroring ``test_pruning_rust.py``'s fixture."""
    return Node(
        name="root",
        branch_length=None,
        children=(
            Node(
                name="ancestor_AB",
                branch_length=0.08,
                children=(
                    Node(name="A", branch_length=0.10),
                    Node(name="B", branch_length=0.25),
                ),
            ),
            Node(
                name="ancestor_CD",
                branch_length=0.05,
                children=(
                    Node(name="C", branch_length=0.15),
                    Node(name="D", branch_length=0.40),
                ),
            ),
            Node(
                name="ancestor_EF",
                branch_length=0.12,
                children=(
                    Node(name="E", branch_length=0.20),
                    Node(name="F", branch_length=0.30),
                ),
            ),
        ),
    )


@pytest.mark.oracle
def test_handle_matches_the_function_bitwise() -> None:
    """Same kernel, same order of operations, so the same ``f64``.

    Asserted with ``==`` rather than a tolerance: a perimeter change that
    moves an answer has changed the model, not the plumbing.
    """
    tau = _tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260923), n_sites=120
    )

    problem = PruningProblem(k, pi, dataset.alignment)
    for rescale in (True, False):
        function = pruning_rust.log_likelihood(
            tau, k, pi, dataset.alignment, rescale=rescale
        )
        held = problem.log_likelihood(tau, rescale=rescale)
        assert held == function, f"rescale={rescale}: {held} against {function}"


@pytest.mark.oracle
def test_handle_matches_numpy_oracle() -> None:
    tau = _tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260924), n_sites=50
    )

    numpy_ll = pruning.log_likelihood(tau, k, pi, dataset.alignment)
    held = PruningProblem(k, pi, dataset.alignment).log_likelihood(tau)

    assert_allclose(held, numpy_ll, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.structural
def test_handle_scores_many_topologies_from_one_alignment() -> None:
    """The shape the handle was built for, and the one the measurement used.

    Every candidate gets the value the function would have given it, so a
    screening loop holding the alignment ranks candidates identically to one
    that does not.
    """
    k = 4
    pi = np.full(k, 0.25)
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(20260925)
    dataset = simulate_alignment(tau=_tree_n6(), k=k, pi=pi, rng=rng, n_sites=80)
    candidates = [
        with_uniform_branch_lengths(random_topology(names, rng), 0.1) for _ in range(8)
    ]

    problem = PruningProblem(k, pi, dataset.alignment)
    held = [problem.log_likelihood(candidate) for candidate in candidates]
    function = [
        pruning_rust.log_likelihood(candidate, k, pi, dataset.alignment)
        for candidate in candidates
    ]

    assert held == function
    assert len(set(held)) > 1, "the fixture should separate the candidates"


@pytest.mark.structural
def test_handle_copies_the_alignment_it_is_given() -> None:
    """The handle is a copy, so a later edit to the caller's array is not seen.

    A cache aliasing its caller's buffer would make a pass depend on when it
    ran, which is the failure mode holding state invites.
    """
    k = 4
    pi = np.full(k, 0.25)
    tau = _tree_n6()
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260926), n_sites=40
    )
    alignment = {name: np.array(row) for name, row in dataset.alignment.items()}

    problem = PruningProblem(k, pi, alignment)
    before = problem.log_likelihood(tau)
    alignment["A"][:] = (alignment["A"] + 1) % k
    after = problem.log_likelihood(tau)

    assert before == after


@pytest.mark.edge_case
def test_handle_rejects_mismatched_pi_shape() -> None:
    alignment = {
        "A": np.zeros(5, dtype=np.int64),
        "B": np.zeros(5, dtype=np.int64),
    }
    with pytest.raises(ValueError, match="pi has shape"):
        PruningProblem(4, np.full(3, 1.0 / 3), alignment)


@pytest.mark.edge_case
def test_handle_rejects_ragged_alignment() -> None:
    alignment = {
        "A": np.zeros(5, dtype=np.int64),
        "B": np.zeros(4, dtype=np.int64),
    }
    with pytest.raises(ValueError, match="the alignment is ragged"):
        PruningProblem(4, np.full(4, 0.25), alignment)


@pytest.mark.edge_case
def test_handle_rejects_a_tree_whose_leaf_is_not_in_the_alignment() -> None:
    """Held once, checked per call: the alignment fixes the leaf set, the tree does not."""
    tau = Node(
        name="root",
        branch_length=None,
        children=(
            Node(name="A", branch_length=0.1),
            Node(name="Z", branch_length=0.2),
        ),
    )
    alignment = {
        "A": np.zeros(5, dtype=np.int64),
        "B": np.zeros(5, dtype=np.int64),
    }
    problem = PruningProblem(4, np.full(4, 0.25), alignment)
    with pytest.raises(ValueError, match="alignment is missing leaf"):
        problem.log_likelihood(tau)


@pytest.mark.edge_case
def test_handle_rejects_non_root_node_without_branch_length() -> None:
    tau = Node(
        name="root",
        branch_length=None,
        children=(
            Node(name="A", branch_length=None),
            Node(name="B", branch_length=0.2),
        ),
    )
    alignment = {
        "A": np.zeros(5, dtype=np.int64),
        "B": np.zeros(5, dtype=np.int64),
    }
    problem = PruningProblem(4, np.full(4, 0.25), alignment)
    with pytest.raises(ValueError, match="has no branch_length"):
        problem.log_likelihood(tau)
