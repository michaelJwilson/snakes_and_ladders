"""Regression tests for ``snakes_and_ladders.likelihood.pruning_rust`` (the Rust CPU backend).

Checks per ``likelihood/CLAUDE.md``, mirroring ``test_pruning_torch.py``'s
structure for the PyTorch backend:

- Agreement with ``pruning.py``, the NumPy oracle every backend is pinned
  against (``test_rust_matches_numpy_oracle``).
- Agreement with ``brute_force.py`` at ``n <= 6`` taxa
  (``test_rust_matches_brute_force``) -- "correctness comes from brute
  force, not from another backend."
- Rescaled and unrescaled Rust paths agreeing
  (``test_rescaled_and_unrescaled_rust_paths_agree``).
- Validation-error parity with the NumPy oracle for malformed inputs,
  mirroring ``test_likelihood_validation.py``'s split for the pure-Python
  backends.

Requires the compiled extension (``maturin develop`` / ``pip install .``),
like ``tests/test_oxiphylo_bindings.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood import pruning, pruning_rust
from snakes_and_ladders.likelihood.brute_force import brute_force_log_likelihood
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.search.rl import with_uniform_branch_lengths
from snakes_and_ladders.search.topology import random_topology
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import FOUR_TAXA, load_fixture

# Relative, not absolute. The log-likelihood is a sum over sites, so an
# absolute bound fixed at one site count does not transfer to another
# (issue #111): the backends agree to ~8e-13 relative at every size, but that
# same agreement reads as 7.4e-07 absolute at 200,000 sites and would fail an
# absolute 1e-9. CROSS_DEVICE_RTOL_FLOAT64 is the float64 implementation-
# agreement bound stated in docs/tex/textbook.tex (sec:tolerance); it is
# never relaxed to accommodate a discrepancy.
_RTOL_ORACLE = CROSS_DEVICE_RTOL_FLOAT64


def _small_tree_n4() -> Node:
    """4-taxon tree with a trifurcating root, mirroring the pruning fixtures."""
    return Node(
        name="root",
        branch_length=None,
        children=(
            Node(name="A", branch_length=0.10),
            Node(name="B", branch_length=0.25),
            Node(
                name="ancestor_CD",
                branch_length=0.05,
                children=(
                    Node(name="C", branch_length=0.15),
                    Node(name="D", branch_length=0.40),
                ),
            ),
        ),
    )


def _small_tree_n6() -> Node:
    """6-taxon, fully binary tree."""
    return Node(
        name="root",
        branch_length=None,
        children=(
            Node(
                name="left",
                branch_length=0.08,
                children=(
                    Node(name="A", branch_length=0.10),
                    Node(name="B", branch_length=0.20),
                ),
            ),
            Node(
                name="right",
                branch_length=0.12,
                children=(
                    Node(
                        name="ancestor_CD",
                        branch_length=0.05,
                        children=(
                            Node(name="C", branch_length=0.15),
                            Node(name="D", branch_length=0.25),
                        ),
                    ),
                    Node(
                        name="ancestor_EF",
                        branch_length=0.05,
                        children=(
                            Node(name="E", branch_length=0.30),
                            Node(name="F", branch_length=0.10),
                        ),
                    ),
                ),
            ),
        ),
    )


@pytest.mark.oracle
def test_rust_matches_numpy_oracle() -> None:
    tau = _small_tree_n4()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260920), n_sites=50
    )

    numpy_ll = pruning.log_likelihood(tau, k, pi, dataset.alignment)
    rust_ll = pruning_rust.log_likelihood(tau, k, pi, dataset.alignment)

    assert_allclose(rust_ll, numpy_ll, rtol=_RTOL_ORACLE)


@pytest.mark.oracle
def test_rust_matches_brute_force() -> None:
    tau = _small_tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260921), n_sites=15
    )

    rust_ll = pruning_rust.log_likelihood(tau, k, pi, dataset.alignment)
    brute = brute_force_log_likelihood(tau, k, pi, dataset.alignment)

    assert_allclose(rust_ll, brute, rtol=_RTOL_ORACLE)


@pytest.mark.mathematical
def test_rescaled_and_unrescaled_rust_paths_agree() -> None:
    tau = _small_tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260922), n_sites=100
    )

    rescaled = pruning_rust.log_likelihood(tau, k, pi, dataset.alignment, rescale=True)
    unrescaled = pruning_rust.log_likelihood(
        tau, k, pi, dataset.alignment, rescale=False
    )

    assert_allclose(rescaled, unrescaled, rtol=1e-10)


@pytest.mark.edge_case
def test_rust_rejects_mismatched_pi_shape() -> None:
    tau = Node(
        name="root",
        branch_length=None,
        children=(
            Node(name="A", branch_length=0.1),
            Node(name="B", branch_length=0.2),
        ),
    )
    alignment = {
        "A": np.zeros(5, dtype=np.int64),
        "B": np.zeros(5, dtype=np.int64),
    }
    with pytest.raises(ValueError, match="pi has shape"):
        pruning_rust.log_likelihood(tau, 4, np.full(3, 1.0 / 3), alignment)


@pytest.mark.edge_case
def test_rust_rejects_alignment_missing_a_leaf() -> None:
    tau = Node(
        name="root",
        branch_length=None,
        children=(
            Node(name="A", branch_length=0.1),
            Node(name="B", branch_length=0.2),
        ),
    )
    alignment = {"A": np.zeros(5, dtype=np.int64)}
    with pytest.raises(ValueError, match="alignment is missing leaf"):
        pruning_rust.log_likelihood(tau, 4, np.full(4, 0.25), alignment)


@pytest.mark.edge_case
def test_rust_rejects_non_root_node_without_branch_length() -> None:
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
    with pytest.raises(ValueError, match="has no branch_length"):
        pruning_rust.log_likelihood(tau, 4, np.full(4, 0.25), alignment)


@pytest.mark.structural
@pytest.mark.release
def test_relative_tolerance_transfers_to_fixture_scale() -> None:
    """The tolerance holds at 200,000 sites, where an absolute one would not.

    This is the claim issue #111 turns on, in executable form. The suite's
    fast tests run at tens of sites because that is cheap, not because the
    tolerance requires it -- and a bound that only holds at tens of sites is
    not a bound on the backend, it is a bound on the test. Release-gated
    rather than deleted: it costs a full-fixture score, and issue #109 keeps
    that off the per-PR path.
    """
    params = load_fixture(FOUR_TAXA)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    alignment = dict(dataset.alignment)

    numpy_ll = pruning.log_likelihood(params.tau, params.k, params.pi, alignment)
    rust_ll = pruning_rust.log_likelihood(params.tau, params.k, params.pi, alignment)

    # The relative bound holds, unchanged from the small-size tests above.
    assert_allclose(rust_ll, numpy_ll, rtol=_RTOL_ORACLE)

    # And the absolute bound this file used to carry would have failed here,
    # on backends that are behaving correctly.
    absolute = abs(rust_ll - numpy_ll)
    assert absolute > 1e-9, (
        "expected the absolute deviation to exceed the old 1e-9 bound at "
        f"fixture scale, got {absolute:.3e}; if this no longer holds the "
        "motivating example in issue #111 needs revisiting"
    )


@pytest.mark.oracle
def test_problem_matches_the_function_bitwise() -> None:
    """The handle returns the function's value exactly, not within a tolerance.

    Both bindings run one kernel over one topology in one order, so the
    log-likelihood is the same ``f64``. That is what lets a caller move from
    one to the other without moving a committed number -- a perimeter change
    that shifts an answer has changed the model, not the plumbing (#444).
    """
    tau = _small_tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260923), n_sites=120
    )

    problem = pruning_rust.PruningProblem(k, pi, dataset.alignment)
    for rescale in (True, False):
        function = pruning_rust.log_likelihood(
            tau, k, pi, dataset.alignment, rescale=rescale
        )
        held = problem.log_likelihood(tau, rescale=rescale)
        assert held == function, f"rescale={rescale}: {held} against {function}"


@pytest.mark.oracle
def test_problem_matches_numpy_oracle() -> None:
    tau = _small_tree_n4()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260924), n_sites=50
    )

    numpy_ll = pruning.log_likelihood(tau, k, pi, dataset.alignment)
    held = pruning_rust.PruningProblem(k, pi, dataset.alignment).log_likelihood(tau)

    assert_allclose(held, numpy_ll, rtol=_RTOL_ORACLE)


@pytest.mark.structural
def test_problem_scores_many_topologies_from_one_alignment() -> None:
    """The reason the handle exists: one alignment, many topologies.

    Every candidate must get the value the function would have given it, so
    a screening loop that holds the alignment ranks candidates identically
    to one that does not.
    """
    k = 4
    pi = np.full(k, 0.25)
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(20260925)
    dataset = simulate_alignment(tau=_small_tree_n6(), k=k, pi=pi, rng=rng, n_sites=80)
    candidates = [
        with_uniform_branch_lengths(random_topology(names, rng), 0.1) for _ in range(8)
    ]

    problem = pruning_rust.PruningProblem(k, pi, dataset.alignment)
    held = [problem.log_likelihood(candidate) for candidate in candidates]
    function = [
        pruning_rust.log_likelihood(candidate, k, pi, dataset.alignment)
        for candidate in candidates
    ]

    assert held == function
    assert len(set(held)) > 1, "the fixture should separate the candidates"


@pytest.mark.edge_case
def test_problem_rejects_mismatched_pi_shape() -> None:
    alignment = {
        "A": np.zeros(5, dtype=np.int64),
        "B": np.zeros(5, dtype=np.int64),
    }
    with pytest.raises(ValueError, match="pi has shape"):
        pruning_rust.PruningProblem(4, np.full(3, 1.0 / 3), alignment)


@pytest.mark.edge_case
def test_problem_rejects_ragged_alignment() -> None:
    alignment = {
        "A": np.zeros(5, dtype=np.int64),
        "B": np.zeros(4, dtype=np.int64),
    }
    with pytest.raises(ValueError, match="the alignment is ragged"):
        pruning_rust.PruningProblem(4, np.full(4, 0.25), alignment)


@pytest.mark.edge_case
def test_problem_rejects_a_tree_whose_leaf_is_not_in_the_alignment() -> None:
    """Held once, checked every call: the alignment fixes the leaf set, the tree does not."""
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
    problem = pruning_rust.PruningProblem(4, np.full(4, 0.25), alignment)
    with pytest.raises(ValueError, match="alignment is missing leaf"):
        problem.log_likelihood(tau)


@pytest.mark.edge_case
def test_problem_rejects_non_root_node_without_branch_length() -> None:
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
    problem = pruning_rust.PruningProblem(4, np.full(4, 0.25), alignment)
    with pytest.raises(ValueError, match="has no branch_length"):
        problem.log_likelihood(tau)


@pytest.mark.structural
def test_problem_copies_the_alignment_it_is_given() -> None:
    """The handle is a copy, so a later edit to the caller's array is not seen.

    A cache that aliased its caller's buffer would make a pass depend on
    when it ran, which is the failure mode state is kept for.
    """
    k = 4
    pi = np.full(k, 0.25)
    tau = _small_tree_n4()
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260926), n_sites=40
    )
    alignment = {name: np.array(row) for name, row in dataset.alignment.items()}

    problem = pruning_rust.PruningProblem(k, pi, alignment)
    before = problem.log_likelihood(tau)
    alignment["A"][:] = (alignment["A"] + 1) % k
    after = problem.log_likelihood(tau)

    assert before == after
