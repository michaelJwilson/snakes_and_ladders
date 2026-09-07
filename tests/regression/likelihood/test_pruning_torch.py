"""Regression tests for ``phylo.likelihood.pruning_torch``.

Four independent checks per issue #70, none sharing an implementation with
the thing it judges:

- Agreement with ``pruning.py``, the NumPy oracle CLAUDE.md requires every
  backend be pinned against (``test_torch_matches_numpy_oracle``).
- Agreement with ``brute_force.py`` at ``n <= 6`` taxa, to machine precision
  (``test_torch_matches_brute_force``) -- a genuinely different algorithm,
  not a second opinion from the same recursion.
- ``torch.autograd.gradcheck`` against central finite differences of the
  NumPy likelihood w.r.t. branch lengths
  (``test_gradient_matches_finite_differences_of_numpy_oracle``).
- Rescaled and unrescaled Torch paths agreeing, the check
  ``likelihood/CLAUDE.md``'s "Rescaling must stay differentiable" calls for
  (``test_rescaled_and_unrescaled_torch_paths_agree``).

A fifth check pins the general ``rate_matrix`` path (``torch.matrix_exp``)
against the closed-form JC path when given the JC generator
(``test_matrix_exp_rate_matrix_path_matches_closed_form``) -- the path issue
#70 asks be exercised for fitting a general Q, even though JC's Q is fully
determined by k.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from phylo.likelihood import pruning, pruning_torch
from phylo.likelihood.brute_force import brute_force_log_likelihood
from phylo.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from phylo.sim.jc import jc_rate_matrix
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import Node

# Relative, not absolute -- see issue #111 and the note in
# test_pruning_rust.py. CROSS_DEVICE_RTOL_FLOAT64 is the float64
# implementation-agreement bound stated in docs/tex/.
_RTOL_ORACLE = CROSS_DEVICE_RTOL_FLOAT64

_FD_EPS = 1e-6

# Finite differences are far less precise than the likelihood itself, so the
# gradient bound is its own number rather than the oracle's. Relative for the
# same reason: the gradient of a sum over sites scales with the site count.
# Measured worst relative disagreement is 6.1e-07 at the step above, so this
# leaves better than an order of magnitude.
_RTOL_GRADIENT = 1e-5


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


def _with_branch_lengths(node: Node, lengths: dict[str, float]) -> Node:
    """Rebuild ``node``'s subtree with each non-root branch length from ``lengths``."""
    new_branch_length = None if node.branch_length is None else lengths[node.name]
    return replace(
        node,
        branch_length=new_branch_length,
        children=tuple(_with_branch_lengths(child, lengths) for child in node.children),
    )


@pytest.mark.oracle
def test_torch_matches_numpy_oracle() -> None:
    tau = _small_tree_n4()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(tau=tau, k=k, pi=pi, seed=20260910, n_sites=50)
    branch_lengths = pruning_torch.branch_lengths_from_tree(tau)

    numpy_ll = pruning.log_likelihood(tau, k, pi, dataset.alignment)
    torch_ll = pruning_torch.log_likelihood(
        tau, k, pi, dataset.alignment, branch_lengths
    )

    assert_allclose(float(torch_ll), numpy_ll, rtol=_RTOL_ORACLE)


@pytest.mark.oracle
def test_torch_matches_brute_force() -> None:
    tau = _small_tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(tau=tau, k=k, pi=pi, seed=20260911, n_sites=15)
    branch_lengths = pruning_torch.branch_lengths_from_tree(tau)

    torch_ll = pruning_torch.log_likelihood(
        tau, k, pi, dataset.alignment, branch_lengths
    )
    brute = brute_force_log_likelihood(tau, k, pi, dataset.alignment)

    assert_allclose(float(torch_ll), brute, rtol=_RTOL_ORACLE)


@pytest.mark.mathematical
def test_rescaled_and_unrescaled_torch_paths_agree() -> None:
    tau = _small_tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(tau=tau, k=k, pi=pi, seed=20260912, n_sites=100)
    branch_lengths = pruning_torch.branch_lengths_from_tree(tau)

    rescaled = pruning_torch.log_likelihood(
        tau, k, pi, dataset.alignment, branch_lengths, rescale=True
    )
    unrescaled = pruning_torch.log_likelihood(
        tau, k, pi, dataset.alignment, branch_lengths, rescale=False
    )

    assert_allclose(float(rescaled), float(unrescaled), rtol=1e-10)


@pytest.mark.oracle
def test_matrix_exp_rate_matrix_path_matches_closed_form() -> None:
    tau = _small_tree_n4()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(tau=tau, k=k, pi=pi, seed=20260913, n_sites=50)
    branch_lengths = pruning_torch.branch_lengths_from_tree(tau)
    rate_matrix = torch.as_tensor(jc_rate_matrix(k), dtype=torch.float64)

    closed_form = pruning_torch.log_likelihood(
        tau, k, pi, dataset.alignment, branch_lengths
    )
    general = pruning_torch.log_likelihood(
        tau, k, pi, dataset.alignment, branch_lengths, rate_matrix=rate_matrix
    )

    assert_allclose(float(general), float(closed_form), rtol=_RTOL_ORACLE)


@pytest.mark.mathematical
@pytest.mark.oracle
def test_gradient_matches_finite_differences_of_numpy_oracle() -> None:
    tau = _small_tree_n4()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(tau=tau, k=k, pi=pi, seed=20260914, n_sites=30)
    order = pruning_torch.branch_order(tau)
    branch_lengths = pruning_torch.branch_lengths_from_tree(tau).requires_grad_(True)

    def _torch_ll(lengths: torch.Tensor) -> torch.Tensor:
        return pruning_torch.log_likelihood(tau, k, pi, dataset.alignment, lengths)

    assert torch.autograd.gradcheck(
        _torch_ll, (branch_lengths,), eps=_FD_EPS, atol=1e-5
    )

    autograd_grad = torch.autograd.grad(_torch_ll(branch_lengths), branch_lengths)[
        0
    ].numpy()

    lengths0 = branch_lengths.detach().numpy()

    def _numpy_ll(lengths: np.ndarray) -> float:
        tau_perturbed = _with_branch_lengths(
            tau, dict(zip(order, lengths, strict=True))
        )
        return pruning.log_likelihood(tau_perturbed, k, pi, dataset.alignment)

    finite_diff_grad = np.empty_like(lengths0)
    for i in range(len(lengths0)):
        plus, minus = lengths0.copy(), lengths0.copy()
        plus[i] += _FD_EPS
        minus[i] -= _FD_EPS
        finite_diff_grad[i] = (_numpy_ll(plus) - _numpy_ll(minus)) / (2 * _FD_EPS)

    assert_allclose(autograd_grad, finite_diff_grad, rtol=_RTOL_GRADIENT)
