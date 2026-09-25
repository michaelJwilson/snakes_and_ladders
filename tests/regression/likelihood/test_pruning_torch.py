"""Regression tests for ``sal.likelihood.pruning.torch``.

Issue #70's shared checks run over ``ROUTES`` (#982): the oracle in
``test_pruning_common.py``, brute force and rescaling in
``test_likelihood_pruning.py``. Here, the taped route's own: ``gradcheck``
and central differences of the NumPy likelihood; and the general
``rate_matrix`` path (``torch.matrix_exp``) against closed-form JC given the
JC generator, the path #70 asks be exercised for fitting a general Q.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.backend import Backend
from sal.likelihood import pruning
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.pruning import torch as pruning_torch
from sal.sim.jc import jc_rate_matrix
from sal.sim.simulate import simulate_alignment
from sal.sim.tree import Node

from tests._fixtures import SMALL_SITES, load_fixture

# Relative, not absolute -- see issue #111 and the note in
# test_pruning_rust.py. CROSS_DEVICE_RTOL_FLOAT64 is the float64
# implementation-agreement bound stated in docs/tex/textbook.tex (sec:tolerance).
_RTOL_ORACLE = CROSS_DEVICE_RTOL_FLOAT64

_FD_EPS = 1e-6

# Relative, its own number: finite differences are far less precise. Measured
# worst 6.1e-07 at the step above, over an order inside.
_RTOL_GRADIENT = 1e-5


def _with_branch_lengths(node: Node, lengths: dict[str, float]) -> Node:
    """Rebuild ``node``'s subtree with each non-root branch length from ``lengths``."""
    new_branch_length = None if node.branch_length is None else lengths[node.name]
    return replace(
        node,
        branch_length=new_branch_length,
        children=tuple(_with_branch_lengths(child, lengths) for child in node.children),
    )


@pytest.mark.oracle
def test_matrix_exp_rate_matrix_path_matches_closed_form() -> None:
    tau = load_fixture(SMALL_SITES).tau
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260913), n_sites=50
    )
    branch_lengths = pruning_torch.branch_lengths_from_tree(tau)
    rate_matrix = torch.as_tensor(jc_rate_matrix(k), dtype=torch.float64)

    closed_form = pruning_torch.log_likelihood(
        tau, k, pi, dataset.alignment, branch_lengths
    )
    general = pruning_torch.log_likelihood(
        tau, k, pi, dataset.alignment, branch_lengths, rate_matrix=rate_matrix
    )

    assert_allclose(float(general), float(closed_form), rtol=_RTOL_ORACLE)


@pytest.mark.analytic
@pytest.mark.oracle
def test_gradient_matches_finite_differences_of_numpy_oracle() -> None:
    tau = load_fixture(SMALL_SITES).tau
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260914), n_sites=30
    )
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


@pytest.mark.oracle
def test_the_batched_transition_matrices_are_the_scalar_ones() -> None:
    # #264 batches P(t): JC is elementwise, so `torch.equal`; `matrix_exp`
    # picks a Pade degree per norm and batches differently, so GTR agrees to
    # 2.9e-13 absolute (stable), held to 1e-12.
    lengths = torch.tensor([0.01, 0.1, 0.35, 1.2, 3.0], dtype=torch.float64)
    rate = torch.tensor(
        [
            [-1.1, 0.5, 0.3, 0.3],
            [0.2, -0.9, 0.4, 0.3],
            [0.3, 0.4, -1.0, 0.3],
            [0.3, 0.3, 0.2, -0.8],
        ],
        dtype=torch.float64,
    )

    batched_jc = pruning_torch.transition_probabilities(lengths, 4, None)
    batched_gtr = pruning_torch.transition_probabilities(lengths, 4, rate)

    assert batched_jc.shape == batched_gtr.shape == (5, 4, 4)
    for position, length in enumerate(lengths):
        assert torch.equal(
            batched_jc[position],
            pruning_torch.transition_probabilities(length, 4, None),
        )
        scalar_gtr = pruning_torch.transition_probabilities(length, 4, rate)
        assert float((batched_gtr[position] - scalar_gtr).abs().max()) < 1e-12


@pytest.mark.oracle
@pytest.mark.backend
def test_the_gateways_torch_door_is_the_taped_value_at_the_oracles_tolerance() -> None:
    # Issue #1059: `pruning.log_likelihood(backend=TORCH)` is this module at
    # the lengths `tau` carries, detached; bitwise that, and within the
    # float64 agreement bound of the NumPy oracle.
    tau = load_fixture(SMALL_SITES).tau
    k = 4
    pi = np.full(k, 0.25)
    alignment = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(1059), n_sites=50
    ).alignment
    through = pruning.log_likelihood(tau, k, pi, alignment, backend=Backend.TORCH)
    taped = pruning_torch.log_likelihood(
        tau, k, pi, alignment, pruning_torch.branch_lengths_from_tree(tau)
    )
    assert through == float(taped)
    assert_allclose(
        through, pruning.log_likelihood(tau, k, pi, alignment), rtol=_RTOL_ORACLE
    )
