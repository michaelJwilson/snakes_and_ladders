"""Regression tests for ``snakes_and_ladders.likelihood.pruning_torch``.

Issue #70's checks that every route shares are one body each, parametrised
over ``ROUTES`` (issue #982): agreement with the NumPy oracle, bitwise, in
``test_pruning_common.py``; with ``brute_force.py`` at ``n <= 6`` taxa and
rescaled against unrescaled --- the check ``likelihood/CLAUDE.md``'s
"Rescaling must stay differentiable" calls for --- in
``test_likelihood_pruning.py``. What is left is the taped route's own:
``torch.autograd.gradcheck`` and central finite differences of the NumPy
likelihood w.r.t. branch lengths
(``test_gradient_matches_finite_differences_of_numpy_oracle``).

A further check pins the general ``rate_matrix`` path (``torch.matrix_exp``)
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
from snakes_and_ladders.likelihood import pruning, pruning_torch
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.sim.jc import jc_rate_matrix
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import SMALL_SITES, load_fixture

# Relative, not absolute -- see issue #111 and the note in
# test_pruning_rust.py. CROSS_DEVICE_RTOL_FLOAT64 is the float64
# implementation-agreement bound stated in docs/tex/textbook.tex (sec:tolerance).
_RTOL_ORACLE = CROSS_DEVICE_RTOL_FLOAT64

_FD_EPS = 1e-6

# Finite differences are far less precise than the likelihood itself, so the
# gradient bound is its own number rather than the oracle's. Relative for the
# same reason: the gradient of a sum over sites scales with the site count.
# Measured worst relative disagreement is 6.1e-07 at the step above, so this
# leaves better than an order of magnitude.
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
    # #264 computes every branch's P(t) in one call. The JC closed form is
    # elementwise, so each matrix of the batch equals the scalar call's
    # bitwise -- `torch.equal`, not a tolerance. `matrix_exp` is not: torch
    # picks its Pade degree per input norm and its batched kernel is a
    # different code path, so the GTR matrices agree to 2.9e-13 absolute
    # (measured, stable across runs) and are held to 1e-12 rather than to
    # equality that would assert something false.
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
