"""The PyTorch patterns issue #544 measured, and what each decline rests on.

Each declined arm reproduces the in-tree value, so `STATUS.md`'s timings are
the whole difference. The arms are the few lines a proposer would write, not
sandbox routes. `tests/benchmarks/test_torch_patterns_bench.py` times them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from sal.likelihood.torch.pruning import (
    branch_lengths_from_tree,
    log_likelihood,
    transition_probabilities,
)
from sal.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, simulated_alignment

#: Sites enough that the recursion dominates the Python around it, few enough
#: that every test here is under a second.
_SITES = 2000


def _dataset() -> tuple[Node, int, np.ndarray, dict[str, np.ndarray], torch.Tensor]:
    params, alignment = simulated_alignment(EIGHT_TAXA, _SITES)
    return (
        params.tau,
        params.k,
        params.pi,
        alignment,
        branch_lengths_from_tree(params.tau),
    )


def _eigen_transitions(rate_matrix: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``P(t)`` by one eigendecomposition of ``Q``, not one ``matrix_exp`` per branch."""
    values, vectors = torch.linalg.eig(rate_matrix)
    inverse = torch.linalg.inv(vectors)
    scaled = torch.exp(values[None, :] * t[:, None].to(values.dtype))
    transitions: torch.Tensor = (
        (vectors[None] * scaled[:, None, :]) @ inverse[None]
    ).real
    return transitions


@pytest.mark.analytic
def test_inference_mode_returns_the_value_no_grad_returns() -> None:
    """Item 1: the two evaluation modes differ in bookkeeping, not in arithmetic.

    Bitwise: version counters and view tracking enter no reduction.
    """
    tau, k, pi, alignment, lengths = _dataset()
    with torch.no_grad():
        taped_off = float(log_likelihood(tau, k, pi, alignment, lengths))
    with torch.inference_mode():
        inferred = float(log_likelihood(tau, k, pi, alignment, lengths))
    assert inferred == taped_off


@pytest.mark.analytic
def test_vmap_over_theta_reproduces_the_sequential_objective() -> None:
    """Item 2: batching over starting points changes no value.

    L-BFGS' line search branches per start, so the optimizer declines it, not this.
    """
    tau, k, pi, alignment, _ = _dataset()
    objective = BranchLengthObjective(tau, k, pi, alignment)
    theta = objective.initial()
    batch = torch.stack([theta + 0.01 * offset for offset in range(4)])

    batched = torch.vmap(objective)(batch)
    sequential = torch.stack([objective(row) for row in batch])
    assert_allclose(batched.numpy(), sequential.numpy(), rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.oracle
def test_eigendecomposition_reproduces_matrix_exp() -> None:
    """Item 6: the closed form declined on its share, not on its accuracy.

    ``matrix_exp`` is the oracle here, being the path in the tree.
    """
    tau, k, _, alignment, lengths = _dataset()
    objective = SubstitutionModelObjective(tau, k, alignment)
    rate_matrix = objective.rate_matrix(objective.initial()).detach()

    expected = transition_probabilities(lengths, k, rate_matrix)
    actual = _eigen_transitions(rate_matrix, lengths)
    assert_allclose(actual.numpy(), expected.numpy(), rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.analytic
def test_eigendecomposition_gives_a_transition_matrix() -> None:
    """Rows sum to 1 and no entry is negative, whichever route built them."""
    tau, k, _, alignment, lengths = _dataset()
    objective = SubstitutionModelObjective(tau, k, alignment)
    rate_matrix = objective.rate_matrix(objective.initial()).detach()

    transitions = _eigen_transitions(rate_matrix, lengths)
    assert_allclose(
        transitions.sum(dim=-1).numpy(),
        np.ones((lengths.shape[0], k)),
        rtol=CROSS_DEVICE_RTOL_FLOAT64,
    )
    assert float(transitions.min()) >= 0.0


@pytest.mark.analytic
def test_the_transposed_operand_is_the_contiguous_one() -> None:
    """Item 5: ``transitions[i].T`` is a view, and BLAS reads it as one.

    Bitwise here over 200 draws at 1-8 threads, not on a runner: tolerance, seeded.
    """
    tau, k, _, _, lengths = _dataset()
    transitions = transition_probabilities(lengths, k, None)
    generator = torch.Generator().manual_seed(544)
    partial = torch.rand((_SITES, k), dtype=torch.float64, generator=generator)

    view = partial @ transitions[0].T
    copied = partial @ transitions[0].T.contiguous()
    assert_allclose(view.numpy(), copied.numpy(), rtol=CROSS_DEVICE_RTOL_FLOAT64)
