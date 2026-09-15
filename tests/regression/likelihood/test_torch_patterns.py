"""The PyTorch patterns issue #544 measured, and what each decline rests on.

Every pattern in the ticket was declined on its measurement, and a decline on
speed is only a decline on speed if the alternative computes the same thing.
That is what this module pins: each arm reproduces the value the path in the
tree produces, so `STATUS.md`'s numbers are the whole of the difference
between them and nothing is hiding in the third decimal.

The alternatives are written out here rather than conserved in
`snakes_and_ladders.sandbox`, whose rule admits a *finished* route and reports
an unfinished one. None is an implementation; each is the two or three lines a
proposer would write, kept beside the assertion that says what it costs.

`tests/benchmarks/test_torch_patterns_bench.py` times the same arms.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from snakes_and_ladders.likelihood.pruning_torch import (
    branch_lengths_from_tree,
    log_likelihood,
    transition_probabilities,
)
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, load_fixture

#: Sites enough that the recursion dominates the Python around it, few enough
#: that every test here is under a second.
_SITES = 2000


def _dataset() -> tuple[Node, int, np.ndarray, dict[str, np.ndarray], torch.Tensor]:
    params = load_fixture(EIGHT_TAXA)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=_SITES,
    )
    return (
        params.tau,
        params.k,
        params.pi,
        dataset.alignment,
        branch_lengths_from_tree(params.tau),
    )


def _eigen_transitions(rate_matrix: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``P(t)`` by one eigendecomposition of ``Q``, not one ``matrix_exp`` per branch.

    The alternative item 6 names: ``Q`` is decomposed once and only the
    eigenvalues are exponentiated per branch, so the per-branch cost falls from
    a matrix exponential to a vector one.
    """
    values, vectors = torch.linalg.eig(rate_matrix)
    inverse = torch.linalg.inv(vectors)
    scaled = torch.exp(values[None, :] * t[:, None].to(values.dtype))
    transitions: torch.Tensor = (
        (vectors[None] * scaled[:, None, :]) @ inverse[None]
    ).real
    return transitions


@pytest.mark.mathematical
def test_inference_mode_returns_the_value_no_grad_returns() -> None:
    """Item 1: the two evaluation modes differ in bookkeeping, not in arithmetic.

    Bitwise, not to a tolerance: ``inference_mode`` switches off the version
    counter and view tracking, and neither enters a reduction.
    """
    tau, k, pi, alignment, lengths = _dataset()
    with torch.no_grad():
        taped_off = float(log_likelihood(tau, k, pi, alignment, lengths))
    with torch.inference_mode():
        inferred = float(log_likelihood(tau, k, pi, alignment, lengths))
    assert inferred == taped_off


@pytest.mark.mathematical
def test_vmap_over_theta_reproduces_the_sequential_objective() -> None:
    """Item 2: batching over starting points changes no value.

    What declines the pattern is the optimizer, not the arithmetic: L-BFGS'
    strong-Wolfe line search branches on each start's own values, so a batched
    objective has no batched consumer. This says the objective itself batches
    exactly.
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


@pytest.mark.mathematical
def test_eigendecomposition_gives_a_transition_matrix() -> None:
    """Rows sum to 1 and no entry is negative, whichever route built them.

    Agreement with ``matrix_exp`` would be satisfied by two implementations
    wrong the same way; this is the property a transition matrix has on its
    own.
    """
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


@pytest.mark.mathematical
def test_the_transposed_operand_is_the_contiguous_one() -> None:
    """Item 5: ``transitions[i].T`` is a view, and BLAS reads it as one.

    Bitwise equality is the claim. A copy that changed the product would mean
    the reduction order moved, which is what the memory-layout rule is
    measured against rather than asserted.
    """
    tau, k, _, _, lengths = _dataset()
    transitions = transition_probabilities(lengths, k, None)
    partial = torch.rand((_SITES, k), dtype=torch.float64)

    view = partial @ transitions[0].T
    copied = partial @ transitions[0].T.contiguous()
    assert torch.equal(view, copied)
