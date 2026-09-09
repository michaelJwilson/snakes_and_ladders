"""The analytic two-pass backward, against the taped gradient it would replace.

``pruning_torch`` is the oracle and stays (``likelihood/CLAUDE.md``). What is
checked here is that the closed form of ``alg:pruning-backward`` computes the
same derivative: against the tape, against central differences, and through
``torch.autograd.gradcheck``, all in ``float64``.

**The central-difference tolerance is derived, not chosen.** Sweeping the step
over ``1e-4, 1e-5, 1e-6, 1e-7`` at 4 and 8 taxa and 2,000 and 20,000 sites, the
largest relative deviation of *any* of the three routes from the difference
quotient is 9.5e-4 at ``h = 1e-4`` and 1.007e-6 at ``h = 1e-6``, where the
quotient's truncation and its cancellation are balanced. The three routes
deviate by the same amount to four significant figures at every step, which
says the deviation is the quotient's and not any gradient's. The bound below is
``h = 1e-6`` and twice that worst case.

Agreement with the taped gradient is a different question and a much tighter
one: the worst observed relative difference is 5.9e-13, inside
``CROSS_DEVICE_RTOL_FLOAT64``, which this module reads from
``likelihood.device`` rather than retyping.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood import pruning_analytic, pruning_torch
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.patterns import compress
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, SMALL_SITES, load_fixture

#: Step at which the difference quotient is most accurate here, and the
#: relative bound the module docstring derives for it.
CENTRAL_DIFFERENCE_STEP = 1e-6
CENTRAL_DIFFERENCE_RTOL = 2e-6


def _case(
    name: str, n_sites: int
) -> tuple[Node, int, np.ndarray, dict[str, np.ndarray], torch.Tensor]:
    params = load_fixture(name)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=n_sites,
    )
    return (
        params.tau,
        params.k,
        params.pi,
        dataset.alignment,
        pruning_torch.branch_lengths_from_tree(params.tau),
    )


def _gradient(evaluate, case, **kwargs) -> tuple[float, np.ndarray]:  # type: ignore[no-untyped-def]
    tau, k, pi, alignment, lengths = case
    branch_lengths = lengths.clone().requires_grad_(True)
    value = evaluate(tau, k, pi, alignment, branch_lengths, **kwargs)
    value.backward()
    assert branch_lengths.grad is not None
    return float(value.detach()), branch_lengths.grad.numpy().copy()


@pytest.mark.oracle
@pytest.mark.parametrize(("fixture_name", "n_sites"), [(SMALL_SITES, 2000), (EIGHT_TAXA, 2000)])
def test_value_is_the_taped_path_s(fixture_name: str, n_sites: int) -> None:
    """The forward value is ``pruning_torch``'s, which is the oracle."""
    case = _case(fixture_name, n_sites)
    tau, k, pi, alignment, lengths = case
    expected = float(pruning_torch.log_likelihood(tau, k, pi, alignment, lengths))
    actual = float(pruning_analytic.log_likelihood(tau, k, pi, alignment, lengths))
    assert_allclose(actual, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.oracle
@pytest.mark.parametrize(("fixture_name", "n_sites"), [(SMALL_SITES, 2000), (EIGHT_TAXA, 2000)])
def test_gradient_matches_the_taped_gradient(fixture_name: str, n_sites: int) -> None:
    """The gradient the tape produces, to the float64 agreement tolerance."""
    case = _case(fixture_name, n_sites)
    _, expected = _gradient(pruning_torch.log_likelihood, case)
    _, actual = _gradient(pruning_analytic.log_likelihood, case)
    assert_allclose(actual, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.oracle
@pytest.mark.parametrize(("fixture_name", "n_sites"), [(SMALL_SITES, 2000), (EIGHT_TAXA, 2000)])
def test_gradient_matches_central_differences(fixture_name: str, n_sites: int) -> None:
    """Against a quotient of the forward pass alone, at the derived tolerance."""
    tau, k, pi, alignment, lengths = _case(fixture_name, n_sites)
    _, gradient = _gradient(
        pruning_analytic.log_likelihood, (tau, k, pi, alignment, lengths)
    )

    def value(at: torch.Tensor) -> float:
        return float(pruning_torch.log_likelihood(tau, k, pi, alignment, at))

    quotient = np.empty_like(gradient)
    for position in range(len(lengths)):
        step = torch.zeros_like(lengths)
        step[position] = CENTRAL_DIFFERENCE_STEP
        quotient[position] = (value(lengths + step) - value(lengths - step)) / (
            2.0 * CENTRAL_DIFFERENCE_STEP
        )
    assert_allclose(gradient, quotient, rtol=CENTRAL_DIFFERENCE_RTOL)


@pytest.mark.mathematical
def test_gradcheck_in_float64() -> None:
    """``torch.autograd.gradcheck``, which is the referee this route is held to."""
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 200)
    assert torch.autograd.gradcheck(
        lambda at: pruning_analytic.log_likelihood(tau, k, pi, alignment, at),
        (lengths.clone().requires_grad_(True),),
        eps=1e-6,
        atol=1e-7,
        rtol=1e-5,
    )


@pytest.mark.mathematical
def test_weighted_patterns_give_the_uncompressed_gradient() -> None:
    """The compressed alignment with its weights is the full alignment's gradient.

    The weights are constants of the data, so the gradient is the weighted sum
    of the site gradients -- the same claim ``pruning_torch`` makes for the
    value, now for the derivative.
    """
    tau, k, pi, alignment, lengths = _case(EIGHT_TAXA, 2000)
    compressed = compress(alignment)
    _, full = _gradient(pruning_analytic.log_likelihood, (tau, k, pi, alignment, lengths))
    _, weighted = _gradient(
        pruning_analytic.log_likelihood,
        (tau, k, pi, compressed.alignment, lengths),
        weights=compressed.weights,
    )
    assert_allclose(weighted, full, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.mathematical
def test_a_general_rate_matrix_agrees_with_the_taped_path() -> None:
    """``dP/dt = Q P(t)`` reproduces the tape's gradient through ``matrix_exp``."""
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 500)
    off_diagonal = torch.full((k, k), 0.3, dtype=torch.float64) * (
        1.0 - torch.eye(k, dtype=torch.float64)
    )
    rate_matrix = off_diagonal - torch.diag(off_diagonal.sum(dim=1))
    case = (tau, k, pi, alignment, lengths)
    _, expected = _gradient(
        pruning_torch.log_likelihood, case, rate_matrix=rate_matrix
    )
    _, actual = _gradient(
        pruning_analytic.log_likelihood, case, rate_matrix=rate_matrix
    )
    assert_allclose(actual, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.mathematical
def test_a_zero_message_does_not_produce_a_nan_gradient() -> None:
    """A site an observation forbids has a finite derivative, not a NaN.

    The sibling product is taken by a scan rather than by dividing the parent's
    partial by the child's message, and this is the case that separates the
    two: at a site where one leaf's state forces a zero, the division is 0/0.
    """
    tau = Node(
        name="root",
        children=(
            Node(name="A", children=(), branch_length=1e-9),
            Node(
                name="ancestor",
                children=(
                    Node(name="B", children=(), branch_length=1e-9),
                    Node(name="C", children=(), branch_length=0.2),
                ),
                branch_length=0.2,
            ),
        ),
        branch_length=None,
    )
    alignment = {
        "A": np.array([0, 1, 2]),
        "B": np.array([1, 2, 3]),
        "C": np.array([2, 3, 0]),
    }
    lengths = pruning_torch.branch_lengths_from_tree(tau)
    case = (tau, 4, np.full(4, 0.25), alignment, lengths)
    _, expected = _gradient(pruning_torch.log_likelihood, case)
    _, actual = _gradient(pruning_analytic.log_likelihood, case)
    assert np.all(np.isfinite(actual))
    assert_allclose(actual, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.edge_case
def test_a_rate_matrix_that_requires_a_gradient_is_refused() -> None:
    """Refused rather than silently returning no gradient for it."""
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 100)
    rate_matrix = torch.eye(k, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="branch_lengths only"):
        pruning_analytic.log_likelihood(
            tau, k, pi, alignment, lengths, rate_matrix=rate_matrix
        )


@pytest.mark.edge_case
def test_branch_lengths_of_the_wrong_length_are_refused() -> None:
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 100)
    with pytest.raises(ValueError, match="branch_order"):
        pruning_analytic.log_likelihood(tau, k, pi, alignment, lengths[:-1])


@pytest.mark.structural
def test_the_graph_is_one_node_whatever_the_tree() -> None:
    """The count issue #443 attacks: the taped graph tracks the tree, this does not.

    Measured here rather than asserted as a constant: what matters is that the
    count stops growing with the topology, which is the mechanism the route
    claims.
    """

    def graph_nodes(value: torch.Tensor) -> int:
        seen: set[object] = set()
        stack = [value.grad_fn]
        while stack:
            node = stack.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            stack.extend(parent for parent, _ in node.next_functions)
        return len(seen)

    counts = []
    for fixture_name in (SMALL_SITES, EIGHT_TAXA):
        tau, k, pi, alignment, lengths = _case(fixture_name, 100)
        at = lengths.clone().requires_grad_(True)
        counts.append(
            (
                graph_nodes(pruning_torch.log_likelihood(tau, k, pi, alignment, at)),
                graph_nodes(pruning_analytic.log_likelihood(tau, k, pi, alignment, at)),
            )
        )
    (taped_small, analytic_small), (taped_large, analytic_large) = counts
    assert taped_large > taped_small
    assert analytic_large == analytic_small
