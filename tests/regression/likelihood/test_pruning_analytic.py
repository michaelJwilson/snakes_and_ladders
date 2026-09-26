"""The analytic two-pass backward, against the taped gradient it would replace.

``pruning_torch`` is the oracle (``likelihood/CLAUDE.md``); ``alg:pruning-backward``
is checked against the tape, central differences and ``gradcheck`` in
``float64``. The difference tolerance is derived: over steps ``1e-4`` to
``1e-7``, 4 and 8 taxa, 2,000 and 20,000 sites, the worst deviation of all
three routes is 9.5e-4 at ``h = 1e-4`` and 1.007e-6 at ``h = 1e-6``, alike to
four figures (the quotient's error); the bound is twice that at ``1e-6``.
Against the tape: 5.9e-13, inside ``CROSS_DEVICE_RTOL_FLOAT64``. Five checks
run over :data:`ROUTES` with `burn` (#982).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.likelihood import pruning_analytic
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.patterns import compress
from sal.likelihood.pruning import torch as pruning_torch
from sal.sandbox import pruning_burn
from sal.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, SMALL_SITES, simulated_alignment

#: Step at which the difference quotient is most accurate here, and the
#: relative bound the module docstring derives for it.
CENTRAL_DIFFERENCE_STEP = 1e-6
CENTRAL_DIFFERENCE_RTOL = 2e-6

#: The routes held to the tape: the adopted analytic backward, and `burn`'s
#: second tape (issue #449, declined and conserved in the sandbox), which
#: skips unless the extension carries the ``sandbox`` Cargo feature. Merged
#: here from ``tests/regression/sandbox/test_pruning_burn.py`` (issue #982).
ROUTES = [
    pytest.param(pruning_analytic.log_likelihood, id="analytic"),
    pytest.param(
        pruning_burn.log_likelihood,
        id="burn",
        marks=pytest.mark.skipif(
            not pruning_burn.AVAILABLE,
            reason="extension built without the `sandbox` Cargo feature",
        ),
    ),
]
_FIXTURES = pytest.mark.parametrize(
    ("fixture_name", "n_sites"), [(SMALL_SITES, 2000), (EIGHT_TAXA, 2000)]
)


def _case(
    name: str, n_sites: int
) -> tuple[Node, int, np.ndarray, dict[str, np.ndarray], torch.Tensor]:
    params, alignment = simulated_alignment(name, n_sites)
    return (
        params.tau,
        params.n_states,
        params.pi,
        alignment,
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
@pytest.mark.parametrize("route", ROUTES)
@_FIXTURES
def test_value_is_the_taped_path_s(
    route: Callable[..., torch.Tensor], fixture_name: str, n_sites: int
) -> None:
    """The forward value is ``pruning_torch``'s, which is the oracle."""
    case = _case(fixture_name, n_sites)
    tau, k, pi, alignment, lengths = case
    expected = float(pruning_torch.log_likelihood(tau, k, pi, alignment, lengths))
    actual = float(route(tau, k, pi, alignment, lengths))
    assert_allclose(actual, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.oracle
@pytest.mark.parametrize("route", ROUTES)
@_FIXTURES
def test_gradient_matches_the_taped_gradient(
    route: Callable[..., torch.Tensor], fixture_name: str, n_sites: int
) -> None:
    """The gradient the tape produces, to the float64 agreement tolerance.

    For `burn`, the `f64` question: an `f32` tape could not agree.
    """
    case = _case(fixture_name, n_sites)
    _, expected = _gradient(pruning_torch.log_likelihood, case)
    _, actual = _gradient(route, case)
    assert_allclose(actual, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.oracle
@pytest.mark.parametrize("route", ROUTES)
@_FIXTURES
def test_gradient_matches_central_differences(
    route: Callable[..., torch.Tensor], fixture_name: str, n_sites: int
) -> None:
    """Against a quotient of the forward pass alone, at the derived tolerance."""
    tau, k, pi, alignment, lengths = _case(fixture_name, n_sites)
    _, gradient = _gradient(route, (tau, k, pi, alignment, lengths))

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


@pytest.mark.analytic
@pytest.mark.parametrize("route", ROUTES)
def test_gradcheck_in_float64(route: Callable[..., torch.Tensor]) -> None:
    """``torch.autograd.gradcheck``, which is the referee this route is held to."""
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 200)
    assert torch.autograd.gradcheck(
        lambda at: route(tau, k, pi, alignment, at),
        (lengths.clone().requires_grad_(True),),
        eps=1e-6,
        atol=1e-7,
        rtol=1e-5,
    )


@pytest.mark.analytic
@pytest.mark.parametrize("route", ROUTES)
def test_weighted_patterns_give_the_uncompressed_gradient(
    route: Callable[..., torch.Tensor],
) -> None:
    """The compressed alignment with its weights is the full alignment's gradient."""
    tau, k, pi, alignment, lengths = _case(EIGHT_TAXA, 2000)
    compressed = compress(alignment)
    _, full = _gradient(route, (tau, k, pi, alignment, lengths))
    _, weighted = _gradient(
        route,
        (tau, k, pi, compressed.alignment, lengths),
        weights=compressed.weights,
    )
    assert_allclose(weighted, full, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.analytic
def test_a_general_rate_matrix_agrees_with_the_taped_path() -> None:
    """``dP/dt = Q P(t)`` reproduces the tape's gradient through ``matrix_exp``."""
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 500)
    off_diagonal = torch.full((k, k), 0.3, dtype=torch.float64) * (
        1.0 - torch.eye(k, dtype=torch.float64)
    )
    rate_matrix = off_diagonal - torch.diag(off_diagonal.sum(dim=1))
    case = (tau, k, pi, alignment, lengths)
    _, expected = _gradient(pruning_torch.log_likelihood, case, rate_matrix=rate_matrix)
    _, actual = _gradient(
        pruning_analytic.log_likelihood, case, rate_matrix=rate_matrix
    )
    assert_allclose(actual, expected, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.analytic
def test_a_zero_message_does_not_produce_a_nan_gradient() -> None:
    """A site an observation forbids has a finite derivative, not a NaN.

    The sibling product is a scan: dividing would be 0/0 here.
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


@pytest.mark.smoke
def test_a_rate_matrix_that_requires_a_gradient_is_refused() -> None:
    """Refused rather than silently returning no gradient for it."""
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 100)
    rate_matrix = torch.eye(k, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="branch_lengths only"):
        pruning_analytic.log_likelihood(
            tau, k, pi, alignment, lengths, rate_matrix=rate_matrix
        )


@pytest.mark.smoke
def test_branch_lengths_of_the_wrong_length_are_refused() -> None:
    tau, k, pi, alignment, lengths = _case(SMALL_SITES, 100)
    with pytest.raises(ValueError, match="branch_order"):
        pruning_analytic.log_likelihood(tau, k, pi, alignment, lengths[:-1])


@pytest.mark.smoke
def test_the_graph_is_one_node_whatever_the_tree() -> None:
    """The count issue #443 attacks: the taped graph tracks the tree, this does not."""

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
