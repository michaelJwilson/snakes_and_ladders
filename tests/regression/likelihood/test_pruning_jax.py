"""The JAX pruning route against the PyTorch one and brute force (issue #1005).

``pruning_jax`` is ``pruning_torch``'s recursion traced once per topology, so
the referee is ``pruning_torch`` through the objectives' own ``TORCH`` route
(the value, and the gradient by autograd), with brute-force marginalization
over internal states as the exact answer on a small tree. The tolerance is
``CROSS_DEVICE_RTOL_FLOAT64``: the two routes share the operation order but
not the backend, and measured agreement is 8.3e-16 (JC) and 9.8e-13 (GTR)
relative on ``tree_jc/release.yaml`` at 2,000 sites.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.likelihood.brute_force import brute_force_log_likelihood
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.objective import (
    BranchLengthObjective,
    SubstitutionModelObjective,
)
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, FOUR_TAXA, SMALL_SITES, load_fixture

_SITES = 2000


def _alignment(
    name: str, n_sites: int = _SITES
) -> tuple[Node, int, np.ndarray, dict[str, np.ndarray]]:
    params = load_fixture(name)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=n_sites,
    )
    return params.tau, params.k, params.pi, dict(dataset.alignment)


def _points(objective: BranchLengthObjective, tau: Node) -> list[torch.Tensor]:
    """The generating lengths, and every length at 0.1 -- the #443 pair."""
    truth = objective.theta_from_truth(tau)
    return [truth, torch.full_like(truth, float(np.log(0.1)))]


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("name", [SMALL_SITES, FOUR_TAXA, EIGHT_TAXA])
def test_branch_length_value_and_gradient_match_torch(name: str) -> None:
    """JC: the JAX value and gradient are the taped ones, at two points per fixture."""
    tau, k, pi, alignment = _alignment(name)
    torch_route = BranchLengthObjective(tau, k, pi, alignment)
    jax_route = BranchLengthObjective(tau, k, pi, alignment, backend=Backend.JAX)
    for theta in _points(torch_route, tau):
        expected = torch_route.value_and_gradient(theta)
        actual = jax_route.value_and_gradient(theta)
        assert_allclose(
            actual[0].item(), expected[0].item(), rtol=CROSS_DEVICE_RTOL_FLOAT64
        )
        assert_allclose(
            actual[1].numpy(), expected[1].numpy(), rtol=CROSS_DEVICE_RTOL_FLOAT64
        )


@pytest.mark.oracle
@pytest.mark.backend
def test_the_analytic_route_and_jax_agree() -> None:
    """Three routes to one gradient: ``gradient="analytic"`` is the third."""
    tau, k, pi, alignment = _alignment(EIGHT_TAXA)
    analytic = BranchLengthObjective(tau, k, pi, alignment, gradient="analytic")
    jax_route = BranchLengthObjective(tau, k, pi, alignment, backend=Backend.JAX)
    theta = analytic.theta_from_truth(tau)
    assert_allclose(
        jax_route.value_and_gradient(theta)[1].numpy(),
        analytic.value_and_gradient(theta)[1].numpy(),
        rtol=CROSS_DEVICE_RTOL_FLOAT64,
    )


@pytest.mark.oracle
def test_the_jax_value_is_the_brute_force_likelihood() -> None:
    """Enumeration over internal states is the exact answer on a small tree."""
    tau, k, pi, alignment = _alignment(SMALL_SITES, n_sites=200)
    objective = BranchLengthObjective(tau, k, pi, alignment, backend=Backend.JAX)
    value, _ = objective.value_and_gradient(objective.theta_from_truth(tau))
    expected = brute_force_log_likelihood(tau, k, pi, alignment)
    assert_allclose(-value.item(), expected, rtol=1e-12)


@pytest.mark.analytic
def test_the_jax_gradient_passes_finite_differences() -> None:
    """``jax.test_util.check_grads`` on the compiled program, reverse mode."""
    from jax.test_util import check_grads
    from snakes_and_ladders.likelihood import pruning_jax

    tau, k, pi, alignment = _alignment(SMALL_SITES, n_sites=200)
    objective = BranchLengthObjective(tau, k, pi, alignment)
    source, halves = objective.branch_map()
    data, weight = pruning_jax.leaves(tau, k, alignment, None)
    program = pruning_jax.branch_length_program(
        pruning_jax.steps(tau), k, len(source), source, halves
    )

    def value(theta: object) -> object:
        return program(theta, np.asarray(pi, float), data, weight)[0]

    theta = objective.theta_from_truth(tau).numpy()
    check_grads(value, (theta,), order=1, modes=("rev",))  # type: ignore[no-untyped-call]


@pytest.mark.oracle
@pytest.mark.backend
@pytest.mark.parametrize("name", [SMALL_SITES, EIGHT_TAXA])
def test_substitution_model_value_and_gradient_match_torch(name: str) -> None:
    """GTR: ``expm`` under JAX against the taped rate matrix, off the start point."""
    tau, k, _, alignment = _alignment(name)
    torch_route = SubstitutionModelObjective(tau, k, alignment)
    jax_route = SubstitutionModelObjective(tau, k, alignment, backend=Backend.JAX)
    rng = np.random.default_rng(1005)
    theta = torch_route.initial() + torch.as_tensor(
        rng.normal(scale=0.3, size=torch_route.n_parameters)
    )
    expected = torch_route.value_and_gradient(theta)
    actual = jax_route.value_and_gradient(theta)
    assert_allclose(
        actual[0].item(), expected[0].item(), rtol=CROSS_DEVICE_RTOL_FLOAT64
    )
    assert_allclose(actual[1].numpy(), expected[1].numpy(), rtol=1e-10, atol=1e-10)


@pytest.mark.oracle
@pytest.mark.backend
def test_a_fit_reaches_the_same_optimum() -> None:
    """L-BFGS through either route stops at the same lengths and likelihood."""
    tau, k, pi, alignment = _alignment(EIGHT_TAXA)
    results = [
        fit(BranchLengthObjective(tau, k, pi, alignment, backend=backend))
        for backend in (Backend.TORCH, Backend.JAX)
    ]
    assert all(result.converged for result in results)
    assert_allclose(results[1].value, results[0].value, rtol=1e-12)
    assert_allclose(results[1].theta.numpy(), results[0].theta.numpy(), atol=1e-6)


@pytest.mark.smoke
def test_a_backend_other_than_jax_or_torch_is_refused() -> None:
    tau, k, pi, alignment = _alignment(SMALL_SITES, n_sites=10)
    with pytest.raises(ValueError, match="gradient"):
        BranchLengthObjective(tau, k, pi, alignment, backend=Backend.PYTHON)
