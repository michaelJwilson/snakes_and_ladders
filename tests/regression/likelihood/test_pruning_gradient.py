"""What must hold across all three routes to the pruning gradient (issue #449).

Each route is pinned against ``pruning_torch`` in its own module. What is
pinned here is the claim a per-route test cannot make: the three agree with
each other on the same inputs, and a search that fits with any of them returns
the same answer. A gradient that changes which topology
:func:`snakes_and_ladders.search.infer.infer` returns has changed the answer
and not the cost, whatever it did to the wall clock.

The swap is a ``monkeypatch`` and not a seam. ``likelihood.objective`` calls
``pruning_torch.log_likelihood`` directly and keeps calling it; a permanent
switch for running the slow path would be machinery outliving its measurement
(issue #425).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood import objective as objective_module
from snakes_and_ladders.likelihood import pruning_analytic, pruning_burn, pruning_torch
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.search.infer import infer
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import EIGHT_TAXA, SMALL_SITES, load_fixture

_ROUTES: dict[str, Callable[..., torch.Tensor]] = {
    "burn": pruning_burn.log_likelihood,
    "analytic": pruning_analytic.log_likelihood,
}

#: Sites enough to separate topologies, few enough that three searches are
#: seconds. The same size ``tests/regression/search/test_search_infer.py`` runs.
_SITES = 2000


@pytest.mark.oracle
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_every_route_agrees_with_the_taped_gradient(route: str) -> None:
    """One fixture, three gradients, the float64 agreement tolerance."""
    params = load_fixture(EIGHT_TAXA)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=_SITES,
    )
    lengths = pruning_torch.branch_lengths_from_tree(params.tau)

    gradients = []
    for evaluate in (pruning_torch.log_likelihood, _ROUTES[route]):
        at = lengths.clone().requires_grad_(True)
        evaluate(params.tau, params.k, params.pi, dataset.alignment, at).backward()
        assert at.grad is not None
        gradients.append(at.grad.numpy().copy())
    assert_allclose(gradients[1], gradients[0], rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.oracle
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_infer_returns_the_same_topology_and_trace(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The search's answer does not depend on which route produced the gradient."""
    params = load_fixture(SMALL_SITES)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=_SITES,
    )
    alignment = dict(dataset.alignment)

    expected = infer(alignment, params.k, rng=np.random.default_rng(449))
    monkeypatch.setattr(objective_module.pruning_torch, "log_likelihood", _ROUTES[route])
    actual = infer(alignment, params.k, rng=np.random.default_rng(449))

    assert actual.topology == expected.topology
    assert actual.evaluations == expected.evaluations
    assert actual.fits == expected.fits
    assert actual.converged == expected.converged
    assert_allclose(actual.trace, expected.trace, rtol=CROSS_DEVICE_RTOL_FLOAT64)
    assert_allclose(
        actual.log_likelihood, expected.log_likelihood, rtol=CROSS_DEVICE_RTOL_FLOAT64
    )
