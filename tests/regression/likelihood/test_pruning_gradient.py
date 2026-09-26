"""A search fitting with any route to the pruning gradient returns the same answer (issue #449).

Each route's gradient is pinned in its own module; here, a search fitting with
any of them returns the same topology from
:func:`sal.search.infer.infer`. The swap is a ``monkeypatch``
of ``pruning_torch.log_likelihood``, not a permanent seam (issue #425).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.likelihood import pruning_analytic
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.pruning import torch as pruning_torch
from sal.search.infer import infer

from tests._fixtures import SMALL_SITES, simulated_alignment

_ROUTES: dict[str, Callable[..., torch.Tensor]] = {
    "analytic": pruning_analytic.log_likelihood,
}

#: Sites enough to separate topologies, few enough that three searches are
#: seconds. The same size ``tests/regression/search/test_search_infer.py`` runs.
_SITES = 2000


@pytest.mark.oracle
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_infer_returns_the_same_topology_and_trace(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The search's answer does not depend on which route produced the gradient."""
    params, alignment = simulated_alignment(SMALL_SITES, _SITES)

    expected = infer(alignment, params.n_states, rng=np.random.default_rng(449))
    monkeypatch.setattr(pruning_torch, "log_likelihood", _ROUTES[route])
    actual = infer(alignment, params.n_states, rng=np.random.default_rng(449))

    assert actual.topology == expected.topology
    assert actual.evaluations == expected.evaluations
    assert actual.fits == expected.fits
    assert actual.converged == expected.converged
    assert_allclose(actual.trace, expected.trace, rtol=CROSS_DEVICE_RTOL_FLOAT64)
    assert_allclose(
        actual.log_likelihood, expected.log_likelihood, rtol=CROSS_DEVICE_RTOL_FLOAT64
    )
