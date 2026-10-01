"""The count mixture's value and gradient, compiled against autograd (issue #1136).

At the stress mixture's draw --- 3,000 joint count pairs, ten components,
49 parameters --- the gradient a Hamiltonian start takes at every leapfrog
step: ``oxisal.count_mixture_value_and_gradient`` against autograd through
``EmissionMixtureObjective.__call__``.
``tests/regression/opt/test_emission_mixture_robustness.py`` pins the two.
"""

from __future__ import annotations

import pytest
import torch
from sal.opt.emission_mixture import CountPairSeeding, EmissionMixtureObjective
from sal.opt.objective import autograd_value_and_gradient
from sal.search.mixture_starts import emission_objective, instance_from
from sal.sim.emission_mixture import simulate_emission_mixture
from sal.sim.fixtures import fixture


@pytest.fixture(scope="module")
def objective() -> EmissionMixtureObjective:
    """The objective the sampling starts read at the stress mixture."""
    params = fixture("emission_mixture", "stress").params
    instance = instance_from(
        simulate_emission_mixture(params),
        CountPairSeeding(
            float(params.components.total.dispersion.mean()),
            float(params.components.concentration.mean()),
            joint=True,
        ),
    )
    return emission_objective(instance)


@pytest.mark.release
@pytest.mark.benchmark
def test_count_mixture_gradient_autograd(
    benchmark: object, objective: EmissionMixtureObjective
) -> None:
    """The reference: autograd through the torch densities."""
    theta = objective.initial()
    benchmark(autograd_value_and_gradient, objective, theta)  # type: ignore[operator]


@pytest.mark.release
@pytest.mark.benchmark
def test_count_mixture_gradient_compiled(
    benchmark: object, objective: EmissionMixtureObjective
) -> None:
    """One crossing into Rust, the parameter map chained in torch."""
    theta = objective.initial()
    assert torch.isfinite(objective.value_and_gradient(theta)[1]).all()
    benchmark(objective.value_and_gradient, theta)  # type: ignore[operator]
