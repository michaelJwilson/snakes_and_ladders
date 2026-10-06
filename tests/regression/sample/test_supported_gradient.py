"""Each kernel an objective's ``supported_gradient`` names, against autograd through the objective (issue #1220).

A compiled chain runs ``oxisal.SupportedEnergy`` on ``(kernel, data)``, and so
does the objective's own gradient; autograd through ``__call__`` is the
independent reference. At five seeded points per objective the value is
within 1e-10 of autograd's, relative, and the gradient within 1e-10 of its
largest coordinate. Measured: 3.7e-15 in the value and 9.4e-14 in the
gradient at most (the negative binomial mixture), over the seven objectives.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal import oxisal
from sal.backend import Backend
from sal.emissions import CountPairEmission, GaussianEmission, NegativeBinomialEmission
from sal.opt.emission_mixture import EmissionMixtureObjective
from sal.opt.hmm import EmissionHmmObjective, family_start
from sal.opt.mixture import GaussianMixtureObjective
from sal.opt.objective import Objective, autograd_value_and_gradient
from sal.opt.testfunctions import Rosenbrock
from sal.sample import declared
from sal.validation.gaussian import GaussianTarget, dense_precision, diagonal_precision

SEED = 1220
TOLERANCE = 1e-10


def _hmm() -> Objective:
    rng = np.random.default_rng(SEED)
    states = np.zeros((6, 40), dtype=int)
    for t in range(1, 40):
        stay = rng.random(6) < 0.9
        states[:, t] = np.where(stay, states[:, t - 1], 1 - states[:, t - 1])
    observations = np.array([-1.0, 1.0])[states] + 0.6 * rng.normal(size=states.shape)
    return EmissionHmmObjective(
        observations,
        family_start(GaussianEmission, observations, 2),
        backend=Backend.TORCH,
    )


def _mixture() -> Objective:
    rng = np.random.default_rng(SEED)
    labels = rng.choice(3, size=1_500, p=[0.3, 0.3, 0.4])
    return GaussianMixtureObjective(
        np.array([-4.0, 0.0, 5.0])[labels] + rng.normal(size=1_500), 3
    )


def _counts(family: NegativeBinomialEmission | CountPairEmission) -> Objective:
    rng = np.random.default_rng(SEED)
    labels = rng.integers(0, family.n_states, 2_000)
    observations = np.asarray(family.sample(labels, rng), dtype=np.float64)
    return EmissionMixtureObjective(observations, family)


OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "gaussian-diagonal": lambda: GaussianTarget(diagonal_precision(10)),
    "gaussian-dense": lambda: GaussianTarget(
        dense_precision(10, np.random.default_rng(SEED))
    ),
    "rosenbrock": lambda: Rosenbrock(dimension=5, b=10.0),
    "gaussian-mixture": _mixture,
    "gaussian-hmm": _hmm,
    "count-mixture-negative-binomial": lambda: _counts(
        NegativeBinomialEmission([3.0, 8.0, 20.0], [5.0, 20.0, 60.0])
    ),
    "count-mixture-pair": lambda: _counts(
        CountPairEmission(
            [6.0, 12.0], [20.0, 60.0], [2.0, 9.0], [8.0, 3.0], joint=True
        ).rate_concentration()
    ),
}


@pytest.mark.oracle
@pytest.mark.parametrize("name", list(OBJECTIVES))
def test_each_supported_kernel_is_autograd_through_the_objective(name: str) -> None:
    objective = OBJECTIVES[name]()
    supported = declared.declared_energy(objective)
    assert supported is not None
    n = int(objective.initial().shape[0])
    kernel = oxisal.SupportedEnergy(*supported, n)
    rng = np.random.default_rng([SEED, len(name)])
    for _ in range(5):
        theta = objective.initial() + 0.1 * torch.as_tensor(rng.normal(size=n))
        value, gradient = kernel.value_and_gradient(theta.numpy())
        want_value, want_gradient = autograd_value_and_gradient(objective, theta)
        reference = want_gradient.numpy()
        assert abs(value - float(want_value)) <= TOLERANCE * abs(float(want_value))
        assert np.abs(gradient - reference).max() <= TOLERANCE * np.abs(reference).max()


@pytest.mark.smoke
def test_every_named_kernel_builds_and_an_unknown_one_is_refused() -> None:
    # The names `sample.declared` lists are the ones `src/energy.rs` builds.
    named = {
        declared.GAUSSIAN,
        declared.ROSENBROCK,
        declared.GAUSSIAN_MIXTURE,
        declared.GAUSSIAN_HMM,
        declared.COUNT_MIXTURE,
    }
    seen = set()
    for make in OBJECTIVES.values():
        supported = declared.declared_energy(make())
        assert supported is not None
        seen.add(supported[0])
    assert seen == named
    with pytest.raises(ValueError, match="no supported gradient kernel"):
        oxisal.SupportedEnergy("quadratic", {}, 2)
    with pytest.raises(ValueError, match="neither d"):
        oxisal.SupportedEnergy(declared.GAUSSIAN, {"precision": np.ones(3)}, 2)
