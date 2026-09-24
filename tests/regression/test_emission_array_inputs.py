"""An M step and a draw take arrays, and reproduce the tensor call bitwise (issue #1011).

:meth:`~snakes_and_ladders.emissions.EmissionFamily.reestimate` and
:meth:`~snakes_and_ladders.emissions.EmissionFamily.sample` take no
derivative, so a caller holding NumPy arrays hands them over without building
a tensor. Each family converts at entry and computes as it did on a tensor,
so the referee is the tensor call itself: every parameter the re-estimate
returns, every field of its record, and every draw, equal byte for byte.

The two implementers in :mod:`snakes_and_ladders.sim.count_pairs` are listed
with the seven in :mod:`snakes_and_ladders.emissions`: they were the consumers
the tensor-only signature blocked.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.typing import ArrayLike
from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    CountPairEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
    Reestimate,
)
from snakes_and_ladders.sim.count_pairs import IndependentCountPair, ReflectedEmission

N_SEQUENCES, LENGTH = 3, 200

_TOTAL = NegativeBinomialEmission([5.0, 12.0], [20.0, 60.0])
_SUCCESSES = BetaBinomialEmission([30.0, 30.0], [2.0, 9.0], [6.0, 3.0])

#: Each family, and the covariate it draws and re-estimates under, or `None`.
#: A covariate is one per observation with a trailing singleton, or one per
#: channel for a pair.
CASES: dict[
    str, tuple[EmissionFamily, Callable[[np.random.Generator], np.ndarray] | None]
] = {
    "categorical": (
        CategoricalEmission(np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])),
        None,
    ),
    "gaussian": (
        GaussianEmission(
            np.array([[-1.0, 0.0], [2.0, 3.0]]),
            np.array([[0.5, 1.0], [1.5, 0.3]]),
            1e-8,
        ),
        None,
    ),
    "poisson": (PoissonEmission([3.0, 11.0]), None),
    "binomial": (BinomialEmission([30.0, 30.0], [0.2, 0.7]), None),
    "negative_binomial": (
        _TOTAL,
        lambda rng: rng.uniform(0.25, 4.0, (N_SEQUENCES * LENGTH, 1)),
    ),
    "beta_binomial": (
        _SUCCESSES,
        lambda rng: rng.integers(1, 40, (N_SEQUENCES * LENGTH, 1)).astype(np.float64),
    ),
    "count_pair_joint": (
        CountPairEmission(
            [5.0, 12.0], [20.0, 60.0], [2.0, 9.0], [6.0, 3.0], joint=True
        ),
        None,
    ),
    "independent_count_pair": (
        IndependentCountPair(_TOTAL, _SUCCESSES),
        lambda rng: np.stack(
            [
                rng.uniform(0.25, 4.0, N_SEQUENCES * LENGTH),
                rng.integers(1, 40, N_SEQUENCES * LENGTH).astype(np.float64),
            ],
            axis=-1,
        ),
    ),
    "reflected_pair": (
        ReflectedEmission(IndependentCountPair(_TOTAL, _SUCCESSES)),
        lambda rng: np.stack(
            [
                rng.uniform(0.25, 4.0, N_SEQUENCES * LENGTH),
                rng.integers(1, 40, N_SEQUENCES * LENGTH).astype(np.float64),
            ],
            axis=-1,
        ),
    ),
}


def _tensor(values: np.ndarray | None) -> torch.Tensor | None:
    """The tensor a caller built before arrays were accepted."""
    return None if values is None else torch.as_tensor(values)


def _step(
    family: EmissionFamily,
    observations: ArrayLike,
    posterior: ArrayLike,
    covariate: ArrayLike | None,
) -> Reestimate[EmissionFamily]:
    """One M step, with the covariate passed only where there is one."""
    if covariate is None:
        return family.reestimate(observations, posterior)
    return family.reestimate(observations, posterior, covariate=covariate)


def _bytes(step: Reestimate[EmissionFamily]) -> dict[str, bytes]:
    """Every returned parameter and record field, as the bytes compared."""
    named = {
        name: value.detach().numpy().tobytes()
        for name, value in step.emissions.named_parameters().items()
    }
    named["record"] = np.array(
        [step.converged, step.at_boundary, step.iterations, step.residual]
    ).tobytes()
    return named


@pytest.mark.smoke
@pytest.mark.patch
@pytest.mark.parametrize("name", list(CASES))
def test_arrays_reproduce_the_tensor_call_bitwise(name: str) -> None:
    family, draw_covariate = CASES[name]
    rng = np.random.default_rng(1011)
    covariate = None if draw_covariate is None else draw_covariate(rng)
    states = rng.integers(family.n_states, size=N_SEQUENCES * LENGTH)
    by_tensor = family.sample(states, np.random.default_rng(7), _tensor(covariate))
    by_array = family.sample(states, np.random.default_rng(7), covariate)
    assert by_array.tobytes() == by_tensor.tobytes()

    # Observations as the draw returns them, the posterior as a float64 array,
    # and the tensors a caller built from them in the family's own dtype.
    observations = by_array.reshape(N_SEQUENCES, LENGTH, *by_array.shape[1:])
    posterior = rng.dirichlet(np.ones(family.n_states), size=(N_SEQUENCES, LENGTH))
    fitted_covariate = (
        None
        if covariate is None
        else covariate.reshape(N_SEQUENCES, LENGTH, *covariate.shape[1:])
    )
    from_tensors = _step(
        family,
        torch.as_tensor(observations, dtype=family.observation_dtype),
        torch.as_tensor(posterior),
        _tensor(fitted_covariate),
    )
    from_arrays = _step(family, observations, posterior, fitted_covariate)

    assert _bytes(from_arrays) == _bytes(from_tensors)
