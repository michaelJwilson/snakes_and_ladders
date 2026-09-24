"""The NumPy energy seam, ``opt.objective.energy_of`` (issue #1011).

Referees:

- a declared :meth:`energy` is ``__call__`` on the same point: bitwise where
  a sum has at most three terms and no matrix product is taken, and within
  ``rtol = 1e-15`` otherwise, where NumPy's pairwise sum and torch's vectorized one order the
  terms differently --- measured 6.1e-16 at most over 500 points per size,
  ``d`` in 2 to 1000, on Rosenbrock and the diagonal and dense Gaussians;
- an objective declaring none is ``float(__call__)``, bitwise, which is what
  a value-only consumer computed before the seam;
- the warm-up's change of coordinates keeps a declared energy: ``_Scaled``'s
  is the inner one at ``x * scale``, bitwise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.objective import DeclaredEnergy, Objective, energy_of
from snakes_and_ladders.opt.testfunctions import Rastrigin, Rosenbrock
from snakes_and_ladders.sample.chain import _Scaled
from snakes_and_ladders.validation.gaussian import (
    GaussianTarget,
    dense_precision,
    diagonal_precision,
)

from tests._rows import every_value

TARGETS: dict[str, Callable[[int], Objective]] = {
    "rosenbrock": lambda d: Rosenbrock(d),
    "gaussian-diagonal": lambda d: GaussianTarget(diagonal_precision(d)),
    "gaussian-dense": lambda d: GaussianTarget(
        dense_precision(d, np.random.default_rng(963))
    ),
}


def _points(dimension: int) -> np.ndarray:
    return 1.5 * np.random.default_rng(1011).normal(size=(200, dimension))


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(TARGETS))
def test_a_declared_energy_is_the_objectives_value(name: str) -> None:
    def check(dimension: int) -> None:
        objective = TARGETS[name](dimension)
        assert isinstance(objective, DeclaredEnergy)
        for x in _points(dimension):
            expected = float(objective(torch.as_tensor(x)))
            got = energy_of(objective, x)
            assert isinstance(got, float)
            # Three terms or fewer are summed left to right by both; the dense
            # Gaussian's matrix products are ordered by each library's BLAS.
            bitwise = name != "gaussian-dense" and dimension <= 3
            if bitwise:
                assert got == expected
            else:
                np.testing.assert_allclose(got, expected, rtol=1e-15, atol=0)

    every_value([2, 3, 10, 100, 1000], check)


@pytest.mark.oracle
def test_an_undeclared_energy_is_the_objectives_value() -> None:
    objective = Rastrigin(5)
    assert not isinstance(objective, DeclaredEnergy)
    for x in _points(5):
        with torch.no_grad():
            expected = float(objective(torch.as_tensor(x)))
        assert energy_of(objective, x) == expected


@pytest.mark.oracle
@pytest.mark.parametrize("inner", [Rosenbrock(10), Rastrigin(10)], ids=str)
def test_the_scaled_energy_is_the_inner_one_at_the_scaled_point(
    inner: Objective,
) -> None:
    scale = torch.as_tensor(np.linspace(0.5, 2.0, 10))
    scaled = _Scaled(inner, scale)
    for x in _points(10):
        assert energy_of(scaled, x) == energy_of(inner, x * scale.numpy())
