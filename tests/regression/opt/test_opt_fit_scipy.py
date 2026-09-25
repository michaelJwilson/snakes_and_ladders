"""`opt.fit` on the test functions against `scipy.optimize.minimize` (issue #376).

The analytic minimizers already referee the fit; an independent L-BFGS from
the same start, objective and autograd gradient must stop at the same point,
to ``1e-4``. Only unambiguous basins are compared: Rastrigin starts in the
central cell. Skips without ``scipy``, which is undeclared (issue #376).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.opt.fit import fit
from sal.opt.objective import Objective
from sal.opt.testfunctions import Himmelblau, Rastrigin, Rosenbrock

from tests._rows import every_value

optimize = pytest.importorskip("scipy.optimize")


def _value_and_gradient(
    objective: Objective,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """The objective as scipy takes it: a value and a gradient from a NumPy vector."""

    def evaluate(point: np.ndarray) -> tuple[float, np.ndarray]:
        theta = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        value = objective(theta)
        (gradient,) = torch.autograd.grad(value, theta)
        return float(value), gradient.detach().numpy()

    return evaluate


def _scipy_minimizer(objective: Objective) -> np.ndarray:
    start = objective.initial().detach().numpy()
    result = optimize.minimize(
        _value_and_gradient(objective),
        start,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 2000},
    )
    # `success` is scipy's verdict on its own line search; at `gtol` of 1e-12
    # it reports ABNORMAL from the exact minimum of Rastrigin, where the
    # gradient it returns is 2e-9 and the value is 0.0. A vanishing gradient is
    # the criterion independent of either optimizer's stopping rule.
    assert result.success or np.linalg.norm(result.jac) < 1e-6, result.message
    return np.asarray(result.x, dtype=np.float64)


SURFACES: tuple[Objective, ...] = (
    Rosenbrock(dimension=2),
    Rosenbrock(dimension=3),
    Rosenbrock(dimension=5),
    Rastrigin(dimension=2, start=0.3),
    Himmelblau(start=(2.5, 2.5)),
    Himmelblau(start=(-2.0, 2.0)),
    Himmelblau(start=(-3.0, -2.5)),
    Himmelblau(start=(3.2, -1.4)),
)


@pytest.mark.oracle
@pytest.mark.parametrize("objective", SURFACES, ids=lambda o: type(o).__name__)
def test_the_fit_reaches_the_minimizer_scipy_reaches_from_the_same_start(
    objective: Objective,
) -> None:
    ours = fit(objective)
    assert ours.converged

    theirs = _scipy_minimizer(objective)

    np.testing.assert_allclose(ours.theta.detach().numpy(), theirs, rtol=0.0, atol=1e-4)


@pytest.mark.oracle
def test_both_optimizers_reach_rosenbrock_s_analytic_minimizer() -> None:
    # The referee refereed: agreement between two optimizers is agreement, so
    # the pair is anchored once to the closed form they are both aiming at.
    def check(dimension: int) -> None:
        objective = Rosenbrock(dimension=dimension)
        expected = objective.minimizer().numpy()

        np.testing.assert_allclose(
            fit(objective).theta.detach().numpy(), expected, rtol=0.0, atol=1e-5
        )
        np.testing.assert_allclose(_scipy_minimizer(objective), expected, atol=1e-5)

    every_value([2, 3, 5], check)


@pytest.mark.smoke
def test_the_two_land_in_the_same_himmelblau_basin_from_each_start() -> None:
    # Himmelblau has four minima of equal value, so agreeing on the *value*
    # says nothing; this is the assertion that they agree on which one.
    reached = set()
    for start in ((2.5, 2.5), (-2.0, 2.0), (-3.0, -2.5), (3.2, -1.4)):
        objective = Himmelblau(start=start)
        ours, _ = Himmelblau.nearest_minimum(fit(objective).theta.detach())
        theirs, _ = Himmelblau.nearest_minimum(
            torch.tensor(_scipy_minimizer(objective))
        )
        assert ours == theirs, start
        reached.add(ours)
    # All four, or the agreement above is about one basin four times.
    assert reached == {0, 1, 2, 3}
