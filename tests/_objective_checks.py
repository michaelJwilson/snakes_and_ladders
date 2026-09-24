"""Shared derivative check for any ``snakes_and_ladders.opt.objective.Objective``.

`DEV.md`'s Test Layout puts a fixture shared across modules in a top-level
underscore-prefixed module, imported rather than collected. `opt/CLAUDE.md`
makes finite differences the derivative test that matters, and every instance
owes the same check, so the check itself lives here rather than being
transcribed per instance -- and, in that form, it is what a new instance has
to pass to be one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from snakes_and_ladders.opt.objective import Objective


class Counted:
    """An objective that records how often it was evaluated.

    Every call counts, gradient or not: the budget the objective saw (issue #281).
    """

    def __init__(self, inner: Objective) -> None:
        self.inner = inner
        self.calls = 0

    def initial(self) -> torch.Tensor:
        return self.inner.initial()

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.inner.constrain(theta)

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.inner.theta_from(named)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.inner(theta)


class AnalyticGaussian:
    """``-log N(mean, covariance)`` up to a constant: the one analytic target.

    Its Hessian is the precision: a Laplace interval is ``sqrt(diag(covariance))``.
    """

    def __init__(self, mean: list[float], covariance: list[list[float]]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float64)
        self.covariance = torch.tensor(covariance, dtype=torch.float64)
        self._precision = torch.linalg.inv(self.covariance)

    def initial(self) -> torch.Tensor:
        return torch.zeros_like(self.mean)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return named["x"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        deviation = theta - self.mean
        quadratic: torch.Tensor = 0.5 * deviation @ self._precision @ deviation
        return quadratic


def central_difference_gradient(
    objective: Objective, theta: torch.Tensor, step: float
) -> torch.Tensor:
    """Numerical gradient of ``objective`` at ``theta`` by central differences.

    Error ``O(step**2) + O(eps / step)``; in ``float64`` the optimum is near ``1e-5``.
    """
    gradient = torch.zeros_like(theta)
    for i in range(theta.numel()):
        offset = torch.zeros_like(theta)
        offset[i] = step
        gradient[i] = (objective(theta + offset) - objective(theta - offset)) / (
            2.0 * step
        )
    return gradient


def analytic_gradient(objective: Objective, theta: torch.Tensor) -> torch.Tensor:
    """Autograd gradient of ``objective`` at ``theta``."""
    point = theta.detach().clone().requires_grad_(True)
    gradient: torch.Tensor = torch.autograd.grad(objective(point), point)[0]
    return gradient


def assert_gradient_matches_finite_differences(
    objective: Objective, theta: torch.Tensor, step: float, rtol: float
) -> float:
    """Assert autograd and central differences agree; return the realized ratio.

    Relative to the norm: zero entries, and scale with data size (issue #111).
    """
    analytic = analytic_gradient(objective, theta)
    numerical = central_difference_gradient(objective, theta, step)
    scale = float(analytic.abs().max())
    assert scale > 0.0, "gradient is identically zero; the check would be vacuous"
    realized = float((analytic - numerical).abs().max()) / scale
    assert realized <= rtol, f"gradient disagreement {realized:.3e} exceeds {rtol:.3e}"
    return realized
