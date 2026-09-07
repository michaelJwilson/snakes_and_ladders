"""Shared derivative check for any ``snakes_and_ladders.opt.objective.Objective``.

`DEV.md`'s Test Layout puts a fixture shared across modules in a top-level
underscore-prefixed module, imported rather than collected. `opt/CLAUDE.md`
makes finite differences the derivative test that matters, and every instance
owes the same check, so the check itself lives here rather than being
transcribed per instance -- and, in that form, it is what a new instance has
to pass to be one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from snakes_and_ladders.opt.objective import Objective


def central_difference_gradient(
    objective: Objective, theta: torch.Tensor, step: float
) -> torch.Tensor:
    """Numerical gradient of ``objective`` at ``theta`` by central differences.

    Parameters
    ----------
    objective : Objective
        The objective to differentiate.
    theta : torch.Tensor
        Point to differentiate at, 1-D.
    step : float
        Half-width of the difference. Central differences carry a truncation
        error of order ``step**2`` and a rounding error of order
        ``eps / step``, so in ``float64`` the optimum sits near ``1e-5``;
        this is a parameter rather than a constant because the right value
        depends on the curvature of the objective.

    Returns
    -------
    torch.Tensor
        Numerical gradient, same shape as ``theta``.
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
    """Autograd gradient of ``objective`` at ``theta``.

    Parameters
    ----------
    objective : Objective
        The objective to differentiate.
    theta : torch.Tensor
        Point to differentiate at, 1-D.

    Returns
    -------
    torch.Tensor
        Gradient from reverse-mode autodiff, same shape as ``theta``.
    """
    point = theta.detach().clone().requires_grad_(True)
    gradient: torch.Tensor = torch.autograd.grad(objective(point), point)[0]
    return gradient


def assert_gradient_matches_finite_differences(
    objective: Objective, theta: torch.Tensor, step: float, rtol: float
) -> float:
    """Assert autograd and central differences agree, and return the realized ratio.

    The comparison is relative to the **norm** of the gradient rather than
    entrywise. Two reasons, both encountered rather than anticipated: at a
    symmetric starting point (uniform distributions everywhere) many entries
    are exactly zero, so an entrywise relative bound is undefined there; and
    the gradient of a summed log-likelihood scales with the data size, so an
    absolute bound fixed at one fixture size would not transfer to another
    (`DEV.md`, issue #111). Scaling by the gradient's own magnitude is the
    form that survives both.

    Parameters
    ----------
    objective : Objective
        The objective to differentiate.
    theta : torch.Tensor
        Point to differentiate at, 1-D.
    step : float
        Half-width passed to :func:`central_difference_gradient`.
    rtol : float
        Bound on ``max|analytic - numerical| / max|analytic|``.

    Returns
    -------
    float
        The realized ratio, for a test to report in a PR's tolerance table.
    """
    analytic = analytic_gradient(objective, theta)
    numerical = central_difference_gradient(objective, theta, step)
    scale = float(analytic.abs().max())
    assert scale > 0.0, "gradient is identically zero; the check would be vacuous"
    realized = float((analytic - numerical).abs().max()) / scale
    assert realized <= rtol, f"gradient disagreement {realized:.3e} exceeds {rtol:.3e}"
    return realized
