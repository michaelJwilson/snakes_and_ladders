"""Gradient-based fitting, and the intervals that make a fit falsifiable.

Model-agnostic, like everything else in this package: ``fit`` takes an
:class:`~snakes_and_ladders.opt.objective.Objective` and knows nothing about what it
optimizes. ``opt/CLAUDE.md`` makes recovery the acceptance test, and recovery
needs an interval, so the observed-information machinery lives here rather
than in a test -- a standard error computed once in a test is not available
to the next instance.

**Convergence is judged relatively.** The objective is a summed
log-likelihood, so both it and its gradient scale with the data; an absolute
gradient threshold fixed at one fixture size does not transfer to another
(``DEV.md``, issue #111). The criterion is the gradient's infinity norm
against the objective's own magnitude.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from snakes_and_ladders.opt.initialize import Initializer
from snakes_and_ladders.opt.objective import Objective

# Two-sided normal quantile for a 95% interval. Written out rather than
# imported from scipy: one constant does not justify a dependency.
_Z_95 = 1.959963984540054

# L-BFGS steps per outer iteration. More than one so the curvature history
# accumulates; few enough that the convergence test below is checked often.
_INNER_ITERATIONS = 20

# Conditioning floor for the observed information; see parameter_covariance.
_RCOND = 1e-6


@dataclass(frozen=True)
class FitResult:
    """Outcome of a fit.

    Parameters
    ----------
    theta : torch.Tensor
        Fitted unconstrained parameters.
    value : float
        Objective at ``theta``.
    gradient_norm : float
        Infinity norm of the gradient at ``theta``, relative to ``value`` --
        the quantity the convergence test is stated against.
    iterations : int
        Optimizer steps taken.
    converged : bool
        Whether the relative gradient norm fell below the tolerance. A fit
        that ran out of iterations is returned rather than raised, so a
        caller can inspect it; every test here asserts this is ``True``.
    """

    theta: torch.Tensor
    value: float
    gradient_norm: float
    iterations: int
    converged: bool


def fit(
    objective: Objective,
    theta0: torch.Tensor | None = None,
    max_iterations: int = 500,
    gradient_tolerance: float = 1e-8,
) -> FitResult:
    """Minimize ``objective`` by L-BFGS with a strong-Wolfe line search.

    L-BFGS rather than a first-order method because the acceptance test is
    parameter recovery, not a decreasing loss: an interval built from the
    observed information is only meaningful at a point where the gradient is
    actually zero, and reaching that to nine digits with Adam takes orders of
    magnitude more steps.

    Parameters
    ----------
    objective : Objective
        The objective to minimize.
    theta0 : torch.Tensor | None
        Starting point; ``objective.initial()`` when omitted.
    max_iterations : int
        Maximum optimizer steps.
    gradient_tolerance : float
        Convergence threshold on ``max|grad| / max(1, |value|)``.

    Returns
    -------
    FitResult
        The fitted parameters and the state of the convergence test.
    """
    theta = (
        (objective.initial() if theta0 is None else theta0)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    # tolerance_grad and tolerance_change are switched off deliberately.
    # Their defaults (1e-7, 1e-9) are absolute, so L-BFGS would stop its
    # inner loop on a summed log-likelihood long before the gradient is
    # small relative to the objective -- the exact failure #111 describes,
    # inside the optimizer rather than in a test. Convergence is decided
    # below, once, by the relative criterion.
    optimizer = torch.optim.LBFGS(
        [theta],
        max_iter=_INNER_ITERATIONS,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=0.0,
        tolerance_change=0.0,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = objective(theta)
        value.backward()  # type: ignore[no-untyped-call]
        return value

    iterations = 0
    converged = False
    while iterations < max_iterations:
        iterations += 1
        optimizer.step(closure)  # type: ignore[no-untyped-call]
        if _relative_gradient_norm(objective, theta) <= gradient_tolerance:
            converged = True
            break

    return FitResult(
        theta=theta.detach(),
        value=float(objective(theta.detach())),
        gradient_norm=_relative_gradient_norm(objective, theta),
        iterations=iterations,
        converged=converged,
    )


def _relative_gradient_norm(objective: Objective, theta: torch.Tensor) -> float:
    point = theta.detach().clone().requires_grad_(True)
    value = objective(point)
    gradient = torch.autograd.grad(value, point)[0]
    return float(gradient.abs().max()) / max(1.0, abs(float(value.detach())))


def observed_information(objective: Objective, theta: torch.Tensor) -> torch.Tensor:
    """Hessian of the objective at ``theta``.

    The objective is a *negative* log-likelihood by this package's
    convention, so its Hessian is the observed Fisher information directly,
    with no sign flip.

    Parameters
    ----------
    objective : Objective
        The fitted objective.
    theta : torch.Tensor
        Point to evaluate at, normally a fitted ``FitResult.theta``.

    Returns
    -------
    torch.Tensor
        Square matrix of shape ``(len(theta), len(theta))``.
    """
    callable_objective: Callable[[torch.Tensor], torch.Tensor] = objective
    information: torch.Tensor = torch.autograd.functional.hessian(  # type: ignore[no-untyped-call]
        callable_objective, theta.detach()
    )
    return information


def parameter_covariance(
    objective: Objective, theta: torch.Tensor, rcond: float = _RCOND
) -> torch.Tensor:
    """Inverse observed information: the asymptotic covariance of ``theta``.

    Conditioning is checked rather than left to ``torch.linalg.inv``. A model
    with an exactly flat direction produces an information matrix that is
    only *numerically* singular -- rounding leaves its smallest eigenvalue at
    1e-4 rather than 0 -- so the inversion succeeds and returns an
    astronomically large covariance instead of failing. Silently returning a
    meaningless interval is worse than raising.

    Parameters
    ----------
    objective : Objective
        The fitted objective.
    theta : torch.Tensor
        Fitted parameters. This must be an optimum: the observed information
        is a statement about curvature at a maximum, and away from one the
        Hessian need not be positive definite, so a non-optimal point is
        rejected by the same check that catches an unidentifiable model.
    rcond : float
        Smallest acceptable ratio of the smallest to the largest eigenvalue
        of the observed information. The default separates the two cases by
        four orders of magnitude on both sides: a phylogenetic tree whose two
        root branches are confounded realizes 6.7e-08, while the same tree
        with that pair merged realizes 6.1e-02 and an unrooted fixture
        5.6e-02.

    Returns
    -------
    torch.Tensor
        Covariance matrix in the unconstrained coordinates.

    Raises
    ------
    ValueError
        If the information is singular, indefinite, or worse conditioned than
        ``rcond``. All three mean the model as parameterized is not
        identifiable from this data -- a gauge that was not fixed, a
        confounded pair of parameters, or a sample too small to pin them.
        Reported as such rather than as a linear-algebra error, because that
        is the actual fault.
    """
    information = observed_information(objective, theta)
    # Symmetric by construction, so eigvalsh is exact where a general
    # eigensolver would introduce a spurious imaginary part.
    eigenvalues = torch.linalg.eigvalsh(information)
    smallest = float(eigenvalues.min())
    largest = float(eigenvalues.max())
    if largest <= 0.0 or smallest <= rcond * largest:
        msg = (
            f"observed information is not positive definite to within "
            f"rcond={rcond:.0e} (eigenvalue ratio {smallest / largest:.2e}): "
            f"the model is not identifiable from this data"
        )
        raise ValueError(msg)
    covariance: torch.Tensor = torch.linalg.inv(information)
    return covariance


def constrained_standard_errors(
    objective: Objective, theta: torch.Tensor
) -> Mapping[str, torch.Tensor]:
    """Delta-method standard errors of the *constrained* parameters.

    Recovery is stated against the parameters a person named, not against the
    unconstrained vector, so the covariance has to be pushed through the
    constraint map: ``Var(g(theta)) ~ J Sigma J'`` with ``J`` the Jacobian of
    ``g``.

    Parameters
    ----------
    objective : Objective
        The fitted objective.
    theta : torch.Tensor
        Fitted parameters.

    Returns
    -------
    Mapping[str, torch.Tensor]
        One tensor per constrained parameter, shaped like that parameter.
    """
    covariance = parameter_covariance(objective, theta)
    point = theta.detach()
    errors: dict[str, torch.Tensor] = {}
    for name in objective.constrain(point):
        jacobian = torch.autograd.functional.jacobian(  # type: ignore[no-untyped-call]
            lambda t, key=name: objective.constrain(t)[key],
            point,
        )
        flat = jacobian.reshape(-1, point.numel())
        variance = ((flat @ covariance) * flat).sum(dim=1)
        errors[name] = variance.clamp_min(0.0).sqrt().reshape(jacobian.shape[:-1])
    return errors


def covers(
    estimate: torch.Tensor, standard_error: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    """Whether a 95% Wald interval around ``estimate`` contains ``truth``.

    Parameters
    ----------
    estimate : torch.Tensor
        Fitted constrained parameter.
    standard_error : torch.Tensor
        Its standard error, same shape.
    truth : torch.Tensor
        The generating value, same shape.

    Returns
    -------
    torch.Tensor
        Boolean tensor, elementwise.
    """
    half_width = _Z_95 * standard_error
    return (truth >= estimate - half_width) & (truth <= estimate + half_width)


@dataclass(frozen=True)
class MultiStartResult:
    """Every fit an initializer's starts produced, best first.

    Parameters
    ----------
    best : FitResult
        The lowest-valued fit. What a single-start caller would want.
    all_fits : tuple[FitResult, ...]
        Every fit, ordered by value ascending. Reported rather than discarded
        because discarding them hides that the surface was multimodal, which is
        exactly what `test_a_converged_fit_on_rastrigin_is_not_a_global_minimum`
        exists to say out loud: a fit that reports one answer for a surface with
        four basins is the failure this is about.
    spread : float
        ``max(value) - min(value)`` over the fits. Zero means every start
        agreed, which is the evidence that one start would have sufficed;
        anything else is the amount a single fit could have been wrong by.
    """

    best: FitResult
    all_fits: tuple[FitResult, ...]
    spread: float


def fit_from(
    objective: Objective,
    initializer: Initializer,
    max_iterations: int = 500,
    gradient_tolerance: float = 1e-8,
) -> MultiStartResult:
    """Fit from every start an initializer offers, and report all of them.

    A single-start initializer makes this exactly :func:`fit`, so the two are
    not different code paths: `FromObjective` reproduces today's behaviour and
    `spread` is then 0 by construction.

    Parameters
    ----------
    objective : Objective
        What to minimize.
    initializer : Initializer
        Where to start. See `snakes_and_ladders.opt.initialize`.
    max_iterations : int
        Passed to each fit.
    gradient_tolerance : float
        Passed to each fit.

    Returns
    -------
    MultiStartResult
        The best fit, every fit, and the spread across them.

    Raises
    ------
    ValueError
        If the initializer offers no starts. A caller asking for zero fits has
        made a mistake that would otherwise surface as an empty ``min``.
    """
    starts = initializer.starts(objective)
    if not starts:
        msg = f"{type(initializer).__name__} offered no starting points"
        raise ValueError(msg)

    results = [
        fit(objective, theta0, max_iterations, gradient_tolerance) for theta0 in starts
    ]
    ordered = tuple(sorted(results, key=lambda result: result.value))
    spread = float(ordered[-1].value - ordered[0].value)
    return MultiStartResult(best=ordered[0], all_fits=ordered, spread=spread)
