"""Hamiltonian Monte Carlo over an :class:`~snakes_and_ladders.opt.objective.Objective`.

Every continuous result in this repository is a point estimate plus an
interval built from the observed information at the optimum --- a Gaussian
approximation to the posterior, evaluated at one point. This samples the
posterior instead, so an interval becomes a quantile rather than a curvature
estimate, and the two can be compared: where the Laplace approximation is good
they agree, and where it is not the disagreement is the finding.

The objective is read as an unnormalized negative log density. That is a
choice, not an identity, and it is the caller's to justify: for a
log-likelihood plus a proper log prior it is the posterior; for a bare
log-likelihood it is a posterior under an improper flat prior, which may not
be normalizable at all. Nothing here can check that, so nothing here pretends
to --- :func:`sample` reports the chain and the diagnostics, and what the
chain is a sample *of* is stated by whoever built the objective.

`snakes_and_ladders.opt` may import no application module and this needs none: an
`Objective` supplies an unconstrained vector and a differentiable scalar,
which is exactly the interface a gradient-based sampler wants.

See Neal (2011), "MCMC using Hamiltonian dynamics"; Nocedal & Wright for the
leapfrog integrator's symplectic structure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from snakes_and_ladders.opt.objective import Objective

DEFAULT_STEPS = 20


@dataclass(frozen=True)
class WithGaussianPrior:
    """An objective plus an isotropic Gaussian log prior, so it has a posterior.

    A bare negative log-likelihood read as a log density is a posterior under
    an improper flat prior, which for most models is not normalizable --- so
    a chain against it is sampling from nothing well defined, and no
    diagnostic in :func:`sample` can tell. Adding a proper prior fixes that,
    and stating it here rather than inside the sampler keeps it the caller's
    declaration: the sampler still just minimizes what it is handed.

    Isotropic and centred on zero *in unconstrained coordinates*, which is
    weakly informative rather than uninformative --- on a log-simplex
    coordinate it pulls toward the uniform distribution, and on a coupling
    toward zero. That is a modelling choice, and any interval reported from a
    chain against it inherits it.

    Parameters
    ----------
    objective : Objective
        The negative log-likelihood being given a prior.
    scale : float
        Prior standard deviation on every unconstrained coordinate.
    """

    objective: Objective
    scale: float

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            msg = f"prior scale must be positive, got {self.scale}"
            raise ValueError(msg)

    def initial(self) -> torch.Tensor:
        return self.objective.initial()

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.objective.constrain(theta)

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.objective.theta_from(named)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        penalty = (theta * theta).sum() / (2.0 * self.scale**2)
        return self.objective(theta) + penalty


@dataclass(frozen=True)
class HmcChain:
    """A chain, and what it cost to get it.

    Parameters
    ----------
    theta : torch.Tensor
        Draws in unconstrained coordinates, shape ``(n_samples, dimension)``.
    acceptance_rate : float
        Fraction of proposals accepted. Hamiltonian dynamics conserves energy
        exactly, so a correct implementation with a small step size accepts
        nearly everything; a rate near zero means the integrator is diverging
        rather than that the target is hard.
    energy_error : torch.Tensor
        ``|H(proposal) - H(current)|`` per proposal. The diagnostic that
        distinguishes a step size too large from a bug: the first grows
        smoothly with the step size, the second does not.
    """

    theta: torch.Tensor
    acceptance_rate: float
    energy_error: torch.Tensor


def leapfrog(
    objective: Objective,
    theta: torch.Tensor,
    momentum: torch.Tensor,
    step_size: float,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate Hamiltonian dynamics: half-kick, drift, half-kick.

    Separated from :func:`sample` because its two defining properties ---
    reversibility and second-order accuracy --- are exact statements testable
    without any sampling, and they are where an error actually localizes. A
    distributional test says the chain is wrong; these say which half.

    Parameters
    ----------
    objective : Objective
        Read as a negative log density, so its gradient is the force.
    theta, momentum : torch.Tensor
        Position and momentum, both 1-D of the same length.
    step_size : float
        Integrator step. Energy error grows as its square.
    n_steps : int
        Steps per trajectory.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Position and momentum after ``n_steps``.
    """
    position = theta.detach().clone()
    velocity = momentum.detach().clone()

    velocity = velocity - 0.5 * step_size * _gradient(objective, position)
    for step in range(n_steps):
        position = position + step_size * velocity
        if step < n_steps - 1:
            velocity = velocity - step_size * _gradient(objective, position)
    velocity = velocity - 0.5 * step_size * _gradient(objective, position)
    return position, velocity


def hamiltonian(
    objective: Objective, theta: torch.Tensor, momentum: torch.Tensor
) -> float:
    """``U(theta) + K(momentum)``, the quantity the integrator conserves."""
    potential = float(objective(theta.detach()))
    kinetic = 0.5 * float((momentum * momentum).sum())
    return potential + kinetic


def sample(
    objective: Objective,
    seed: int,
    n_samples: int,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    theta0: torch.Tensor | None = None,
    burn_in: int = 0,
) -> HmcChain:
    """Draw ``n_samples`` from the density ``exp(-objective)``.

    Parameters
    ----------
    objective : Objective
        Read as an unnormalized negative log density.
    seed : int
        Seed for ``torch.Generator``, so a chain is reproducible from it.
    n_samples : int
        Draws recorded after burn-in.
    step_size : float
        Leapfrog step. Required rather than defaulted: it is the one
        parameter whose right value depends on the target's scale, and a
        default would be wrong silently. Adaptation is deliberately absent
        --- see the module note below.
    n_steps : int
        Leapfrog steps per proposal.
    theta0 : torch.Tensor | None
        Starting point; ``objective.initial()`` when omitted.
    burn_in : int
        Draws discarded before recording.

    Returns
    -------
    HmcChain
        The draws, the acceptance rate, and the per-proposal energy error.

    Raises
    ------
    ValueError
        If ``step_size`` is not positive or ``n_steps`` is below 1. A
        zero-length trajectory proposes the current point every time, which
        accepts at rate 1 and samples nothing --- a chain that looks healthy
        by every diagnostic and has not moved.
    """
    if step_size <= 0.0:
        msg = f"step_size must be positive, got {step_size}"
        raise ValueError(msg)
    if n_steps < 1:
        msg = (
            f"n_steps must be at least 1, got {n_steps}: a zero-length "
            "trajectory proposes the current point and accepts at rate 1, "
            "which looks healthy and samples nothing"
        )
        raise ValueError(msg)

    generator = torch.Generator().manual_seed(seed)
    position = (
        objective.initial().detach().clone()
        if theta0 is None
        else theta0.detach().clone()
    ).to(torch.float64)

    draws = torch.empty((n_samples, position.shape[0]), dtype=torch.float64)
    errors = torch.empty(n_samples + burn_in, dtype=torch.float64)
    accepted = 0

    for index in range(n_samples + burn_in):
        momentum = torch.randn(position.shape, generator=generator, dtype=torch.float64)
        current = hamiltonian(objective, position, momentum)

        proposal, proposed_momentum = leapfrog(
            objective, position, momentum, step_size, n_steps
        )
        # Negating the momentum makes the proposal symmetric, which is what
        # leaves the acceptance ratio as the energy difference alone. It has
        # no effect on the next iteration, where the momentum is redrawn.
        proposed = hamiltonian(objective, proposal, -proposed_momentum)

        errors[index] = abs(proposed - current)
        uniform = float(torch.rand(1, generator=generator))
        if uniform < float(torch.exp(torch.tensor(current - proposed))):
            position = proposal
            if index >= burn_in:
                accepted += 1
        if index >= burn_in:
            draws[index - burn_in] = position

    return HmcChain(
        theta=draws,
        acceptance_rate=accepted / n_samples if n_samples else 0.0,
        energy_error=errors[burn_in:],
    )


def _gradient(objective: Objective, theta: torch.Tensor) -> torch.Tensor:
    """``dU/dtheta``, by autograd through the objective."""
    point = theta.detach().clone().requires_grad_(True)
    value = objective(point)
    (grad,) = torch.autograd.grad(value, point)
    return grad.detach()
