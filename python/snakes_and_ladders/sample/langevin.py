"""The Metropolis-adjusted Langevin algorithm: HMC at one leapfrog step.

The baseline every Hamiltonian number in this package is read against.
:func:`~snakes_and_ladders.sample.hmc.sample` costs ``n_steps + 1`` gradients a
proposal and buys a trajectory with them; MALA (Roberts & Tweedie, 1996) costs
two and buys one gradient-informed step, so a comparison at equal *gradients*
is the one that says whether the trajectory was worth it. Reported as
effective samples per gradient, never per draw.

**It is the one-step limit, and that is asserted rather than asserted about.**
The proposal ``x' = x - (h^2 / 2) grad U(x) + h sqrt(T) z`` is one leapfrog
step from momentum ``z sqrt(T)``, and the Metropolis ratio of the two Gaussian
transition densities is ``exp(-(H(x', -p') - H(x, p)) / T)`` --- the same
number :func:`~snakes_and_ladders.sample.hmc.sample` accepts on. This module
writes the *densities*, not the trajectory, so the identity is a measurement
between two implementations rather than a transcription of one;
``tests/regression/opt/test_opt_langevin.py`` reads the difference against
``sample(n_steps=1)`` on one generator state.

**Recompute rather than store, and the reason is the warm-up.** A proposal
evaluates the gradient at the current point and at the proposal, and the
second could be carried into the next transition when the proposal is
accepted --- halving the cost. It is not, because the warm-up rebases the
objective onto the metric's coordinates and moves the position with it, so a
gradient cached across transitions would have two invalidation points and no
owner. The cost is therefore two gradients a proposal, which is
``leapfrog.force_evaluations(1)`` exactly, and what makes the two routes
comparable at equal evaluations rather than at equal draws.

**Unadjusted Langevin is a flag, and it costs what MALA costs.** Dropping the
Metropolis correction leaves a chain whose stationary distribution is not the
target, so `opt/CLAUDE.md` requires the bias be reported rather than argued
away: the energy error is the correction the chain declined to apply, and
reporting it needs the gradient at the proposal that the correction needed.
ULA therefore saves no evaluation here --- it buys a bias and nothing else,
which is the finding and not an implementation accident. On a Gaussian of
variance ``s^2`` its stationary variance is ``s^2 / (1 - h^2 / (4 s^2))``,
written out in the test that pins it.

See Roberts & Tweedie (1996) for the algorithm and its stability; Roberts &
Rosenthal (1998) for the 0.574 acceptance the step is adapted toward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from snakes_and_ladders.backend import Backend, refuse_backend
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.sample.accept import accept_ratio, acceptance_probability
from snakes_and_ladders.sample.hmc import (
    Adaptation,
    Adapted,
    DeclaredGaussian,
    Transition,
    _compiled_gaussian_chain,
    _start,
    gradient_at,
    run_chain,
)
from snakes_and_ladders.track import NULL as UNTRACKED
from snakes_and_ladders.track import current as current_tracked

#: The acceptance a MALA step is adapted toward: the optimal scaling of the
#: Langevin diffusion in the limit of many dimensions (Roberts & Rosenthal,
#: 1998), against 0.65 for HMC. The two differ because the proposals do --- a
#: single step's error is order ``h^3`` per coordinate where a trajectory's is
#: bounded --- so a warm-up reaching for HMC's target here adapts to the wrong
#: step. Stated as a constant and passed by the caller through
#: :class:`~snakes_and_ladders.sample.hmc.Adaptation`, whose every field is
#: required.
MALA_TARGET_ACCEPTANCE = 0.574

#: Leapfrog steps a MALA proposal is. One, by the identity this module exists
#: to keep; the gradients it costs are ``leapfrog.force_evaluations(1)``.
LANGEVIN_STEPS = 1

#: Gradients one proposal spends: at the current point and at the proposal.
#: See the module note on why the second is not carried forward.
GRADIENTS_PER_PROPOSAL = 2


@dataclass(frozen=True)
class LangevinChain:
    """A Langevin chain, what it cost, and whether it was corrected.

    :class:`~snakes_and_ladders.sample.hmc.HmcChain`'s fields, plus the flag,
    because an uncorrected chain's acceptance rate is 1 by construction and
    reading the two objects alike would read that 1 as a diagnostic.

    Parameters
    ----------
    theta : torch.Tensor
        Draws in unconstrained coordinates, shape ``(n_samples, dimension)``.
    acceptance_rate : float
        Fraction of proposals accepted. **1.0 by construction when
        ``corrected`` is False**, where it says nothing at all; read
        ``energy_error`` instead.
    energy_error : torch.Tensor
        ``|H(proposal) - H(current)|`` per proposal, ``H`` being the
        Hamiltonian of the equivalent one-step trajectory. For a corrected
        chain it separates a step too large from a bug, as it does for
        :func:`~snakes_and_ladders.sample.hmc.sample`; for an uncorrected one it
        is the correction that was not applied, and so the bias itself.
    force_evaluations : int
        Gradients spent, warm-up and burn-in included, at
        :data:`GRADIENTS_PER_PROPOSAL` a proposal.
    adapted : Adapted | None
        What the warm-up settled on, or ``None`` for a fixed-step chain.
    corrected : bool
        Whether the Metropolis correction was applied. False is ULA, whose
        draws are from a distribution that is not the target.
    """

    theta: torch.Tensor
    acceptance_rate: float
    energy_error: torch.Tensor
    force_evaluations: int
    adapted: Adapted | None
    corrected: bool


def mala(
    objective: Objective,
    generator: torch.Generator,
    n_samples: int,
    *,
    step_size: float,
    theta0: torch.Tensor | None = None,
    burn_in: int = 0,
    temperature: float = 1.0,
    adaptation: Adaptation | None = None,
    corrected: bool = True,
    store_chain: bool = True,
    backend: Backend = Backend.RUST,
) -> LangevinChain:
    """Draw ``n_samples`` from ``exp(-objective / temperature)`` by Langevin steps.

    Parameters
    ----------
    objective : Objective
        Read as an unnormalized negative log density, as
        :func:`~snakes_and_ladders.sample.hmc.sample` reads it. The same warning
        applies: a bare negative log-likelihood here is a posterior under an
        improper flat prior and nothing in this module can tell.
    generator : torch.Generator
        The stream every noise and acceptance draw comes from, passed in
        rather than seeded here.
    n_samples : int
        Draws recorded after burn-in.
    step_size : float
        The Langevin step ``h``. Required rather than defaulted: the drift is
        ``h^2 / 2`` times a gradient whose scale is the target's, so a default
        would be wrong silently. With an ``adaptation`` it is the warm-up's
        starting point.
    theta0 : torch.Tensor | None
        Starting point; ``objective.initial()`` when omitted.
    burn_in : int
        Draws discarded before recording.
    temperature : float
        The chain targets ``exp(-objective / temperature)``; the noise has
        variance ``h^2 T`` and the ratio divides by ``T``, so at 1 both are
        the identity.
    adaptation : Adaptation | None
        A warm-up that sets the step size and the mass diagonal before the
        ``burn_in`` and the draws, the same two windows
        :func:`~snakes_and_ladders.sample.hmc.sample` runs. Its
        ``target_acceptance`` belongs at :data:`MALA_TARGET_ACCEPTANCE` here
        and not at HMC's 0.65.
    corrected : bool
        True is MALA. False is ULA: every proposal is accepted and the chain
        is biased, by an amount ``energy_error`` reports and no diagnostic
        inside the chain corrects.
    store_chain : bool
        Whether the draws are kept. False keeps none --- ``theta`` has zero
        rows --- and the chain reports its acceptance and energy errors
        alone, as :func:`~snakes_and_ladders.sample.hmc.sample`'s does (issues
        #988, #997).
    backend : Backend
        :data:`~snakes_and_ladders.backend.Backend.RUST`, the default since
        issue #997, runs the whole chain in
        ``oxisal.gaussian_hmc`` at one leapfrog step --- the
        identity this module keeps --- when the objective is a
        :class:`~snakes_and_ladders.sample.hmc.DeclaredGaussian` and the chain
        is the plain corrected one: unit temperature, no adaptation, and no
        tracked run. Its draws are its own ChaCha8 stream seeded by one draw
        from ``generator``, so it is pinned to the torch route in
        distribution, as :func:`~snakes_and_ladders.sample.hmc.sample`'s is.
        Any other chain, and
        :data:`~snakes_and_ladders.backend.Backend.PYTHON`, take the torch
        kernel.

    Returns
    -------
    LangevinChain

    Raises
    ------
    ValueError
        If ``step_size`` or ``temperature`` is not positive.
    """
    if step_size <= 0.0:
        msg = f"step_size must be positive, got {step_size}"
        raise ValueError(msg)
    refuse_backend("mala", backend, (Backend.PYTHON, Backend.RUST))
    if (
        backend is Backend.RUST
        and corrected
        and isinstance(objective, DeclaredGaussian)
        and temperature == 1.0
        and adaptation is None
        and current_tracked() is UNTRACKED
    ):
        compiled = _compiled_gaussian_chain(
            objective,
            generator,
            n_samples,
            step_size=step_size,
            n_steps=LANGEVIN_STEPS,
            theta0=_start(objective, theta0),
            burn_in=burn_in,
            store_chain=store_chain,
        )
        return LangevinChain(
            theta=compiled.theta,
            acceptance_rate=compiled.acceptance_rate,
            energy_error=compiled.energy_error,
            force_evaluations=(n_samples + burn_in) * GRADIENTS_PER_PROPOSAL,
            adapted=None,
            corrected=True,
        )

    chain = run_chain(
        _LangevinKernel(corrected=corrected),
        GRADIENTS_PER_PROPOSAL,
        objective,
        generator,
        n_samples,
        step_size=step_size,
        theta0=theta0,
        burn_in=burn_in,
        temperature=temperature,
        adaptation=adaptation,
        store_chain=store_chain,
    )
    return LangevinChain(
        theta=chain.draws,
        acceptance_rate=chain.acceptance_rate,
        energy_error=chain.energy_error,
        force_evaluations=chain.force_evaluations,
        adapted=chain.adapted,
        corrected=corrected,
    )


def _log_proposal_density(
    destination: torch.Tensor,
    origin: torch.Tensor,
    gradient: torch.Tensor,
    step_size: float,
    temperature: float,
) -> float:
    """``log q(destination | origin)`` up to the constant that cancels.

    ``q(. | x) = N(x - (h^2 / 2) grad U(x), h^2 T I)``. The normalizer depends
    on ``h``, ``T`` and the dimension alone, all of which are the same in both
    directions of one Metropolis ratio, so it is not formed.
    """
    residual = destination - origin + 0.5 * step_size * step_size * gradient
    return -float((residual * residual).sum()) / (
        2.0 * step_size * step_size * temperature
    )


@dataclass(frozen=True)
class _LangevinKernel:
    """One Langevin proposal and its Metropolis test: :func:`mala`'s kernel.

    Written as the two transition densities Roberts & Tweedie state the
    algorithm with, rather than as a trajectory, so that its agreement with
    ``sample(n_steps=1)`` is a measurement between two routes. The generator
    is consumed in the order that sampler consumes it --- the normal draw,
    then the uniform --- so the two routes see the same noise.
    """

    corrected: bool

    def __call__(
        self,
        objective: Objective,
        position: torch.Tensor,
        temperature: float,
        generator: torch.Generator,
        step_size: float,
    ) -> Transition:
        gradient = gradient_at(objective, position)
        noise = torch.randn(
            position.shape, generator=generator, dtype=torch.float64
        ) * math.sqrt(temperature)
        proposal = position - 0.5 * step_size * step_size * gradient + step_size * noise
        proposed_gradient = gradient_at(objective, proposal)

        log_ratio = (
            float(objective(position.detach())) - float(objective(proposal.detach()))
        ) / temperature
        log_ratio += _log_proposal_density(
            position, proposal, proposed_gradient, step_size, temperature
        )
        log_ratio -= _log_proposal_density(
            proposal, position, gradient, step_size, temperature
        )

        # ``log_ratio`` is ``-(H(proposal) - H(current)) / T`` by the identity
        # the module states, so the energy error is available here without
        # forming the momenta the trajectory form would carry.
        error = abs(temperature * log_ratio)
        ratio = float(torch.exp(torch.tensor(log_ratio)))
        probability = acceptance_probability(ratio)
        if not self.corrected:
            return Transition(
                position=proposal,
                energy_error=error,
                accepted=1,
                probability=probability,
            )
        uniform = float(torch.rand(1, generator=generator))
        if accept_ratio(ratio, uniform):
            return Transition(
                position=proposal,
                energy_error=error,
                accepted=1,
                probability=probability,
            )
        return Transition(
            position=position, energy_error=error, accepted=0, probability=probability
        )
