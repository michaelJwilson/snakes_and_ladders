"""Random-walk Metropolis over an :class:`~snakes_and_ladders.opt.objective.Objective` (issue #1006).

The gradient-free chain: propose ``y = x + h z`` with ``z ~ N(0, T I)`` and
accept with probability ``min(1, exp((U(x) - U(y)) / T))``. The proposal is
symmetric, so the ratio is the energy difference alone (Metropolis et al.,
1953). One energy evaluation per proposal: the current point's energy is
carried from the step that reached it.

It is a :class:`~snakes_and_ladders.sample.hmc.Kernel`, so
:func:`~snakes_and_ladders.sample.hmc.run_chain` runs its warm-up, burn-in,
draws and operators, as it does for HMC and MALA. The warm-up is
:class:`~snakes_and_ladders.sample.hmc.Adaptation` unchanged: a diagonal
metric is a per-coordinate proposal scale, and dual averaging drives the step
to a target acceptance, which belongs near :data:`RWM_TARGET_ACCEPTANCE`
(Roberts, Gelman & Gilks, 1997) and not at HMC's 0.65.

The compiled route (``oxisal.metropolis``, ``src/metropolis.rs``) runs the
whole chain, the warm-up included, on a declared family
(:mod:`snakes_and_ladders.sample.declared`), from its own ChaCha8 stream.
:func:`replay` is the chain on randomness the caller supplies: the step
BlackJAX's ``rmh`` is pinned to draw for draw.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders import oxisal
from snakes_and_ladders.backend import Backend, refuse_backend
from snakes_and_ladders.emissions import ParameterDomainError
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.sample.accept import accept_ratio, acceptance_probability
from snakes_and_ladders.sample.chain import (
    Adaptation,
    Transition,
    compiled_route,
    run_chain,
    run_compiled,
    start_point,
)
from snakes_and_ladders.sample.declared import declared_energy

#: The asymptotically optimal acceptance for a random walk on a product
#: target as the dimension grows (Roberts, Gelman & Gilks, 1997).
RWM_TARGET_ACCEPTANCE = 0.234

#: Energy evaluations per proposal: the proposal's; the current point's is carried.
EVALUATIONS_PER_PROPOSAL = 1


def _energy(objective: Objective, theta: torch.Tensor) -> float:
    with torch.no_grad():
        return float(objective(theta))


def _decide(
    current: float, proposed: float, temperature: float, uniform: float
) -> tuple[bool, float, float]:
    """Accept or not, the acceptance probability, and the energy error."""
    log_ratio = (current - proposed) / temperature
    # ``math.exp`` overflows where a ratio is only "certainly accept"; a nan
    # energy stays nan, which the comparison refuses.
    ratio = math.exp(log_ratio) if not log_ratio > 700.0 else math.inf
    return (
        accept_ratio(ratio, uniform),
        acceptance_probability(ratio),
        abs(proposed - current),
    )


class _RandomWalkKernel:
    """One Gaussian proposal and its Metropolis test: :func:`random_walk`'s kernel.

    The generator is consumed as the normal increment, then the uniform. The
    current energy is kept beside the tensor it was taken at, and reused
    while the chain has not moved --- the identity of the tensor is the key,
    so a changed objective or a new position is evaluated afresh.
    """

    def __init__(self) -> None:
        self._at: tuple[Objective, torch.Tensor, float] | None = None

    def __call__(
        self,
        objective: Objective,
        position: torch.Tensor,
        temperature: float,
        generator: torch.Generator,
        step_size: float,
    ) -> Transition:
        if (
            self._at is not None
            and self._at[0] is objective
            and self._at[1] is position
        ):
            current = self._at[2]
        else:
            current = _energy(objective, position)
        increment = torch.randn(
            position.shape, generator=generator, dtype=torch.float64
        )
        proposal = position + step_size * math.sqrt(temperature) * increment
        try:
            proposed = _energy(objective, proposal)
        except ParameterDomainError:
            proposed = math.inf
        uniform = float(torch.rand(1, generator=generator))
        take, probability, error = _decide(current, proposed, temperature, uniform)
        if take:
            self._at = (objective, proposal, proposed)
            return Transition(proposal, error, 1, probability)
        self._at = (objective, position, current)
        return Transition(position, error, 0, probability)


def random_walk(
    objective: Objective,
    generator: torch.Generator,
    n_samples: int,
    *,
    step_size: float,
    theta0: torch.Tensor | None = None,
    burn_in: int = 0,
    temperature: float = 1.0,
    adaptation: Adaptation | None = None,
    store_chain: bool = True,
    operators: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    backend: Backend = Backend.RUST,
) -> hmc.Chain:
    """Draw ``n_samples`` from ``exp(-objective / temperature)`` by random-walk Metropolis.

    Parameters
    ----------
    objective : Objective
        Read as an unnormalized negative log density; only its value is used.
    generator : torch.Generator
        The stream every increment and uniform is drawn from.
    n_samples : int
        Draws recorded after burn-in.
    step_size : float
        The proposal's standard deviation per coordinate, in the metric's
        coordinates. Required: its right value is the target's scale. With an
        ``adaptation`` it is the warm-up's starting point.
    theta0, burn_in, temperature, store_chain, operators
        As :func:`snakes_and_ladders.sample.hmc.sample`.
    adaptation : Adaptation | None
        The two-window warm-up :func:`~snakes_and_ladders.sample.hmc.sample`
        runs: dual averaging on the step, the variance for the metric. Its
        ``target_acceptance`` belongs near :data:`RWM_TARGET_ACCEPTANCE`.
    backend : Backend
        :data:`~snakes_and_ladders.backend.Backend.RUST`, the default, runs
        the chain and its warm-up in ``oxisal.metropolis`` when the objective
        declares a family (:func:`~snakes_and_ladders.sample.declared.declared_energy`)
        and the chain is at unit temperature in no tracked run; its
        ``operators`` observe the draws as :func:`~snakes_and_ladders.sample.hmc.run_compiled`
        states, so a
        chain with ``store_chain=False`` holds that many draws at most. Its stream is ChaCha8 seeded by one draw from
        ``generator``, so it is pinned to the torch route in distribution.
        Any other chain, and :data:`~snakes_and_ladders.backend.Backend.PYTHON`,
        run the torch kernel.

    Returns
    -------
    ~snakes_and_ladders.sample.hmc.Chain
        The draws, acceptance, per-proposal energy error, energy evaluations
        spent (warm-up included) and what the warm-up settled on.

    Raises
    ------
    ValueError
        If ``step_size`` or ``temperature`` is not positive.
    """
    if not step_size > 0.0:
        msg = f"step_size must be positive, got {step_size}"
        raise ValueError(msg)
    refuse_backend("random_walk", backend, (Backend.PYTHON, Backend.RUST))
    declared = declared_energy(objective)
    if compiled_route(backend, temperature) and declared is not None:
        return run_compiled(
            oxisal.MetropolisWalk,
            declared,
            (),
            EVALUATIONS_PER_PROPOSAL,
            generator,
            n_samples,
            step_size=step_size,
            theta0=start_point(objective, theta0),
            burn_in=burn_in,
            adaptation=adaptation,
            store_chain=store_chain,
            operators=operators,
        )
    return run_chain(
        _RandomWalkKernel(),
        EVALUATIONS_PER_PROPOSAL,
        objective,
        generator,
        n_samples,
        step_size=step_size,
        theta0=theta0,
        burn_in=burn_in,
        temperature=temperature,
        adaptation=adaptation,
        store_chain=store_chain,
        operators=operators,
    )


@dataclass(frozen=True)
class Replayed:
    """A chain :func:`replay` ran: every position and whether each proposal was taken."""

    draws: np.ndarray
    accepted: np.ndarray


def replay(
    objective: Objective,
    theta0: np.ndarray,
    step_size: float | np.ndarray,
    increments: np.ndarray,
    uniforms: np.ndarray,
) -> Replayed:
    """The chain at unit temperature on the caller's increments ``(n, d)`` and uniforms ``(n,)``.

    :class:`_RandomWalkKernel`'s step with the draws supplied rather than
    taken from a generator, so an external implementation given the same
    randomness is compared draw for draw. ``step_size`` is a scalar or one
    scale per coordinate.
    """
    position = torch.as_tensor(np.asarray(theta0, dtype=np.float64)).clone()
    scale = torch.as_tensor(np.asarray(step_size, dtype=np.float64))
    current = _energy(objective, position)
    draws = np.empty((len(uniforms), position.shape[0]))
    accepted = np.zeros(len(uniforms), dtype=bool)
    for index, (increment, uniform) in enumerate(
        zip(increments, uniforms, strict=True)
    ):
        proposal = position + scale * torch.as_tensor(increment)
        proposed = _energy(objective, proposal)
        take, _, _ = _decide(current, proposed, 1.0, float(uniform))
        if take:
            position, current = proposal, proposed
        accepted[index] = take
        draws[index] = position.numpy()
    return Replayed(draws, accepted)
