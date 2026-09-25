"""Random-walk Metropolis over an :class:`~sal.opt.objective.Objective` (issue #1006).

The gradient-free chain: propose ``y = x + h z`` with ``z ~ N(0, T I)`` and
accept with probability ``min(1, exp((U(x) - U(y)) / T))``. The proposal is
symmetric, so the ratio is the energy difference alone (Metropolis et al.,
1953). One energy evaluation per proposal: the current point's energy is
carried from the step that reached it.

It is a :class:`~sal.sample.hmc.Kernel`, so
:func:`~sal.sample.hmc.run_chain` runs its warm-up, burn-in,
draws and operators, as it does for HMC and MALA. The warm-up is
:class:`~sal.sample.hmc.Adaptation` unchanged: a diagonal
metric is a per-coordinate proposal scale, and dual averaging drives the step
to a target acceptance, which belongs near :data:`RWM_TARGET_ACCEPTANCE`
(Roberts, Gelman & Gilks, 1997) and not at HMC's 0.65.

No derivative is taken, so the chain reads the objective through
:func:`~sal.opt.objective.energy_of` on arrays and draws from
a :class:`numpy.random.Generator` (issue #1011): an objective declaring a
NumPy :meth:`~sal.opt.objective.DeclaredEnergy.energy` is
evaluated without the tensor type, and any other through ``__call__``.

The compiled route (``oxisal.metropolis``, ``src/metropolis.rs``) runs the
whole chain, the warm-up included, on a declared family
(:mod:`sal.sample.declared`), from its own ChaCha8 stream.
:func:`replay` is the chain on randomness the caller supplies: the step
BlackJAX's ``rmh`` is pinned to draw for draw.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sal import oxisal
from sal.backend import Backend, refuse_backend
from sal.emissions import ParameterDomainError
from sal.opt.objective import Objective, energy_of
from sal.sample import hmc
from sal.sample.accept import accept_ratio, acceptance_probability
from sal.sample.chain import (
    Adaptation,
    Transition,
    compiled_route,
    on_buffer,
    run_chain,
    run_compiled,
    start_point,
)
from sal.sample.declared import declared_energy

if TYPE_CHECKING:
    import torch

#: The asymptotically optimal acceptance for a random walk on a product
#: target as the dimension grows (Roberts, Gelman & Gilks, 1997).
RWM_TARGET_ACCEPTANCE = 0.234

#: Energy evaluations per proposal: the proposal's; the current point's is carried.
EVALUATIONS_PER_PROPOSAL = 1


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
    step is on arrays: the loop's tensor is read through its own buffer and
    the proposal handed back on its buffer, so neither is copied. The current
    energy is kept beside the tensor it was taken at, and reused while the
    chain has not moved --- the identity of the tensor is the key, so a
    changed objective or a new position is evaluated afresh.
    """

    def __init__(self) -> None:
        self._at: tuple[Objective, torch.Tensor, float] | None = None

    def __call__(
        self,
        objective: Objective,
        position: torch.Tensor,
        temperature: float,
        generator: np.random.Generator,
        step_size: float,
    ) -> Transition:
        if (
            self._at is not None
            and self._at[0] is objective
            and self._at[1] is position
        ):
            current = self._at[2]
        else:
            current = energy_of(objective, position.numpy())
        increment = generator.standard_normal(position.shape[0])
        proposal = position.numpy() + step_size * math.sqrt(temperature) * increment
        try:
            proposed = energy_of(objective, proposal)
        except ParameterDomainError:
            proposed = math.inf
        uniform = float(generator.random())
        take, probability, error = _decide(current, proposed, temperature, uniform)
        if take:
            moved = on_buffer(proposal)
            self._at = (objective, moved, proposed)
            return Transition(moved, error, 1, probability)
        self._at = (objective, position, current)
        return Transition(position, error, 0, probability)


def random_walk(
    objective: Objective,
    rng: np.random.Generator,
    n_samples: int,
    *,
    step_size: float,
    theta0: np.ndarray | None = None,
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
        Read as an unnormalized negative log density; only its value is
        used, through :func:`~sal.opt.objective.energy_of`.
    rng : numpy.random.Generator
        The stream every increment and uniform is drawn from (issue #1011).
    n_samples : int
        Draws recorded after burn-in.
    step_size : float
        The proposal's standard deviation per coordinate, in the metric's
        coordinates. Required: its right value is the target's scale. With an
        ``adaptation`` it is the warm-up's starting point.
    theta0 : numpy.ndarray | None
        Starting point, copied; ``objective.initial()`` when omitted. An
        array, as the chain takes no derivative (issue #1059).
    burn_in, temperature, store_chain, operators
        As :func:`sal.sample.hmc.sample`.
    adaptation : Adaptation | None
        The two-window warm-up :func:`~sal.sample.hmc.sample`
        runs: dual averaging on the step, the variance for the metric. Its
        ``target_acceptance`` belongs near :data:`RWM_TARGET_ACCEPTANCE`.
    backend : Backend
        :data:`~sal.backend.Backend.RUST`, the default, runs
        the chain and its warm-up in ``oxisal.metropolis`` when the objective
        declares a family (:func:`~sal.sample.declared.declared_energy`)
        and the chain is at unit temperature in no tracked run; its
        ``operators`` observe the draws as :func:`~sal.sample.hmc.run_compiled`
        states, so a
        chain with ``store_chain=False`` holds that many draws at most. Its stream is ChaCha8 seeded by one draw from
        ``rng``, so it is pinned to the Python route in distribution.
        Any other chain, and :data:`~sal.backend.Backend.PYTHON`,
        run the NumPy kernel.

    Returns
    -------
    ~sal.sample.hmc.Chain
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
            rng,
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
        rng,
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
    position = np.array(theta0, dtype=np.float64)
    scale = np.asarray(step_size, dtype=np.float64)
    current = energy_of(objective, position)
    draws = np.empty((len(uniforms), position.shape[0]))
    accepted = np.zeros(len(uniforms), dtype=bool)
    for index, (increment, uniform) in enumerate(
        zip(increments, uniforms, strict=True)
    ):
        proposal = position + scale * np.asarray(increment, dtype=np.float64)
        proposed = energy_of(objective, proposal)
        take, _, _ = _decide(current, proposed, 1.0, float(uniform))
        if take:
            position, current = proposal, proposed
        accepted[index] = take
        draws[index] = position
    return Replayed(draws, accepted)
