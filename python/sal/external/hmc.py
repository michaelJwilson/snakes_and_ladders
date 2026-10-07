"""The Hamiltonian chain's external solver, in ``sample.hmc``'s terms (issue #1282, step 6).

``sal.external`` is namespaced by problem family: this module holds
:func:`sample`, BlackJAX's HMC (:attr:`~sal.external.solvers.Solver.BLACKJAX_HMC`)
in a subprocess, with :func:`sal.sample.hmc.sample`'s positional arguments in
its order and a :class:`~sal.external.solvers.Solver` after them. It returns
the sibling's :class:`~sal.sample.hmc.HmcChain` as :class:`ExternalChain`,
which adds a :class:`~sal.opt.termination.Termination`, the seconds the
script measured and the :class:`~sal.external.solvers.Provenance`.

**Targets.** A torch closure cannot cross the process boundary, so the
target is what the objective declares through
:meth:`~sal.sample.declared.SupportedGradient.supported_gradient`, rebuilt in
JAX by the script. Four kernels are rebuilt, each a capability BlackJAX
declares: ``gaussian`` (a precision, dense or diagonal), ``rosenbrock``
(``a``, ``b``), ``gaussian_mixture`` (``k`` and one channel of observations)
and ``gaussian_hmm`` on segments of equal length. A ragged Gaussian HMM, a
count mixture, a count HMM and an objective that declares no kernel each need
a :class:`~sal.external.solvers.Capability` BlackJAX does not declare
(:func:`target_capabilities`), and are refused before any subprocess starts.
The script also returns ``-log p`` at every draw, so a caller checks the
rebuilt target against the objective's own value
(``tests/validation/test_blackjax.py``).

**The bytes.** The target and the chain are posed with
:func:`~sal.external.hmc_inputs.target_inputs` and
:func:`~sal.external.hmc_inputs.chain_inputs`, which the adapter
:mod:`sal.validation.blackjax` sends too, so a chain is the adapter's bitwise
on one key. ``rng`` is drawn from once, for the integer JAX builds its key
from; no generator crosses the boundary.

**Adaptation.** :class:`~sal.sample.chain.Adaptation` maps to
``blackjax.window_adaptation`` with a diagonal mass, starting from
``step_size``, at ``target_acceptance``, over ``warmup`` transitions. Both
regularize the inverse mass as Stan does, with pseudo-count 5 (#1207), and
both drive the acceptance probability by dual averaging toward
``log(10 * step)``. The windows differ: the package runs two windows of equal
length, BlackJAX Stan's schedule --- a fast window of 75 steps, slow windows
doubling from 25, a final fast window of 50, scaled to 15%, 75% and 10% of a
shorter warm-up --- and adapts no mass below 20 steps, which is refused.
BlackJAX's step is fixed, so a ``step_jitter`` other than 0 needs
:attr:`~sal.external.solvers.Capability.STEP_JITTER` and is refused.
:class:`~sal.sample.chain.Adapted` reports the final fast window's mean
acceptance probability and the coordinates the last slow window did not move.

**Cost.** ``spent`` is in :attr:`~sal.cost.Cost.GRADIENTS`, as the
sibling's: one at the start, and ``n_steps`` per transition, the warm-up's
included, since BlackJAX's velocity Verlet carries the gradient a trajectory
ends on into the next.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
import torch

from sal.cost import Cost
from sal.external.hmc_inputs import chain_inputs, target_inputs
from sal.external.sessions import Session, served_by
from sal.external.solvers import (
    Capability,
    ExternalUnavailable,
    Provenance,
    Solver,
    available,
    invoke,
    provenance,
    require,
)
from sal.opt.objective import Objective
from sal.opt.termination import Termination
from sal.sample.chain import Adaptation, Adapted, start_point
from sal.sample.declared import declared_energy
from sal.sample.hmc import DEFAULT_STEPS, HmcChain

#: The unit ``spent`` counts, as :func:`sal.sample.hmc.sample`'s.
UNIT = Cost.GRADIENTS

#: The fewest warm-up steps over which BlackJAX's schedule adapts a mass.
MIN_WARMUP = 20

#: The kernels the script rebuilds in JAX, and the capability each needs.
_REBUILT: Mapping[str, Capability] = {
    "gaussian": Capability.GAUSSIAN_TARGET,
    "rosenbrock": Capability.ROSENBROCK_TARGET,
    "gaussian_mixture": Capability.GAUSSIAN_MIXTURE_TARGET,
    "gaussian_hmm": Capability.GAUSSIAN_HMM_TARGET,
}

#: The declared kernels it does not.
_NOT_REBUILT: Mapping[str, Capability] = {
    "count_mixture": Capability.COUNT_MIXTURE_TARGET,
    "count_hmm": Capability.COUNT_HMM_TARGET,
}


def target_capabilities(objective: object) -> frozenset[Capability]:
    """The :class:`~sal.external.solvers.Capability` set a solver needs to sample ``objective``.

    Read from its declared kernel (:func:`~sal.sample.declared.declared_energy`):
    an objective with none needs
    :attr:`~sal.external.solvers.Capability.UNDECLARED_TARGET`, and a
    Gaussian HMM over segments of unequal lengths
    :attr:`~sal.external.solvers.Capability.RAGGED_SEGMENTS` beside its own.
    """
    declared = declared_energy(objective)
    if declared is None:
        return frozenset({Capability.UNDECLARED_TARGET})
    kernel, data = declared
    if kernel in _NOT_REBUILT:
        return frozenset({_NOT_REBUILT[kernel]})
    if kernel not in _REBUILT:
        return frozenset({Capability.UNDECLARED_TARGET})
    needs = {_REBUILT[kernel]}
    if kernel == "gaussian_hmm" and len(set(_lengths(data))) > 1:
        needs.add(Capability.RAGGED_SEGMENTS)
    return frozenset(needs)


def _lengths(data: Mapping[str, Any]) -> tuple[int, ...]:
    """A Gaussian HMM's segment lengths: ``lengths``, or one per row of ``observations``."""
    lengths = data.get("lengths")
    if lengths is not None:
        return tuple(int(n) for n in np.asarray(lengths).reshape(-1))
    rows = np.asarray(data["observations"])
    return (int(rows.shape[-1]),) * (int(rows.shape[0]) if rows.ndim == 2 else 1)


def _target(objective: object) -> dict[str, np.ndarray]:
    """The script's target inputs for an objective :func:`target_capabilities` admitted."""
    declared = declared_energy(objective)
    assert declared is not None  # admitted: a rebuilt kernel was declared
    kernel, data = declared
    if kernel == "gaussian":
        return target_inputs(np.asarray(data["precision"]))
    if kernel == "rosenbrock":
        return target_inputs(rosenbrock=(float(data["a"]), float(data["b"])))
    if kernel == "gaussian_mixture":
        values = np.asarray(data["observations"]).reshape(-1)
        return target_inputs(mixture=(int(data["k"]), values))
    lengths = _lengths(data)
    rows = np.asarray(data["observations"]).reshape(len(lengths), lengths[0])
    return target_inputs(hmm=(int(data["m"]), rows))


def seed_of(rng: np.random.Generator | torch.Generator) -> int:
    """One draw from ``rng``: the integer the subprocess builds its JAX key from."""
    if isinstance(rng, torch.Generator):
        return int(torch.randint(0, 2**31 - 1, (), generator=rng))
    return int(rng.integers(0, 2**31 - 1))


@dataclass(frozen=True)
class ExternalChain(HmcChain):
    """A :class:`~sal.sample.hmc.HmcChain` from an external solver, with its termination, seconds and provenance."""

    #: The mean Metropolis acceptance probability over the draws, the
    #: statistic BlackJAX reports; ``acceptance_rate`` is the fraction accepted.
    acceptance_probability: float = dataclass_field(kw_only=True)
    #: ``n_samples`` transitions, the chain's budget, run to its end.
    termination: Termination = dataclass_field(kw_only=True)
    #: Wall seconds of the compiled chain, compilation excluded, as the script measured them.
    seconds: float = dataclass_field(kw_only=True)
    #: The framework, installed version and licence the chain came from.
    provenance: Provenance = dataclass_field(kw_only=True)


def _needs(objective: object, adaptation: Adaptation | None) -> set[Capability]:
    """What a call asks of its solver: the task, the target, and the warm-up's parts."""
    needs = {Capability.HMC_SAMPLE, *target_capabilities(objective)}
    if adaptation is not None:
        needs.add(Capability.WINDOW_ADAPTATION)
        if adaptation.step_jitter != 0.0:
            needs.add(Capability.STEP_JITTER)
    return needs


def _checked_chain(
    step_size: object, n_steps: int, n_samples: int, adaptation: Adaptation | None
) -> float:
    """``step_size`` as a ``float``, once the chain's sizes are admitted; raises :class:`ValueError`."""
    if not isinstance(step_size, int | float) or isinstance(step_size, bool):
        msg = (
            f"BlackJAX takes a given step, a positive float; got {step_size!r} "
            "(sample.hmc's 'auto' pilot runs in the package alone)"
        )
        raise ValueError(msg)
    if not step_size > 0.0:
        msg = f"step_size must be positive, got {step_size}"
        raise ValueError(msg)
    if n_steps < 1:
        msg = f"n_steps must be at least 1, got {n_steps}"
        raise ValueError(msg)
    if n_samples < 1:
        msg = f"n_samples must be at least 1, got {n_samples}"
        raise ValueError(msg)
    if adaptation is not None and adaptation.warmup < MIN_WARMUP:
        msg = (
            f"BlackJAX's window adaptation adapts no mass below {MIN_WARMUP} "
            f"warm-up steps; got {adaptation.warmup}"
        )
        raise ValueError(msg)
    return float(step_size)


def sample(
    objective: Objective,
    rng: np.random.Generator | torch.Generator,
    n_samples: int,
    solver: Solver,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    start: torch.Tensor | None = None,
    adaptation: Adaptation | None = None,
    timeout: float = 600.0,
    session: Session | None = None,
) -> ExternalChain:
    """``solver``'s HMC chain on ``exp(-objective)``, as :func:`sal.sample.hmc.sample` returns one.

    The sibling's three positional arguments in its order, the solver after
    them, and of its keywords those BlackJAX's ``hmc`` honours: a given step,
    the leapfrog count, the start and a warm-up. The rest set the package's
    own route --- an integrator, a temperature, a burn-in, operators, a
    backend --- and BlackJAX has none of them here.

    Parameters
    ----------
    objective : Objective
        Read as an unnormalized negative log density, through the kernel it
        declares (:func:`target_capabilities`).
    rng : np.random.Generator | torch.Generator
        Drawn from once, for the integer the subprocess keys JAX from.
    n_samples : int
        Transitions recorded, at least one.
    solver : Solver
        One that declares :attr:`~sal.external.solvers.Capability.HMC_SAMPLE`
        and the target's capability.
    step_size : float
        The leapfrog step; with an ``adaptation`` the warm-up's first.
    n_steps : int
        Leapfrog steps per proposal; the sibling's default.
    start : torch.Tensor | None
        The starting point; ``objective.initial()`` when omitted.
    adaptation : Adaptation | None
        ``blackjax.window_adaptation`` over ``warmup`` transitions at
        ``target_acceptance``, with ``step_jitter`` 0; the module docstring
        states what differs from the package's warm-up.
    timeout : float
        Seconds the subprocess may take.
    session : Session | None
        A worker :func:`sal.external.session` opened on ``solver``.

    Returns
    -------
    ExternalChain
        The draws at unit mass or the adapted one, the fraction accepted and
        the mean acceptance probability, ``|H(proposal) - H(current)|`` per
        transition, ``spent`` gradients, the :class:`~sal.sample.chain.Adapted`
        warm-up if one ran, a :class:`~sal.opt.termination.Termination` on the
        budget after ``n_samples``, the script's seconds and the
        :class:`~sal.external.solvers.Provenance`.

    Raises
    ------
    CapabilityRefused
        If the target, or a jittered step, needs what ``solver`` does not
        declare, or ``solver`` does not sample.
    ExternalUnavailable
        If the framework is not installed.
    ValueError
        If ``step_size`` is not a positive float, ``n_steps`` or
        ``n_samples`` is below 1, the warm-up is shorter than
        :data:`MIN_WARMUP`, or ``session`` serves another solver.
    ScriptError
        If BlackJAX fails.
    """
    needs = _needs(objective, adaptation)
    require(solver, needs)
    step = _checked_chain(step_size, n_steps, n_samples, adaptation)
    served_by(session, solver)
    if session is None and not available(solver):
        raise ExternalUnavailable(solver)

    position = start_point(objective, start).numpy()
    warmup = 0 if adaptation is None else adaptation.warmup
    inputs = chain_inputs(
        _target(objective),
        position,
        step,
        n_steps,
        n_samples,
        seed_of(rng),
        warmup=None if adaptation is None else warmup,
        target_acceptance=None if adaptation is None else adaptation.target_acceptance,
        initial_step_size=None if adaptation is None else step,
    )
    result = (
        invoke(solver, needs, inputs, timeout=timeout)
        if session is None
        else session.invoke(needs, inputs, timeout=timeout)
    )
    out = result.outputs
    adapted = None
    if adaptation is not None:
        adapted = Adapted(
            step_size=float(out["step_size"]),
            mass_diagonal=torch.from_numpy(1.0 / out["inverse_mass_matrix"]),
            warmup_acceptance=float(out["warmup_acceptance"]),
            force_evaluations=warmup * n_steps,
            flat=tuple(int(i) for i in out["flat"]),
        )
    return ExternalChain(
        draws=torch.from_numpy(out["draws"].reshape(n_samples, position.shape[0])),
        acceptance_rate=float(out["accepted"]),
        energy_error=torch.from_numpy(out["energy_error"]),
        spent=1 + (warmup + n_samples) * n_steps,
        unit=UNIT,
        adapted=adapted,
        acceptance_probability=float(out["acceptance"]),
        # A chain of a fixed count has no criterion: it ends on its budget.
        termination=Termination.after(n_samples, converged=False),
        seconds=result.seconds,
        provenance=provenance(solver),
    )
