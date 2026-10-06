"""Hamiltonian Monte Carlo over an :class:`~sal.opt.objective.Objective`.

Every other continuous result here is a point estimate plus an interval from
the observed information at the optimum --- a Gaussian approximation to the
posterior at one point. This samples the posterior instead, so an interval is
a quantile rather than a curvature estimate, and where the two disagree the
disagreement is the finding.

The objective is read as an unnormalized negative log density. That is the
caller's claim to justify: for a log-likelihood plus a proper log prior it is
the posterior; for a bare log-likelihood it is a posterior under an improper
flat prior, which may not be normalizable. Nothing here can check it, so
:func:`sample` reports the chain and the diagnostics and leaves what the chain
samples to whoever built the objective.

`sal.opt` may import no application module and this needs none:
an `Objective` supplies an unconstrained vector and a differentiable scalar.

**The integrator is a composition, not a procedure.** Every method here is a
sequence of second-order kick-drift-kick sub-steps differing only in their
lengths, so :class:`Integrator` carries the weights and one driver runs them
all. A higher order costs more force evaluations per step, so the choice
between them is a measurement at equal *evaluations* and never at equal steps
(`search/CLAUDE.md`'s budget rule, and why
:meth:`Integrator.force_evaluations` exists).

**Temperature is the momentum's variance.** The tempered target
``exp(-U / T)`` is the marginal of ``exp(-(U + K) / T)``, whose momentum is
``N(0, T)``; Hamilton's equations for ``(U + K) / T`` are the untempered ones
in rescaled time, so the *same* integrator with the *same* step serves every
temperature and only the momentum draw and the acceptance ratio change (issue
#267). At ``T = 1`` every operation is the identity bitwise, so :func:`anneal`
is this sampler on a schedule rather than a second sampler and chains drawn
before the temperature existed are unchanged. A quasi-Newton fit has no
acceptance ratio to temper, which is why ``fit`` takes no schedule.

**Adaptation is a warm-up, and the chain is drawn after it.**
:class:`Adaptation` asks :func:`sample` for a warm-up that sets the mass
diagonal from the warm-up sample variance, regularized as Stan does
(:func:`~sal.sample.chain.regularized_variance`; issue #1207), and the step size by dual averaging
toward a stated acceptance (Hoffman & Gelman, 2014, §3.2;
``eq:dual-averaging``), then draws the chain at those *fixed* values, so the
draws are a Markov chain with the target as its stationary distribution.
Opt-in and reported on the result (:class:`Adapted`), because a chain whose
parameters are not stated cannot be reproduced. Without an
:class:`Adaptation` every draw is the one the same seed gave before,
bitwise. :func:`parallel_tempering` runs the same warm-up once per rung
before its rounds; :func:`anneal` re-tunes the step along its schedule,
which is a heuristic, since an annealing run has no stationary law to adapt
to (issue #1208).

**A supported objective anneals and tempers in ``oxisal``.** Where the
objective has :meth:`~sal.sample.declared.SupportedGradient.supported_gradient`
and the integrator is leapfrog, :func:`anneal` and :func:`parallel_tempering`
run :func:`sal.sample.loop.anneal` and :func:`~sal.sample.loop.temper` over a
step that advances an ``oxisal.HmcWalk`` one transition at the temperature it
is handed, and hands it another rung's position and carried ``(U, grad U)``
after an exchange (issue #1249). The loops are shared; the draws come from
the walk's ChaCha8 stream, so the torch route is matched in distribution,
as :func:`sample`'s compiled route is.

See Neal (2011), "MCMC using Hamiltonian dynamics"; Yoshida (1990) for the
fourth-order composition and Suzuki (1991) for why its middle coefficient
must be negative; Nocedal & Wright for the symplectic structure; Kirkpatrick,
Gelatt & Vecchi (1983) for annealing; Hoffman & Gelman (2014) for dual
averaging; Geyer (1992) for the effective sample size.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sal import oxisal
from sal.backend import Backend, refuse_backend
from sal.cost import Cost
from sal.emissions import ParameterDomainError
from sal.opt.objective import (
    Objective,
    value_and_gradient,
)
from sal.opt.termination import Termination
from sal.sample import loop
from sal.sample.accept import (
    accept_ratio,
    accept_with,
    acceptance_probability,
)
from sal.sample.chain import (
    BLOCK,
    DUAL_AVERAGING_GAMMA,
    DUAL_AVERAGING_KAPPA,
    DUAL_AVERAGING_T0,
    Adaptation,
    Adapted,
    Chain,
    DualAveraging,
    Kernel,
    Scaled,
    Transition,
    compiled_route,
    compiled_walk,
    gradient_at,
    jittered,
    run_chain,
    run_compiled,
    start_point,
    torch_stream,
    warm_up,
)
from sal.sample.declared import (
    Power,
    declared_energy,
    declared_jax_energy,
)
from sal.sample.hmc.jax import JaxWalk
from sal.sample.loop import Exchanging, Moved, temper
from sal.sample.schedule import (
    Annealed,
    Monotone,
    TempSchedule,
    check_ladder,
    ladder,
)
from sal.sample.schedule import Tempered as TemperedRun
from sal.sample.tempered import ExchangeSeries

# `current` is aliased: `_coefficients` already binds that name to a
# sub-step length, and one of the two has to give.
from sal.track import TrackedOptimization
from sal.track import current as current_tracked

#: The public surface, and the names #1010 moved to
#: :mod:`sal.sample.chain`, exported from here for one release.
__all__ = [
    "BLOCK",
    "DEFAULT_STEPS",
    "DUAL_AVERAGING_GAMMA",
    "DUAL_AVERAGING_KAPPA",
    "DUAL_AVERAGING_T0",
    "LEAPFROG_WEIGHTS",
    "YOSHIDA_WEIGHTS",
    "Adaptation",
    "Adapted",
    "AnnealedTheta",
    "Chain",
    "HmcChain",
    "Integrator",
    "Kernel",
    "PhaseSpace",
    "Tempered",
    "Transition",
    "WithGaussianPrior",
    "anneal",
    "compiled_trajectory",
    "effective_sample_size",
    "gradient_at",
    "hamiltonian",
    "leapfrog",
    "parallel_tempering",
    "run_chain",
    "run_compiled",
    "sample",
    "start_point",
    "yoshida",
]

DEFAULT_STEPS = 20


@dataclass(frozen=True)
class WithGaussianPrior(Objective):
    """An objective plus an isotropic Gaussian log prior, so it has a posterior.

    A bare negative log-likelihood read as a log density is a posterior under
    an improper flat prior, which for most models is not normalizable, and no
    diagnostic in :func:`sample` can tell. A proper prior fixes that, and
    stating it here keeps it the caller's declaration.

    Isotropic and centred on zero *in unconstrained coordinates*, which is
    weakly informative rather than uninformative --- on a log-simplex
    coordinate it pulls toward the uniform distribution, on a coupling toward
    zero. Any interval reported from a chain against it inherits that.

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
class HmcChain(Chain):
    """A Hamiltonian chain, and what it cost to get it (issue #1090).

    A :class:`~sal.sample.chain.Chain`, ``spent`` in gradients, warm-up and
    burn-in included, so an effective sample size divided by it is the cost
    of a draw and two chains compare at equal evaluations rather than equal
    samples. The acceptance rate reads as Hamiltonian dynamics has it:
    energy is conserved exactly, so a correct implementation with a small
    step accepts nearly everything, and a rate near zero means the
    integrator is diverging rather than that the target is hard. The
    energy error is the diagnostic that tells a step too large from a bug:
    the first grows smoothly with the step, the second does not.
    """


#: The cube root that Yoshida's fourth-order composition is built from.
_CUBE_ROOT_OF_TWO = 2.0 ** (1.0 / 3.0)

#: Yoshida's (1990) fourth-order weights: three second-order sub-steps of
#: lengths ``w1``, ``w0``, ``w1`` with ``2 * w1 + w0 == 1``.
#:
#: **``w0`` is negative --- the middle sub-step runs backwards in time.** That
#: is not a sign error and not avoidable: no composition of a second-order
#: symmetric method reaches fourth order with positive coefficients (Suzuki,
#: 1991). The trajectory is therefore non-monotone in time, which is worth
#: knowing before plotting one and concluding the integrator is broken.
YOSHIDA_WEIGHTS = (
    1.0 / (2.0 - _CUBE_ROOT_OF_TWO),
    -_CUBE_ROOT_OF_TWO / (2.0 - _CUBE_ROOT_OF_TWO),
    1.0 / (2.0 - _CUBE_ROOT_OF_TWO),
)

#: The trivial composition: one second-order sub-step of full length.
LEAPFROG_WEIGHTS = (1.0,)


@dataclass(frozen=True)
class PhaseSpace:
    """A point of the phase space an :class:`Integrator` moves through.

    Parameters
    ----------
    position : torch.Tensor
        ``theta``, 1-D.
    momentum : torch.Tensor
        ``p``, the same length. The trajectory's, not the negated one the
        acceptance ratio reads: the negation is the caller's, and the
        Hamiltonian is even in it.
    """

    position: torch.Tensor
    momentum: torch.Tensor

    def __iter__(self) -> Iterator[Any]:
        """``(position, momentum)``: the order callers unpack.

        ``Any`` for :meth:`Transition.__iter__`'s reason.
        """
        yield from (self.position, self.momentum)


@dataclass(frozen=True)
class Integrator:
    """A symplectic integrator, as the composition of sub-steps it is.

    Every method here is a composition of the second-order kick-drift-kick
    step, differing only in the sub-step lengths. Carrying the weights rather
    than a procedure buys:

    * **one implementation.** The kicks between adjacent sub-steps merge, so a
      hand-written composition is the same arithmetic with more places to put
      a wrong coefficient. :func:`leapfrog` is this object at ``(1.0,)`` and
      reproduces the previous implementation exactly;
    * **a declared cost**, since a comparison between integrators is only
      meaningful at equal force evaluations;
    * **a declared order**, so a test can assert the one it was built for
      rather than the one it happens to achieve.

    Parameters
    ----------
    name : str
        For error messages and reports.
    weights : tuple[float, ...]
        Sub-step lengths, summing to 1. Reversibility requires the sequence be
        a palindrome; both compositions here are.
    order : int
        The order of accuracy of the energy error in the step size.

    Raises
    ------
    ValueError
        If the weights do not sum to 1, or are not a palindrome. The first
        integrates the wrong amount of time while leaving reversibility
        intact, which is the one arithmetic slip a reversibility check does
        not catch.
    """

    name: str
    weights: tuple[float, ...]
    order: int

    def __post_init__(self) -> None:
        total = math.fsum(self.weights)
        if abs(total - 1.0) > 1e-12:
            msg = (
                f"{self.name}: sub-step weights sum to {total!r}, expected 1.0; "
                f"a composition that does not integrates the wrong interval "
                f"while remaining perfectly reversible"
            )
            raise ValueError(msg)
        if self.weights != tuple(reversed(self.weights)):
            msg = f"{self.name}: weights must be a palindrome to be reversible"
            raise ValueError(msg)

    def force_evaluations(self, n_steps: int, *, carried: bool = False) -> int:
        """Gradient evaluations one trajectory of ``n_steps`` costs.

        ``len(weights) * n_steps + 1``: the kicks at the join between two
        sub-steps merge into one, so a composition of ``s`` sub-steps costs
        ``s`` gradients per step rather than ``2 s``. ``carried`` is a
        trajectory whose first kick reads ``grad U`` at its start from the
        transition that landed there, and costs ``len(weights) * n_steps``
        (issue #1222).
        """
        return len(self.weights) * n_steps + (0 if carried else 1)

    def __call__(
        self,
        objective: Objective,
        theta: torch.Tensor,
        momentum: torch.Tensor,
        step_size: float,
        n_steps: int,
    ) -> PhaseSpace:
        """Integrate Hamiltonian dynamics over ``n_steps`` steps.

        Separated from :func:`sample` because its two defining properties ---
        reversibility and order of accuracy --- are exact statements testable
        without sampling. A distributional test says the chain is wrong; these
        say which half.

        Parameters
        ----------
        objective : Objective
            Read as a negative log density, so its gradient is the force.
        theta, momentum : torch.Tensor
            Position and momentum, both 1-D of the same length.
        step_size : float
            Integrator step. Energy error grows as its ``order`` power.
        n_steps : int
            Steps per trajectory.

        Returns
        -------
        PhaseSpace
            Position and momentum after ``n_steps``.
        """
        position = theta.detach()
        return self._scored(
            objective,
            position,
            momentum,
            step_size,
            n_steps,
            gradient_at(objective, position),
        )[0]

    def _scored(
        self,
        objective: Objective,
        theta: torch.Tensor,
        momentum: torch.Tensor,
        step_size: float,
        n_steps: int,
        force: torch.Tensor,
    ) -> tuple[PhaseSpace, float, torch.Tensor]:
        """:meth:`__call__` from ``force``, ``grad U`` at ``theta``; and ``(U, grad U)`` at the end point.

        The last kick takes the value beside the gradient through
        :func:`~sal.opt.objective.value_and_gradient`, so the acceptance
        test does not evaluate the end point a second time (issue #1217),
        and both are returned for the next transition to carry (issue
        #1222). The gradient is :func:`gradient_at`'s wherever that reads
        ``value_and_gradient``'s or a declared gradient alone, which is
        every objective but one that declares both by different arithmetic.
        """
        position = theta.detach().clone()
        velocity = momentum.detach().clone()
        kicks, drifts = _coefficients(self.weights, n_steps)

        velocity = velocity - kicks[0] * step_size * force
        for drift, kick in zip(drifts[:-1], kicks[1:-1], strict=True):
            position = position + drift * step_size * velocity
            velocity = velocity - kick * step_size * gradient_at(objective, position)
        position = position + drifts[-1] * step_size * velocity
        potential, force = value_and_gradient(objective, position)
        velocity = velocity - kicks[-1] * step_size * force
        return PhaseSpace(position=position, momentum=velocity), float(potential), force


def _coefficients(
    weights: tuple[float, ...], n_steps: int
) -> tuple[list[float], list[float]]:
    """Kick and drift coefficients for ``n_steps`` of a composition.

    Each sub-step is a kick-drift-kick of half, full, half its length, and
    the trailing half-kick of one sub-step merges with the leading half-kick
    of the next. That leaves one more kick than sub-steps, which is where
    :meth:`Integrator.force_evaluations` comes from.
    """
    sub_steps = list(weights) * n_steps
    kicks = [0.5 * sub_steps[0]]
    drifts = []
    for current, following in itertools.pairwise(sub_steps):
        drifts.append(current)
        kicks.append(0.5 * (current + following))
    drifts.append(sub_steps[-1])
    kicks.append(0.5 * sub_steps[-1])
    return kicks, drifts


#: The second-order kick-drift-kick method. The default everywhere, and the
#: reference every other integrator is checked against.
leapfrog = Integrator(name="leapfrog", weights=LEAPFROG_WEIGHTS, order=2)

#: Yoshida's fourth-order triple jump. Three force evaluations per step
#: against leapfrog's one, so whether it pays is a measurement at equal
#: evaluations rather than a consequence of the higher order.
yoshida = Integrator(name="yoshida", weights=YOSHIDA_WEIGHTS, order=4)


def hamiltonian(
    objective: Objective, theta: torch.Tensor, momentum: torch.Tensor
) -> float:
    """``U(theta) + K(momentum)``, the quantity the integrator conserves."""
    return _potential(objective, theta) + _kinetic(momentum)


def _potential(objective: Objective, theta: torch.Tensor) -> float:
    """``U(theta)``, by the objective's forward: :func:`hamiltonian`'s first term."""
    return float(objective(theta.detach()))


def _kinetic(momentum: torch.Tensor) -> float:
    """``K(p) = p'p / 2`` at unit mass: :func:`hamiltonian`'s second term."""
    return 0.5 * float((momentum * momentum).sum())


def sample(
    objective: Objective,
    rng: np.random.Generator | torch.Generator,
    n_samples: int,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    start: torch.Tensor | None = None,
    burn_in: int = 0,
    integrator: Integrator = leapfrog,
    temperature: float = 1.0,
    adaptation: Adaptation | None = None,
    store_chain: bool = True,
    operators: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    backend: Backend = Backend.RUST,
) -> HmcChain:
    """Draw ``n_samples`` from the density ``exp(-objective / temperature)``.

    Parameters
    ----------
    objective : Objective
        Read as an unnormalized negative log density.
    rng : np.random.Generator | torch.Generator
        The stream every momentum and acceptance draw comes from, passed in
        rather than seeded here (`sim/CLAUDE.md`): a chain is reproducible
        from ``torch.Generator().manual_seed(seed)`` at the call site, and two
        chains drawn from one generator are two chains.
    n_samples : int
        Draws recorded after burn-in.
    step_size : float
        Leapfrog step. Required rather than defaulted: its right value depends
        on the target's scale, so a default would be wrong silently. With an
        ``adaptation`` it is the warm-up's starting point.
    n_steps : int
        Leapfrog steps per proposal.
    start : torch.Tensor | None
        Starting point; ``objective.initial()`` when omitted.
    burn_in : int
        Draws discarded before recording.
    integrator : Integrator
        The symplectic method. ``leapfrog`` by default, which every chain in
        this repository was drawn with. A higher-order method takes more force
        evaluations per step, so choose it only against a comparison at equal
        evaluations.
    temperature : float
        The chain targets ``exp(-objective / temperature)``; 1 is the
        objective as declared. Whether that is a tempered *energy* or a power
        posterior is the caller's to say (`sal.sample.schedule`).
    adaptation : Adaptation | None
        A warm-up that sets the step size and the mass diagonal before the
        ``burn_in`` and the draws, both of which then run at fixed values.
        ``None`` runs the fixed-parameter chain at unit mass, bitwise what it
        was before adaptation existed.
    store_chain : bool
        Keep the draws (issue #988). ``False`` keeps none --- ``draws`` has
        zero rows --- and the chain holds memory of the order of one draw
        rather than ``n_samples`` of them; what it was for is then
        ``operators``' expectations.
    operators : Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None
        Functions of a draw, in the caller's coordinates, each fed to a
        :class:`~sal.sample.expectation.KalmanMean` at every
        recorded draw; the burn-in's are not observed. ``None`` observes
        nothing, the chain as it was.
    backend : Backend
        :data:`~sal.backend.Backend.RUST`, the default since
        issue #986, runs the whole chain in ``oxisal.HmcWalk`` (issue #1008)
        when the objective supports a kernel
        (:func:`~sal.sample.declared.declared_energy`) and the
        chain is leapfrog, at any temperature (issue #1220), in no enclosing
        :func:`sal.track.track`; the warm-up and ``Power``
        operators' filters run there too, and other operators as
        :func:`run_compiled` states. Its momenta and uniforms come
        from ChaCha8 seeded by one draw from ``generator``, so it is its own
        stream: reproducible from the generator, not the torch route's draws.
        Any other chain, and :data:`~sal.backend.Backend.PYTHON`
        always, is the torch route, whose integrator pins the compiled one.

    Returns
    -------
    HmcChain
        The draws, the acceptance rate, the per-proposal energy error, the
        gradients spent, what the warm-up settled on if there was one, and
        the operators' expectations.

    Raises
    ------
    ValueError
        If ``step_size`` or ``temperature`` is not positive, or ``n_steps``
        is below 1. A zero-length trajectory proposes the current point every
        time: it accepts at rate 1 and samples nothing, looking healthy by
        every diagnostic.
    """
    generator = torch_stream(rng)
    _check_trajectory(step_size, n_steps)
    refuse_backend("hmc.sample", backend, (Backend.PYTHON, Backend.RUST))
    declared = declared_energy(objective)
    traced = None if declared is not None else declared_jax_energy(objective)
    if (
        compiled_route(backend)
        and integrator is leapfrog
        and (declared is not None or traced is not None)
        and (
            declared is not None
            or all(isinstance(o, Power) for o in (operators or {}).values())
        )
    ):
        chain = run_compiled(
            oxisal.HmcWalk if declared is not None else JaxWalk,
            declared if declared is not None else traced,  # type: ignore[arg-type]
            (n_steps,),
            integrator.force_evaluations(n_steps, carried=True),
            generator,
            n_samples,
            per_start=1,
            unit=Cost.GRADIENTS,
            step_size=step_size,
            start=start_point(objective, start),
            burn_in=burn_in,
            temperature=temperature,
            adaptation=adaptation,
            store_chain=store_chain,
            operators=operators,
        )
    else:
        chain = run_chain(
            _HamiltonianKernel(n_steps=n_steps, integrator=integrator),
            integrator.force_evaluations(n_steps, carried=True),
            objective,
            generator,
            n_samples,
            per_start=1,
            unit=Cost.GRADIENTS,
            step_size=step_size,
            start=start,
            burn_in=burn_in,
            temperature=temperature,
            adaptation=adaptation,
            store_chain=store_chain,
            operators=operators,
        )
    return HmcChain(
        draws=chain.draws,
        acceptance_rate=chain.acceptance_rate,
        energy_error=chain.energy_error,
        spent=chain.spent,
        unit=chain.unit,
        adapted=chain.adapted,
        expectations=chain.expectations,
    )


@dataclass(frozen=True, kw_only=True)
class AnnealedTheta(Annealed[torch.Tensor]):
    """What one annealing run found, and what it cost (issue #1090).

    An :class:`~sal.sample.schedule.Annealed` over points in unconstrained
    coordinates: ``best`` is the lowest-valued point visited, ``final``
    where the chain ended, and ``spent`` the gradients, so the run is
    comparable to any other optimizer at equal evaluations.

    Parameters
    ----------
    value : float
        The objective at ``best``, which the objective minimizes.
    acceptance_rate : float
        Over the whole schedule. Near zero at the cold end is the symptom of
        a step too large for the final temperature.
    step_sizes : tuple[float, ...] | None
        With an ``adaptation``, the averaged step each window of
        ``adaptation.warmup`` proposals ended on, in schedule order; ``None``
        for a fixed step.
    """

    value: float
    acceptance_rate: float
    step_sizes: tuple[float, ...] | None = None


def anneal(
    objective: Objective,
    schedule: TempSchedule,
    rng: np.random.Generator | torch.Generator,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    start: torch.Tensor | None = None,
    integrator: Integrator = leapfrog,
    adaptation: Adaptation | None = None,
    backend: Backend = Backend.RUST,
) -> AnnealedTheta:
    """Simulated annealing with Hamiltonian proposals: :func:`sample` on a schedule.

    One proposal per schedule step at that step's temperature, tracking the
    lowest objective seen. Each transition is the one :func:`sample` runs at a
    constant temperature, so a constant schedule reproduces a chain draw for
    draw from generators seeded alike. What the falling temperature buys is
    measured against the alternatives at equal force evaluations.

    Parameters
    ----------
    objective : Objective
        What to minimize. Read as an energy, so ``T`` is physical; a negative
        log-likelihood here is a power posterior and the caller should know
        which they meant.
    schedule : TempSchedule
        Temperature per proposal. Its length is the budget in proposals;
        ``spent`` on the result is the budget in gradients.
    rng : np.random.Generator | torch.Generator
        As :func:`sample`.
    step_size, n_steps, start, integrator
        As :func:`sample`. The step needs no rescaling with temperature ---
        see the module note --- but a step that is stable at the hot end can
        still reject at the cold end, which the acceptance rate reports.
    adaptation : Adaptation | None
        Re-tunes the step along the schedule (issue #1208): dual averaging
        toward ``adaptation.target_acceptance`` runs over each window of
        ``adaptation.warmup`` proposals, restarted at every window from the
        averaged step the last one ended on, and every proposal's step is
        jittered by ``adaptation.step_jitter``. The mass stays unit, and
        ``step_size`` is the first window's starting point. **This is a
        heuristic, not a warm-up**: annealing is not a stationary chain, so
        no window's step is the one a fixed temperature would settle on, and
        the windows spend the schedule's proposals rather than discarded
        ones. ``None`` runs the fixed step, bitwise as before it existed.
    backend : Backend
        ``Backend.RUST``, the default, runs the schedule on an
        ``oxisal.HmcWalk`` where the objective supports a kernel, the
        integrator is leapfrog, ``adaptation`` is ``None`` and no run is
        tracked (issue #1249); otherwise, and always under
        ``Backend.PYTHON``, the torch transition runs it. The compiled walk
        draws from its own ChaCha8 stream, seeded by one draw of ``rng``.

    Returns
    -------
    AnnealedTheta
    """
    generator = torch_stream(rng)
    _check_trajectory(step_size, n_steps)
    refuse_backend("hmc.anneal", backend, (Backend.PYTHON, Backend.RUST))
    position = start_point(objective, start)
    declared = _compiled(objective, integrator, backend)
    if declared is not None and adaptation is None:
        compiled = _CompiledHamiltonian(
            compiled_walk(
                oxisal.HmcWalk,
                declared,
                (n_steps,),
                generator,
                step_size=step_size,
                start=position.numpy(),
                temperature=1.0,
                adaptation=None,
            ),
            integrator.force_evaluations(n_steps, carried=True),
        )
        # The walk evaluated `(U, grad U)` at the start: one gradient.
        ran = loop.anneal(compiled, schedule, compiled.start(1), generator, np.copy)
        return AnnealedTheta(
            best=torch.from_numpy(ran.best),
            value=ran.energy,
            final=torch.from_numpy(ran.final),
            acceptance_rate=compiled.accepted / schedule.n_steps,
            spent=ran.spent,
            unit=Cost.GRADIENTS,
            termination=ran.termination,
        )
    step = _Hamiltonian(
        objective,
        integrator,
        n_steps,
        step_size,
        jitter=None if adaptation is None else adaptation.step_jitter,
        tuning=adaptation,
        n_total=schedule.n_steps,
    )
    # `U` at the start; `grad U` is the first trajectory's, charged there.
    origin: Moved[torch.Tensor, torch.Tensor | None] = Moved(
        position, float(objective(position)), None, 0
    )
    walked = loop.anneal(step, schedule, origin, generator, torch.Tensor.clone)
    return AnnealedTheta(
        best=walked.best,
        value=walked.energy,
        final=walked.final,
        acceptance_rate=step.accepted / schedule.n_steps,
        spent=walked.spent,
        unit=Cost.GRADIENTS,
        termination=walked.termination,
        step_sizes=None if adaptation is None else tuple(step.step_sizes),
    )


@dataclass
class _Hamiltonian:
    """One Hamiltonian transition as a :data:`~sal.sample.loop.Step`, charged in gradients.

    Unadapted it carries ``grad U`` beside the position, so a trajectory
    costs ``integrator.force_evaluations(n_steps, carried=True)`` after the
    first (issue #1222). With an ``adapted`` warm-up it runs at that step on
    the metric ``Scaled(objective, scale)`` and evaluates ``U`` and ``grad U`` afresh at
    ``position / scale``, which may differ from the carried point in the last
    place; a rejection keeps ``position`` itself. A ``jitter`` draws every
    step; a ``tuning`` re-tunes it by dual averaging per window of
    ``tuning.warmup`` steps, as :func:`anneal` states.
    """

    objective: Objective
    integrator: Integrator
    n_steps: int
    step_size: float
    jitter: float | None = None
    adapted: Adapted | None = None
    tuning: Adaptation | None = None
    n_total: int = 0
    accepted: int = 0
    steps: int = 0
    step_sizes: list[float] = field(default_factory=list)
    scale: torch.Tensor | None = field(init=False, default=None)
    metric: Scaled | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.adapted is not None:
            self.step_size = self.adapted.step_size
            self.scale = self.adapted.mass_diagonal.rsqrt()
            self.metric = Scaled(self.objective, self.scale)
        self.averaging = (
            None
            if self.tuning is None
            else DualAveraging(self.step_size, self.tuning.target_acceptance)
        )

    def __call__(
        self,
        position: torch.Tensor,
        potential: float,
        force: torch.Tensor | None,
        temperature: float,
        generator: torch.Generator,
        /,
    ) -> Moved[torch.Tensor, torch.Tensor | None]:
        """One transition at ``temperature``."""
        size = (
            self.step_size
            if self.jitter is None
            else jittered(self.step_size, self.jitter, generator)
        )
        scale = self.scale
        # `force` is `None` on a metric: its steps carry nothing.
        taken, value, landed = _transition(
            self.objective if self.metric is None else self.metric,
            position if scale is None else position / scale,
            temperature,
            generator,
            size,
            self.n_steps,
            self.integrator,
            potential=potential if scale is None else None,
            force=force,
        )
        spent = self.integrator.force_evaluations(
            self.n_steps, carried=force is not None
        )
        moved = Moved(taken.position, value, landed, spent)
        if scale is not None:
            moved = Moved(position, potential, None, spent)
            if taken.accepted:
                moved = Moved(taken.position * scale, value, None, spent)
        self.accepted += taken.accepted
        self.steps += 1
        if self.averaging is not None and self.tuning is not None:
            # At a window's end the averaged step is kept and the averaging
            # restarts from it, the temperature it was tuned at having moved.
            self.step_size = self.averaging.update(taken.probability)
            if self.steps % self.tuning.warmup == 0 or self.steps == self.n_total:
                self.step_size = self.averaging.averaged
                self.step_sizes.append(self.step_size)
                self.averaging = DualAveraging(
                    self.step_size, self.tuning.target_acceptance
                )
        return moved


@dataclass(frozen=True, kw_only=True)
class Tempered(TemperedRun[torch.Tensor]):
    """What one parallel-tempering run found, and what it cost (issue #1090).

    A :class:`~sal.sample.schedule.Tempered` over points: ``best`` is the
    lowest-valued point visited at any temperature, ``spent`` the gradients
    over every replica, so the run is comparable to any other optimizer at
    equal evaluations, and ``swap_acceptance`` an array, the NumPy the
    exchange bookkeeping already kept.

    Parameters
    ----------
    value : float
        The objective at ``best``, which the objective minimizes.
    positions : torch.Tensor
        Every replica after every round, shape ``(n_rounds, n_replicas,
        dimension)``; replica ``r`` sits at ``temperatures[r]`` throughout,
        an exchange swapping *positions* between temperatures rather than
        moving a chain along the ladder. Recorded so a replica's marginal can
        be checked against the tempered target.
    acceptance_rate : torch.Tensor
        Fraction of Hamiltonian proposals accepted per replica, shape
        ``(n_replicas,)``.
    walkers : np.ndarray
        ``walkers[t, w]`` is the rung walker ``w`` sat at, at round ``t``,
        shape ``(n_rounds, n_replicas)``. The other reading of the same run:
        a *replica* is a temperature positions pass through, a *walker* is a
        position followed through the exchanges, and a round trip is a
        statement about the second.
        :func:`sal.sample.tempered.round_trips` and
        :func:`sal.sample.tempered.up_fraction` read it, as
        they read the trace the discrete temperings carry --- an integer
        trace rather than a tensor, because it is bookkeeping and nothing
        differentiates it (issue #861).
    adapted : tuple[Adapted, ...] | None
        What each rung's warm-up settled on, in the ladder's order, or
        ``None`` without an ``adaptation`` (issue #1208).
    """

    value: float
    positions: torch.Tensor
    acceptance_rate: torch.Tensor
    walkers: np.ndarray
    adapted: tuple[Adapted, ...] | None = None


def parallel_tempering(
    objective: Objective,
    temperatures: TempSchedule | Sequence[float],
    rng: np.random.Generator | torch.Generator,
    n_rounds: int,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    start: torch.Tensor | Sequence[torch.Tensor] | None = None,
    integrator: Integrator = leapfrog,
    adaptation: Adaptation | None = None,
    deadline: float | None = None,
    backend: Backend = Backend.RUST,
) -> Tempered:
    """Replicas at fixed temperatures, exchanging positions by Metropolis.

    Each round is one :func:`sample` transition per replica at its own
    temperature, then every adjacent pair proposes to exchange positions and
    accepts on ``(beta_i - beta_j)(U_i - U_j)``. The hot replicas cross
    barriers the cold one cannot, and an exchange carries what they find
    down the ladder (Swendsen & Wang, 1986; Geyer, 1991; Earl & Deem, 2005).

    **The exchange is one loop, and this is one of its instantiations**:
    :func:`sal.sample.loop.temper` runs the rounds,
    the swaps and the walker trace (issue #1218), and what is supplied here is
    the step --- one Hamiltonian transition per rung, carrying ``grad U`` ---
    and the draw the swap
    uniform comes from, which is this module's torch stream (issue #861).
    The discrete counterpart,
    :func:`sal.sample.potts_mcmc.parallel_tempering`, stays
    its own entry point over the same loop: it moves spins rather than a
    vector and is refereed against enumeration where this is refereed
    against quadrature.

    **The replicas must not share a stream and must be reproducible from one
    generator.** The caller's ``generator`` draws a seed per replica and then
    only the exchange uniforms, so the replicas are independent streams and
    one generator state reproduces the run. Sharing one stream would correlate
    them while every diagnostic looked healthy.

    Parameters
    ----------
    objective : Objective
        What to minimize. Read as an energy, so ``T`` is physical; a negative
        log-likelihood here is a power posterior and the caller should know
        which they meant.
    temperatures : TempSchedule | Sequence[float]
        The ladder, coldest first; at least two, all positive, strictly
        increasing so that adjacent pairs are the ones that exchange.
    rng : np.random.Generator | torch.Generator
        The parent stream, seeded by the caller (issue #337); it draws the
        replicas' seeds and the exchange uniforms.
    n_rounds : int
        Transitions per replica, at least one. The budget in proposals is
        ``n_rounds * len(temperatures)``; ``spent`` on the result
        is the budget in gradients.
    step_size, n_steps, integrator
        As :func:`sample`. With an ``adaptation``, ``step_size`` is every
        rung's warm-up starting point.
    start : torch.Tensor | Sequence[torch.Tensor] | None
        A tensor is the point every replica starts at; a sequence is one
        point per rung, in the ladder's order, as the Potts
        :func:`~sal.sample.potts_mcmc.parallel_tempering` takes one labelling
        per rung (issue #1157); ``None`` is ``objective.initial()`` for all.
    adaptation : Adaptation | None
        A warm-up per rung before the rounds (issue #1208): each replica runs
        :class:`Adaptation`'s two windows at its own temperature on its own
        stream, setting its own step and mass diagonal, and the rounds then
        run every rung at the values it settled on, each proposal's step
        jittered as the warm-up's was. No step is scaled with temperature:
        the step at which a chain's acceptance falls to 0.65 was measured
        flat to falling over ``T`` = 1 to 64 (#1195), so each rung adapts on
        its own. The warm-up stops before the first exchange, so the rounds
        are a fixed-parameter chain on the product law; its positions are
        discarded and its gradients are in ``spent``. ``None`` runs every rung
        at ``step_size`` and unit mass, bitwise as before it existed.
    deadline : float | None
        A :func:`time.perf_counter` reading. A round after the first starts
        only if the longest round so far would end by it, so ``n_rounds`` is
        a ceiling and the result's ``positions`` and ``force_evaluations``
        count the rounds run (issue #902). The first round always runs: a
        start is a point, and no round has yet been measured to predict it.
        A wall clock reads no replica's state, which is what makes it a stop
        a sweep may take. ``None`` runs ``n_rounds``, bitwise as before it
        existed.
    backend : Backend
        ``Backend.RUST``, the default, runs every rung on its own
        ``oxisal.HmcWalk`` where the objective supports a kernel, the
        integrator is leapfrog and no run is tracked (issue #1249): an
        exchange hands each walk the other's position and carried ``(U,
        grad U)``, and a rung's warm-up is the walk's own, at its temperature.
        A warmed-up rung then carries ``grad U`` across its metric, so its
        rounds cost ``n_steps`` gradients where the torch route's cost
        ``n_steps + 1``. Otherwise, and always under ``Backend.PYTHON``, the
        torch transition runs every rung.

    Returns
    -------
    Tempered

    Raises
    ------
    ValueError
        If fewer than two temperatures are given --- a ladder of one has
        nothing to exchange and is :func:`sample` --- if any is not positive
        or the ladder is not increasing, if ``n_rounds`` is below one, or if
        a sequence ``start`` does not give one point per rung.
    """
    generator = torch_stream(rng)
    _check_trajectory(step_size, n_steps)
    temperatures = check_ladder(
        ladder(temperatures),
        needed_by="parallel tempering",
        monotone=Monotone.INCREASING,
    )
    if n_rounds < 1:
        msg = f"n_rounds must be at least 1, got {n_rounds}"
        raise ValueError(msg)
    refuse_backend("hmc.parallel_tempering", backend, (Backend.PYTHON, Backend.RUST))

    parent = generator
    n_replicas = len(temperatures)
    children = [
        torch.Generator().manual_seed(int(child))
        for child in torch.randint(0, 2**31 - 1, (n_replicas,), generator=parent)
    ]
    origins = _rung_starts(objective, start, n_replicas)
    declared = _compiled(objective, integrator, backend)
    if declared is not None:
        return _tempered_compiled(
            declared,
            temperatures,
            parent,
            children,
            origins,
            n_rounds,
            step_size=step_size,
            n_steps=n_steps,
            adaptation=adaptation,
            deadline=deadline,
        )
    adapted: tuple[Adapted, ...] | None = None
    if adaptation is not None:
        kernel = _HamiltonianKernel(n_steps=n_steps, integrator=integrator)
        reports = []
        for index, (temperature, child) in enumerate(
            zip(temperatures, children, strict=True)
        ):
            report, origins[index] = warm_up(
                kernel,
                integrator.force_evaluations(n_steps, carried=True),
                1,
                objective,
                origins[index],
                temperature,
                child,
                step_size,
                adaptation,
            )
            reports.append(report)
        adapted = tuple(reports)
    # Each rung's metric is a change of coordinates (`Scaled`): a replica's
    # position is kept in `theta`, which is what an exchange swaps.
    jitter = None if adaptation is None else adaptation.step_jitter
    steps = [
        _Hamiltonian(objective, integrator, n_steps, step_size, jitter, report)
        for report in adapted or [None] * n_replicas
    ]
    # One evaluation where every replica starts at one point, as before
    # per-rung starts existed: a counted objective sees the same calls.
    shared = adapted is None and (start is None or isinstance(start, torch.Tensor))
    first = float(objective(origins[0])) if shared else math.nan
    starts: list[Moved[torch.Tensor, torch.Tensor | None]] = [
        Moved(origin.clone(), first if shared else float(objective(origin)), None, 0)
        for origin in origins
    ]
    # `energy` is the best value so far, which is `Tempered.value`; the rest
    # of the series is the exchange's (`ExchangeSeries`).
    tracked: TrackedOptimization = current_tracked()
    series = ExchangeSeries()
    recorded: list[torch.Tensor] = []

    def observe(sweep: int, run: Exchanging[torch.Tensor]) -> None:
        tracked.record(sweep, energy=run.energy)
        series(sweep, run)

    run = temper(
        steps,
        temperatures,
        starts,
        children,
        # One torch uniform per proposal, always drawn, on this module's stream.
        lambda ratio: accept_with(ratio, float(torch.rand(1, generator=parent))),
        n_rounds,
        keep=torch.Tensor.clone,
        record=lambda states, _: recorded.append(torch.stack(list(states))),
        observe=observe,
        stop=None if deadline is None else _before(deadline),
    )
    series.close(run)
    accepted = torch.tensor([float(s.accepted) for s in steps], dtype=torch.float64)
    rounds = torch.stack(recorded)
    tracked.record_cost(max(run.rounds - 1, 0), rounds.nbytes)
    return Tempered(
        best=run.best,
        value=run.energy,
        positions=rounds,
        acceptance_rate=accepted / run.rounds,
        temperatures=tuple(temperatures),
        swap_acceptance=run.swap_acceptance,
        # The warm-ups are charged beside the rungs' own trajectories.
        spent=run.spent + sum(report.force_evaluations for report in adapted or ()),
        unit=Cost.GRADIENTS,
        termination=Termination.after(run.rounds, converged=False),
        walkers=run.walkers,
        adapted=adapted,
    )


def _compiled(
    objective: Objective, integrator: Integrator, backend: Backend
) -> tuple[str, Mapping[str, Any]] | None:
    """The supported kernel an anneal or a tempering runs compiled, or ``None`` for the torch route."""
    if compiled_route(backend) and integrator is leapfrog:
        return declared_energy(objective)
    return None


@dataclass
class _CompiledHamiltonian:
    """One ``oxisal.HmcWalk`` transition as a :data:`~sal.sample.loop.Step` (issue #1249).

    The walk keeps where it is and the ``(U, grad U)`` it carries there; a
    state the step did not hand back last --- another rung's, after an
    exchange --- is handed to the walk with what it carries, so nothing is
    evaluated afresh. The stream argument is unread: the walk draws from its own
    stream. Charged ``per_proposal`` gradients, the carried trajectory's.
    """

    walk: Any
    per_proposal: int
    accepted: int = 0
    _at: np.ndarray | None = field(default=None, init=False, repr=False)

    def start(self, spent: int) -> Moved[np.ndarray, np.ndarray]:
        """Where the walk is and what it carries, charged ``spent``."""
        position, value, gradient = self.walk.state()
        self._at = position
        return Moved(position, value, gradient, spent)

    def __call__(
        self,
        position: np.ndarray,
        potential: float,
        force: np.ndarray,
        temperature: float,
        _generator: torch.Generator,
        /,
    ) -> Moved[np.ndarray, np.ndarray]:
        """One transition at ``temperature`` from ``position``."""
        if position is not self._at:
            self.walk.set_state(position, potential, force)
        landed, value, gradient, taken = self.walk.advance_at(1, temperature)
        self._at = landed
        self.accepted += taken
        return Moved(landed, value, gradient, self.per_proposal)


def _tempered_compiled(
    declared: tuple[str, Mapping[str, Any]],
    temperatures: Sequence[float],
    parent: torch.Generator,
    children: Sequence[torch.Generator],
    origins: Sequence[torch.Tensor],
    n_rounds: int,
    *,
    step_size: float,
    n_steps: int,
    adaptation: Adaptation | None,
    deadline: float | None,
) -> Tempered:
    """:func:`parallel_tempering` on one ``oxisal.HmcWalk`` per rung, over :func:`~sal.sample.loop.temper` (issue #1249).

    Each rung's walk is seeded by one draw of its child stream and runs its
    warm-up, where there is one, at its own temperature on construction;
    the exchange uniforms are drawn from ``parent`` as the torch route draws
    them. A walk evaluates ``(U, grad U)`` at its start once, charged to its
    warm-up where there is one, as :func:`~sal.sample.chain.run_compiled`
    charges it.
    """
    per_proposal = leapfrog.force_evaluations(n_steps, carried=True)
    steps = [
        _CompiledHamiltonian(
            compiled_walk(
                oxisal.HmcWalk,
                declared,
                (n_steps,),
                child,
                step_size=step_size,
                start=origin.numpy(),
                temperature=temperature,
                adaptation=adaptation,
            ),
            per_proposal,
        )
        for origin, temperature, child in zip(
            origins, temperatures, children, strict=True
        )
    ]
    warmup = 0 if adaptation is None else adaptation.warmup * per_proposal + 1
    adapted = (
        None
        if adaptation is None
        else tuple(
            Adapted(
                step_size=step.walk.step_size,
                mass_diagonal=torch.from_numpy(step.walk.mass_diagonal),
                warmup_acceptance=step.walk.warmup_acceptance,
                force_evaluations=warmup,
                flat=tuple(step.walk.flat),
            )
            for step in steps
        )
    )
    starts = [step.start(1 if adaptation is None else 0) for step in steps]
    tracked: TrackedOptimization = current_tracked()
    series = ExchangeSeries()
    recorded: list[np.ndarray] = []

    def observe(sweep: int, run: Exchanging[np.ndarray]) -> None:
        tracked.record(sweep, energy=run.energy)
        series(sweep, run)

    run = temper(
        steps,
        temperatures,
        starts,
        children,
        lambda ratio: accept_with(ratio, float(torch.rand(1, generator=parent))),
        n_rounds,
        keep=np.copy,
        record=lambda states, _: recorded.append(np.stack(states)),
        observe=observe,
        stop=None if deadline is None else _before(deadline),
    )
    series.close(run)
    accepted = torch.tensor([float(s.accepted) for s in steps], dtype=torch.float64)
    rounds = torch.from_numpy(np.stack(recorded))
    tracked.record_cost(max(run.rounds - 1, 0), rounds.nbytes)
    return Tempered(
        best=torch.from_numpy(run.best),
        value=run.energy,
        positions=rounds,
        acceptance_rate=accepted / run.rounds,
        temperatures=tuple(temperatures),
        swap_acceptance=run.swap_acceptance,
        spent=run.spent + warmup * len(steps),
        unit=Cost.GRADIENTS,
        termination=Termination.after(run.rounds, converged=False),
        walkers=run.walkers,
        adapted=adapted,
    )


def _rung_starts(
    objective: Objective,
    start: torch.Tensor | Sequence[torch.Tensor] | None,
    n_replicas: int,
) -> list[torch.Tensor]:
    """One starting point per rung: ``start`` for each, or the sequence's own."""
    if start is None or isinstance(start, torch.Tensor):
        origin = start_point(objective, start)
        return [origin.clone() for _ in range(n_replicas)]
    if len(start) != n_replicas:
        msg = f"start gives {len(start)} points for a ladder of {n_replicas} rungs"
        raise ValueError(msg)
    return [start_point(objective, point) for point in start]


def _before(deadline: float) -> Callable[[], bool]:
    """A stop that ends the rounds when the longest so far would pass ``deadline``.

    Asked between rounds, so the interval between two askings is one round
    and the first is timed from here, which the caller builds just before
    the loop starts.
    """
    mark = time.perf_counter()
    longest = 0.0

    def stop() -> bool:
        nonlocal mark, longest
        now = time.perf_counter()
        longest = max(longest, now - mark)
        mark = now
        return now + longest > deadline

    return stop


def _check_trajectory(step_size: float, n_steps: int) -> None:
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


@dataclass
class _HamiltonianKernel:
    """:func:`_transition` with its trajectory bound: :func:`sample`'s :class:`Kernel`.

    The current point's ``(U, grad U)`` is kept beside the tensor it was
    taken at and handed to the next transition while the chain has not left
    it, as :class:`~sal.sample.metropolis` keeps its energy: the identity of
    the objective and of the tensor is the key, so the warm-up's change of
    coordinates, or a new position, is evaluated afresh (issues #1217,
    #1222). A chain therefore evaluates ``grad U`` at a start once per
    objective it is handed --- once unadapted, three times with a warm-up
    --- which :func:`~sal.sample.chain.run_chain`'s ``per_start`` charges.
    """

    n_steps: int
    integrator: Integrator
    _at: tuple[Objective, torch.Tensor, float, torch.Tensor | None] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __call__(
        self,
        objective: Objective,
        position: torch.Tensor,
        temperature: float,
        generator: torch.Generator,
        step_size: float,
    ) -> Transition:
        held = (
            self._at
            if self._at is not None
            and self._at[0] is objective
            and self._at[1] is position
            else None
        )
        step, potential, force = _transition(
            objective,
            position,
            temperature,
            generator,
            step_size,
            self.n_steps,
            self.integrator,
            potential=None if held is None else held[2],
            force=None if held is None else held[3],
        )
        self._at = (objective, step.position, potential, force)
        return step


def _transition(
    objective: Objective,
    position: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
    step_size: float,
    n_steps: int,
    integrator: Integrator,
    *,
    potential: float | None = None,
    force: torch.Tensor | None = None,
) -> tuple[Transition, float, torch.Tensor | None]:
    """One Metropolis step with a Hamiltonian proposal at ``temperature``.

    Momentum is drawn with variance ``temperature`` and the acceptance ratio
    divides the energy difference by it; the integrator itself is untempered.
    At ``temperature = 1.0`` both are the identity bitwise, so this *is* the
    untempered transition and not an approximation of it.

    ``U`` and ``grad U`` are evaluated nowhere the chain has already
    evaluated them (issues #1217, #1222): at the current point they are
    ``potential`` and ``force`` where the caller carries them, and at the
    proposal they are the integrator's last force evaluation's. A carried
    ``force`` spares the trajectory's first gradient, so the step costs
    ``integrator.force_evaluations(n_steps, carried=True)``.

    Parameters
    ----------
    potential : float | None
        ``U(position)`` where the caller holds it, from the transition that
        landed there; ``None`` evaluates it.
    force : torch.Tensor | None
        ``grad U(position)`` on the same terms; ``None`` evaluates it.

    Returns
    -------
    tuple[Transition, float, torch.Tensor | None]
        The new position, the absolute energy error of the proposal, 1 if
        it was accepted, and the Metropolis acceptance probability
        ``min(1, exp(-dH / T))`` --- the statistic dual averaging drives,
        which has less variance than the accept/reject outcome. A proposal
        whose energy is not finite has probability 0, and so does one whose
        trajectory left the objective's domain
        (:class:`~sal.emissions.ParameterDomainError`). Beside it, ``U`` and
        ``grad U`` at the new position, for the caller to carry into the
        next step; the gradient is ``None`` only where the current point's
        own raised :class:`~sal.emissions.ParameterDomainError`.
    """
    momentum = torch.randn(
        position.shape, generator=generator, dtype=torch.float64
    ) * math.sqrt(temperature)
    here = _potential(objective, position) if potential is None else potential
    current = here + _kinetic(momentum)

    start = force
    try:
        if start is None:
            start = gradient_at(objective, position.detach())
        trajectory, there, landed = integrator._scored(
            objective, position, momentum, step_size, n_steps, start
        )
        proposal = trajectory.position
        # Negating the momentum makes the proposal symmetric, which is what
        # leaves the acceptance ratio as the energy difference alone. It has
        # no effect on the next iteration, where the momentum is redrawn.
        proposed = there + _kinetic(-trajectory.momentum)
    except ParameterDomainError:
        # A divergent trajectory: it drove a parameter out of the family's
        # domain, where the energy does not exist. Rejected as a proposal of
        # infinite energy is, the uniform drawn as that path draws it (#912).
        torch.rand(1, generator=generator)
        return (
            Transition(
                position=position, energy_error=math.inf, accepted=0, probability=0.0
            ),
            here,
            start,
        )

    error = abs(proposed - current)
    uniform = float(torch.rand(1, generator=generator))
    ratio = float(torch.exp(torch.tensor((current - proposed) / temperature)))
    probability = acceptance_probability(ratio)
    if accept_ratio(ratio, uniform):
        return (
            Transition(
                position=proposal,
                energy_error=error,
                accepted=1,
                probability=probability,
            ),
            there,
            landed,
        )
    return (
        Transition(
            position=position, energy_error=error, accepted=0, probability=probability
        ),
        here,
        start,
    )


def compiled_trajectory(
    objective: Objective,
    position: torch.Tensor,
    momentum: torch.Tensor,
    step_size: float,
    n_steps: int,
) -> PhaseSpace:
    """:func:`leapfrog` on a supported kernel, in ``oxisal`` at unit mass (issues #986, #1008, #1220).

    The trajectory is the one arithmetic the compiled chain and the torch
    route share, so it is what pins ``src/hmc.rs`` to :func:`leapfrog` step
    for step; the chains themselves differ in their streams.

    Raises
    ------
    TypeError
        If the objective supports no kernel
        (:func:`~sal.sample.declared.declared_energy`).
    """
    declared = declared_energy(objective)
    if declared is None:
        msg = f"{type(objective).__name__} supports no kernel a compiled trajectory can run"
        raise TypeError(msg)
    end, velocity = oxisal.leapfrog_trajectory(
        declared[0],
        declared[1],
        np.ascontiguousarray(position.detach().numpy(), dtype=np.float64),
        np.ascontiguousarray(momentum.detach().numpy(), dtype=np.float64),
        step_size,
        n_steps,
    )
    return PhaseSpace(torch.from_numpy(end), torch.from_numpy(velocity))


def effective_sample_size(draws: torch.Tensor | np.ndarray) -> np.ndarray:
    """Effective sample size per coordinate, by Geyer's initial positive sequence.

    The integrated autocorrelation time ``tau = 1 + 2 sum_k rho_k`` is
    estimated by summing the autocorrelations in adjacent pairs
    ``Gamma_k = rho_2k + rho_2k+1`` and stopping at the first non-positive
    pair (Geyer, 1992, §3.3): for a reversible chain every ``Gamma_k`` is
    positive, so the first non-positive one is noise. The size is ``n / tau``,
    and divided by the gradients a chain cost it is the number two samplers
    are compared on.

    Parameters
    ----------
    draws : torch.Tensor | np.ndarray
        Shape ``(n, dimension)``, one chain.

    Returns
    -------
    np.ndarray
        Shape ``(dimension,)``, a diagnostic and so NumPy, whichever the draws
        came as (issue #1092). A coordinate that did not move has no
        autocorrelation and is reported as ``n``.

    Raises
    ------
    ValueError
        If fewer than 4 draws are given, which is fewer than the two pairs
        the truncation rule needs.
    """
    draws = torch.as_tensor(draws)
    n = int(draws.shape[0])
    if n < 4:
        msg = f"effective sample size needs at least 4 draws, got {n}"
        raise ValueError(msg)
    centred = draws.to(torch.float64) - draws.to(torch.float64).mean(dim=0)
    padded = 1 << (2 * n - 1).bit_length()
    spectrum = torch.fft.rfft(centred, n=padded, dim=0)
    autocovariance = torch.fft.irfft(spectrum * spectrum.conj(), n=padded, dim=0)[:n]
    autocovariance = autocovariance / n

    sizes = torch.empty(draws.shape[1], dtype=torch.float64)
    for coordinate in range(draws.shape[1]):
        if not autocovariance[0, coordinate] > 0.0:
            sizes[coordinate] = float(n)
            continue
        rho = autocovariance[:, coordinate] / autocovariance[0, coordinate]
        pairs = rho[0 : n - n % 2 : 2] + rho[1 : n - n % 2 : 2]
        negative = torch.nonzero(pairs <= 0.0).flatten()
        cutoff = int(negative[0]) if negative.numel() else int(pairs.shape[0])
        tau = -1.0 + 2.0 * float(pairs[:cutoff].sum())
        sizes[coordinate] = n / tau
    return sizes.numpy()
