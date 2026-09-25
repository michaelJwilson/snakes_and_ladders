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
diagonal from the warm-up sample variance and the step size by dual averaging
toward a stated acceptance (Hoffman & Gelman, 2014, §3.2;
``eq:dual-averaging``), then draws the chain at those *fixed* values, so the
draws are a Markov chain with the target as its stationary distribution.
Opt-in and reported on the result (:class:`Adapted`), because a chain whose
parameters are not stated cannot be reproduced. Without an
:class:`Adaptation` every draw is the one the same seed gave before,
bitwise.

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
from sal.emissions import ParameterDomainError
from sal.opt.objective import (
    Objective,
)
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
    Kernel,
    Transition,
    compiled_route,
    gradient_at,
    run_chain,
    run_compiled,
    start_point,
)
from sal.sample.declared import (
    Power,
    declared_energy,
    declared_jax_energy,
)
from sal.sample.expectation import Expectation
from sal.sample.hmc_jax import JaxWalk
from sal.sample.schedule import (
    Monotone,
    TempSchedule,
    check_ladder,
    ladder,
)
from sal.sample.tempered import exchange

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
    "Annealed",
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
class HmcChain:
    """A chain, and what it cost to get it.

    Parameters
    ----------
    draws : torch.Tensor
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
    force_evaluations : int
        Gradients spent, warm-up and burn-in included, so an effective
        sample size divided by it is the cost of a draw and two chains
        compare at equal evaluations rather than equal samples.
    adapted : Adapted | None
        What the warm-up settled on, when :func:`sample` was given an
        :class:`Adaptation`; ``None`` for a fixed-parameter chain.
    expectations : Mapping[str, Expectation]
        Each operator's expectation over the recorded draws, by the Kalman
        filter of :mod:`sal.sample.expectation`, keyed as the
        ``operators`` given to :func:`sample`; empty when none were.
    """

    draws: torch.Tensor
    acceptance_rate: float
    energy_error: torch.Tensor
    force_evaluations: int
    adapted: Adapted | None
    expectations: Mapping[str, Expectation] = field(default_factory=dict)


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

    def force_evaluations(self, n_steps: int) -> int:
        """Gradient evaluations one trajectory of ``n_steps`` costs.

        ``len(weights) * n_steps + 1``: the kicks at the join between two
        sub-steps merge into one, so a composition of ``s`` sub-steps costs
        ``s`` gradients per step rather than ``2 s``.
        """
        return len(self.weights) * n_steps + 1

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
        position = theta.detach().clone()
        velocity = momentum.detach().clone()
        kicks, drifts = _coefficients(self.weights, n_steps)

        velocity = velocity - kicks[0] * step_size * gradient_at(objective, position)
        for drift, kick in zip(drifts, kicks[1:], strict=True):
            position = position + drift * step_size * velocity
            velocity = velocity - kick * step_size * gradient_at(objective, position)
        return PhaseSpace(position=position, momentum=velocity)


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
    potential = float(objective(theta.detach()))
    kinetic = 0.5 * float((momentum * momentum).sum())
    return potential + kinetic


def sample(
    objective: Objective,
    generator: torch.Generator,
    n_samples: int,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    theta0: torch.Tensor | None = None,
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
    generator : torch.Generator
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
    theta0 : torch.Tensor | None
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
        when the objective declares an energy
        (:func:`~sal.sample.declared.declared_energy`) and the
        chain is leapfrog at unit temperature in no enclosing
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
    _check_trajectory(step_size, n_steps)
    refuse_backend("hmc.sample", backend, (Backend.PYTHON, Backend.RUST))
    declared = declared_energy(objective)
    traced = None if declared is not None else declared_jax_energy(objective)
    if (
        compiled_route(backend, temperature)
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
            integrator.force_evaluations(n_steps),
            generator,
            n_samples,
            step_size=step_size,
            theta0=start_point(objective, theta0),
            burn_in=burn_in,
            adaptation=adaptation,
            store_chain=store_chain,
            operators=operators,
        )
    else:
        chain = run_chain(
            _HamiltonianKernel(n_steps=n_steps, integrator=integrator),
            integrator.force_evaluations(n_steps),
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
    return HmcChain(
        draws=chain.draws,
        acceptance_rate=chain.acceptance_rate,
        energy_error=chain.energy_error,
        force_evaluations=chain.force_evaluations,
        adapted=chain.adapted,
        expectations=chain.expectations,
    )


@dataclass(frozen=True)
class Annealed:
    """What one annealing run found, and what it cost.

    Parameters
    ----------
    theta : torch.Tensor
        The lowest-valued point visited, in unconstrained coordinates. The
        *best* rather than the last: the final proposals run cold but not at
        zero, so the chain can leave its best point.
    value : float
        The objective there.
    final : torch.Tensor
        Where the chain ended.
    acceptance_rate : float
        Over the whole schedule. Near zero at the cold end is the symptom of
        a step too large for the final temperature.
    force_evaluations : int
        Gradients spent, so the run is comparable to any other optimizer at
        equal evaluations.
    """

    theta: torch.Tensor
    value: float
    final: torch.Tensor
    acceptance_rate: float
    force_evaluations: int


def anneal(
    objective: Objective,
    schedule: TempSchedule,
    generator: torch.Generator,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    theta0: torch.Tensor | None = None,
    integrator: Integrator = leapfrog,
) -> Annealed:
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
        ``force_evaluations`` on the result is the budget in gradients.
    generator : torch.Generator
        As :func:`sample`.
    step_size, n_steps, theta0, integrator
        As :func:`sample`. The step needs no rescaling with temperature ---
        see the module note --- but a step that is stable at the hot end can
        still reject at the cold end, which the acceptance rate reports.

    Returns
    -------
    Annealed
    """
    _check_trajectory(step_size, n_steps)
    position = start_point(objective, theta0)

    best, best_value = position.clone(), float(objective(position))
    accepted = 0
    # `energy` is the best value so far, which is what `Annealed.value`
    # returns: the series ends at the field rather than at the last visited
    # point, which the result does not report. `best` is the state passed,
    # for the same reason.
    tracked: TrackedOptimization = current_tracked()
    for step in range(schedule.n_steps):
        temperature = schedule(step)
        taken = _transition(
            objective,
            position,
            temperature,
            generator,
            step_size,
            n_steps,
            integrator,
        )
        position = taken.position
        accepted += taken.accepted
        value = float(objective(position))
        if value < best_value:
            best, best_value = position.clone(), value
        tracked.record(step, state=best, temperature=temperature, energy=best_value)
    tracked.record_cost(max(schedule.n_steps - 1, 0), best.nbytes)
    return Annealed(
        theta=best,
        value=best_value,
        final=position,
        acceptance_rate=accepted / schedule.n_steps,
        force_evaluations=schedule.n_steps * integrator.force_evaluations(n_steps),
    )


@dataclass(frozen=True)
class Tempered:
    """What one parallel-tempering run found, and what it cost.

    Parameters
    ----------
    theta : torch.Tensor
        The lowest-valued point visited at any temperature.
    value : float
        The objective there.
    positions : torch.Tensor
        Every replica after every round, shape ``(n_rounds, n_replicas,
        dimension)``; replica ``r`` sits at ``temperatures[r]`` throughout,
        an exchange swapping *positions* between temperatures rather than
        moving a chain along the ladder. Recorded so a replica's marginal can
        be checked against the tempered target.
    acceptance_rate : torch.Tensor
        Fraction of Hamiltonian proposals accepted per replica, shape
        ``(n_replicas,)``.
    swap_acceptance : torch.Tensor
        Fraction of proposed exchanges accepted per adjacent pair, shape
        ``(n_replicas - 1,)``. Near zero means the ladder has a gap no
        position crosses and the replicas are independent chains; near one
        means two temperatures are close enough that one is redundant.
    force_evaluations : int
        Gradients spent over every replica, so the run is comparable to any
        other optimizer at equal evaluations.
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
    """

    theta: torch.Tensor
    value: float
    positions: torch.Tensor
    acceptance_rate: torch.Tensor
    swap_acceptance: torch.Tensor
    force_evaluations: int
    walkers: np.ndarray


def parallel_tempering(
    objective: Objective,
    temperatures: TempSchedule | Sequence[float],
    generator: torch.Generator,
    n_rounds: int,
    *,
    step_size: float,
    n_steps: int = DEFAULT_STEPS,
    theta0: torch.Tensor | None = None,
    integrator: Integrator = leapfrog,
    deadline: float | None = None,
) -> Tempered:
    """Replicas at fixed temperatures, exchanging positions by Metropolis.

    Each round is one :func:`sample` transition per replica at its own
    temperature, then every adjacent pair proposes to exchange positions and
    accepts on ``(beta_i - beta_j)(U_i - U_j)``. The hot replicas cross
    barriers the cold one cannot, and an exchange carries what they find
    down the ladder (Swendsen & Wang, 1986; Geyer, 1991; Earl & Deem, 2005).

    **The exchange is one loop, and this is one of its instantiations**:
    :func:`sal.sample.tempered.exchange` runs the rounds,
    the swaps and the walker trace, and what is supplied here is the step
    --- one Hamiltonian transition per replica --- and the draw the swap
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
    generator : torch.Generator
        The parent stream, seeded by the caller (issue #337); it draws the
        replicas' seeds and the exchange uniforms.
    n_rounds : int
        Transitions per replica, at least one. The budget in proposals is
        ``n_rounds * len(temperatures)``; ``force_evaluations`` on the result
        is the budget in gradients.
    step_size, n_steps, theta0, integrator
        As :func:`sample`; every replica starts at ``theta0``.
    deadline : float | None
        A :func:`time.perf_counter` reading. A round after the first starts
        only if the longest round so far would end by it, so ``n_rounds`` is
        a ceiling and the result's ``positions`` and ``force_evaluations``
        count the rounds run (issue #902). The first round always runs: a
        start is a point, and no round has yet been measured to predict it.
        A wall clock reads no replica's state, which is what makes it a stop
        a sweep may take. ``None`` runs ``n_rounds``, bitwise as before it
        existed.

    Returns
    -------
    Tempered

    Raises
    ------
    ValueError
        If fewer than two temperatures are given --- a ladder of one has
        nothing to exchange and is :func:`sample` --- if any is not positive
        or the ladder is not increasing, or if ``n_rounds`` is below one.
    """
    _check_trajectory(step_size, n_steps)
    temperatures = check_ladder(
        ladder(temperatures),
        needed_by="parallel tempering",
        monotone=Monotone.INCREASING,
    )
    if n_rounds < 1:
        msg = f"n_rounds must be at least 1, got {n_rounds}"
        raise ValueError(msg)

    parent = generator
    n_replicas = len(temperatures)
    children = [
        torch.Generator().manual_seed(int(child))
        for child in torch.randint(0, 2**31 - 1, (n_replicas,), generator=parent)
    ]
    start = start_point(objective, theta0)
    positions = [start.clone() for _ in range(n_replicas)]
    value = float(objective(start))
    best, best_value = start.clone(), value
    accepted = torch.zeros(n_replicas, dtype=torch.float64)
    # A list rather than a tensor sized to `n_rounds`: under a deadline that
    # count is a ceiling, and the stacked rounds are the same values.
    recorded: list[torch.Tensor] = []
    # The ladder is strictly increasing, so a replica's temperature names it:
    # the loop hands the step a temperature and a generator, not an index.
    rung = {temperature: index for index, temperature in enumerate(temperatures)}
    round_index = 0
    # `energy` is the best value so far, which is `Tempered.value`; the rest
    # of the series is the exchange loop's, recorded where it runs. A
    # per-replica energy series is not recorded because the result reports
    # the minimum over replicas.
    tracked: TrackedOptimization = current_tracked()

    def transition(
        position: torch.Tensor,
        _: float,
        temperature: float,
        child: torch.Generator,
    ) -> tuple[torch.Tensor, float]:
        """One Hamiltonian transition, and the log-density where it landed."""
        step = _transition(
            objective, position, temperature, child, step_size, n_steps, integrator
        )
        accepted[rung[temperature]] += step.accepted
        return step.position, -float(objective(step.position))

    def observe(states: Sequence[torch.Tensor], densities: Sequence[float]) -> None:
        """The round the exchange closed: the positions it left, and the best point seen."""
        nonlocal best, best_value, round_index
        recorded.append(torch.stack(list(states)))
        lowest = max(range(n_replicas), key=densities.__getitem__)
        if -densities[lowest] < best_value:
            best, best_value = states[lowest].clone(), -densities[lowest]
        tracked.record(round_index, energy=best_value)
        round_index += 1

    def swap(log_ratio: float) -> bool:
        """The exchange's accept step on this module's stream: one torch uniform, always drawn."""
        return accept_with(log_ratio, float(torch.rand(1, generator=parent)))

    ensemble = exchange(
        transition,
        None,
        positions,
        [-value] * n_replicas,
        temperatures,
        children,
        swap,
        n_rounds,
        0,
        1,
        observe,
        None if deadline is None else _before(deadline),
    )
    rounds = torch.stack(recorded)
    # The loop records the ensemble it built; the tempering records the
    # positions it returns, which is the state its result holds.
    tracked.record_cost(max(round_index - 1, 0), rounds.nbytes)

    return Tempered(
        theta=best,
        value=best_value,
        positions=rounds,
        acceptance_rate=accepted / round_index,
        swap_acceptance=torch.tensor(ensemble.swap_acceptance, dtype=torch.float64),
        force_evaluations=round_index
        * n_replicas
        * integrator.force_evaluations(n_steps),
        walkers=ensemble.walkers,
    )


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


@dataclass(frozen=True)
class _HamiltonianKernel:
    """:func:`_transition` with its trajectory bound: :func:`sample`'s :class:`Kernel`."""

    n_steps: int
    integrator: Integrator

    def __call__(
        self,
        objective: Objective,
        position: torch.Tensor,
        temperature: float,
        generator: torch.Generator,
        step_size: float,
    ) -> Transition:
        return _transition(
            objective,
            position,
            temperature,
            generator,
            step_size,
            self.n_steps,
            self.integrator,
        )


def _transition(
    objective: Objective,
    position: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
    step_size: float,
    n_steps: int,
    integrator: Integrator,
) -> Transition:
    """One Metropolis step with a Hamiltonian proposal at ``temperature``.

    Momentum is drawn with variance ``temperature`` and the acceptance ratio
    divides the energy difference by it; the integrator itself is untempered.
    At ``temperature = 1.0`` both are the identity bitwise, so this *is* the
    untempered transition and not an approximation of it.

    Returns
    -------
    Transition
        The new position, the absolute energy error of the proposal, 1 if
        it was accepted, and the Metropolis acceptance probability
        ``min(1, exp(-dH / T))`` --- the statistic dual averaging drives,
        which has less variance than the accept/reject outcome. A proposal
        whose energy is not finite has probability 0, and so does one whose
        trajectory left the objective's domain
        (:class:`~sal.emissions.ParameterDomainError`).
    """
    momentum = torch.randn(
        position.shape, generator=generator, dtype=torch.float64
    ) * math.sqrt(temperature)
    current = hamiltonian(objective, position, momentum)

    try:
        trajectory = integrator(objective, position, momentum, step_size, n_steps)
        proposal = trajectory.position
        # Negating the momentum makes the proposal symmetric, which is what
        # leaves the acceptance ratio as the energy difference alone. It has
        # no effect on the next iteration, where the momentum is redrawn.
        proposed = hamiltonian(objective, proposal, -trajectory.momentum)
    except ParameterDomainError:
        # A divergent trajectory: it drove a parameter out of the family's
        # domain, where the energy does not exist. Rejected as a proposal of
        # infinite energy is, the uniform drawn as that path draws it (#912).
        torch.rand(1, generator=generator)
        return Transition(
            position=position, energy_error=math.inf, accepted=0, probability=0.0
        )

    error = abs(proposed - current)
    uniform = float(torch.rand(1, generator=generator))
    ratio = float(torch.exp(torch.tensor((current - proposed) / temperature)))
    probability = acceptance_probability(ratio)
    if accept_ratio(ratio, uniform):
        return Transition(
            position=proposal, energy_error=error, accepted=1, probability=probability
        )
    return Transition(
        position=position, energy_error=error, accepted=0, probability=probability
    )


def compiled_trajectory(
    objective: Objective,
    position: torch.Tensor,
    momentum: torch.Tensor,
    step_size: float,
    n_steps: int,
) -> PhaseSpace:
    """:func:`leapfrog` on a declared energy, in ``oxisal`` at unit mass (issues #986, #1008).

    The trajectory is the one arithmetic the compiled chain and the torch
    route share, so it is what pins ``src/hmc.rs`` to :func:`leapfrog` step
    for step; the chains themselves differ in their streams.

    Raises
    ------
    TypeError
        If the objective declares no energy
        (:func:`~sal.sample.declared.declared_energy`).
    """
    declared = declared_energy(objective)
    if declared is None:
        msg = f"{type(objective).__name__} declares no energy a compiled trajectory can run"
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


def effective_sample_size(draws: torch.Tensor) -> torch.Tensor:
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
    draws : torch.Tensor
        Shape ``(n, dimension)``, one chain.

    Returns
    -------
    torch.Tensor
        Shape ``(dimension,)``. A coordinate that did not move has no
        autocorrelation and is reported as ``n``.

    Raises
    ------
    ValueError
        If fewer than 4 draws are given, which is fewer than the two pairs
        the truncation rule needs.
    """
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
    return sizes
