"""Hamiltonian Monte Carlo over an :class:`~snakes_and_ladders.opt.objective.Objective`.

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

`snakes_and_ladders.opt` may import no application module and this needs none:
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
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.backend import Backend, refuse_backend
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.sample.accept import (
    accept_ratio,
    accept_with,
    acceptance_probability,
)
from snakes_and_ladders.sample.expectation import Expectation, KalmanMean
from snakes_and_ladders.sample.schedule import (
    Monotone,
    TempSchedule,
    check_ladder,
    ladder,
)
from snakes_and_ladders.sample.tempered import _exchange

# `current` is aliased: `_coefficients` already binds that name to a
# sub-step length, and one of the two has to give.
from snakes_and_ladders.track import NULL as UNTRACKED
from snakes_and_ladders.track import TrackedOptimization
from snakes_and_ladders.track import current as current_tracked

DEFAULT_STEPS = 20


@runtime_checkable
class DeclaredGradient(Protocol):
    """An objective that states its own gradient, which :func:`gradient_at` then reads.

    Autograd records and replays a graph per call, 57 µs at d = 10 where the
    closed form of a Gaussian takes 1.7 µs (issue #991); an objective whose
    gradient has a closed form states it here (issue #986). It must equal
    autograd's through ``__call__``, which each implementation's test pins.
    """

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        """``dU/dtheta`` at ``theta``, detached."""
        ...


@runtime_checkable
class DeclaredGaussian(Protocol):
    """An objective that declares itself ``U(x) = x' P x / 2``, a zero-mean Gaussian.

    What :func:`sample` compiles (issue #986): a torch closure cannot cross
    the FFI boundary, so the compiled chain runs a declared family, and an
    objective opts in by stating its precision rather than by being
    recognized. ``P`` is ``(d,)`` for a diagonal or ``(d, d)`` symmetric.
    """

    @property
    def gaussian_precision(self) -> torch.Tensor:
        """``P``."""
        ...


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


#: Hoffman & Gelman's (2014, §3.2) dual-averaging constants: the shrinkage
#: strength toward ``mu``, the iterations that stabilize the early estimates,
#: and the exponent that forgets them. They set the *rate* of adaptation, not
#: what it converges to, which is why they are constants and the target
#: acceptance is not. ``t0`` and ``kappa`` are as published.
#:
#: **``gamma`` is not, and the deviation is measured.** The published 0.05
#: was set for a statistic averaged over a NUTS trajectory; a single
#: Metropolis probability per proposal has a standard deviation near 0.4,
#: and the iterates' spread in ``log step_size`` scales as
#: ``sd / (gamma * sqrt(m))``, so at 0.05 the averaged step lands where the
#: acceptance is not the target. On the analytic Gaussian at a target of
#: 0.65 with jitter 0.4 and a 300-proposal warm-up, the drawn chain's
#: acceptance over 10 seeds was 0.815 at 0.05, 0.691 at 0.1, 0.643 at 0.2,
#: 0.610 at 0.5 and 0.555 at 1.0 --- at the larger values the iteration
#: has not converged from ``mu`` within the warm-up. 0.2 is the value at
#: which the drawn chain's acceptance is the target, on that posterior and
#: on the four-taxon tree posterior (0.672), and the tests pin what it
#: achieves rather than the constant.
DUAL_AVERAGING_GAMMA = 0.2
DUAL_AVERAGING_T0 = 10.0
DUAL_AVERAGING_KAPPA = 0.75


@dataclass(frozen=True)
class Transition:
    """What one Metropolis step leaves behind, for the loop that drives it.

    Parameters
    ----------
    position : torch.Tensor
        Where the chain is after the step: the proposal if it was accepted,
        the current point if it was not.
    energy_error : float
        ``|H(proposal) - H(current)|``, the diagnostic a step size too large
        moves and an acceptance rate does not.
    accepted : int
        1 if the proposal was accepted, 0 otherwise. Summed over a chain, so
        an ``int`` rather than a ``bool``.
    probability : float
        ``min(1, exp(-dH / T))``, the statistic dual averaging drives. It has
        less variance than ``accepted``, which is why it is carried beside it.
    """

    position: torch.Tensor
    energy_error: float
    accepted: int
    probability: float

    def __iter__(self) -> Iterator[Any]:
        """``(position, energy_error, accepted, probability)``: the order callers unpack.

        ``Any`` and not a union: an unpacking gives every name the element
        type, so a union would mistype each of them.
        """
        yield from (self.position, self.energy_error, self.accepted, self.probability)


class Kernel(Protocol):
    """One Metropolis transition at a step size, as a warm-up has to see it.

    The warm-up and the chain loop below are statements about a *step size*
    and a *mass diagonal*, not about Hamiltonian dynamics: dual averaging
    needs an acceptance probability per proposal and the metric needs the
    positions a proposal leaves. Both hold of any sampler whose move is
    parameterized by one scale, so the loop takes the transition as an
    argument and :func:`sample` and
    :func:`snakes_and_ladders.sample.langevin.mala` share it rather than
    running two copies that drift.

    An implementation draws from ``generator`` and from nothing else, and
    consumes it in one order for one call, or a chain stops being
    reproducible from a seed.
    """

    def __call__(
        self,
        objective: Objective,
        position: torch.Tensor,
        temperature: float,
        generator: torch.Generator,
        step_size: float,
    ) -> Transition:
        """The new position, the energy error, 1 if accepted, and the probability.

        The fourth is ``min(1, exp(-dH / T))``, the statistic dual averaging
        drives; it has less variance than the accept/reject outcome, which is
        why it is returned beside it.
        """
        ...  # pragma: no cover


@dataclass(frozen=True)
class Adaptation:
    """A warm-up that sets the step size and the mass diagonal, then stops.

    The warm-up runs ``warmup`` proposals in two windows of equal length.
    The first adapts the step size at unit mass and records the positions of
    its second half; their per-coordinate variance is the inverse mass
    diagonal. The second adapts the step size again on that metric, which has
    changed the coordinates' scale and so the step right for them. Both adapt
    by dual averaging (Hoffman & Gelman, 2014, §3.2; ``eq:dual-averaging``):
    the running estimate ``h`` of ``target_acceptance - alpha`` is driven to
    zero by shrinking ``log step_size`` toward ``mu = log(10 * step_size_0)``,
    and the iterate reported is the polynomially averaged one. Every draw
    after the warm-up is at the values it ended on.

    **The step is jittered, and that is what makes the target reachable.**
    On a locally quadratic target the leapfrog energy error is bounded and
    oscillatory, so the acceptance stays high up to the stability limit
    ``step_size * omega < 2`` on the stiffest direction and falls to zero past
    it: measured on the analytic Gaussian at 20 steps, 0.86 at a step of 0.8,
    0.75 at 1.2 and 0.00 at 1.4. A target of 0.65 is then *on* the cliff, and
    dual averaging oscillates across it. Drawing each proposal's step
    uniformly from ``step_size * (1 +/- step_jitter)`` (Neal, 2011, §5.4.2.2)
    averages the acceptance over the band, turning the cliff into a slope with
    one root, and breaks the periodicity that makes a fixed step resonate.
    Each step is a kernel with the right stationary distribution drawn
    independently of the state, so it costs nothing in validity; the draws
    after warm-up use the same jitter, since that is the step the acceptance
    was adapted for.

    Every field is required: 0.65 is Hoffman & Gelman's for HMC, the optimum
    for a Gaussian in the limit of many dimensions (Beskos et al., 2013), a
    warm-up right for one target is too short for the next, and a jitter right
    on a cliff is wasted on a slope.

    Parameters
    ----------
    warmup : int
        Proposals spent adapting, at least 8 so the variance is over more
        than one draw. Discarded.
    target_acceptance : float
        The Metropolis acceptance probability the step size is driven to,
        strictly between 0 and 1.
    step_jitter : float
        Half-width of the uniform band each proposal's step is drawn from,
        as a fraction of the adapted step; in ``[0, 1)``. Zero is a fixed
        step.

    Raises
    ------
    ValueError
        If any is out of range.
    """

    warmup: int
    target_acceptance: float
    step_jitter: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.step_jitter < 1.0:
            msg = (
                f"step_jitter must lie in [0, 1), got {self.step_jitter}: at 1 "
                "the band reaches a zero step, which proposes the current point"
            )
            raise ValueError(msg)
        if self.warmup < 8:
            msg = (
                f"warmup must be at least 8 proposals, got {self.warmup}: the "
                "mass diagonal is a variance over the first window's second "
                "half, which is one draw below that"
            )
            raise ValueError(msg)
        if not 0.0 < self.target_acceptance < 1.0:
            msg = (
                f"target_acceptance must lie strictly between 0 and 1, got "
                f"{self.target_acceptance}: at 1 the step size shrinks without "
                "bound and at 0 it grows without bound"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class Adapted:
    """What a warm-up settled on, so the chain it produced can be reproduced.

    Parameters
    ----------
    step_size : float
        The dual-averaged step, in the coordinates the mass diagonal scales
        --- on coordinate ``i`` the step in ``theta`` is
        ``step_size / sqrt(mass_diagonal[i])`` --- and the centre of the
        jitter band each proposal draws from.
    mass_diagonal : torch.Tensor
        The diagonal of the mass matrix, ``1 / variance`` of the first
        window's second half, shape ``(dimension,)``. Every entry is finite
        and positive, because a coordinate the warm-up did not move is
        refused rather than given an infinite mass.
    warmup_acceptance : float
        Mean Metropolis acceptance probability over the second window ---
        the statistic dual averaging drives to the target, so its distance
        from the target is the warm-up's residual.
    force_evaluations : int
        Gradients the warm-up spent.
    """

    step_size: float
    mass_diagonal: torch.Tensor
    warmup_acceptance: float
    force_evaluations: int


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
    force_evaluations : int
        Gradients spent, warm-up and burn-in included, so an effective
        sample size divided by it is the cost of a draw and two chains
        compare at equal evaluations rather than equal samples.
    adapted : Adapted | None
        What the warm-up settled on, when :func:`sample` was given an
        :class:`Adaptation`; ``None`` for a fixed-parameter chain.
    expectations : Mapping[str, Expectation]
        Each operator's expectation over the recorded draws, by the Kalman
        filter of :mod:`snakes_and_ladders.sample.expectation`, keyed as the
        ``operators`` given to :func:`sample`; empty when none were.
    """

    theta: torch.Tensor
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
        posterior is the caller's to say (`snakes_and_ladders.sample.schedule`).
    adaptation : Adaptation | None
        A warm-up that sets the step size and the mass diagonal before the
        ``burn_in`` and the draws, both of which then run at fixed values.
        ``None`` runs the fixed-parameter chain at unit mass, bitwise what it
        was before adaptation existed.
    store_chain : bool
        Keep the draws (issue #988). ``False`` keeps none --- ``theta`` has
        zero rows --- and the chain holds memory of the order of one draw
        rather than ``n_samples`` of them; what it was for is then
        ``operators``' expectations.
    operators : Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None
        Functions of a draw, in the caller's coordinates, each fed to a
        :class:`~snakes_and_ladders.sample.expectation.KalmanMean` at every
        recorded draw; the burn-in's are not observed. ``None`` observes
        nothing, the chain as it was.
    backend : Backend
        :data:`~snakes_and_ladders.backend.Backend.RUST`, the default since
        issue #986, runs the whole chain in
        ``oxi_snakes_and_ladders.gaussian_hmc`` when the objective is a
        :class:`DeclaredGaussian` and the chain is the plain one: leapfrog,
        unit temperature, no adaptation, no operators, and no enclosing
        :func:`snakes_and_ladders.track.track`. Its momenta and uniforms come
        from ChaCha8 seeded by one draw from ``generator``, so it is its own
        stream: reproducible from the generator, not the torch route's draws.
        Any other chain, and :data:`~snakes_and_ladders.backend.Backend.PYTHON`
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
    if (
        backend is Backend.RUST
        and isinstance(objective, DeclaredGaussian)
        and integrator is leapfrog
        and temperature == 1.0
        and adaptation is None
        and not operators
        and current_tracked() is UNTRACKED
    ):
        return _compiled_gaussian_chain(
            objective,
            generator,
            n_samples,
            step_size=step_size,
            n_steps=n_steps,
            theta0=_start(objective, theta0),
            burn_in=burn_in,
            store_chain=store_chain,
        )
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
        theta=chain.draws,
        acceptance_rate=chain.acceptance_rate,
        energy_error=chain.energy_error,
        force_evaluations=chain.force_evaluations,
        adapted=chain.adapted,
        expectations=chain.expectations,
    )


@dataclass(frozen=True)
class Chain:
    """What a run of :func:`run_chain` drew, whatever kernel drew it.

    The fields :class:`HmcChain` and
    :class:`~snakes_and_ladders.sample.langevin.LangevinChain` are built
    from, in the unit each sampler spends: the evaluations are gradients for
    both kernels here.

    Parameters
    ----------
    draws : torch.Tensor
        Recorded draws, shape ``(n_samples, dimension)``, in the coordinates
        the caller asked for --- a mass diagonal is undone before they are
        returned.
    acceptance_rate : float
        Fraction of the recorded proposals accepted; the burn-in's are not
        counted.
    energy_error : torch.Tensor
        ``|H(proposal) - H(current)|`` per recorded proposal.
    force_evaluations : int
        Evaluations spent, the warm-up's and the burn-in's included.
    adapted : Adapted | None
        What the warm-up settled on, or ``None`` for a fixed-parameter chain.
    """

    draws: torch.Tensor
    acceptance_rate: float
    energy_error: torch.Tensor
    force_evaluations: int
    adapted: Adapted | None
    #: Each operator's expectation over the recorded draws (issue #988).
    expectations: Mapping[str, Expectation] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Any]:
        """``(draws, acceptance_rate, energy_error, force_evaluations, adapted, expectations)``.

        The order callers unpack. ``Any`` for :meth:`Transition.__iter__`'s
        reason.
        """
        yield from (
            self.draws,
            self.acceptance_rate,
            self.energy_error,
            self.force_evaluations,
            self.adapted,
            self.expectations,
        )


def run_chain(
    kernel: Kernel,
    per_proposal: int,
    objective: Objective,
    generator: torch.Generator,
    n_samples: int,
    *,
    step_size: float,
    theta0: torch.Tensor | None,
    burn_in: int,
    temperature: float,
    adaptation: Adaptation | None,
    store_chain: bool = True,
    operators: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> Chain:
    """The warm-up, the burn-in and the recorded draws, for any :class:`Kernel`.

    Extracted from :func:`sample` when a second kernel wanted the same three:
    a warm-up that is discarded, a chain at the values it ended on, and a cost
    in evaluations that includes what the warm-up spent. The arithmetic is
    :func:`sample`'s, unchanged --- a chain drawn from a seeded generator is
    the one that seed gave before the extraction, bitwise.

    Parameters
    ----------
    kernel : Kernel
        The transition, already carrying whatever the sampler needs beyond a
        step size.
    per_proposal : int
        Evaluations one proposal costs, in the unit the sampler is compared
        on: gradients for :func:`sample` and
        :func:`snakes_and_ladders.sample.langevin.mala`.
    objective, generator, n_samples, step_size, theta0, burn_in, temperature, adaptation
        As :func:`sample`.
    store_chain, operators
        As :func:`sample` (issue #988).

    Returns
    -------
    Chain
        The draws, the acceptance rate over them, the per-proposal energy
        error over them, the evaluations spent including the warm-up's, and
        what the warm-up settled on.

    Raises
    ------
    ValueError
        If ``temperature`` is not positive.
    """
    if not temperature > 0.0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)

    position = _start(objective, theta0)

    adapted: Adapted | None = None
    target: Objective = objective
    scale: torch.Tensor | None = None
    if adaptation is not None:
        adapted, position = _warm_up(
            kernel,
            per_proposal,
            objective,
            position,
            temperature,
            generator,
            step_size,
            adaptation,
        )
        step_size = adapted.step_size
        scale = adapted.mass_diagonal.rsqrt()
        target = _Scaled(objective, scale)
        position = position / scale

    draws = torch.empty(
        (n_samples if store_chain else 0, position.shape[0]), dtype=torch.float64
    )
    filters = {name: KalmanMean() for name in (operators or {})}
    errors = torch.empty(n_samples + burn_in, dtype=torch.float64)
    accepted = 0

    jitter = adaptation.step_jitter if adaptation is not None else 0.0
    # One lookup for the chain (`snakes_and_ladders.track`) and one `record`
    # per draw. The counters the result is built from are read rather than
    # recomputed: at the last draw each series equals the field `HmcChain`
    # returns. The position is passed so a bound `Metrics` reports what the
    # draw means beside them.
    tracked: TrackedOptimization = current_tracked()
    started = time.perf_counter()
    warmup_evaluations = adapted.force_evaluations if adapted is not None else 0
    for index in range(n_samples + burn_in):
        step = kernel(
            target,
            position,
            temperature,
            generator,
            _jittered(step_size, jitter, generator),
        )
        position, error = step.position, step.energy_error
        errors[index] = error
        if index >= burn_in:
            drawn = index - burn_in
            accepted += step.accepted
            if store_chain:
                draws[drawn] = position
            if filters:
                drawn_at = position * scale if scale is not None else position
                for name, kalman in filters.items():
                    kalman.update(operators[name](drawn_at))  # type: ignore[index]
            tracked.record(
                drawn,
                state=position,
                acceptance_so_far=accepted / (drawn + 1),
                energy_error=float(error),
                force_evaluations=(index + 1) * per_proposal + warmup_evaluations,
                wall_s=time.perf_counter() - started,
            )

    if scale is not None:
        draws = draws * scale
    tracked.record_cost(max(n_samples - 1, 0), draws.nbytes)
    return Chain(
        draws=draws,
        acceptance_rate=accepted / n_samples if n_samples else 0.0,
        energy_error=errors[burn_in:],
        force_evaluations=(n_samples + burn_in) * per_proposal + warmup_evaluations,
        adapted=adapted,
        expectations={name: kalman.estimate() for name, kalman in filters.items()},
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
    position = _start(objective, theta0)

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
        :func:`snakes_and_ladders.sample.tempered.round_trips` and
        :func:`snakes_and_ladders.sample.tempered.up_fraction` read it, as
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
    :func:`snakes_and_ladders.sample.tempered._exchange` runs the rounds,
    the swaps and the walker trace, and what is supplied here is the step
    --- one Hamiltonian transition per replica --- and the draw the swap
    uniform comes from, which is this module's torch stream (issue #861).
    The discrete counterpart,
    :func:`snakes_and_ladders.sample.potts_mcmc.parallel_tempering`, stays
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
    start = _start(objective, theta0)
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

    ensemble = _exchange(
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


def _start(objective: Objective, theta0: torch.Tensor | None) -> torch.Tensor:
    return (
        objective.initial().detach().clone()
        if theta0 is None
        else theta0.detach().clone()
    ).to(torch.float64)


def _compiled_gaussian_chain(
    objective: DeclaredGaussian,
    generator: torch.Generator,
    n_samples: int,
    *,
    step_size: float,
    n_steps: int,
    theta0: torch.Tensor,
    burn_in: int,
    store_chain: bool,
) -> HmcChain:
    """:func:`sample`'s plain chain on a :class:`DeclaredGaussian`, in one compiled call."""
    precision = objective.gaussian_precision.detach().numpy()
    dimension = int(theta0.shape[0])
    seed = int(torch.randint(0, 2**62, (1,), generator=generator))
    draws, accepted, errors = oxi_snakes_and_ladders.gaussian_hmc(
        np.ascontiguousarray(precision, dtype=np.float64).reshape(-1),
        np.ascontiguousarray(theta0.detach().numpy(), dtype=np.float64),
        n_samples,
        burn_in,
        step_size,
        n_steps,
        seed,
        store_chain,
    )
    return HmcChain(
        theta=torch.from_numpy(draws.reshape(-1, dimension)),
        acceptance_rate=accepted / n_samples if n_samples else 0.0,
        energy_error=torch.from_numpy(errors),
        force_evaluations=(n_samples + burn_in) * leapfrog.force_evaluations(n_steps),
        adapted=None,
    )


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
        whose energy is not finite has probability 0.
    """
    momentum = torch.randn(
        position.shape, generator=generator, dtype=torch.float64
    ) * math.sqrt(temperature)
    current = hamiltonian(objective, position, momentum)

    trajectory = integrator(objective, position, momentum, step_size, n_steps)
    proposal = trajectory.position
    # Negating the momentum makes the proposal symmetric, which is what
    # leaves the acceptance ratio as the energy difference alone. It has
    # no effect on the next iteration, where the momentum is redrawn.
    proposed = hamiltonian(objective, proposal, -trajectory.momentum)

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


def gradient_at(objective: Objective, theta: torch.Tensor) -> torch.Tensor:
    """``dU/dtheta``: the objective's own where it is a :class:`DeclaredGradient`, else by autograd."""
    if isinstance(objective, DeclaredGradient):
        return objective.gradient(theta.detach())
    point = theta.detach().clone().requires_grad_(True)
    value = objective(point)
    (grad,) = torch.autograd.grad(value, point)
    return grad.detach()


@dataclass(frozen=True)
class _Scaled(Objective):
    """``objective`` in the coordinates ``phi = theta / scale``.

    Hamiltonian dynamics with a diagonal mass ``M`` on ``theta`` is the
    unit-mass dynamics on ``phi = M^(1/2) theta``: substituting
    ``p = M^(1/2) q`` into ``U(theta) + p' M^-1 p / 2`` gives
    ``U(M^(-1/2) phi) + q'q / 2``, and the leapfrog steps map term for term
    (``theta += eps M^-1 p`` is ``phi += eps q``). The mass matrix is
    therefore a change of coordinates on the objective and the integrator is
    untouched; the Hamiltonian is the same number in both, so the energy error
    reported is the one on ``theta``. ``scale`` is ``M^(-1/2)``, the warm-up's
    standard deviation per coordinate.
    """

    objective: Objective
    scale: torch.Tensor

    def initial(self) -> torch.Tensor:
        return self.objective.initial() / self.scale

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.objective.constrain(theta * self.scale)

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.objective.theta_from(named) / self.scale

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return self.objective(theta * self.scale)


class _DualAveraging:
    """Hoffman & Gelman's (2014, §3.2) step-size iteration, ``eq:dual-averaging``.

    ``update`` takes the acceptance probability of the proposal just made
    and returns the step for the next; ``averaged`` is the polynomially
    averaged iterate that is used once adaptation stops.
    """

    def __init__(self, step_size: float, target: float) -> None:
        self.mu = math.log(10.0 * step_size)
        self.target = target
        self.h_bar = 0.0
        self.log_step = math.log(step_size)
        self.log_averaged = 0.0
        self.iteration = 0

    def update(self, probability: float) -> float:
        self.iteration += 1
        m = self.iteration
        weight = 1.0 / (m + DUAL_AVERAGING_T0)
        self.h_bar = (1.0 - weight) * self.h_bar + weight * (self.target - probability)
        self.log_step = self.mu - math.sqrt(m) / DUAL_AVERAGING_GAMMA * self.h_bar
        forget = m**-DUAL_AVERAGING_KAPPA
        self.log_averaged = forget * self.log_step + (1.0 - forget) * self.log_averaged
        return math.exp(self.log_step)

    @property
    def averaged(self) -> float:
        return math.exp(self.log_averaged)


def _warm_up(
    kernel: Kernel,
    per_proposal: int,
    objective: Objective,
    position: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
    step_size: float,
    adaptation: Adaptation,
) -> tuple[Adapted, torch.Tensor]:
    """The two windows :class:`Adaptation` describes; returns the report and where the chain is."""
    first = adaptation.warmup // 2
    second = adaptation.warmup - first

    # Window one: the step at unit mass, recording the second half.
    averaging = _DualAveraging(step_size, adaptation.target_acceptance)
    recorded = []
    jitter = adaptation.step_jitter
    for index in range(first):
        step = kernel(
            objective,
            position,
            temperature,
            generator,
            _jittered(step_size, jitter, generator),
        )
        position = step.position
        step_size = averaging.update(step.probability)
        if index >= first // 2:
            recorded.append(position)
    variance = torch.stack(recorded).var(dim=0, unbiased=True)
    if not bool((variance > 0.0).all()):
        stuck = torch.nonzero(~(variance > 0.0)).flatten().tolist()
        msg = (
            f"warm-up variance is zero on coordinate(s) {stuck} over the "
            f"{len(recorded)} recorded proposals: the chain did not move there, "
            "so no mass can be estimated; lengthen the warm-up or start the "
            "step size smaller"
        )
        raise ValueError(msg)
    scale = variance.sqrt()

    # Window two: the step again, on the metric, from where window one ended.
    scaled = _Scaled(objective, scale)
    position = position / scale
    averaging = _DualAveraging(averaging.averaged, adaptation.target_acceptance)
    step_size = averaging.averaged
    total = 0.0
    for _ in range(second):
        step = kernel(
            scaled,
            position,
            temperature,
            generator,
            _jittered(step_size, jitter, generator),
        )
        position = step.position
        step_size = averaging.update(step.probability)
        total += step.probability

    report = Adapted(
        step_size=averaging.averaged,
        mass_diagonal=1.0 / variance,
        warmup_acceptance=total / second,
        force_evaluations=adaptation.warmup * per_proposal,
    )
    return report, position * scale


def _jittered(step_size: float, jitter: float, generator: torch.Generator) -> float:
    """A step drawn uniformly from ``step_size * (1 +/- jitter)``.

    At zero jitter it is ``step_size`` and draws nothing, so a chain without
    jitter consumes the stream a chain before jitter existed did.
    """
    if jitter == 0.0:
        return step_size
    uniform = float(torch.rand(1, generator=generator))
    return step_size * (1.0 + jitter * (2.0 * uniform - 1.0))


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
