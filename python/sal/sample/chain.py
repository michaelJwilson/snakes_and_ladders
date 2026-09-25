"""The chain every continuous sampler runs, whatever its proposal (issues #1006, #1008, #1010).

A sampler here is a :class:`Kernel` --- one Metropolis transition at a step
size --- and :func:`run_chain` is the loop around it: the warm-up
:class:`Adaptation` describes (two windows of dual averaging, the second on
the metric the first estimates), the burn-in, the recorded draws, and the
operators' Kalman-filtered expectations. :func:`run_compiled` is the same
loop compiled (``src/chain.rs``), over a walk ``oxisal`` or
:mod:`sal.sample.jax.hmc` builds on a declared energy.

HMC (:mod:`sal.sample.hmc`), MALA
(:mod:`sal.sample.langevin`) and random-walk Metropolis
(:mod:`sal.sample.metropolis`) each supply a kernel and share
the rest. Moved out of ``hmc`` by issue #1010, whose module held it beside
the integrators, annealing and tempering; ``hmc`` still exports every public
name for one release.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np
import torch

from sal.backend import Backend
from sal.opt.objective import (
    Objective,
    declares_gradient,
    energy_of,
    value_and_gradient,
)
from sal.sample.declared import Power
from sal.sample.expectation import Expectation, KalmanMean
from sal.track import NULL as UNTRACKED
from sal.track import TrackedOptimization
from sal.track import current as current_tracked

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

#: The stream a chain draws from: torch's for a kernel that differentiates,
#: NumPy's for one that reads values alone (root ``CLAUDE.md``, "No autodiff
#: package where no derivative is taken"; issue #1011). The loop draws from
#: it through :func:`_uniform` and :func:`_seed` and is otherwise blind to it.
Stream = TypeVar("Stream", torch.Generator, np.random.Generator)


def torch_stream(rng: np.random.Generator) -> torch.Generator:
    """A torch :data:`Stream` seeded by one draw from ``rng``, so one seed runs a torch chain.

    The one derivation a caller holding a NumPy generator makes before a
    torch-kernel sampler; ``search.projection`` and ``search.mixture_starts``
    each wrote it until #1059.
    """
    return torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))


_KernelStream_contra = TypeVar(
    "_KernelStream_contra", torch.Generator, np.random.Generator, contravariant=True
)


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


class Kernel(Protocol[_KernelStream_contra]):
    """One Metropolis transition at a step size, as a warm-up has to see it.

    The warm-up and the chain loop below are statements about a *step size*
    and a *mass diagonal*, not about Hamiltonian dynamics: dual averaging
    needs an acceptance probability per proposal and the metric needs the
    positions a proposal leaves. Both hold of any sampler whose move is
    parameterized by one scale, so the loop takes the transition as an
    argument and :func:`sample` and
    :func:`sal.sample.langevin.mala` share it rather than
    running two copies that drift.

    An implementation draws from ``generator`` and from nothing else, and
    consumes it in one order for one call, or a chain stops being
    reproducible from a seed. The stream is torch's or NumPy's
    (:data:`Stream`), the one the kernel's own draws are taken from.
    """

    def __call__(
        self,
        objective: Objective,
        position: torch.Tensor,
        temperature: float,
        generator: _KernelStream_contra,
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
class Chain:
    """What a run of :func:`run_chain` drew, whatever kernel drew it.

    The fields :class:`HmcChain` and
    :class:`~sal.sample.langevin.LangevinChain` are built
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
    kernel: Kernel[Stream],
    per_proposal: int,
    objective: Objective,
    generator: Stream,
    n_samples: int,
    *,
    step_size: float,
    theta0: torch.Tensor | np.ndarray | None,
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
        :func:`sal.sample.langevin.mala`.
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

    position = start_point(objective, theta0)

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
    # One lookup for the chain (`sal.track`) and one `record`
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
                    kalman.update(
                        operators[name](drawn_at).detach().cpu().numpy()  # type: ignore[index]
                    )
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


def compiled_route(backend: Backend, temperature: float) -> bool:
    """Whether a chain may take a compiled walk: the Rust backend, unit temperature, no tracked run (issue #1010).

    What every sampler's compiled route asks before its own conditions
    (HMC's integrator, MALA's correction, the declared energy); a tracked
    run records per draw, which only :func:`run_chain`'s loop does.
    """
    return (
        backend is Backend.RUST
        and temperature == 1.0
        and current_tracked() is UNTRACKED
    )


#: Draws a compiled chain hands back per block when operators observe it:
#: the memory an unstored chain holds is of the order of this many draws.
BLOCK = 1_024


def run_compiled(
    walk_class: Callable[..., Any],
    declared: tuple[Any, Any],
    extra: tuple[Any, ...],
    per_proposal: int,
    generator: Stream,
    n_samples: int,
    *,
    step_size: float,
    theta0: torch.Tensor,
    burn_in: int,
    adaptation: Adaptation | None,
    store_chain: bool,
    operators: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None,
) -> Chain:
    """A chain on a declared family, the warm-up included, compiled (issues #1006, #1008).

    ``walk_class`` is ``oxisal.MetropolisWalk`` or ``oxisal.HmcWalk``, built
    with its own arguments ``extra`` after the shared ones; ``per_proposal``
    is what one proposal costs, as :func:`run_chain` takes it. The warm-up,
    the burn-in, the draws and the filters are ``src/chain.rs``'s, one loop
    for both kernels as :func:`run_chain` is one for the torch ones.

    A :class:`~sal.sample.declared.Power` operator is
    evaluated and Kalman-filtered in the compiled loop, and only its six
    statistics per coordinate come back. Any other operator is a Python
    callable, so the chain is advanced :data:`BLOCK` draws at a time for it
    and each block is filtered here and dropped unless ``store_chain`` keeps
    it (issues #988, #1006).
    """
    family, parameters = declared
    dimension = int(theta0.shape[0])
    seed = _seed(generator)
    declared_operators = {
        name: operator
        for name, operator in (operators or {}).items()
        if isinstance(operator, Power)
    }
    walk = walk_class(
        family,
        parameters,
        np.ascontiguousarray(theta0.numpy(), dtype=np.float64),
        step_size,
        seed,
        0 if adaptation is None else adaptation.warmup,
        0.5 if adaptation is None else adaptation.target_acceptance,
        0.0 if adaptation is None else adaptation.step_jitter,
        (DUAL_AVERAGING_GAMMA, DUAL_AVERAGING_T0, DUAL_AVERAGING_KAPPA),
        [operator.exponent for operator in declared_operators.values()],
        *extra,
    )
    walk.advance(burn_in, False, False)
    filters = {
        name: KalmanMean()
        for name in (operators or {})
        if name not in declared_operators
    }
    blocks: list[torch.Tensor] = []
    errors: list[np.ndarray] = []
    accepted = 0
    remaining = n_samples
    while remaining > 0 or not errors:
        size = min(remaining, BLOCK) if filters else remaining
        draws, taken, error = walk.advance(size, store_chain or bool(filters), True)
        accepted += taken
        errors.append(error)
        block = torch.from_numpy(draws.reshape(-1, dimension))
        for name, kalman in filters.items() if block.shape[0] else ():
            kalman.update_block(
                np.stack(
                    [
                        operators[name](row).detach().cpu().numpy()  # type: ignore[index]
                        for row in block
                    ]
                )
            )
        if store_chain:
            blocks.append(block)
        remaining -= size
    for index, name in enumerate(declared_operators):
        filters[name] = KalmanMean.from_statistics(*walk.statistics(index))
    warmup_evaluations = 0 if adaptation is None else adaptation.warmup * per_proposal
    return Chain(
        # One block is the chain as Rust built it; `cat` would copy it.
        draws=blocks[0]
        if len(blocks) == 1
        else torch.cat(blocks)
        if blocks
        else torch.empty((0, dimension)),
        acceptance_rate=accepted / n_samples if n_samples else 0.0,
        energy_error=torch.from_numpy(np.concatenate(errors)),
        force_evaluations=(n_samples + burn_in) * per_proposal + warmup_evaluations,
        adapted=None
        if adaptation is None
        else Adapted(
            step_size=walk.step_size,
            mass_diagonal=torch.from_numpy(walk.mass_diagonal),
            warmup_acceptance=walk.warmup_acceptance,
            force_evaluations=warmup_evaluations,
        ),
        expectations={name: filters[name].estimate() for name in (operators or {})},
    )


def start_point(
    objective: Objective, theta0: torch.Tensor | np.ndarray | None
) -> torch.Tensor:
    """Where a chain starts: ``theta0``, or ``objective.initial()``, detached, in ``float64``.

    An array ``theta0`` is copied into the loop's tensor, so a sampler that
    takes no derivative takes its start as an array (issue #1059).
    """
    if isinstance(theta0, np.ndarray):
        return torch.tensor(theta0, dtype=torch.float64)
    return (
        objective.initial().detach().clone()
        if theta0 is None
        else theta0.detach().clone()
    ).to(torch.float64)


def on_buffer(array: np.ndarray) -> torch.Tensor:
    """``array`` as the loop's position, on its own buffer and uncopied.

    The one conversion a kernel that steps on arrays needs to hand a
    :class:`Transition` back, kept here so its module imports no torch
    (root ``CLAUDE.md``, "No autodiff package where no derivative is taken").
    """
    return torch.from_numpy(array)


def gradient_at(objective: Objective, theta: torch.Tensor) -> torch.Tensor:
    """``dU/dtheta``: the objective's declared gradient, its declared value and gradient, or autograd."""
    if declares_gradient(objective):
        return objective.gradient(theta.detach())  # type: ignore[attr-defined, no-any-return]
    return value_and_gradient(objective, theta)[1]


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

    def energy(self, x: np.ndarray) -> float:
        # The same product on the tensor's buffer, so a declared NumPy
        # energy survives the change of coordinates (issue #1011).
        return energy_of(self.objective, x * self.scale.numpy())

    def value_and_gradient(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The chain rule through `theta * scale`, so a declared gradient
        # survives the change of coordinates.
        value, gradient = value_and_gradient(self.objective, theta * self.scale)
        return value, gradient * self.scale

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        # A kick needs the gradient alone: through `gradient_at` the inner
        # objective's declared gradient is taken without its value, which
        # `value_and_gradient` would evaluate beside it (issue #1008).
        return gradient_at(self.objective, theta * self.scale) * self.scale


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
    kernel: Kernel[Stream],
    per_proposal: int,
    objective: Objective,
    position: torch.Tensor,
    temperature: float,
    generator: Stream,
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


def _jittered(
    step_size: float, jitter: float, generator: torch.Generator | np.random.Generator
) -> float:
    """A step drawn uniformly from ``step_size * (1 +/- jitter)``.

    At zero jitter it is ``step_size`` and draws nothing, so a chain without
    jitter consumes the stream a chain before jitter existed did.
    """
    if jitter == 0.0:
        return step_size
    return step_size * (1.0 + jitter * (2.0 * _uniform(generator) - 1.0))


def _uniform(generator: torch.Generator | np.random.Generator) -> float:
    """One uniform on ``[0, 1)`` from either :data:`Stream`: one ``torch.rand`` or one ``random()``."""
    if isinstance(generator, np.random.Generator):
        return float(generator.random())
    return float(torch.rand(1, generator=generator))


def _seed(generator: torch.Generator | np.random.Generator) -> int:
    """A compiled chain's ChaCha8 seed on ``[0, 2**62)``, one draw from either :data:`Stream`."""
    if isinstance(generator, np.random.Generator):
        return int(generator.integers(0, 2**62))
    return int(torch.randint(0, 2**62, (1,), generator=generator))
