"""``log Z`` by annealing: importance sampling, a resampled population, and the tempering that reads their rungs back.

Three estimators on one ladder of inverse temperatures, from ``beta = 0`` ---
where ``log Z_0 = n log q`` is exact, the uniform law over ``q ** n``
configurations --- to the target. Each steps the shipped kernel
:func:`~snakes_and_ladders.sample.potts_mcmc._sweep_for` once per rung, so
every move set :class:`~snakes_and_ladders.sample.potts_mcmc.PottsMove`
declares is admissible here and no second physics enters. What is new is the
bookkeeping around the sweep, never the sweep.

:func:`annealed_importance_sampling` (Neal 2001) carries an unnormalized
importance weight down the ladder and reads ``log Z`` off it;
:func:`population_annealing` (Hukushima & Iba 2003, Machta 2010) carries a
*population*, resampling it by those weights at every rung and accumulating
``log Z`` from the per-rung normalizers. The two are one estimator with the
resampling on or off: at :data:`Resampling.NONE` the population's product
telescopes into the importance weight, which is what
`tests/regression/search/test_search_annealed.py` pins.

**An estimate of ``log Z`` without its standard error is not one.** Both
return :class:`LogPartition`, which carries the error and the effective
sample size the error corresponds to; the estimator is a ratio of an average,
so a run whose weight is concentrated on one member is an estimate with one
sample in it however large the population, and the ESS is what says so. Both
are judged against :func:`~snakes_and_ladders.likelihood.potts.strip_log_partition`
and against enumeration, never against each other.

:func:`simulated_tempering` (Marinari & Parisi 1992) is the consumer of those
rungs. One walker moves over the ladder with the rung as a dynamic variable,
and the rung weights ``g_k`` it needs are ``-log Z_k`` --- which is what
:attr:`LogPartition.rung_log_z` holds, so a pilot run of either estimator
configures the sampler. Its walker trace is a one-column
:attr:`~snakes_and_ladders.sample.tempered.TemperedEnsemble.walkers`, so
:func:`~snakes_and_ladders.sample.tempered.round_trips` reads it unchanged.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from snakes_and_ladders.backend import Backend
from snakes_and_ladders.numerics import logsumexp
from snakes_and_ladders.sample.accept import accept
from snakes_and_ladders.sample.potts_mcmc import (
    PottsMove,
    _refuse_negative_coupling,
    _sweep_for,
    energies,
)
from snakes_and_ladders.sample.schedule import (
    ExponentialTempSchedule,
    TempSchedule,
    beta_ladder,
    temperatures,
)
from snakes_and_ladders.sim.graph import PottsGraph
from snakes_and_ladders.sim.potts import site_field
from snakes_and_ladders.track import TrackedOptimization, current


class Resampling(StrEnum):
    """How a population is resampled by its weights at a rung.

    ``SYSTEMATIC`` is the default: it draws one uniform per rung rather than
    ``n_replicas`` of them, gives every member the same expected number of
    copies as ``MULTINOMIAL``, and never splits a member's expected count by
    more than one --- so its resampling variance is the smaller of the two at
    equal population (Kitagawa 1996). ``NONE`` is the ablation: with no
    resampling :func:`population_annealing` *is*
    :func:`annealed_importance_sampling`, the per-rung normalizers
    telescoping into the importance weight.
    """

    SYSTEMATIC = "systematic"
    MULTINOMIAL = "multinomial"
    NONE = "none"


@dataclass(frozen=True)
class LogPartition:
    """``log Z`` from an annealed run, and what the weights behind it were worth.

    Parameters
    ----------
    log_z : float
        The estimate at the ladder's last rung.
    stderr : float
        Its standard error, by the delta method on the weight average:
        ``sqrt(1 / ess - 1 / n_replicas)``. Zero at a one-rung ladder, where
        every weight is one and the estimate is the exact ``n log q``.
    ess : float
        The effective sample size the error corresponds to,
        ``1 / (stderr ** 2 + 1 / n_replicas)`` --- for
        :func:`annealed_importance_sampling` exactly
        ``(sum w) ** 2 / sum w ** 2``. It is at most ``n_replicas``, and a run
        whose weight sits on one member has an ESS near 1 however large the
        population.
    rung_log_z : np.ndarray
        ``log Z`` at every rung of the ladder, shape ``(n_rungs,)``, the first
        entry the exact ``n log q`` and the last :attr:`log_z`. These are the
        ``-g_k`` :func:`simulated_tempering` takes.
    log_weights : np.ndarray
        The accumulated log importance weight of each member of the final
        population, shape ``(n_replicas,)``; for
        :func:`population_annealing` the normalized log weights carried since
        the last resampling. ``logsumexp`` of these minus ``log n_replicas``
        is the correction a plain annealing run drops, so this is what an
        ablation reads.
    family_entropy : float
        Entropy of the family sizes in nats, at most ``log n_replicas``: the
        population's spread over the runs it descends from. Exactly
        ``log n_replicas`` for :func:`annealed_importance_sampling`, where
        every trajectory is its own family and nothing is resampled, and
        falling as a resampled population collapses onto fewer ancestors.
    """

    log_z: float
    stderr: float
    ess: float
    rung_log_z: np.ndarray
    log_weights: np.ndarray
    family_entropy: float


def geometric_betas(beta: float, n_rungs: int, *, beta_min: float) -> tuple[float, ...]:
    """A ladder of inverse temperatures: exactly ``0``, then geometric to ``beta``.

    The zero rung is prepended rather than approached, because a geometric
    sequence never reaches zero and zero is the one rung whose ``log Z`` is
    exact. The rest is
    :class:`~snakes_and_ladders.sample.schedule.ExponentialTempSchedule`'s own
    interpolation, so both of its ends come out bitwise and the geometric
    spacing is written once.

    Parameters
    ----------
    beta : float
        The target inverse temperature, positive.
    n_rungs : int
        Rungs in all, the zero one included; at least 2.
    beta_min : float
        The first non-zero rung, in ``(0, beta]``. The step from 0 to it is
        the one step of the ladder the geometric spacing cannot take, so it
        is stated rather than derived. A two-rung ladder is ``(0, beta)`` and
        has no rung for it to name.

    Returns
    -------
    tuple[float, ...]

    Raises
    ------
    ValueError
        If fewer than two rungs are asked for, or ``beta_min`` is not inside
        ``(0, beta]``.
    """
    if n_rungs < 2:
        msg = f"a ladder needs at least two rungs, got {n_rungs}: one is beta = 0"
        raise ValueError(msg)
    if not 0.0 < beta_min <= beta:
        msg = f"beta_min must lie in (0, {beta}], got {beta_min}"
        raise ValueError(msg)
    if n_rungs == 2:
        return (0.0, beta)
    return (0.0, *temperatures(ExponentialTempSchedule(beta_min, beta, n_rungs - 1)))


def _check_betas(betas: Sequence[float], *, from_zero: bool) -> tuple[float, ...]:
    """The ladder, validated: increasing, non-negative, and anchored at zero where it must be."""
    ladder = tuple(float(value) for value in betas)
    least = 1 if from_zero else 2
    if len(ladder) < least:
        msg = f"a ladder needs at least {least} rungs, got {len(ladder)}"
        raise ValueError(msg)
    if from_zero and ladder[0] != 0.0:
        msg = (
            f"the ladder must start at beta = 0, got {ladder[0]}: that rung is "
            "where log Z = n log q is exact, and the estimate is anchored on it"
        )
        raise ValueError(msg)
    if ladder[0] < 0.0:
        msg = f"every inverse temperature must be >= 0, got {ladder[0]}"
        raise ValueError(msg)
    if any(later <= earlier for earlier, later in itertools.pairwise(ladder)):
        msg = f"the ladder must be strictly increasing in beta, got {ladder}"
        raise ValueError(msg)
    return ladder


def _population(
    graph: PottsGraph,
    field: np.ndarray,
    rng: np.random.Generator,
    n_replicas: int,
    move: PottsMove,
    backend: Backend,
) -> tuple[
    np.ndarray,
    list[np.random.Generator],
    Callable[[np.ndarray, np.random.Generator, float], int],
    np.ndarray,
]:
    """A population drawn from the ``beta = 0`` law, its generators, and the sweep.

    **Every replica draws from its own generator**, spawned from the parent
    that then draws only the resampling uniforms, where a caller asked for
    any --- the rule
    :func:`~snakes_and_ladders.sample.potts_mcmc.parallel_tempering` states
    and for the reason it gives. The states are one contiguous block so
    :func:`~snakes_and_ladders.sim.potts.energies` scores the population in
    one call and each row is still a buffer the kernel can borrow.
    """
    _refuse_negative_coupling(move, graph)
    if n_replicas < 2:
        msg = (
            f"a standard error needs at least two replicas, got {n_replicas}: "
            "one carries an estimate and no statement about it"
        )
        raise ValueError(msg)
    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    n_states = int(rows.shape[1])
    children = list(rng.spawn(n_replicas))
    states = np.ascontiguousarray(
        np.stack(
            [child.integers(0, n_states, size=graph.n_nodes) for child in children]
        ),
        dtype=np.int64,
    )
    offsets, neighbours, couplings = graph.compressed_adjacency()
    advance = _sweep_for(move, graph, rows, offsets, neighbours, couplings, backend)
    return states, children, advance, rows


def _log_z_zero(graph: PottsGraph, rows: np.ndarray) -> float:
    """``n log q``: the exact normalizer of the uniform law at ``beta = 0``."""
    return float(graph.n_nodes) * math.log(float(rows.shape[1]))


def _entropy(labels: np.ndarray, n_replicas: int) -> float:
    """Entropy of the family sizes in nats, ``log n_replicas`` where every label is distinct."""
    sizes = np.bincount(labels, minlength=n_replicas)
    fraction = sizes[sizes > 0] / n_replicas
    return float(-(fraction * np.log(fraction)).sum())


def annealed_importance_sampling(
    graph: PottsGraph,
    field: np.ndarray,
    betas: TempSchedule | Sequence[float],
    rng: np.random.Generator,
    n_replicas: int,
    *,
    move: PottsMove = PottsMove.SINGLE_SITE,
    backend: Backend = Backend.RUST,
) -> LogPartition:
    """``log Z`` from independent annealing runs, weighted by what each one cost (Neal 2001).

    Each replica starts from the uniform law at ``beta = 0``, whose
    ``log Z_0 = n log q`` is exact, and is carried to the target by one sweep
    of ``move`` per rung. Between rungs it accumulates the log weight
    ``-(beta_k - beta_(k-1)) E(x)``, and ``log Z`` is
    ``log Z_0 + logsumexp(w) - log N``. The average is of the *weights*, not
    of their logs: ``mean(log w)`` is Jensen's lower bound and is what a plain
    annealing run reports, which
    `tests/regression/search/test_search_annealed.py` ablates.

    The transitions need only leave each rung's law invariant, so the ladder
    length trades against the population: a coarse ladder is not wrong, it is
    a wider weight distribution and a smaller :attr:`LogPartition.ess`.

    Parameters
    ----------
    graph : PottsGraph
        The lattice. Couplings of either sign, subject to ``move``'s own
        refusal.
    field : np.ndarray
        External field ``h``, shape ``(n_states,)`` or ``(n_nodes, n_states)``.
    betas : TempSchedule | Sequence[float]
        The ladder of inverse temperatures, starting at ``0.0`` and strictly
        increasing; :func:`geometric_betas` builds one. A
        :class:`~snakes_and_ladders.sample.schedule.TempSchedule` is read as
        temperatures and inverted, so it carries no ``beta = 0`` rung and is
        refused on the anchor (issue #861).
    rng : np.random.Generator
        The parent: one child per replica, and nothing else drawn from it
        here.
    n_replicas : int
        Independent runs, at least 2.
    move : PottsMove
        The move set each rung's sweep uses.
    backend : Backend
        As :func:`~snakes_and_ladders.sample.potts_mcmc.sample_potts`.

    Returns
    -------
    LogPartition
        With :attr:`LogPartition.family_entropy` exactly ``log n_replicas``:
        nothing is resampled, so every trajectory is its own family.

    Raises
    ------
    ValueError
        If the ladder does not start at zero, is not strictly increasing, or
        fewer than two replicas are asked for; and as
        :func:`~snakes_and_ladders.sample.potts_mcmc.sample_potts` for a
        Fortuin-Kasteleyn cluster move on a negative coupling.
    """
    ladder = _check_betas(beta_ladder(betas), from_zero=True)
    states, children, advance, rows = _population(
        graph, field, rng, n_replicas, move, backend
    )
    log_zero = _log_z_zero(graph, rows)
    log_n = np.log(n_replicas)

    log_w = np.zeros(n_replicas)
    rung_log_z = [log_zero]
    # One lookup, one record per rung (`snakes_and_ladders.track`, issue
    # #799): the estimate, its error and its effective sample size read from
    # the weights at that rung by the closing formulas below, so the last
    # entry is the result's own number bitwise. Assembled only when a run
    # listens, since the reductions are the cost of the estimate itself.
    tracked: TrackedOptimization = current()
    started = time.perf_counter()
    for rung in range(1, len(ladder)):
        log_w = log_w - (ladder[rung] - ladder[rung - 1]) * energies(
            graph, rows, states
        )
        rung_log_z.append(log_zero + float(logsumexp(log_w, axis=0) - log_n))
        if not tracked.is_null:
            so_far = logsumexp(log_w, axis=0)
            relative_so_far = math.expm1(
                float(log_n + logsumexp(2.0 * log_w, axis=0) - 2.0 * so_far)
            )
            tracked.record(
                rung,
                state=states[0],
                log_z=rung_log_z[-1],
                log_z_stderr=math.sqrt(max(relative_so_far, 0.0) / n_replicas),
                ess=n_replicas / (relative_so_far + 1.0),
                wall_s=time.perf_counter() - started,
            )
        for replica in range(n_replicas):
            advance(states[replica], children[replica], ladder[rung])
    tracked.record_cost(max(len(ladder) - 1, 0), states.nbytes)

    total = logsumexp(log_w, axis=0)
    square = logsumexp(2.0 * log_w, axis=0)
    # `expm1` of the log ratio rather than the difference of two reciprocals:
    # at a one-rung ladder every weight is 1, the log ratio is 0 bitwise and
    # the error is 0, where `1 / ess - 1 / n` reads 6.9e-18 off the rounding
    # of `exp(log n)` and reports a positive error on an exact answer.
    relative = math.expm1(float(log_n + square - 2.0 * total))
    return LogPartition(
        log_z=log_zero + float(total - log_n),
        stderr=math.sqrt(max(relative, 0.0) / n_replicas),
        ess=n_replicas / (relative + 1.0),
        rung_log_z=np.array(rung_log_z),
        log_weights=log_w,
        family_entropy=float(log_n),
    )


def _resampled(
    weights: np.ndarray, rng: np.random.Generator, scheme: Resampling
) -> np.ndarray:
    """Indices of the resampled population, one entry per member kept."""
    n_replicas = weights.shape[0]
    if scheme is Resampling.MULTINOMIAL:
        return np.asarray(rng.choice(n_replicas, size=n_replicas, p=weights))
    positions = (rng.random() + np.arange(n_replicas)) / n_replicas
    drawn = np.searchsorted(np.cumsum(weights), positions)
    return np.asarray(drawn.clip(max=n_replicas - 1))


def population_annealing(
    graph: PottsGraph,
    field: np.ndarray,
    betas: TempSchedule | Sequence[float],
    rng: np.random.Generator,
    n_replicas: int,
    *,
    move: PottsMove = PottsMove.SINGLE_SITE,
    backend: Backend = Backend.RUST,
    resample: Resampling = Resampling.SYSTEMATIC,
) -> LogPartition:
    """``log Z`` from a population resampled at every rung (Hukushima & Iba 2003; Machta 2010).

    The ladder, the kernel and the weights are
    :func:`annealed_importance_sampling`'s. What differs is that the weight is
    *spent* at each rung rather than carried: ``log Z`` takes the rung's
    normalizer ``log sum_i W_i w_i``, the population is resampled by the
    normalized weights, and the weights are reset. Resampling kills the
    trajectories that carry no weight and copies the ones that do, so the
    population stays in the region that matters and the estimate does not
    degenerate onto one member --- at the price that the copies share an
    ancestor, which is what :attr:`LogPartition.family_entropy` reports.

    At :data:`Resampling.NONE` the product of the rung normalizers telescopes
    into ``logsumexp(w) - log N`` and this is Neal's estimator on the same
    trajectories, bitwise the same sweeps from the same seed.

    Parameters
    ----------
    graph, field, betas, rng, n_replicas, move, backend
        As :func:`annealed_importance_sampling`, with the parent generator
        drawing the resampling uniforms beside spawning the children.
    resample : Resampling
        The scheme, or :data:`Resampling.NONE` for the ablation.

    Returns
    -------
    LogPartition
        With :attr:`LogPartition.stderr` the square root of the summed
        per-rung relative variance ``1 / ess_k - 1 / N`` --- the free energy
        being a sum over rungs, where importance sampling's is one ratio ---
        and :attr:`LogPartition.ess` read back from it by the one relation
        that dataclass states.

    Raises
    ------
    ValueError
        As :func:`annealed_importance_sampling`.
    """
    ladder = _check_betas(beta_ladder(betas), from_zero=True)
    states, children, advance, rows = _population(
        graph, field, rng, n_replicas, move, backend
    )
    log_zero = _log_z_zero(graph, rows)
    log_n = np.log(n_replicas)

    families = np.arange(n_replicas)
    log_weights = np.full(n_replicas, -log_n)
    log_z = log_zero
    rung_log_z = [log_zero]
    variance = 0.0
    # As in `annealed_importance_sampling`: one record per rung, the running
    # estimate, error and effective size by the closing formulas (issue #799).
    tracked: TrackedOptimization = current()
    started = time.perf_counter()
    for rung in range(1, len(ladder)):
        log_weights = log_weights - (ladder[rung] - ladder[rung - 1]) * energies(
            graph, rows, states
        )
        normalizer = float(logsumexp(log_weights, axis=0))
        log_z += normalizer
        rung_log_z.append(log_z)
        log_weights = log_weights - normalizer
        variance += (
            math.expm1(float(log_n + logsumexp(2.0 * log_weights, axis=0))) / n_replicas
        )
        if not tracked.is_null:
            stderr_so_far = math.sqrt(max(variance, 0.0))
            tracked.record(
                rung,
                state=states[0],
                log_z=log_z,
                log_z_stderr=stderr_so_far,
                ess=1.0 / (stderr_so_far**2 + 1.0 / n_replicas),
                wall_s=time.perf_counter() - started,
            )
        if resample is not Resampling.NONE:
            kept = _resampled(np.exp(log_weights), rng, resample)
            states = np.ascontiguousarray(states[kept])
            families = families[kept]
            log_weights = np.full(n_replicas, -log_n)
        for replica in range(n_replicas):
            advance(states[replica], children[replica], ladder[rung])

    tracked.record_cost(max(len(ladder) - 1, 0), states.nbytes)
    stderr = math.sqrt(max(variance, 0.0))
    return LogPartition(
        log_z=log_z,
        stderr=stderr,
        ess=1.0 / (stderr**2 + 1.0 / n_replicas),
        rung_log_z=np.array(rung_log_z),
        log_weights=log_weights,
        family_entropy=_entropy(families, n_replicas),
    )


@dataclass(frozen=True)
class SimulatedTempered:
    """What one simulated-tempering walker did, at which rung it did it.

    Parameters
    ----------
    betas : tuple[float, ...]
        The ladder, as given; rung ``k`` is ``betas[k]``, and rung 0 is the
        hottest because the ladder increases in ``beta``.
    weights : np.ndarray
        The rung weights ``g_k`` the run was given, shape ``(n_rungs,)``.
    rungs : np.ndarray
        The rung at every recorded sweep, shape ``(n_recorded,)``.
    states : np.ndarray
        The configuration at every recorded sweep, shape
        ``(n_recorded, n_nodes)``. Read together with :attr:`rungs`: the law
        of the states recorded at rung ``k`` is the Boltzmann law at
        ``betas[k]``, which is what refereeing this against enumeration means.
    occupation : np.ndarray
        Fraction of recorded sweeps spent at each rung, shape
        ``(n_rungs,)``. Uniform where ``g_k = -log Z_k``, which is the whole
        point of the weights; a rung the walker never reaches is a ladder the
        temperature move cannot cross.
    acceptance : float
        Fraction of the rung moves accepted, the proposals off the ends of
        the ladder included --- they are proposals the ladder refuses, and
        hiding them would report an acceptance for a move nobody made.
    """

    betas: tuple[float, ...]
    weights: np.ndarray
    rungs: np.ndarray
    states: np.ndarray
    occupation: np.ndarray
    acceptance: float

    @property
    def walkers(self) -> np.ndarray:
        """The rung trace as one walker's column, shape ``(n_recorded, 1)``.

        The shape
        :func:`~snakes_and_ladders.sample.tempered.round_trips` reads, so a
        round trip over a simulated-tempering ladder is counted by the one
        definition this package has rather than by a second one here.
        """
        return self.rungs.reshape(-1, 1)


def rung_weights(estimate: LogPartition) -> np.ndarray:
    """The ``g_k = -log Z_k`` a pilot run leaves for :func:`simulated_tempering`.

    Marinari & Parisi's weights are the free energies of the rungs, so an
    estimator of ``log Z`` per rung *is* the pilot: no separate calibration
    run, and the quality of the weights is the quality of that estimate,
    reported by its own standard error.
    """
    return -np.asarray(estimate.rung_log_z, dtype=float)


def simulated_tempering(
    graph: PottsGraph,
    field: np.ndarray,
    betas: TempSchedule | Sequence[float],
    weights: np.ndarray,
    rng: np.random.Generator,
    n_sweeps: int,
    burn_in: int = 0,
    thin: int = 1,
    *,
    move: PottsMove = PottsMove.SINGLE_SITE,
    backend: Backend = Backend.RUST,
) -> SimulatedTempered:
    """One walker over the ladder, with the rung as a sampled variable (Marinari & Parisi 1992).

    The target is the joint ``pi(x, k) ~ exp(-beta_k E(x) + g_k)`` over
    configurations and rungs. A step is one sweep of ``move`` at the current
    rung, then a Metropolis proposal to the rung above or below with equal
    probability --- symmetric, so the ratio is the target's alone, and a
    proposal off either end is refused rather than reflected, which would not
    be symmetric.

    Where replica exchange runs one chain per rung, this runs one chain over
    all of them, so the ladder costs one sweep per step rather than
    ``n_rungs``; what it needs in exchange is the ``g_k``, which replica
    exchange does not. With ``g_k = -log Z_k`` the rung marginal is uniform,
    and :func:`rung_weights` takes those from a pilot
    :func:`annealed_importance_sampling` or :func:`population_annealing` run.

    Parameters
    ----------
    graph, field, move, backend
        As :func:`annealed_importance_sampling`.
    betas : TempSchedule | Sequence[float]
        Read by
        :func:`~snakes_and_ladders.sample.schedule.beta_ladder`: a
        :class:`~snakes_and_ladders.sample.schedule.TempSchedule` as
        temperatures inverted rung by rung (issue #827), a sequence as the
        inverse temperatures given.
        The ladder, at least two rungs, non-negative and strictly increasing.
        It need not start at zero: nothing here is anchored on an exact
        normalizer.
    weights : np.ndarray
        The rung weights ``g_k``, shape ``(n_rungs,)``. Required rather than
        defaulted to zero: a run at ``g = 0`` samples the rungs in proportion
        to their ``Z_k``, which for a ladder worth having is one rung.
    rng : np.random.Generator
        The parent: one child steps the configuration, and the parent draws
        the rung proposals and their uniforms, so the sweep's stream is the
        one a fixed-rung chain would consume.
    n_sweeps, burn_in, thin : int
        As :func:`~snakes_and_ladders.sample.potts_mcmc.sample_potts`. A
        chi-square over the recorded states assumes independent draws, so the
        thinning is part of that test rather than a speed knob.

    Returns
    -------
    SimulatedTempered

    Raises
    ------
    ValueError
        If the ladder is unusable, ``weights`` does not carry one entry per
        rung, or ``n_sweeps`` or ``thin`` is below 1 or ``burn_in`` below 0.
    """
    ladder = _check_betas(beta_ladder(betas), from_zero=False)
    g = np.asarray(weights, dtype=float)
    if g.shape != (len(ladder),):
        msg = f"weights must carry one g_k per rung, got {g.shape} for {len(ladder)} rungs"
        raise ValueError(msg)
    if n_sweeps < 1 or thin < 1 or burn_in < 0:
        msg = f"n_sweeps {n_sweeps} and thin {thin} must be >= 1 and burn_in {burn_in} >= 0"
        raise ValueError(msg)

    rows = site_field(np.asarray(field, dtype=float), graph.n_nodes)
    n_states = int(rows.shape[1])
    (child,) = rng.spawn(1)
    state = np.ascontiguousarray(
        rng.integers(0, n_states, size=graph.n_nodes), dtype=np.int64
    )
    offsets, neighbours, couplings = graph.compressed_adjacency()
    _refuse_negative_coupling(move, graph)
    advance = _sweep_for(move, graph, rows, offsets, neighbours, couplings, backend)

    rung = 0
    accepted = 0
    proposed = 0
    recorded_rungs = np.empty(n_sweeps, dtype=np.int64)
    recorded_states = np.empty((n_sweeps, graph.n_nodes), dtype=np.int64)
    # One record per recorded sweep: the rung the walker is at, the rung-move
    # acceptance so far and the sweep rate; the occupation per rung once, at
    # the end, under the rung's context (issue #799).
    tracked: TrackedOptimization = current()
    started = time.perf_counter()
    for step in range(-burn_in * thin, n_sweeps * thin):
        advance(state, child, ladder[rung])
        candidate = rung + (1 if rng.random() < 0.5 else -1)
        proposed += 1
        if 0 <= candidate < len(ladder):
            energy = float(energies(graph, rows, state[np.newaxis])[0])
            log_ratio = (ladder[rung] - ladder[candidate]) * energy + (
                g[candidate] - g[rung]
            )
            if accept(log_ratio, rng):
                accepted += 1
                rung = candidate
        if step >= 0 and (step + 1) % thin == 0:
            recorded_rungs[step // thin] = rung
            recorded_states[step // thin] = state
            wall = time.perf_counter() - started
            tracked.record(
                step // thin,
                state=state,
                rung=float(rung),
                acceptance=accepted / proposed,
                sweeps_per_second=(step + burn_in * thin + 1) / wall if wall else 0.0,
                wall_s=wall,
            )
    occupation = np.bincount(recorded_rungs, minlength=len(ladder)) / n_sweeps
    if not tracked.is_null:
        for index, fraction in enumerate(occupation):
            tracked.record(
                n_sweeps - 1, context={"rung": index}, occupation=float(fraction)
            )
    tracked.record_cost(n_sweeps - 1, recorded_states.nbytes)
    return SimulatedTempered(
        betas=ladder,
        weights=g,
        rungs=recorded_rungs,
        states=recorded_states,
        occupation=occupation,
        acceptance=accepted / proposed,
    )
