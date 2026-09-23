"""Relabelling the draws of a mixture posterior (issue #964).

A mixture's likelihood is invariant under permuting its components, and so is
its posterior under an exchangeable prior, so a chain that mixes visits all
``K!`` relabellings of every mode. A summary over draws --- a posterior mean,
an interval, a per-observation allocation --- then averages different
components together. A relabelling algorithm returns one permutation per
draw that puts the draws in a common labelling. The six here are the ones
``label.switching`` (Papastamoulis 2016) implements, plus the ordering
constraint that serves as their baseline:

- **STEPHENS** (Stephens 2000), the default: minimize the Kullback--Leibler
  divergence from each draw's allocation probabilities to their mean over
  the relabelled draws, alternating the mean and the permutations.
- **ECR** (Papastamoulis & Iliopoulos 2010): maximize each draw's agreement
  with a pivot allocation.
- **ECR-ITERATIVE-1**: ECR with the pivot re-estimated as the mode of the
  relabelled allocations, iterated.
- **ECR-ITERATIVE-2**: ECR with the pivot re-estimated as the argmax of the
  mean relabelled allocation probabilities, iterated.
- **PRA** (Marin, Mengersen & Robert 2005): maximize each draw's inner product
  with a pivot draw's parameters.
- **SJW** (Sperrin, Jaki & Wit 2010): EM over the latent permutation of each
  draw, weighted by the complete-data likelihood at the current estimate.
- **AIC**: order the components by one parameter.

**One permutation convention.** ``permutations[t]`` is an *order*, as
:func:`~snakes_and_ladders.opt.hmm.align_by_key` returns: component ``k`` of
draw ``t`` after relabelling is component ``permutations[t, k]`` before. A
parameter array is relabelled by indexing, an allocation by the inverse
(:func:`permute_parameters`, :func:`permute_allocations`). The source
``label.switching`` publishes applies the order itself, not its inverse, to
the allocations inside ECR-ITERATIVE-1 and SJW; the two agree only where a
permutation is its own inverse, and this module keeps one convention
throughout.

**Every assignment step is a linear assignment**, solved by
:func:`scipy.optimize.linear_sum_assignment` in ``O(K^3)`` per draw; each is
refereed against brute force over ``K!`` in the suite. SJW alone enumerates
``K!`` permutations per draw, as published, and refuses past
:data:`MAX_SJW_COMPONENTS`.

**What each needs.** Allocations ``(m, n)`` (the ECR family, SJW),
allocation probabilities ``(m, n, K)`` (STEPHENS, ECR-ITERATIVE-2), per-draw
parameters ``(m, K, J)`` (PRA, SJW, AIC). :func:`allocation_draws` builds the
first two from any :class:`~snakes_and_ladders.emissions.EmissionFamily`
through its ``log_density``, so every algorithm serves any emission mixture,
the Gaussian one included.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from snakes_and_ladders.emissions import EmissionFamily

#: SJW enumerates every permutation of every draw: ``8! = 40,320`` per draw
#: is the largest set it is run over.
MAX_SJW_COMPONENTS = 8

#: STEPHENS clamps each allocation probability into ``[FLOOR, 1 - FLOOR]``
#: and renormalizes before taking logarithms, as the published
#: implementation does, so a probability of zero does not make a cost
#: infinite.
STEPHENS_FLOOR = 1e-6


class RelabelMethod(StrEnum):
    """The relabelling algorithm :func:`relabel` runs."""

    STEPHENS = "stephens"
    ECR = "ecr"
    ECR_ITERATIVE_1 = "ecr-iterative-1"
    ECR_ITERATIVE_2 = "ecr-iterative-2"
    PRA = "pra"
    SJW = "sjw"
    AIC = "aic"


@dataclass(frozen=True)
class Relabelling:
    """One permutation per draw, and how the algorithm reached them.

    Parameters
    ----------
    permutations : np.ndarray
        Shape ``(m, K)``: component ``k`` of draw ``t`` after relabelling is
        component ``permutations[t, k]`` before.
    method : RelabelMethod
        The algorithm.
    iterations : int
        Passes over the draws; ``1`` for a method that does not iterate.
    converged : bool
        Whether the stopping rule was met before ``max_iterations``.
    objective : tuple[float, ...]
        The method's own objective after each pass: the summed divergence
        for STEPHENS, the summed agreement for the ECR family, the summed
        inner product for PRA, the largest change in the estimate for SJW.
    weights : np.ndarray | None
        SJW only: the probability of each permutation of
        :func:`all_permutations` per draw, shape ``(m, K!)``.
    """

    permutations: np.ndarray
    method: RelabelMethod
    iterations: int
    converged: bool
    objective: tuple[float, ...]
    weights: np.ndarray | None = None


@dataclass(frozen=True)
class AllocationDraws:
    """Per-draw allocation probabilities and one allocation drawn from them.

    Parameters
    ----------
    probabilities : np.ndarray
        ``P(component | observation, draw)``, shape ``(m, n, K)``.
    allocations : np.ndarray
        One allocation per observation and draw, shape ``(m, n)``.
    """

    probabilities: np.ndarray
    allocations: np.ndarray


def all_permutations(n_components: int) -> np.ndarray:
    """Every permutation of ``range(n_components)``, lexicographic, shape ``(K!, K)``."""
    return np.array(list(itertools.permutations(range(n_components))), dtype=np.int64)


def inverse(permutations: np.ndarray) -> np.ndarray:
    """The inverse of each row: ``inverse(p)[t, p[t, k]] == k``."""
    orders = np.asarray(permutations, dtype=np.int64)
    inverted = np.empty_like(orders)
    rows = np.arange(orders.shape[0])[:, None]
    inverted[rows, orders] = np.arange(orders.shape[1])[None, :]
    return inverted


def permute_parameters(values: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    """Per-draw parameters relabelled: ``values[t, permutations[t]]``, shape ``(m, K, ...)``."""
    rows = np.arange(values.shape[0])[:, None]
    return np.asarray(np.asarray(values)[rows, np.asarray(permutations)])


def permute_probabilities(
    probabilities: np.ndarray, permutations: np.ndarray
) -> np.ndarray:
    """Allocation probabilities ``(m, n, K)`` relabelled along the component axis."""
    rows = np.arange(probabilities.shape[0])[:, None, None]
    columns = np.asarray(permutations)[:, None, :]
    positions = np.arange(probabilities.shape[1])[None, :, None]
    return np.asarray(np.asarray(probabilities)[rows, positions, columns])


def permute_allocations(
    allocations: np.ndarray, permutations: np.ndarray
) -> np.ndarray:
    """Allocations ``(m, n)`` relabelled: the inverse permutation applied to each label."""
    inverted = inverse(permutations)
    rows = np.arange(allocations.shape[0])[:, None]
    return np.asarray(inverted[rows, np.asarray(allocations)])


def random_permutations(
    n_draws: int, n_components: int, rng: np.random.Generator
) -> np.ndarray:
    """One uniform permutation per draw: the random permutation sampler's move.

    Applied after each transition of a sampler whose target is invariant
    under relabelling, a uniform relabelling leaves the target invariant and
    makes every draw's labels arbitrary (Fruhwirth-Schnatter 2001); applied
    to recorded draws it gives the same joint law, since the transition is
    equivariant under relabelling. The permutations applied are then known,
    which is what :func:`planted_recovery` scores a relabelling against.
    """
    return np.argsort(rng.random((n_draws, n_components)), axis=1)


def planted_recovery(recovered: np.ndarray, planted: np.ndarray) -> float:
    """Fraction of draws whose relabelling undoes the planted permutation.

    Draws permuted by ``planted`` (as :func:`permute_parameters` applies it)
    and relabelled by ``recovered`` are in the original labelling up to one
    permutation shared by every draw, ``planted[t][recovered[t]]``. The
    fraction is the share of draws at that composite's mode: ``1.0`` when
    every draw is put back in one labelling, whichever it is.
    """
    composite = permute_parameters(np.asarray(planted), np.asarray(recovered))
    _, counts = np.unique(composite, axis=0, return_counts=True)
    return float(counts.max() / composite.shape[0])


def allocation_draws(
    observations: np.ndarray | torch.Tensor,
    log_weights: np.ndarray | torch.Tensor,
    families: Sequence[EmissionFamily],
    rng: np.random.Generator,
) -> AllocationDraws:
    """Allocation probabilities and an allocation per draw, for any emission family.

    Parameters
    ----------
    observations : np.ndarray | torch.Tensor
        Shape ``(n,)`` or ``(n, channels)``.
    log_weights : np.ndarray | torch.Tensor
        Log mixing weights per draw, shape ``(m, K)``.
    families : Sequence[EmissionFamily]
        The components at each draw, ``m`` of them.
    rng : np.random.Generator
        The stream the allocations are drawn from, one uniform per
        observation and draw.

    Returns
    -------
    AllocationDraws
    """
    values = torch.as_tensor(observations)
    weights = torch.as_tensor(log_weights, dtype=torch.float64)
    if weights.shape[0] != len(families):
        msg = f"{weights.shape[0]} weight draws against {len(families)} families"
        raise ValueError(msg)
    probabilities = np.stack(
        [
            torch.softmax(family.log_density(values) + weight, dim=-1).numpy()
            for family, weight in zip(families, weights, strict=True)
        ]
    )
    uniforms = rng.random(probabilities.shape[:2])
    cumulative = np.cumsum(probabilities, axis=-1)
    allocations = (cumulative[..., :-1] < uniforms[..., None]).sum(axis=-1)
    return AllocationDraws(probabilities, allocations.astype(np.int64))


def _assign(scores: np.ndarray) -> np.ndarray:
    """Per draw, the order maximizing ``sum_k scores[t, k, order[k]]``, shape ``(m, K)``."""
    return np.stack(
        [linear_sum_assignment(score, maximize=True)[1] for score in scores]
    ).astype(np.int64)


def _achieved(scores: np.ndarray, orders: np.ndarray) -> float:
    """``sum_t sum_k scores[t, k, orders[t, k]]``."""
    rows = np.arange(scores.shape[0])[:, None]
    labels = np.arange(scores.shape[1])[None, :]
    return float(scores[rows, labels, orders].sum())


def _agreement(
    allocations: np.ndarray, pivot: np.ndarray, n_components: int
) -> np.ndarray:
    """``A[t, k, j]``: observations draw ``t`` allocates to ``j`` and the pivot to ``k``."""
    m = allocations.shape[0]
    flat = (
        np.arange(m)[:, None] * n_components + pivot[None, :]
    ) * n_components + allocations
    counts = np.bincount(flat.reshape(-1), minlength=m * n_components * n_components)
    return counts.reshape(m, n_components, n_components).astype(np.float64)


def _check_allocations(allocations: np.ndarray, n_components: int) -> np.ndarray:
    labels = np.asarray(allocations, dtype=np.int64)
    if labels.ndim != 2:
        msg = f"allocations are (draws, observations); got shape {labels.shape}"
        raise ValueError(msg)
    if labels.min() < 0 or labels.max() >= n_components:
        msg = f"an allocation lies outside [0, {n_components})"
        raise ValueError(msg)
    return labels


def ecr(allocations: np.ndarray, pivot: np.ndarray, n_components: int) -> Relabelling:
    """ECR: each draw's permutation maximizes its agreement with ``pivot``.

    Draw ``t``'s relabelled allocations agree with the pivot on
    ``sum_k A[t, k, order[k]]`` observations, where ``A[t, k, j]`` counts
    those the pivot places in ``k`` and the draw in ``j``; the order
    maximizing it is a linear assignment.

    Parameters
    ----------
    allocations : np.ndarray
        Shape ``(m, n)``, labels in ``[0, n_components)``.
    pivot : np.ndarray
        One allocation, shape ``(n,)``: the maximum a posteriori draw's is
        the published choice.
    n_components : int
        ``K``.

    Returns
    -------
    Relabelling
    """
    labels = _check_allocations(allocations, n_components)
    centre = np.asarray(pivot, dtype=np.int64)
    if centre.shape != (labels.shape[1],):
        msg = f"the pivot has shape {centre.shape}, the draws {labels.shape[1]} observations"
        raise ValueError(msg)
    scores = _agreement(labels, centre, n_components)
    orders = _assign(scores)
    return Relabelling(orders, RelabelMethod.ECR, 1, True, (_achieved(scores, orders),))


def _mode(labels: np.ndarray, n_components: int) -> np.ndarray:
    """Per observation, the most frequent label over draws; ties to the smallest."""
    n = labels.shape[1]
    flat = np.arange(n)[None, :] * n_components + labels
    counts = np.bincount(flat.reshape(-1), minlength=n * n_components)
    return counts.reshape(n, n_components).argmax(axis=1)


def ecr_iterative_1(
    allocations: np.ndarray,
    n_components: int,
    *,
    start: np.ndarray | None = None,
    max_iterations: int = 100,
) -> Relabelling:
    """ECR-ITERATIVE-1: ECR against the mode of the relabelled allocations, iterated.

    Coordinate ascent on the summed agreement: the mode maximizes it over
    the pivot at fixed permutations, ECR over the permutations at a fixed
    pivot, so it never decreases. It stops when a pass leaves the
    permutations unchanged or does not raise the agreement.

    Parameters
    ----------
    allocations : np.ndarray
        Shape ``(m, n)``.
    n_components : int
        ``K``.
    start : np.ndarray | None
        Initial permutations ``(m, K)``; the identity when omitted.
    max_iterations : int
        Passes allowed.

    Returns
    -------
    Relabelling
    """
    labels = _check_allocations(allocations, n_components)
    orders = (
        np.tile(np.arange(n_components), (labels.shape[0], 1))
        if start is None
        else np.asarray(start, dtype=np.int64)
    )
    history: list[float] = []
    converged = False
    for _ in range(max_iterations):
        pivot = _mode(permute_allocations(labels, orders), n_components)
        scores = _agreement(labels, pivot, n_components)
        proposed = _assign(scores)
        value = _achieved(scores, proposed)
        if history and value <= history[-1]:
            converged = True
            break
        unchanged = np.array_equal(proposed, orders)
        orders = proposed
        history.append(value)
        if unchanged:
            converged = True
            break
    return Relabelling(
        orders, RelabelMethod.ECR_ITERATIVE_1, len(history), converged, tuple(history)
    )


def _check_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3:
        msg = f"probabilities are (draws, observations, components); got {values.shape}"
        raise ValueError(msg)
    return values


def ecr_iterative_2(
    allocations: np.ndarray,
    probabilities: np.ndarray,
    *,
    max_iterations: int = 100,
    threshold: float = 1e-6,
) -> Relabelling:
    """ECR-ITERATIVE-2: ECR against the argmax of the mean relabelled probabilities.

    From the identity: the pivot allocates each observation to the component
    of largest mean relabelled probability, and ECR relabels against it;
    iterated until the summed agreement changes by at most ``threshold`` or
    the permutations repeat. Not monotone in general, which is why the stop
    is on the change.

    Parameters
    ----------
    allocations : np.ndarray
        Shape ``(m, n)``.
    probabilities : np.ndarray
        Shape ``(m, n, K)``.
    max_iterations : int
        Passes allowed.
    threshold : float
        The change in agreement that stops it.

    Returns
    -------
    Relabelling
    """
    values = _check_probabilities(probabilities)
    n_components = values.shape[2]
    labels = _check_allocations(allocations, n_components)
    orders = np.tile(np.arange(n_components), (labels.shape[0], 1))
    history: list[float] = []
    converged = False
    for _ in range(max_iterations):
        pivot = permute_probabilities(values, orders).mean(axis=0).argmax(axis=1)
        scores = _agreement(labels, pivot, n_components)
        proposed = _assign(scores)
        value = _achieved(scores, proposed)
        unchanged = np.array_equal(proposed, orders)
        orders = proposed
        history.append(value)
        if unchanged or (
            len(history) > 1 and abs(history[-1] - history[-2]) <= threshold
        ):
            converged = True
            break
    return Relabelling(
        orders, RelabelMethod.ECR_ITERATIVE_2, len(history), converged, tuple(history)
    )


def stephens(
    probabilities: np.ndarray,
    *,
    start: np.ndarray | None = None,
    max_iterations: int = 100,
    threshold: float = 1e-6,
) -> Relabelling:
    """STEPHENS: the permutations minimizing the divergence to the mean relabelled probabilities.

    Minimizes ``sum_t sum_i KL(p_t,i relabelled || q_i)`` over the
    permutations and ``q`` by alternation: ``q`` is the mean of the
    relabelled probabilities, the minimizer at fixed permutations, and each
    draw's permutation is a linear assignment on
    ``cost[t, k, l] = sum_i p[t, i, l] (log p[t, i, l] - log q[i, k])``.
    Both steps lower the objective, so it is non-increasing; it stops when
    the permutations repeat or the decrease is at most ``threshold``.

    Parameters
    ----------
    probabilities : np.ndarray
        Shape ``(m, n, K)``; clamped to ``[STEPHENS_FLOOR, 1 - STEPHENS_FLOOR]``
        and renormalized first.
    start : np.ndarray | None
        Initial permutations ``(m, K)``; the identity when omitted.
    max_iterations : int
        Passes allowed.
    threshold : float
        The decrease that stops it.

    Returns
    -------
    Relabelling
    """
    values = np.clip(
        _check_probabilities(probabilities), STEPHENS_FLOOR, 1.0 - STEPHENS_FLOOR
    )
    values = values / values.sum(axis=2, keepdims=True)
    n_components = values.shape[2]
    entropy = (values * np.log(values)).sum(axis=1)
    orders = (
        np.tile(np.arange(n_components), (values.shape[0], 1))
        if start is None
        else np.asarray(start, dtype=np.int64)
    )
    history: list[float] = []
    converged = False
    for _ in range(max_iterations):
        log_mean = np.log(permute_probabilities(values, orders).mean(axis=0))
        # cost[t, k, l]: draw t's component l given label k.
        cost = entropy[:, None, :] - np.einsum("til,ik->tkl", values, log_mean)
        proposed = _assign(-cost)
        value = -_achieved(-cost, proposed)
        unchanged = np.array_equal(proposed, orders)
        orders = proposed
        history.append(value)
        if unchanged or (len(history) > 1 and history[-2] - history[-1] <= threshold):
            converged = True
            break
    return Relabelling(
        orders, RelabelMethod.STEPHENS, len(history), converged, tuple(history)
    )


def _check_parameters(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3:
        msg = f"parameters are (draws, components, J); got {values.shape}"
        raise ValueError(msg)
    return values


def pra(parameters: np.ndarray, pivot: np.ndarray) -> Relabelling:
    """PRA: each draw's permutation maximizes its inner product with ``pivot``.

    ``sum_k <pivot[k], parameters[t, order[k]]>`` is separable in the
    components, so the published enumeration over ``K!`` is a linear
    assignment. The parameters' scale is the caller's: a column in larger
    units weighs more.

    Parameters
    ----------
    parameters : np.ndarray
        Shape ``(m, K)`` or ``(m, K, J)``.
    pivot : np.ndarray
        Shape ``(K,)`` or ``(K, J)``: the maximum a posteriori draw is the
        published choice.

    Returns
    -------
    Relabelling
    """
    values = _check_parameters(parameters)
    centre = np.asarray(pivot, dtype=np.float64).reshape(values.shape[1], -1)
    if centre.shape != values.shape[1:]:
        msg = f"the pivot has shape {centre.shape}, a draw {values.shape[1:]}"
        raise ValueError(msg)
    scores = np.einsum("kj,tlj->tkl", centre, values)
    orders = _assign(scores)
    return Relabelling(orders, RelabelMethod.PRA, 1, True, (_achieved(scores, orders),))


def aic(parameters: np.ndarray, column: int = 0) -> Relabelling:
    """AIC: the components ordered by one parameter, ascending, in every draw.

    The identifiability constraint the other methods are compared against:
    it relabels well only where that parameter separates the components in
    every draw.
    """
    values = _check_parameters(parameters)
    orders = np.argsort(values[:, :, column], axis=1, kind="stable").astype(np.int64)
    return Relabelling(orders, RelabelMethod.AIC, 1, True, ())


def sjw(
    parameters: np.ndarray,
    allocations: np.ndarray,
    scores: Callable[[np.ndarray], np.ndarray],
    *,
    start: int | None = None,
    max_iterations: int = 100,
    threshold: float = 1e-6,
) -> Relabelling:
    """SJW: EM over each draw's latent permutation (Sperrin, Jaki & Wit 2010).

    The E step weighs every permutation ``v`` of draw ``t`` by the
    complete-data likelihood of its relabelled allocations at the estimate,
    ``w[t, v] ~ exp(sum_i S[i, v^-1(z[t, i])])``; the M step sets the
    estimate to the weighted mean of the relabelled parameters. It stops when
    the estimate moves by at most ``threshold``. Each draw's permutation is
    its most probable one, where the published version samples it; the
    weights are returned.

    Parameters
    ----------
    parameters : np.ndarray
        Shape ``(m, K)`` or ``(m, K, J)``.
    allocations : np.ndarray
        Shape ``(m, n)``.
    scores : Callable[[np.ndarray], np.ndarray]
        ``estimate (K, J) -> S (n, K)``: the complete-data log-likelihood of
        observation ``i`` in component ``k`` at the estimate, weight
        included. A mixture's complete-data log-likelihood is additive over
        observations, so this is the published ``complete`` factorized.
    start : int | None
        The draw the estimate starts at; the mean over draws when omitted.
    max_iterations : int
        EM iterations allowed.
    threshold : float
        The largest change in the estimate that stops it.

    Returns
    -------
    Relabelling
        With ``weights`` over :func:`all_permutations`.

    Raises
    ------
    ValueError
        Past :data:`MAX_SJW_COMPONENTS` components.
    """
    values = _check_parameters(parameters)
    n_components = values.shape[1]
    if n_components > MAX_SJW_COMPONENTS:
        msg = (
            f"SJW enumerates {math.factorial(n_components):,} permutations per draw "
            f"at K = {n_components}; the limit is K = {MAX_SJW_COMPONENTS}"
        )
        raise ValueError(msg)
    labels = _check_allocations(allocations, n_components)
    m = labels.shape[0]
    orders = all_permutations(n_components)
    inverted = inverse(orders)
    estimate = values.mean(axis=0) if start is None else values[start].copy()
    draws = np.arange(m)[:, None]
    history: list[float] = []
    converged = False
    weights = np.full((m, orders.shape[0]), 1.0 / orders.shape[0])
    for _ in range(max_iterations):
        score = np.asarray(scores(estimate), dtype=np.float64)
        # G[t, j, k]: draw t's observations allocated to j, scored in k.
        grouped = np.zeros((m, n_components, n_components))
        np.add.at(
            grouped,
            (np.repeat(draws, labels.shape[1], axis=1), labels),
            score[None, :, :],
        )
        log_weight = grouped[:, np.arange(n_components)[None, :], inverted].sum(axis=2)
        log_weight -= log_weight.max(axis=1, keepdims=True)
        weights = np.exp(log_weight)
        weights /= weights.sum(axis=1, keepdims=True)
        relabelled = values[:, orders]  # (m, K!, K, J)
        updated = np.einsum("tv,tvkj->kj", weights, relabelled) / m
        change = float(np.abs(updated - estimate).max())
        estimate = updated
        history.append(change)
        if change <= threshold:
            converged = True
            break
    return Relabelling(
        orders[weights.argmax(axis=1)],
        RelabelMethod.SJW,
        len(history),
        converged,
        tuple(history),
        weights,
    )


def relabel(
    method: RelabelMethod = RelabelMethod.STEPHENS,
    *,
    probabilities: np.ndarray | None = None,
    allocations: np.ndarray | None = None,
    parameters: np.ndarray | None = None,
    pivot: np.ndarray | None = None,
    scores: Callable[[np.ndarray], np.ndarray] | None = None,
    max_iterations: int = 100,
    threshold: float = 1e-6,
) -> Relabelling:
    """Relabel a mixture's draws by ``method``; STEPHENS by default.

    STEPHENS is the default because it reads only the allocation
    probabilities, which :func:`allocation_draws` gives for any family,
    needs no pivot, and minimizes a stated objective monotonically. Each
    method refuses a call missing what it reads.

    Parameters
    ----------
    method : RelabelMethod
        The algorithm.
    probabilities : np.ndarray | None
        ``(m, n, K)``: STEPHENS, ECR-ITERATIVE-2.
    allocations : np.ndarray | None
        ``(m, n)``: ECR, ECR-ITERATIVE-1, ECR-ITERATIVE-2, SJW.
    parameters : np.ndarray | None
        ``(m, K, J)``: PRA, SJW, AIC.
    pivot : np.ndarray | None
        ECR's pivot allocation ``(n,)`` or PRA's pivot parameters ``(K, J)``.
    scores : Callable | None
        SJW's complete-data scores, as :func:`sjw` states them.
    max_iterations, threshold
        For the iterative methods.

    Returns
    -------
    Relabelling
    """

    def needs(value: np.ndarray | None, what: str) -> np.ndarray:
        if value is None:
            msg = f"{method} needs {what}"
            raise ValueError(msg)
        return value

    match method:
        case RelabelMethod.STEPHENS:
            return stephens(
                needs(probabilities, "probabilities"),
                max_iterations=max_iterations,
                threshold=threshold,
            )
        case RelabelMethod.ECR:
            labels = needs(allocations, "allocations")
            centre = needs(pivot, "a pivot allocation")
            n_components = (
                int(probabilities.shape[2])
                if probabilities is not None
                else int(max(labels.max(), centre.max())) + 1
            )
            return ecr(labels, centre, n_components)
        case RelabelMethod.ECR_ITERATIVE_1:
            labels = needs(allocations, "allocations")
            n_components = (
                int(probabilities.shape[2])
                if probabilities is not None
                else int(labels.max()) + 1
            )
            return ecr_iterative_1(labels, n_components, max_iterations=max_iterations)
        case RelabelMethod.ECR_ITERATIVE_2:
            return ecr_iterative_2(
                needs(allocations, "allocations"),
                needs(probabilities, "probabilities"),
                max_iterations=max_iterations,
                threshold=threshold,
            )
        case RelabelMethod.PRA:
            return pra(needs(parameters, "parameters"), needs(pivot, "a pivot draw"))
        case RelabelMethod.SJW:
            if scores is None:
                msg = f"{method} needs complete-data scores"
                raise ValueError(msg)
            return sjw(
                needs(parameters, "parameters"),
                needs(allocations, "allocations"),
                scores,
                max_iterations=max_iterations,
                threshold=threshold,
            )
        case RelabelMethod.AIC:
            return aic(needs(parameters, "parameters"))
    msg = f"unknown method {method!r}"  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover
