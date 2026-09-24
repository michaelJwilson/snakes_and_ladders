"""The count families' M-step solves, their identifiability bounds, and the backend they run on.

The negative-binomial dispersion and the beta-binomial rate and concentration
have no closed-form maximum, so each is a bisection on a weighted score. The
batched ``torch`` solves are the oracle; the compiled kernels in
``src/count_mstep.rs`` are the default route, chosen by :data:`M_STEP_BACKEND`
(issue #922).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.backend import Backend

#: How far below :func:`identifiable_dispersion_bound` the dispersion solve's
#: bracket reaches. Nine decades: the score diverges to ``+inf`` as ``r -> 0``,
#: so the lower end only has to put the root above it, and bisection pays for
#: the width in ``log2`` -- three extra iterations per three decades.
_DISPERSION_BRACKET_RATIO = 1e-9


#: How far a re-estimated binomial probability is held from 0 and 1. A
#: weighted mean of exactly 0 or exactly n makes ``log p`` or ``log(1 - p)``
#: infinite, and a state that took no weight produces one; the margin is
#: below any probability a fixture of a realistic size can resolve.
PROBABILITY_MARGIN = 1e-12


#: How far below :func:`identifiable_concentration_bound` the beta-binomial
#: solve's bracket reaches, and the iteration cap each bisection runs under.
#: Nine decades halved 60 times leaves a bracket far below any tolerance a
#: caller states.
_CONCENTRATION_BRACKET_RATIO = 1e-9


_MAX_BISECTIONS = 60


@dataclass(frozen=True)
class SolvedBetaBinomial:
    """One state's ``(a, b)`` solve."""

    alpha: float
    beta: float
    at_boundary: bool
    converged: bool
    iterations: int
    residual: float


def _beta_binomial_rate_score(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float | torch.Tensor,
    rate: float,
    concentration: float,
) -> float:
    """Score in the mean rate ``p`` at fixed concentration, divided by ``M``.

    ``trials`` is a scalar where the family fixes it and a tensor of the same
    shape as ``values`` where the observation supplies it
    (:class:`CountPairEmission`'s joint form); every term broadcasts either
    way.
    """
    alpha = torch.tensor(rate * concentration, dtype=values.dtype)
    beta = torch.tensor((1.0 - rate) * concentration, dtype=values.dtype)
    return float(
        (
            weights
            * (
                torch.digamma(values + alpha)
                - torch.digamma(alpha)
                - torch.digamma(trials - values + beta)
                + torch.digamma(beta)
            )
        ).sum()
    )


def _beta_binomial_concentration_score(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float | torch.Tensor,
    rate: float,
    concentration: float,
) -> float:
    """Score in the concentration ``M = a + b`` at fixed mean rate."""
    total = torch.tensor(concentration, dtype=values.dtype)
    alpha = torch.tensor(rate * concentration, dtype=values.dtype)
    beta = torch.tensor((1.0 - rate) * concentration, dtype=values.dtype)
    return float(
        (
            weights
            * (
                rate * (torch.digamma(values + alpha) - torch.digamma(alpha))
                + (1.0 - rate)
                * (torch.digamma(trials - values + beta) - torch.digamma(beta))
                - (
                    torch.digamma(torch.as_tensor(trials, dtype=values.dtype) + total)
                    - torch.digamma(total)
                )
            )
        ).sum()
    )


def _rate_score_at(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float | torch.Tensor,
    concentration: float,
) -> Callable[[float], float]:
    """The rate score as a function of the rate alone, at a held concentration."""

    def score(rate: float) -> float:
        return _beta_binomial_rate_score(values, weights, trials, rate, concentration)

    return score


def _concentration_score_at(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float | torch.Tensor,
    rate: float,
) -> Callable[[float], float]:
    """The concentration score as a function of ``log M`` alone, at a held rate."""

    def score(log_concentration: float) -> float:
        return _beta_binomial_concentration_score(
            values, weights, trials, rate, math.exp(log_concentration)
        )

    return score


def _bisect(
    score: Callable[[float], float], low: float, high: float, tolerance: float
) -> float:
    """Root of a decreasing ``score`` bracketed by ``[low, high]``."""
    for _ in range(_MAX_BISECTIONS):
        if high - low <= tolerance:
            break
        middle = 0.5 * (low + high)
        if score(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def solve_beta_binomial(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: float | torch.Tensor,
    rate: float,
    concentration: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 60,
) -> SolvedBetaBinomial:
    """Maximize the weighted likelihood in ``(p, M)`` by alternating bisection.

    Minka's fixed point for the Polya distribution was tried first and
    rejected on measurement: monotone but linearly convergent, at a
    concentration of 120 it was still moving in the third decimal place after
    500 iterations, returning an unconverged answer that looked like an
    estimate. Alternating bisection brackets the root instead of stepping
    toward it, so the failure mode becomes "the root is outside the bracket"
    --- a fact worth reporting.

    Each coordinate's score is decreasing in that coordinate over its bracket,
    with the mean rate bracketed by ``(0, 1)`` and the concentration by
    ``(0, bound]``, so each inner solve is unconditional.
    """
    bound = identifiable_concentration_bound(
        _effective_trials(trials, weights), float(weights.sum())
    )
    concentration = min(concentration, bound)
    at_boundary = False
    residual = float("inf")
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        previous = (rate, concentration)
        rate = _bisect(
            _rate_score_at(values, weights, trials, concentration),
            PROBABILITY_MARGIN,
            1.0 - PROBABILITY_MARGIN,
            tolerance,
        )
        if (
            _beta_binomial_concentration_score(values, weights, trials, rate, bound)
            > 0.0
        ):
            concentration, at_boundary = bound, True
        else:
            at_boundary = False
            low, high = (
                math.log(bound) + math.log(_CONCENTRATION_BRACKET_RATIO),
                math.log(bound),
            )
            concentration = math.exp(
                _bisect(
                    _concentration_score_at(values, weights, trials, rate),
                    low,
                    high,
                    tolerance,
                )
            )
        residual = max(
            abs(rate - previous[0]),
            abs(concentration - previous[1]) / concentration,
        )
        if residual <= tolerance:
            break
    return SolvedBetaBinomial(
        alpha=rate * concentration,
        beta=(1.0 - rate) * concentration,
        at_boundary=at_boundary,
        converged=residual <= tolerance,
        iterations=iterations,
        residual=residual,
    )


def weighted_histogram(
    values: torch.Tensor, columns: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The distinct entries of ``values`` and, per row of ``columns``, the weight summed at each.

    Parameters
    ----------
    values : torch.Tensor
        Shape ``(n,)``.
    columns : torch.Tensor
        Shape ``(K, n)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The distinct values, shape ``(U,)``, and the weights, shape ``(K, U)``.
    """
    distinct, inverse = torch.unique(values, return_inverse=True)
    summed = torch.zeros(
        (columns.shape[0], distinct.shape[0]), dtype=columns.dtype
    ).index_add_(1, inverse, columns)
    return distinct, summed


def solve_beta_binomial_batched(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: torch.Tensor | Sequence[float],
    rates: Sequence[float],
    concentrations: Sequence[float],
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 60,
) -> list[SolvedBetaBinomial]:
    """:func:`solve_beta_binomial` for every component at once, in lockstep (issue #892).

    Two cuts. **Less work:** within one solve the weights are fixed, so each
    score's sum over the observations is a sum over each channel's distinct
    values --- the successes, the failures and the depths --- weighted by the
    responsibility summed there, and a bisection step evaluates ``digamma``
    on those alone: 846 values against 9,000 terms per component on
    ``emission_mixture/stress``. **One call:** the per-component solves are
    independent and identically shaped, so one alternating bisection runs
    over all of them on ``(components, values)`` tensors. Every component
    keeps its own bracket, its own count of outer iterations and its own
    stop, and one that has stopped is carried unchanged while the others
    move. The scalars :func:`solve_beta_binomial` takes through :mod:`math`
    --- the bound's logarithm, the concentration's exponential --- are taken
    through :mod:`math` here too. That function stays as the oracle this is
    pinned against; the sums are reordered, so the pin is a tolerance, not
    bitwise.

    Parameters
    ----------
    values : torch.Tensor
        Success counts, shape ``(n,)``.
    weights : torch.Tensor
        Posterior weights, shape ``(n, K)``.
    trials : torch.Tensor | Sequence[float]
        One count per observation, shape ``(n,)``, shared by every component;
        or one per component, length ``K``.
    rates, concentrations : Sequence[float]
        Each component's starting mean rate and concentration.

    Returns
    -------
    list[SolvedBetaBinomial]
        One per component, in order.
    """
    n_components = weights.shape[1]
    shared = isinstance(trials, torch.Tensor)
    if shared:
        assert isinstance(trials, torch.Tensor)
        per: list[float | torch.Tensor] = [trials] * n_components
    else:
        per = [float(t) for t in trials]
    # Within one solve the weights are fixed, so each score's sum over the
    # observations is a sum over each channel's distinct values, weighted by
    # the responsibility summed at that value: the digammas are evaluated on
    # the distinct values only (issue #892).
    columns = weights.T.contiguous()
    total_weight = columns.sum(dim=1)
    counts, count_weight = weighted_histogram(values, columns)
    if shared:
        assert isinstance(trials, torch.Tensor)
        grid_trials = trials.to(values.dtype)
        rest, rest_weight = weighted_histogram(grid_trials - values, columns)
        depth, depth_weight = weighted_histogram(grid_trials, columns)
        rest_grid = rest.reshape(1, -1)
        depth_grid = depth.reshape(1, -1)
    else:
        depth_column = torch.tensor(per, dtype=values.dtype).reshape(-1, 1)
        # A fixed trial count per component: the remainder's distinct values
        # are that count less the success's, one row per component.
        rest_grid = depth_column - counts.reshape(1, -1)
        rest_weight = count_weight
        depth_grid = depth_column
        depth_weight = total_weight.reshape(-1, 1)
    count_grid = counts.reshape(1, -1)
    bounds = [
        identifiable_concentration_bound(
            _effective_trials(per[k], weights[:, k]), float(weights[:, k].sum())
        )
        for k in range(n_components)
    ]
    rate = torch.tensor(list(rates), dtype=values.dtype)
    concentration = torch.tensor(
        [min(c, b) for c, b in zip(concentrations, bounds, strict=True)],
        dtype=values.dtype,
    )
    bound = torch.tensor(bounds, dtype=values.dtype)
    log_bound = [math.log(b) for b in bounds]
    log_low = torch.tensor(
        [lb + math.log(_CONCENTRATION_BRACKET_RATIO) for lb in log_bound],
        dtype=values.dtype,
    )
    log_high = torch.tensor(log_bound, dtype=values.dtype)

    def success_term(alpha: torch.Tensor) -> torch.Tensor:
        """``sum_i w_i (digamma(y_i + a) - digamma(a))`` per component."""
        return (count_weight * torch.digamma(count_grid + alpha.reshape(-1, 1))).sum(
            dim=1
        ) - total_weight * torch.digamma(alpha)

    def failure_term(beta: torch.Tensor) -> torch.Tensor:
        """``sum_i w_i (digamma(n_i - y_i + b) - digamma(b))`` per component."""
        return (rest_weight * torch.digamma(rest_grid + beta.reshape(-1, 1))).sum(
            dim=1
        ) - total_weight * torch.digamma(beta)

    def rate_score(at_rate: torch.Tensor, *, held: torch.Tensor) -> torch.Tensor:
        return success_term(at_rate * held) - failure_term((1.0 - at_rate) * held)

    def concentration_score(held: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        depth_term = (
            depth_weight * torch.digamma(depth_grid + total.reshape(-1, 1))
        ).sum(dim=1) - total_weight * torch.digamma(total)
        return (
            held * success_term(held * total)
            + (1.0 - held) * failure_term((1.0 - held) * total)
            - depth_term
        )

    def _at_log(
        score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        exponent: Callable[[torch.Tensor], torch.Tensor],
        held: torch.Tensor,
        log_total: torch.Tensor,
    ) -> torch.Tensor:
        return score(held, exponent(log_total))

    def exp_each(log_values: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [math.exp(v) for v in log_values.tolist()], dtype=values.dtype
        )

    def bisect(
        score: Callable[[torch.Tensor], torch.Tensor],
        low: torch.Tensor,
        high: torch.Tensor,
        moving: torch.Tensor,
    ) -> torch.Tensor:
        low, high = low.clone(), high.clone()
        for _ in range(_MAX_BISECTIONS):
            step = moving & (high - low > tolerance)
            if not bool(step.any()):
                break
            middle = 0.5 * (low + high)
            above = score(middle) > 0.0
            low = torch.where(step & above, middle, low)
            high = torch.where(step & ~above, middle, high)
        return 0.5 * (low + high)

    active = torch.ones(n_components, dtype=torch.bool)
    at_boundary = torch.zeros(n_components, dtype=torch.bool)
    residual = torch.full((n_components,), float("inf"), dtype=values.dtype)
    iterations = torch.zeros(n_components, dtype=torch.int64)
    lows = torch.full((n_components,), PROBABILITY_MARGIN, dtype=values.dtype)
    highs = torch.full((n_components,), 1.0 - PROBABILITY_MARGIN, dtype=values.dtype)
    for _ in range(max_iterations):
        if not bool(active.any()):
            break
        iterations = iterations + active.to(torch.int64)
        previous_rate, previous_concentration = rate, concentration
        held = concentration
        new_rate = bisect(functools.partial(rate_score, held=held), lows, highs, active)
        rate = torch.where(active, new_rate, rate)
        pinned = concentration_score(rate, bound) > 0.0
        solving = active & ~pinned
        at_rate = rate
        solved = exp_each(
            bisect(
                functools.partial(_at_log, concentration_score, exp_each, at_rate),
                log_low,
                log_high,
                solving,
            )
        )
        concentration = torch.where(
            active, torch.where(pinned, bound, solved), concentration
        )
        at_boundary = torch.where(active, pinned, at_boundary)
        moved = torch.maximum(
            (rate - previous_rate).abs(),
            (concentration - previous_concentration).abs() / concentration,
        )
        residual = torch.where(active, moved, residual)
        active = active & (residual > tolerance)
    return [
        SolvedBetaBinomial(
            alpha=float(rate[k]) * float(concentration[k]),
            beta=(1.0 - float(rate[k])) * float(concentration[k]),
            at_boundary=bool(at_boundary[k]),
            converged=float(residual[k]) <= tolerance,
            iterations=int(iterations[k]),
            residual=float(residual[k]),
        )
        for k in range(n_components)
    ]


def _effective_trials(trials: float | torch.Tensor, weights: torch.Tensor) -> float:
    """The one trial count :func:`identifiable_concentration_bound` is read at.

    A fixed trial count is itself. A per-observation one has no single value,
    so the bound is taken at the *posterior-weighted mean* depth: the bound
    scales as ``n - 1``, and those same weights set each observation's
    contribution to the concentration's score.

    **A constant per-observation count reduces exactly**, rather than through
    that mean: it *is* a fixed trial count, and the weighted mean of a constant
    is only the constant to within rounding, which would move the bracket
    ``identifiable_concentration_bound`` returns for no reason.

    It does **not** make the M step bitwise, and the reason is data-dependent.
    The score functions sum ``w_i f(y_i, n_i)`` over a vector ``n`` rather than
    folding a scalar, so the summation order differs; on some draws the
    alternating bisection still lands on the same root and on others the fitted
    ``alpha`` moves a **relative 1.0e-06**, four orders outside
    ``solve_beta_binomial``'s own 1e-10 tolerance. Issue #648 carries that.
    Scoring *is* bitwise --- one broadcast expression with no reduction --- and
    is the half of #631's bit-for-bit claim that holds today.
    """
    if isinstance(trials, torch.Tensor):
        first = trials.reshape(-1)[0]
        if bool((trials == first).all()):
            return float(first)
        return float((weights * trials).sum() / weights.sum())
    return trials


def identifiable_concentration_bound(trials: float, weight: float) -> float:
    """The concentration above which this much data cannot tell ``a + b`` from infinity.

    A beta-binomial exceeds its binomial variance by a factor
    ``(n + a + b) / (1 + a + b)``, which is ``1 + (n - 1) / (a + b)`` to
    leading order in large ``a + b``. The sampling noise on a variance from
    ``W`` observations is a relative ``sqrt(2 / W)``. Setting the excess equal
    to the noise gives ``a + b = (n - 1) sqrt(W / 2)``: the same construction
    as :func:`identifiable_dispersion_bound`, one family over.

    Parameters
    ----------
    trials : float
        The state's trial count, at least 2 --- at ``n = 1`` a beta-binomial
        *is* a Bernoulli whatever its concentration, so nothing identifies it.
    weight : float
        Total posterior weight on the state, its effective sample size.

    Returns
    -------
    float
        The bound, strictly positive.

    Raises
    ------
    ValueError
        If the trial count is below 2 or the weight is not positive.
    """
    if trials < 2.0 or weight <= 0.0:
        msg = f"trials must be >= 2 and weight positive, got {trials} and {weight}"
        raise ValueError(msg)
    return (trials - 1.0) * math.sqrt(weight / 2.0)


@dataclass(frozen=True)
class SolvedDispersion:
    """One state's dispersion solve."""

    value: float
    at_boundary: bool
    iterations: int
    residual: float


def _weighted_dispersion_score(
    values: torch.Tensor,
    weights: torch.Tensor,
    dispersion: float,
    mean: float | torch.Tensor,
) -> float:
    """The posterior-weighted score in ``r``, with the mean profiled out.

    ``sum_t w_t (digamma(y_t + r) - digamma(r)) + W log(r / (r + mu))``. The
    term in ``(mu - y_t) / (r + mu)`` that appears in the full derivative
    vanishes because ``mu`` is the weighted mean of ``y``, which is what
    "profiled out" buys.

    Under an exposure ``mean`` is the **rate** ``e_t mu`` per observation, and
    the last term stops factoring out of the sum: it is
    ``sum_t w_t log(r / (r + e_t mu))``. The profiling identity does **not**
    survive a varying exposure: ``sum_t w_t (e_t mu - y_t) / (r + e_t mu)``
    vanishes only where ``e_t`` is constant, so it is added wherever the rate
    varies, and the solve is the likelihood's maximum in ``r`` at the given
    ``mu`` (issue #933; until then the fitted ``r`` sat a relative 5.6e-4
    off it on a 1,500-observation draw). A constant rate keeps the profiled
    form, so its fits are unchanged bitwise.
    """
    r = torch.tensor(dispersion, dtype=values.dtype)
    weighted = (weights * (torch.digamma(values + r) - torch.digamma(r))).sum()
    if isinstance(mean, torch.Tensor):
        score = weighted + (weights * torch.log(dispersion / (dispersion + mean))).sum()
        if not _is_constant(mean):
            score = score + (weights * (mean - values) / (dispersion + mean)).sum()
        return float(score)
    return float(weighted + weights.sum() * math.log(dispersion / (dispersion + mean)))


def solve_dispersion(
    values: torch.Tensor,
    weights: torch.Tensor,
    mean: float | torch.Tensor,
    *,
    tolerance: float = 1e-12,
) -> SolvedDispersion:
    """Maximize the weighted likelihood in ``r`` by bisection on ``log r``.

    ``mean`` is a per-state mean, or the **rate** ``e_t mu`` per observation
    under an exposure. It is formed once by the caller rather than rebuilt
    here, because the solve is at fixed ``mu`` and the product does not move
    across the bisection's steps.

    **That buys nothing measurable, and the measurement is the point.** Issue
    #631 predicted the repeated multiply would dominate. At ``n = 200,000`` and
    45 bisection steps it is **1.2 ms** against the two ``digamma`` calls per
    step at **97.8 ms**, in a solve of roughly **270 ms** --- under half a
    percent. Root ``CLAUDE.md``'s profile-first rule says a term that small
    pays for no optimization, so this form is kept for being the clearer one
    and not for a speed-up it does not deliver. The term that would pay is the
    ``digamma`` pair.
    """
    upper = identifiable_dispersion_bound(
        _effective_rate(mean, weights), float(weights.sum())
    )
    lower = upper * _DISPERSION_BRACKET_RATIO
    if _weighted_dispersion_score(values, weights, upper, mean) > 0.0:
        return SolvedDispersion(upper, at_boundary=True, iterations=0, residual=0.0)
    if _weighted_dispersion_score(values, weights, lower, mean) < 0.0:
        return SolvedDispersion(lower, at_boundary=True, iterations=0, residual=0.0)

    low, high = math.log(lower), math.log(upper)
    iterations = 0
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if _weighted_dispersion_score(values, weights, math.exp(middle), mean) > 0.0:
            low = middle
        else:
            high = middle
        iterations += 1
    dispersion = math.exp(0.5 * (low + high))
    return SolvedDispersion(
        dispersion,
        at_boundary=False,
        iterations=iterations,
        residual=abs(_weighted_dispersion_score(values, weights, dispersion, mean))
        / float(weights.sum()),
    )


def solve_dispersion_batched(
    values: torch.Tensor,
    weights: torch.Tensor,
    means: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> list[SolvedDispersion]:
    """:func:`solve_dispersion` for every state at once, in lockstep, on the distinct counts (issue #918).

    The cut #892 made for the beta-binomial. Within one solve the weights are
    fixed, so ``sum_t w_t digamma(y_t + r)`` is a sum over the distinct counts
    weighted by the responsibility summed at each, and a bisection step
    evaluates ``digamma`` on those alone. Every state keeps its own bracket,
    derived as the oracle derives it, and its own stop; the bracket's width
    in ``log r`` is the same for every state, so the interior ones stop
    together. The scalars the oracle takes through :mod:`math` --- the
    bracket's logarithms, each midpoint's exponential --- are taken through
    :mod:`math` here. :func:`solve_dispersion` stays as the oracle; the sums
    are reordered, so the pin is a tolerance.

    Parameters
    ----------
    values : torch.Tensor
        Counts, shape ``(n,)``.
    weights : torch.Tensor
        Posterior weights, shape ``(n, K)``.
    means : Sequence[float]
        Each state's profiled mean, length ``K``.

    Returns
    -------
    list[SolvedDispersion]
        One per state, in order.
    """
    n_states = weights.shape[1]
    columns = weights.T.contiguous()
    total = columns.sum(dim=1)
    distinct, summed = weighted_histogram(values, columns)
    grid = distinct.reshape(1, -1)
    mean = torch.tensor(list(means), dtype=values.dtype)

    def score(r: torch.Tensor) -> torch.Tensor:
        """The profiled score of every state at its own ``r``."""
        return (
            (summed * torch.digamma(grid + r.reshape(-1, 1))).sum(dim=1)
            - total * torch.digamma(r)
            + total * torch.log(r / (r + mean))
        )

    def exp_each(log_values: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [math.exp(v) for v in log_values.tolist()], dtype=values.dtype
        )

    uppers = [
        identifiable_dispersion_bound(float(mean[k]), float(weights[:, k].sum()))
        for k in range(n_states)
    ]
    lowers = [u * _DISPERSION_BRACKET_RATIO for u in uppers]
    upper = torch.tensor(uppers, dtype=values.dtype)
    lower = torch.tensor(lowers, dtype=values.dtype)
    at_upper = score(upper) > 0.0
    at_lower = ~at_upper & (score(lower) < 0.0)
    moving = ~(at_upper | at_lower)
    low = torch.tensor([math.log(v) for v in lowers], dtype=values.dtype)
    high = torch.tensor([math.log(v) for v in uppers], dtype=values.dtype)
    iterations = torch.zeros(n_states, dtype=torch.int64)
    while True:
        step = moving & (high - low > tolerance)
        if not bool(step.any()):
            break
        middle = 0.5 * (low + high)
        above = score(exp_each(middle)) > 0.0
        low = torch.where(step & above, middle, low)
        high = torch.where(step & ~above, middle, high)
        iterations = iterations + step.to(torch.int64)
    dispersion = torch.where(
        at_upper, upper, torch.where(at_lower, lower, exp_each(0.5 * (low + high)))
    )
    residual = torch.where(
        moving, score(dispersion).abs() / total, torch.zeros_like(total)
    )
    return [
        SolvedDispersion(
            float(dispersion[k]),
            at_boundary=bool(not moving[k]),
            iterations=int(iterations[k]),
            residual=float(residual[k]),
        )
        for k in range(n_states)
    ]


def solve_dispersion_tied(
    values: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    offsets: torch.Tensor | None,
    *,
    tolerance: float = 1e-12,
) -> SolvedDispersion:
    """One dispersion for every state: bisection on ``log r`` of the summed score (issue #933).

    The score is :func:`_weighted_dispersion_score` summed over the states,
    each at its own profiled mean. Its ``digamma`` half pools: ``sum_k sum_t
    w_tk (digamma(y_t + r) - digamma(r))`` is one sum at the weight each
    observation carries over all states, so without an exposure it is taken
    on the distinct counts as :func:`solve_dispersion_batched` takes it. The
    ``log`` half keeps a term per state, and per observation under an
    exposure. A sum of decreasing scores is decreasing, so the bracket is
    unconditional; it is :func:`identifiable_dispersion_bound` at the pooled
    rate and weight, as one state holding all the data would read it.

    Returns
    -------
    SolvedDispersion
    """
    per_state = weights.sum(dim=0)
    pooled = weights.sum(dim=1)
    total = float(per_state.sum())
    if offsets is None:
        distinct, summed = weighted_histogram(values, pooled.reshape(1, -1))
        grid, mass = distinct, summed[0]

        def score(r: float) -> float:
            at = torch.tensor(r, dtype=values.dtype)
            return float(
                (mass * torch.digamma(grid + at)).sum()
                - total * torch.digamma(at)
                + (per_state * torch.log(r / (r + means))).sum()
            )

        rate = float((per_state * means).sum()) / total
    else:
        rates = offsets.reshape(-1, 1) * means.reshape(1, -1)
        # The term profiling drops, kept where the exposure varies, as
        # `_weighted_dispersion_score` keeps it.
        varying = not _is_constant(offsets)
        observed = values.reshape(-1, 1)

        def score(r: float) -> float:
            at = torch.tensor(r, dtype=values.dtype)
            summed = (
                pooled * (torch.digamma(values + at) - torch.digamma(at))
            ).sum() + (weights * torch.log(r / (r + rates))).sum()
            if varying:
                summed = summed + (weights * (rates - observed) / (r + rates)).sum()
            return float(summed)

        rate = float((weights * rates).sum()) / total

    upper = identifiable_dispersion_bound(rate, total)
    lower = upper * _DISPERSION_BRACKET_RATIO
    if score(upper) > 0.0:
        return SolvedDispersion(upper, at_boundary=True, iterations=0, residual=0.0)
    if score(lower) < 0.0:
        return SolvedDispersion(lower, at_boundary=True, iterations=0, residual=0.0)
    low, high = math.log(lower), math.log(upper)
    iterations = 0
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if score(math.exp(middle)) > 0.0:
            low = middle
        else:
            high = middle
        iterations += 1
    dispersion = math.exp(0.5 * (low + high))
    return SolvedDispersion(
        dispersion,
        at_boundary=False,
        iterations=iterations,
        residual=abs(score(dispersion)) / total,
    )


@dataclass(frozen=True)
class SolvedTiedBetaBinomial:
    """The tied beta-binomial solve: every state's ``(a, b)`` at one concentration."""

    alpha: torch.Tensor
    beta: torch.Tensor
    at_boundary: bool
    converged: bool
    iterations: int
    residual: float


def solve_beta_binomial_tied(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: torch.Tensor | Sequence[float],
    rates: Sequence[float],
    concentration: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 60,
) -> SolvedTiedBetaBinomial:
    """:func:`solve_beta_binomial` with one concentration for every state (issue #933).

    The same alternation: each state's rate by bisection at the held
    concentration, then the concentration by bisection on ``log M`` of the
    concentration score summed over the states at their rates. Each score is
    decreasing in its own coordinate, and a sum of decreasing scores is too.
    The bound is :func:`identifiable_concentration_bound` at the pooled
    weight and the weighted mean trial count.

    Returns
    -------
    SolvedTiedBetaBinomial
    """
    n_states = weights.shape[1]
    per_observation = isinstance(trials, torch.Tensor)

    def trials_of(state: int) -> float | torch.Tensor:
        return trials if per_observation else float(trials[state])  # type: ignore[return-value]

    per_state = weights.sum(dim=0)
    if per_observation:
        effective = _effective_trials(trials, weights.sum(dim=1))  # type: ignore[arg-type]
    else:
        depth = torch.tensor(list(trials), dtype=weights.dtype)
        first = depth[0]
        effective = (
            float(first)
            if bool((depth == first).all())
            else float((per_state * depth).sum() / per_state.sum())
        )
    bound = identifiable_concentration_bound(effective, float(per_state.sum()))
    concentration = min(concentration, bound)
    rate = list(rates)
    at_boundary = False
    residual = float("inf")
    iterations = 0

    def summed(log_concentration: float) -> float:
        held = math.exp(log_concentration)
        return sum(
            _beta_binomial_concentration_score(
                values, weights[:, k], trials_of(k), rate[k], held
            )
            for k in range(n_states)
        )

    while iterations < max_iterations:
        iterations += 1
        previous = ([*rate], concentration)
        rate = [
            _bisect(
                _rate_score_at(values, weights[:, k], trials_of(k), concentration),
                PROBABILITY_MARGIN,
                1.0 - PROBABILITY_MARGIN,
                tolerance,
            )
            for k in range(n_states)
        ]
        if summed(math.log(bound)) > 0.0:
            concentration, at_boundary = bound, True
        else:
            at_boundary = False
            concentration = math.exp(
                _bisect(
                    summed,
                    math.log(bound) + math.log(_CONCENTRATION_BRACKET_RATIO),
                    math.log(bound),
                    tolerance,
                )
            )
        residual = max(
            *(abs(a - b) for a, b in zip(rate, previous[0], strict=True)),
            abs(concentration - previous[1]) / concentration,
        )
        if residual <= tolerance:
            break
    shares = torch.tensor(rate, dtype=torch.float64)
    return SolvedTiedBetaBinomial(
        alpha=shares * concentration,
        beta=(1.0 - shares) * concentration,
        at_boundary=at_boundary,
        converged=residual <= tolerance,
        iterations=iterations,
        residual=residual,
    )


#: Where the count families' M-step solves run (issue #922). The compiled
#: kernel is the default: on one thread it measured 12.3x the batched torch
#: beta-binomial solve on `emission_mixture/stress` and 19.7x on the
#: `spatio_sequential_counts/release` projection, and 7.2x and 3.7x the
#: negative-binomial one, agreeing with them bitwise and to 1.2e-12 relative.
#: ``Backend.PYTHON`` runs the batched torch solves, which stay as its oracle.
M_STEP_BACKEND: Backend = Backend.RUST


def solve_dispersion_m_step(
    values: torch.Tensor, weights: torch.Tensor, means: Sequence[float]
) -> list[SolvedDispersion]:
    """The dispersion solve on :data:`M_STEP_BACKEND`.

    Returns
    -------
    list[SolvedDispersion]
    """
    if M_STEP_BACKEND is Backend.RUST:
        try:
            return solve_dispersion_rust(values, weights, means)
        except NoTails:
            pass
    return solve_dispersion_batched(values, weights, means)


def solve_beta_binomial_m_step(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: torch.Tensor | Sequence[float],
    rates: Sequence[float],
    concentrations: Sequence[float],
) -> list[SolvedBetaBinomial]:
    """The beta-binomial solve on :data:`M_STEP_BACKEND`.

    Returns
    -------
    list[SolvedBetaBinomial]
    """
    if M_STEP_BACKEND is Backend.RUST:
        try:
            return solve_beta_binomial_rust(
                values, weights, trials, rates, concentrations
            )
        except NoTails:
            pass
    return solve_beta_binomial_batched(values, weights, trials, rates, concentrations)


class NoTails(Exception):
    """A weighted count outside the support, which has no tails to sum."""


def weight_tails(counts: torch.Tensor, columns: torch.Tensor) -> np.ndarray:
    """``T[k, j] = sum_{u > j} w[k, u]``: the tails of each row's weights over integer counts (issue #922).

    ``sum_u w_u (digamma(u + x) - digamma(x))`` is ``sum_j T_j / (x + j)`` for
    integer ``u``, which is what the compiled M step evaluates in place of
    ``digamma``.

    Parameters
    ----------
    counts : torch.Tensor
        Integer-valued, shape ``(n,)`` shared by every row, or ``(K, n)``.
    columns : torch.Tensor
        The weights, shape ``(K, n)``.

    Returns
    -------
    np.ndarray
        Shape ``(K, max count)``, C-contiguous ``float64``.

    Raises
    ------
    NoTails
        If a count below zero carries weight.
    """
    index = counts.to(torch.int64)
    weights = columns.to(torch.float64)
    # A count below zero is a success above its own trial count: outside the
    # support, so the E step gave it no weight, and it is dropped. One that
    # carries weight has no tail, and the caller solves by the oracle.
    outside = index < 0
    if bool(outside.any()):
        if bool((weights * outside).any()):
            raise NoTails
        weights = weights * ~outside
        index = index.clamp(min=0)
    width = int(index.max()) + 1 if index.numel() else 1
    dense = torch.zeros((columns.shape[0], width), dtype=torch.float64)
    if index.dim() == 1:
        dense.index_add_(1, index, weights)
    else:
        dense.scatter_add_(1, index.expand_as(weights), weights)
    tails = torch.flip(torch.cumsum(torch.flip(dense, [1]), dim=1), [1])[:, 1:]
    return np.ascontiguousarray(tails.numpy())


def solve_dispersion_rust(
    values: torch.Tensor,
    weights: torch.Tensor,
    means: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> list[SolvedDispersion]:
    """:func:`solve_dispersion_batched` in the compiled kernel, each score by the reciprocal-sum identity (issue #922).

    Returns
    -------
    list[SolvedDispersion]
    """
    from snakes_and_ladders import oxisal

    n_states = weights.shape[1]
    columns = weights.T.contiguous()
    uppers = [
        identifiable_dispersion_bound(float(means[k]), float(weights[:, k].sum()))
        for k in range(n_states)
    ]
    value = np.empty(n_states)
    at_boundary = np.empty(n_states, dtype=np.uint8)
    iterations = np.empty(n_states, dtype=np.uint32)
    residual = np.empty(n_states)
    oxisal.negative_binomial_dispersions(
        weight_tails(values, columns).reshape(-1),
        columns.sum(dim=1).numpy().astype(np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray([u * _DISPERSION_BRACKET_RATIO for u in uppers]),
        np.asarray(uppers, dtype=np.float64),
        tolerance,
        value,
        at_boundary,
        iterations,
        residual,
    )
    return [
        SolvedDispersion(
            float(value[k]),
            at_boundary=bool(at_boundary[k]),
            iterations=int(iterations[k]),
            residual=float(residual[k]),
        )
        for k in range(n_states)
    ]


#: The widest tails, as a multiple of the observations, the exposed kernel
#: takes before the oracle's per-observation digamma is cheaper (issue #933).
EXPOSED_TAIL_RATIO = 8.0


def solve_dispersion_exposed_rust(
    values: torch.Tensor,
    weights: torch.Tensor,
    means: Sequence[float],
    offsets: torch.Tensor,
    *,
    tolerance: float = 1e-12,
) -> list[SolvedDispersion]:
    """:func:`solve_dispersion` under an exposure, every state in the compiled kernel (issue #933).

    The digamma half by the reciprocal-sum identity on the tails, as
    :func:`solve_dispersion_rust`; the log half and the term a varying
    exposure keeps per observation. Each state's bracket is the oracle's.

    **The tails are as wide as the largest count, and past a width of**
    :data:`EXPOSED_TAIL_RATIO` **times the observations they cost more than
    the digamma calls they replace**, so the solve falls back to the oracle
    there. Measured at four states, 2,000 observations and exposures in
    [0.5, 2], best of three: 5.4-5.9x the torch solve at widths under one
    observation, 2.2x at 8.3, 1.3x at 16.7, parity at 23.7 and 0.4x at 59.
    The cap is where the kernel still clears the 2x root ``CLAUDE.md`` keeps
    a compiled path for.

    Returns
    -------
    list[SolvedDispersion]

    Raises
    ------
    NoTails
        Where a weighted count has no tails, as :func:`weight_tails` says,
        or they are wider than the cap.
    """
    from snakes_and_ladders import oxisal

    if values.numel() and float(values.max()) > EXPOSED_TAIL_RATIO * values.numel():
        raise NoTails
    n_states = weights.shape[1]
    columns = weights.T.contiguous().to(torch.float64)
    exposures = offsets.to(torch.float64)
    uppers = [
        identifiable_dispersion_bound(
            _effective_rate(exposures * float(means[k]), weights[:, k]),
            float(weights[:, k].sum()),
        )
        for k in range(n_states)
    ]
    value = np.empty(n_states)
    at_boundary = np.empty(n_states, dtype=np.uint8)
    iterations = np.empty(n_states, dtype=np.uint32)
    residual = np.empty(n_states)
    oxisal.negative_binomial_dispersions_exposed(
        weight_tails(values, columns).reshape(-1),
        np.ascontiguousarray(columns.numpy()).reshape(-1),
        np.ascontiguousarray(exposures.numpy()),
        np.ascontiguousarray(values.to(torch.float64).numpy()),
        columns.sum(dim=1).numpy().astype(np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray([u * _DISPERSION_BRACKET_RATIO for u in uppers]),
        np.asarray(uppers, dtype=np.float64),
        tolerance,
        value,
        at_boundary,
        iterations,
        residual,
    )
    return [
        SolvedDispersion(
            float(value[k]),
            at_boundary=bool(at_boundary[k]),
            iterations=int(iterations[k]),
            residual=float(residual[k]),
        )
        for k in range(n_states)
    ]


def solve_beta_binomial_rust(
    values: torch.Tensor,
    weights: torch.Tensor,
    trials: torch.Tensor | Sequence[float],
    rates: Sequence[float],
    concentrations: Sequence[float],
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 60,
) -> list[SolvedBetaBinomial]:
    """:func:`solve_beta_binomial_batched` in the compiled kernel, each score by the reciprocal-sum identity (issue #922).

    Returns
    -------
    list[SolvedBetaBinomial]
    """
    from snakes_and_ladders import oxisal

    n_components = weights.shape[1]
    columns = weights.T.contiguous()
    total = columns.sum(dim=1)
    if isinstance(trials, torch.Tensor):
        depth_counts = trials.to(values.dtype)
        per: list[float | torch.Tensor] = [trials] * n_components
        failure = weight_tails(depth_counts - values, columns)
        depth = weight_tails(depth_counts, columns)
    else:
        per = [float(t) for t in trials]
        fixed = torch.tensor(per, dtype=values.dtype).reshape(-1, 1)
        failure = weight_tails(fixed - values.reshape(1, -1), columns)
        depth = weight_tails(fixed.to(torch.int64), total.reshape(-1, 1))
    bounds = [
        identifiable_concentration_bound(
            _effective_trials(per[k], weights[:, k]), float(weights[:, k].sum())
        )
        for k in range(n_components)
    ]
    out = np.empty(6 * n_components)
    oxisal.beta_binomial_parameters(
        weight_tails(values, columns).reshape(-1),
        failure.reshape(-1),
        depth.reshape(-1),
        total.numpy().astype(np.float64),
        np.asarray(list(rates), dtype=np.float64),
        np.asarray(
            [min(c, b) for c, b in zip(concentrations, bounds, strict=True)],
            dtype=np.float64,
        ),
        np.asarray(bounds, dtype=np.float64),
        np.asarray(
            [math.log(b) + math.log(_CONCENTRATION_BRACKET_RATIO) for b in bounds]
        ),
        tolerance,
        max_iterations,
        _MAX_BISECTIONS,
        PROBABILITY_MARGIN,
        out,
    )
    rows = out.reshape(n_components, 6)
    return [
        SolvedBetaBinomial(
            alpha=float(row[0]),
            beta=float(row[1]),
            at_boundary=bool(row[2]),
            converged=bool(row[3]),
            iterations=int(row[4]),
            residual=float(row[5]),
        )
        for row in rows
    ]


def _is_constant(values: torch.Tensor) -> bool:
    """Whether every entry of ``values`` is its first."""
    return bool((values == values.reshape(-1)[0]).all())


def _effective_rate(mean: float | torch.Tensor, weights: torch.Tensor) -> float:
    """The one mean :func:`identifiable_dispersion_bound` is read at.

    A per-state mean is itself. A per-observation rate has no single value, so
    the bound is taken at the *posterior-weighted mean* rate, which is the
    construction :func:`_effective_trials` already uses one family over. A
    constant rate reduces exactly, for the reason given there.

    Returns
    -------
    float
    """
    if not isinstance(mean, torch.Tensor):
        return mean
    first = mean.reshape(-1)[0]
    if bool((mean == first).all()):
        return float(first)
    return float((weights * mean).sum() / weights.sum())


def identifiable_dispersion_bound(mean: float, weight: float) -> float:
    """The dispersion above which this much data cannot tell ``r`` from infinity.

    A negative binomial exceeds its Poisson variance by ``mu**2 / r``. The
    sampling noise on a variance estimated from ``W`` observations is about
    ``Var sqrt(2 / W)``, which near the Poisson limit is ``mu sqrt(2 / W)``.
    Setting the excess equal to the noise gives ``r = mu sqrt(W / 2)``: beyond
    it the overdispersion the model is *for* is smaller than the error on
    measuring it, and a maximum reported there is a bound rather than an
    estimate. Derived rather than fixed, so a constant cap cannot flag an
    identified fixture at one size and miss an unidentified one at another.

    Parameters
    ----------
    mean : float
        The state's posterior-weighted mean count.
    weight : float
        Total posterior weight on the state, its effective sample size.

    Returns
    -------
    float
        The bound, strictly positive.

    Raises
    ------
    ValueError
        If the mean or the weight is not positive.
    """
    if mean <= 0.0 or weight <= 0.0:
        msg = f"mean and weight must be positive, got {mean} and {weight}"
        raise ValueError(msg)
    return mean * math.sqrt(weight / 2.0)
