"""Locally balanced proposals over single-variable changes, and their accept step.

Zanella (2020) writes a proposal over a neighbourhood as
``Q(s -> s') = g(pi(s') / pi(s)) / Z(s)`` for a balancing function satisfying
``g(t) = t g(1 / t)``. The Metropolis-Hastings ratio then collapses:

    pi(s') Q(s' -> s) / (pi(s) Q(s -> s')) = Z(s) / Z(s'),

for **any** such ``g``, because ``pi(s') g(1 / t) = pi(s) g(t)`` cancels the
weights against the targets. The balancing function therefore decides how the
proposal spends its mass and nothing about the correction.

Grathwohl et al. (2021) take the same proposal with
``g(t) = sqrt(t)`` and replace ``log pi(s') - log pi(s)`` by its first-order
Taylor estimate at the current state, which is one gradient rather than one
energy per neighbour. The correction no longer collapses --- the estimate is
not the ratio it stands for --- so :func:`log_metropolis_ratio` is written in
the general form and the collapse is what a test asserts where it holds.

Two consumers, one kernel: the Potts lattice
(:mod:`snakes_and_ladders.sample.potts_mcmc`) and the factor graph
(:mod:`snakes_and_ladders.sample.gibbs`). Both present a neighbourhood as one
row per variable and one column per value, ``-inf`` where a variable has no
such value, so a ragged cardinality reaches the same rectangular arithmetic.

See ``docs/tex/textbook.tex``, ``sec:potts``; `REFERENCES.md` carries both
papers.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from snakes_and_ladders.numerics import logsumexp

__all__ = [
    "BalancingFunction",
    "draw_change",
    "log_balanced_weights",
    "log_metropolis_ratio",
    "log_normalizer",
    "log_ratios",
]


class BalancingFunction(StrEnum):
    """Which ``g`` weights the neighbourhood.

    A ``StrEnum`` for the reason
    :class:`~snakes_and_ladders.sample.potts_mcmc.PottsMove` is one: an
    unrecognized choice is refused by ``mypy --strict`` at the call site.
    """

    #: ``g(t) = sqrt(t)``, so ``log g = (log pi(s') - log pi(s)) / 2``.
    SQRT = "sqrt"
    #: ``g(t) = t / (1 + t)``, Barker's, bounded by 1.
    BARKER = "barker"


def log_ratios(conditionals: np.ndarray, state: np.ndarray) -> np.ndarray:
    """``log pi(s') - log pi(s)`` for every single-variable change of ``state``.

    A variable's value appears only in the factors that touch it --- its own
    field and its incident edges, on a Potts lattice --- so changing it moves
    the log weight by the difference of that variable's own conditional. No
    enumeration and no second energy: the whole neighbourhood in one
    subtraction.

    Parameters
    ----------
    conditionals : np.ndarray
        Every variable's unnormalized log conditional, one row per variable,
        already tempered by whatever ``beta`` the chain runs at, and ``-inf``
        past a variable's cardinality.
    state : np.ndarray
        The current value per variable.

    Returns
    -------
    np.ndarray
        Shape ``conditionals.shape``; zero on the current value, whose
        "change" is the identity.
    """
    ratios: np.ndarray = (
        conditionals - conditionals[np.arange(state.shape[0]), state][:, None]
    )
    return ratios


def log_balanced_weights(
    ratios: np.ndarray,
    state: np.ndarray,
    *,
    function: BalancingFunction = BalancingFunction.SQRT,
) -> np.ndarray:
    """``log g(pi(s') / pi(s))`` for every single-variable change.

    Parameters
    ----------
    ratios : np.ndarray
        ``log pi(s') - log pi(s)`` per ``(variable, value)``, shape
        ``(n_variables, n_values)``; ``-inf`` where a variable has no such
        value. For Gibbs-with-gradients this is the Taylor estimate of that
        difference rather than the difference.
    state : np.ndarray
        The current value per variable, shape ``(n_variables,)``.
    function : BalancingFunction
        ``SQRT`` by default: it is unbounded, so the weight keeps the order of
        the differences instead of saturating as Barker's does above one, and
        its log is the differences halved --- the conditional a heat-bath
        sweep already forms, with no second expression to keep in step.

    Returns
    -------
    np.ndarray
        Shape ``ratios.shape``, ``-inf`` at each variable's current value:
        a proposal to change nothing is not a neighbour, and admitting it
        would put mass on the identity move rather than on the neighbourhood.
    """
    weights: np.ndarray
    if function is BalancingFunction.SQRT:
        weights = 0.5 * ratios
    else:
        # `log(t / (1 + t))` in the log domain is `d - log(1 + exp(d))`, which
        # `logaddexp` computes without forming `exp(d)` and overflowing at the
        # large differences a frustrated lattice produces.
        weights = ratios - np.logaddexp(0.0, ratios)
    weights[np.arange(state.shape[0]), state] = -np.inf
    return weights


def log_normalizer(log_weights: np.ndarray) -> float:
    """``log Z(s)``: the log of the neighbourhood's total weight.

    Taken as its own function because the accept step needs it at the proposed
    state, where nothing is drawn, and a draw there would consume a uniform
    and move the stream.
    """
    return float(logsumexp(log_weights.reshape(-1), axis=0))


def draw_change(
    log_weights: np.ndarray, total: float, rng: np.random.Generator
) -> tuple[int, int]:
    """One ``(variable, value)`` drawn proportional to ``exp(log_weights)``.

    One uniform and a search of the cumulative sum, the arithmetic every
    single-site update in this package performs, so a proposal costs one draw
    rather than a categorical object per step. ``total`` is
    :func:`log_normalizer` of the same weights, passed in because the accept
    step needs it too.
    """
    probabilities = np.exp(log_weights.reshape(-1) - total)
    cumulative = np.cumsum(probabilities)
    position = int(np.searchsorted(cumulative, rng.random() * cumulative[-1]))
    variable, value = divmod(position, log_weights.shape[1])
    return variable, value


def log_metropolis_ratio(
    log_ratio: float,
    forward: float,
    forward_total: float,
    reverse: float,
    reverse_total: float,
) -> float:
    """``log`` of the Metropolis-Hastings ratio of one single-variable change.

    ``log pi(s') - log pi(s) + log Q(s' -> s) - log Q(s -> s')``, with each
    proposal log-probability given as its unnormalized weight and normalizer.

    Parameters
    ----------
    log_ratio : float
        ``log pi(s') - log pi(s)``, from the energy and **not** from the
        estimate the proposal used: the correction is what makes the estimate
        admissible, so an estimate on both sides would leave the chain
        stationary at neither distribution.
    forward, forward_total : float
        ``log g`` of the proposed change and ``log Z(s)``.
    reverse, reverse_total : float
        ``log g`` of the change back and ``log Z(s')``.

    Returns
    -------
    float
        Where the weights are the exact ratios under a balancing function,
        this is ``log Z(s) - log Z(s')`` --- the collapse this module's
        docstring derives, which
        `tests/regression/search/test_potts_mcmc.py` pins on the Potts energy.
    """
    return log_ratio + (reverse - reverse_total) - (forward - forward_total)
