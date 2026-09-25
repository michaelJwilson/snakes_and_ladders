"""The Metropolis accept step, in one place (issue #857).

``log u < log alpha`` was written thirteen times across this package and `search` --- in the
two parallel temperings, simulated tempering, the exchange, the two balanced
sweeps, the cluster recolouring, Niedermayer's accept, MALA, HMC's transition,
a topology step and one learned environment --- in four spellings that differ
in what they draw and when.

**What differs between the sites is the draw, not the test.** A site draws
from a NumPy generator, from a torch one, through a callable that must not be
called where the difference is non-negative, or not at all where the limit is
exact. So the draw is the parameter and the comparison is the body: each arm
below names how its uniform arrives and nothing else.

**An arm draws what the site it folds drew, in the same order.** The
short-circuit on a non-negative ``log_ratio`` is a saved *draw* and not only a
saved exponential, so it belongs to the arms whose site had it and is absent
from the arms whose site drew unconditionally. A generator's stream is part of
what a seeded chain returns, and every chain in
``tests/regression/sample/`` pins one.

The exponential is :func:`numpy.exp`, which is what ten of the twelve sites
read; the two that reach it through :mod:`torch` exponentiate on their own
side and hand the ratio to :func:`accept_ratio`, so a batched, differentiable
kernel is not routed through NumPy for the last bit of its acceptance
probability.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def accept(log_ratio: float, rng: np.random.Generator) -> bool:
    """Accept with probability ``min(1, exp(log_ratio))``, drawing at most one uniform.

    Parameters
    ----------
    log_ratio : float
        The log Metropolis ratio. Non-negative accepts without a draw.
    rng : np.random.Generator
        Drawn from once, and only where ``log_ratio`` is negative.

    Returns
    -------
    bool
        Whether the proposal is accepted.
    """
    return accept_drawn(log_ratio, rng.random)


def accept_drawn(log_ratio: float, draw: Callable[[], float]) -> bool:
    """:func:`accept` with the uniform behind a call.

    The arm for a site whose uniform is *conditional*: the generator it would
    come from is keyed on the move, or costs something to reach, so taking one
    eagerly would advance a stream that a non-negative ``log_ratio`` never
    touched --- ``potts_mcmc.sweeps._recolour_drawn``'s contract, which the Rust
    hand-back path reads (issues #599, #754).

    Parameters
    ----------
    log_ratio : float
        As :func:`accept`.
    draw : Callable[[], float]
        Returns one uniform on ``[0, 1)``. Called once where ``log_ratio`` is
        negative and never otherwise.

    Returns
    -------
    bool
        Whether the proposal is accepted.
    """
    return log_ratio >= 0.0 or bool(draw() < np.exp(log_ratio))


def accept_with(log_ratio: float, uniform: float) -> bool:
    """:func:`accept` on a uniform the caller has already drawn.

    The arm for a site whose uniform comes from a :class:`torch.Generator`:
    the draw stays on the caller's stream and this decides. No draw happens
    here, so the short-circuit saves an exponential and no uniform.

    Parameters
    ----------
    log_ratio : float
        As :func:`accept`.
    uniform : float
        A draw on ``[0, 1)``.

    Returns
    -------
    bool
        Whether the proposal is accepted.
    """
    return log_ratio >= 0.0 or bool(uniform < np.exp(log_ratio))


def accept_ratio(ratio: float, uniform: float) -> bool:
    """Accept on a ratio the caller has already exponentiated.

    The arm for the two gradient samplers: they form ``exp(-dH / T)`` with
    :func:`torch.exp` on a batched, differentiable side, so the ratio arrives
    formed rather than as a log and its last bit is the one those kernels
    computed. The comparison is against the unclamped ratio, which is
    ``min(1, ratio)`` for a uniform on ``[0, 1)``, and a ``nan`` ratio --- a
    proposal whose energy is not finite --- is refused by the comparison
    itself.

    There is no short-circuit: the sites this folds draw their uniform before
    they compare, and skipping a draw would move the stream.

    Parameters
    ----------
    ratio : float
        ``exp(log_ratio)``, possibly above one or ``nan``.
    uniform : float
        A draw on ``[0, 1)``.

    Returns
    -------
    bool
        Whether the proposal is accepted.
    """
    return uniform < ratio


def acceptance_probability(ratio: float) -> float:
    """``min(1, ratio)``, and zero where the ratio is ``nan``.

    Reported beside the outcome by the samplers :func:`accept_ratio` serves:
    dual averaging drives this statistic rather than the accept itself,
    because it carries less variance. Separate from :func:`accept_ratio` so
    the uncorrected Langevin path, which draws no uniform, reports it too.

    Parameters
    ----------
    ratio : float
        As :func:`accept_ratio`.

    Returns
    -------
    float
        The acceptance probability.
    """
    return 0.0 if math.isnan(ratio) else min(1.0, ratio)


def accept_at(beta: float, difference: float, rng: np.random.Generator) -> bool:
    """:func:`accept` at inverse temperature ``beta``, with the ``beta = inf`` limit exact.

    A non-negative ``difference`` is accepted without a draw, and at
    ``beta = inf`` a negative one is refused without a draw --- the ``T = 0``
    limit, where a step that lowers the score is refused rather than accepted
    with probability zero.

    Parameters
    ----------
    beta : float
        Inverse temperature, non-negative; ``inf`` is the zero-temperature
        limit.
    difference : float
        The score change the move proposes, in the direction that goes up.
    rng : np.random.Generator
        Drawn from once, and only where the difference is negative and
        ``beta`` is finite.

    Returns
    -------
    bool
        Whether the proposal is accepted.
    """
    return difference >= 0.0 or (
        bool(np.isfinite(beta)) and accept(beta * difference, rng)
    )


def accept_at_temperature(
    temperature: float, difference: float, draw: Callable[[], float]
) -> bool:
    """:func:`accept_at` read in temperature, with the uniform behind a call.

    ``temperature = 0`` is ``beta = inf`` and refuses a negative difference.
    The quotient is ``difference / temperature`` rather than a reciprocal
    times the difference: the two disagree in the last bit, and the site this
    folds --- ``learn.potts_nd``'s flip, whose generator is keyed on the state
    and the action and so is built only where a draw is needed --- reads the
    quotient.

    Parameters
    ----------
    temperature : float
        Non-negative; ``0`` is the zero-temperature limit.
    difference : float
        As :func:`accept_at`.
    draw : Callable[[], float]
        As :func:`accept_drawn`.

    Returns
    -------
    bool
        Whether the proposal is accepted.
    """
    if difference >= 0.0:
        return True
    if temperature == 0.0:
        return False
    return accept_drawn(difference / temperature, draw)
