"""BlackJAX as an oracle for the integrators and a timing reference for HMC (issue #963).

BlackJAX's integrators and HMC kernel share no code with
:mod:`snakes_and_ladders.sample.hmc`, and run on JAX, a second autodiff stack
the package does not take on: autodiff here is PyTorch by decision, so JAX
runs only in ``scripts/blackjax.py``, in a subprocess.

:func:`integrate` runs the package's :class:`~snakes_and_ladders.sample.hmc.Integrator`
composition in BlackJAX from one phase-space point. The sub-step weights are
converted to BlackJAX's alternating momentum and position coefficients: a
kick of half the first weight, a drift of it, a kick of the mean of it and
the next, and so on. The package merges the half-kicks between steps and
BlackJAX does not, so the two agree to rounding rather than bitwise.
:func:`sample` runs ``blackjax.hmc`` at unit mass.

The target is a zero-mean Gaussian given by its precision, dense or diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snakes_and_ladders.validation.runner import run

#: The script this adapter runs.
SCRIPT = "blackjax"


@dataclass(frozen=True)
class Trajectory:
    """BlackJAX's endpoint of one trajectory."""

    position: np.ndarray
    momentum: np.ndarray
    #: Wall seconds of the compiled trajectory, compilation excluded.
    seconds: float


@dataclass(frozen=True)
class Chain:
    """BlackJAX's HMC draws, one row per transition."""

    draws: np.ndarray
    #: The mean acceptance probability over the transitions.
    acceptance: float
    #: Wall seconds of the compiled chain, compilation excluded.
    seconds: float
    #: Peak resident bytes the compiled chain added, compilation excluded.
    peak_bytes: int


def coefficients(weights: tuple[float, ...]) -> np.ndarray:
    """BlackJAX's alternating kick and drift coefficients for a composition.

    ``weights`` are the sub-step lengths of
    :class:`~snakes_and_ladders.sample.hmc.Integrator`, each a kick-drift-kick
    of half, full, half its length; adjacent half-kicks merge.
    """
    sequence = [0.5 * weights[0]]
    for index, weight in enumerate(weights):
        following = weights[index + 1] if index + 1 < len(weights) else 0.0
        sequence += [weight, 0.5 * (weight + following)]
    return np.asarray(sequence, dtype=np.float64)


def integrate(
    precision: np.ndarray,
    position: np.ndarray,
    momentum: np.ndarray,
    step_size: float,
    n_steps: int,
    weights: tuple[float, ...],
) -> Trajectory:
    """``n_steps`` steps of the composition ``weights`` in BlackJAX."""
    result = run(
        SCRIPT,
        {
            "mode": np.asarray(0, dtype=np.int64),
            "precision": np.ascontiguousarray(precision, dtype=np.float64),
            "position": np.ascontiguousarray(position, dtype=np.float64),
            "momentum": np.ascontiguousarray(momentum, dtype=np.float64),
            "step_size": np.asarray(step_size, dtype=np.float64),
            "n_steps": np.asarray(n_steps, dtype=np.int64),
            "coefficients": coefficients(weights),
        },
    )
    out = result.outputs
    return Trajectory(out["position"], out["momentum"], result.seconds)


def sample(
    precision: np.ndarray,
    position: np.ndarray,
    step_size: float,
    n_steps: int,
    n_draws: int,
    key: int,
) -> Chain:
    """``n_draws`` BlackJAX HMC transitions at unit mass, from ``jax.random.key(key)``.

    ``key`` is the integer JAX builds its own PRNG key from, in the
    subprocess; no NumPy or torch generator crosses the boundary.
    """
    result = run(
        SCRIPT,
        {
            "mode": np.asarray(1, dtype=np.int64),
            "precision": np.ascontiguousarray(precision, dtype=np.float64),
            "position": np.ascontiguousarray(position, dtype=np.float64),
            "step_size": np.asarray(step_size, dtype=np.float64),
            "n_steps": np.asarray(n_steps, dtype=np.int64),
            "n_draws": np.asarray(n_draws, dtype=np.int64),
            "key": np.asarray(key, dtype=np.int64),
        },
    )
    out = result.outputs
    return Chain(
        out["draws"],
        float(out["acceptance"]),
        result.seconds,
        int(result.peak_bytes or 0),
    )
