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
    #: Peak resident bytes of the first call: compilation, and the buffers
    #: XLA allocates then and reuses after, included (issue #997).
    first_peak_bytes: int = 0


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
    precision: np.ndarray | None,
    position: np.ndarray,
    momentum: np.ndarray,
    step_size: float,
    n_steps: int,
    weights: tuple[float, ...],
    *,
    rosenbrock: tuple[float, float] | None = None,
) -> Trajectory:
    """``n_steps`` steps of the composition ``weights`` in BlackJAX, on the Gaussian or on Rosenbrock's function."""
    result = run(
        SCRIPT,
        {
            "mode": np.asarray(0, dtype=np.int64),
            **_target(precision, rosenbrock),
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
    precision: np.ndarray | None,
    position: np.ndarray,
    step_size: float,
    n_steps: int,
    n_draws: int,
    key: int,
    *,
    store_chain: bool = True,
    rosenbrock: tuple[float, float] | None = None,
) -> Chain:
    """``n_draws`` BlackJAX HMC transitions at unit mass, from ``jax.random.key(key)``.

    ``key`` is the integer JAX builds its own PRNG key from, in the
    subprocess; no NumPy or torch generator crosses the boundary. With
    ``store_chain`` false no draw is kept and ``draws`` has zero rows
    (issue #997). ``rosenbrock = (a, b)`` targets Rosenbrock's function in
    place of the Gaussian of ``precision`` (issue #1008).
    """
    result = run(
        SCRIPT,
        {
            "mode": np.asarray(1, dtype=np.int64),
            **_target(precision, rosenbrock),
            "position": np.ascontiguousarray(position, dtype=np.float64),
            "step_size": np.asarray(step_size, dtype=np.float64),
            "n_steps": np.asarray(n_steps, dtype=np.int64),
            "n_draws": np.asarray(n_draws, dtype=np.int64),
            "key": np.asarray(key, dtype=np.int64),
            "store_chain": np.asarray(store_chain),
        },
    )
    out = result.outputs
    return Chain(
        out["draws"].reshape(-1, position.shape[0])
        if store_chain
        else out["draws"].reshape(0, position.shape[0]),
        float(out["acceptance"]),
        result.seconds,
        int(result.peak_bytes or 0),
        int(out["first_peak_bytes"]),
    )


def mala(
    precision: np.ndarray,
    position: np.ndarray,
    step_size: float,
    n_draws: int,
    key: int,
    *,
    store_chain: bool = True,
) -> Chain:
    """``n_draws`` BlackJAX MALA transitions at the package's Langevin step ``h`` (issue #997).

    BlackJAX's ``step_size`` is ``epsilon`` in ``x + epsilon grad log p +
    sqrt(2 epsilon) xi``, so it is passed ``h^2 / 2``: the same proposal
    :func:`snakes_and_ladders.sample.langevin.mala` makes at ``h``.
    """
    result = run(
        SCRIPT,
        {
            "mode": np.asarray(2, dtype=np.int64),
            "precision": np.ascontiguousarray(precision, dtype=np.float64),
            "position": np.ascontiguousarray(position, dtype=np.float64),
            "step_size": np.asarray(step_size**2 / 2.0, dtype=np.float64),
            "n_steps": np.asarray(1, dtype=np.int64),
            "n_draws": np.asarray(n_draws, dtype=np.int64),
            "key": np.asarray(key, dtype=np.int64),
            "store_chain": np.asarray(store_chain),
        },
    )
    out = result.outputs
    return Chain(
        out["draws"].reshape(-1, position.shape[0])
        if store_chain
        else out["draws"].reshape(0, position.shape[0]),
        float(out["acceptance"]),
        result.seconds,
        int(result.peak_bytes or 0),
        int(out["first_peak_bytes"]),
    )


def _target(
    precision: np.ndarray | None, rosenbrock: tuple[float, float] | None
) -> dict[str, np.ndarray]:
    """The script's target inputs: a Gaussian's precision or Rosenbrock's ``(a, b)``."""
    if rosenbrock is not None:
        return {
            "target": np.asarray(1, dtype=np.int64),
            "constants": np.asarray(rosenbrock, dtype=np.float64),
        }
    return {
        "target": np.asarray(0, dtype=np.int64),
        "precision": np.ascontiguousarray(precision, dtype=np.float64),
    }


def random_walk(
    position: np.ndarray,
    step_size: float | np.ndarray,
    n_draws: int,
    key: int,
    *,
    precision: np.ndarray | None = None,
    rosenbrock: tuple[float, float] | None = None,
    store_chain: bool = True,
) -> Chain:
    """``n_draws`` transitions of ``blackjax.additive_step_random_walk`` with a normal step (issue #1006).

    ``step_size`` is the step's standard deviation, a scalar or one per
    coordinate: the package's ``step_size`` times its metric's scale. The
    target is the Gaussian of ``precision`` or Rosenbrock's function with
    ``rosenbrock = (a, b)``.
    """
    result = run(
        SCRIPT,
        {
            "mode": np.asarray(3, dtype=np.int64),
            **_target(precision, rosenbrock),
            "position": np.ascontiguousarray(position, dtype=np.float64),
            "step_size": np.asarray(step_size, dtype=np.float64),
            "n_steps": np.asarray(1, dtype=np.int64),
            "n_draws": np.asarray(n_draws, dtype=np.int64),
            "key": np.asarray(key, dtype=np.int64),
            "store_chain": np.asarray(store_chain),
        },
    )
    out = result.outputs
    return Chain(
        out["draws"].reshape(-1 if store_chain else 0, position.shape[0]),
        float(out["acceptance"]),
        result.seconds,
        int(result.peak_bytes or 0),
        int(out["first_peak_bytes"]),
    )


@dataclass(frozen=True)
class Replay:
    """BlackJAX's ``rmh`` chain on supplied increments, and the uniforms it compared."""

    draws: np.ndarray
    uniforms: np.ndarray


def replay(
    position: np.ndarray,
    increments: np.ndarray,
    key: int,
    *,
    precision: np.ndarray | None = None,
    rosenbrock: tuple[float, float] | None = None,
) -> Replay:
    """BlackJAX's ``build_rmh`` kernel on ``increments``, one row per transition (issue #1006).

    Returns every position and the uniform each acceptance was decided on,
    so :func:`snakes_and_ladders.sample.metropolis.replay` runs the same chain.
    """
    result = run(
        SCRIPT,
        {
            "mode": np.asarray(4, dtype=np.int64),
            **_target(precision, rosenbrock),
            "position": np.ascontiguousarray(position, dtype=np.float64),
            "increments": np.ascontiguousarray(increments, dtype=np.float64),
            "step_size": np.asarray(1.0),
            "n_steps": np.asarray(1, dtype=np.int64),
            "n_draws": np.asarray(increments.shape[0], dtype=np.int64),
            "key": np.asarray(key, dtype=np.int64),
        },
    )
    return Replay(result.outputs["draws"], result.outputs["uniforms"])


@dataclass(frozen=True)
class AdaptedChain:
    """BlackJAX's HMC after ``window_adaptation``, and what the adaptation settled on."""

    chain: Chain
    step_size: float
    inverse_mass_matrix: np.ndarray


def adapted_sample(
    position: np.ndarray,
    step_size: float,
    n_steps: int,
    warmup: int,
    n_draws: int,
    key: int,
    target_acceptance: float,
    *,
    precision: np.ndarray | None = None,
    rosenbrock: tuple[float, float] | None = None,
    store_chain: bool = True,
) -> AdaptedChain:
    """``blackjax.window_adaptation`` over ``warmup`` steps, then ``n_draws`` HMC transitions (issue #1008).

    The warm-up and the draws are one compiled call, as the package's
    compiled chain runs them.
    """
    result = run(
        SCRIPT,
        {
            "mode": np.asarray(5, dtype=np.int64),
            **_target(precision, rosenbrock),
            "position": np.ascontiguousarray(position, dtype=np.float64),
            "step_size": np.asarray(step_size, dtype=np.float64),
            "n_steps": np.asarray(n_steps, dtype=np.int64),
            "warmup": np.asarray(warmup, dtype=np.int64),
            "n_draws": np.asarray(n_draws, dtype=np.int64),
            "key": np.asarray(key, dtype=np.int64),
            "target_acceptance": np.asarray(target_acceptance, dtype=np.float64),
            "store_chain": np.asarray(store_chain),
        },
    )
    out = result.outputs
    return AdaptedChain(
        Chain(
            out["draws"].reshape(-1 if store_chain else 0, position.shape[0]),
            float(out["acceptance"]),
            result.seconds,
            int(result.peak_bytes or 0),
            int(out["first_peak_bytes"]),
        ),
        float(out["step_size"]),
        np.asarray(out["inverse_mass_matrix"]),
    )
