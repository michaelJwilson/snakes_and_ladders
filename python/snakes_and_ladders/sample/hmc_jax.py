"""HMC on an objective with a traceable JAX energy, the chain under ``jit`` (issue #1008).

The compiled Rust walks run declared families; an objective whose energy is a
JAX function (:class:`~snakes_and_ladders.sample.declared.DeclaredJaxEnergy`,
the HMM objectives under ``Backend.JAX``) cannot cross into Rust, but its
whole chain can be traced: the leapfrog under ``fori_loop``, the transitions
under ``lax.scan``, as BlackJAX runs its own. :class:`JaxWalk` is that chain
with the face of ``oxisal.HmcWalk`` --- built with the warm-up run,
``advance`` in blocks, the ``Power`` operators' Kalman statistics, the step
and metric it settled on --- so :func:`snakes_and_ladders.sample.hmc.run_compiled`
drives it unchanged.

The arithmetic is ``src/hmc.rs``'s and ``src/chain.rs``'s: the gradient at
the current point carried from the trajectory that reached it, drifts of
``s * p`` and kicks of ``s * grad U`` on the metric of scale ``s``, the two
dual-averaging windows of ``chain._warm_up`` (Welford over the first window's
second half), and ``KalmanMean``'s six statistics per operator. The stream is
JAX's, keyed by one draw from the caller's generator, so the torch route is
matched in distribution.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np


def _jax() -> Any:
    import jax  # a core dependency, imported where it is used

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    return jax


@functools.cache
def _programs(
    energy: Callable[[Any, Any], Any], n_steps: int, jitter: bool
) -> tuple[Any, Any, Any]:
    """The warm-up window, the block of transitions and the first gradient, compiled per energy."""
    jax = _jax()
    jnp = jax.numpy
    value_and_grad = jax.value_and_grad(energy)

    def transition(
        state: tuple[Any, Any, Any], key: Any, step: Any, scale: Any, data: Any
    ) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
        x, u, g = state
        momentum_key, uniform_key, jitter_key = jax.random.split(key, 3)
        if jitter:
            step = step[0] * (
                1.0 + step[1] * (2.0 * jax.random.uniform(jitter_key) - 1.0)
            )
        else:
            step = step[0]
        p = jax.random.normal(momentum_key, x.shape)
        current = u + 0.5 * jnp.dot(p, p)
        p1 = p - 0.5 * step * scale * g

        def body(
            index: Any, carry: tuple[Any, Any, Any, Any]
        ) -> tuple[Any, Any, Any, Any]:
            y, q, _, _ = carry
            y = y + step * scale * q
            value, gradient = value_and_grad(y, data)
            kick = jnp.where(index + 1 == n_steps, 0.5, 1.0)
            return y, q - kick * step * scale * gradient, gradient, value

        y, q, gradient, value = jax.lax.fori_loop(0, n_steps, body, (x, p1, g, u))
        proposed = value + 0.5 * jnp.dot(q, q)
        log_ratio = current - proposed
        ratio = jnp.exp(log_ratio)
        probability = jnp.where(jnp.isnan(ratio), 0.0, jnp.minimum(ratio, 1.0))
        take = jax.random.uniform(uniform_key) < ratio
        state = jax.tree.map(
            lambda new, old: jnp.where(take, new, old), (y, value, gradient), state
        )
        return state, (take, probability, jnp.abs(proposed - current))

    def dual(
        averaging: tuple[Any, ...], probability: Any, target: Any, constants: Any
    ) -> tuple[Any, ...]:
        # `chain._DualAveraging.update`, term for term.
        mu, h_bar, log_step, log_averaged, m = averaging
        gamma, t0, kappa = constants
        m = m + 1.0
        weight = 1.0 / (m + t0)
        h_bar = (1.0 - weight) * h_bar + weight * (target - probability)
        log_step = mu - jnp.sqrt(m) / gamma * h_bar
        forget = m ** (-kappa)
        log_averaged = forget * log_step + (1.0 - forget) * log_averaged
        return mu, h_bar, log_step, log_averaged, m

    @functools.partial(jax.jit, static_argnames=("length", "record_from"))
    def window(
        state: Any,
        keys: Any,
        averaging: Any,
        jitter_width: Any,
        scale: Any,
        target: Any,
        constants: Any,
        data: Any,
        length: int,
        record_from: int,
    ) -> Any:
        d = state[0].shape[0]

        def body(carry: Any, indexed: Any) -> tuple[Any, None]:
            state, averaging, count, mean, m2, total = carry
            index, key = indexed
            step = jnp.stack([jnp.exp(averaging[2]), jitter_width])
            state, (_, probability, _) = transition(state, key, step, scale, data)
            averaging = dual(averaging, probability, target, constants)
            recording = index >= record_from
            count = count + jnp.where(recording, 1.0, 0.0)
            delta = state[0] - mean
            mean = jnp.where(recording, mean + delta / jnp.maximum(count, 1.0), mean)
            m2 = jnp.where(recording, m2 + delta * (state[0] - mean), m2)
            return (state, averaging, count, mean, m2, total + probability), None

        start = (state, averaging, 0.0, jnp.zeros(d), jnp.zeros(d), 0.0)
        return jax.lax.scan(body, start, (jnp.arange(length), keys))[0]

    @functools.partial(jax.jit, static_argnames=("store", "observe", "powers"))
    def block(
        state: Any,
        keys: Any,
        step: Any,
        scale: Any,
        statistics: Any,
        data: Any,
        store: bool,
        observe: bool,
        powers: tuple[int, ...],
    ) -> Any:
        def body(carry: Any, key: Any) -> tuple[Any, Any]:
            state, statistics = carry
            state, (take, _, error) = transition(state, key, step, scale, data)
            if observe:
                updated = []
                for power, (n, total, squares, lagged, first, last) in zip(
                    powers, statistics, strict=True
                ):
                    y = state[0] ** power
                    fresh = n == 0
                    updated.append(
                        (
                            n + 1,
                            total + y,
                            squares + y * y,
                            lagged + jnp.where(fresh, 0.0, y * last),
                            jnp.where(fresh, y, first),
                            y,
                        )
                    )
                statistics = tuple(updated)
            draw = state[0] if store else jnp.zeros((0,))
            return (state, statistics), (draw, take, error)

        return jax.lax.scan(body, (state, statistics), keys)

    start = jax.jit(value_and_grad)
    return window, block, start


class JaxWalk:
    """A chain on a JAX energy with ``oxisal.HmcWalk``'s face; see the module docs.

    ``declared`` is ``(energy, data)`` from ``jax_energy()``; the other
    arguments are the Rust walks', in their order, and ``n_steps`` the
    leapfrog steps per transition.
    """

    def __init__(
        self,
        energy: Callable[[Any, Any], Any],
        data: Any,
        theta0: np.ndarray,
        step_size: float,
        seed: int,
        warmup: int,
        target_acceptance: float,
        step_jitter: float,
        constants: tuple[float, float, float],
        powers: list[int],
        n_steps: int,
    ) -> None:
        if n_steps < 1:
            msg = f"n_steps must be at least 1, got {n_steps}"
            raise ValueError(msg)
        jax = _jax()
        jnp = jax.numpy
        self._jnp = jnp
        self._window, self._block, start = _programs(energy, n_steps, step_jitter > 0.0)
        self._data = data
        self._key = jax.random.key(seed % 2**63)
        self._powers = tuple(powers)
        x = jnp.asarray(theta0, dtype=jnp.float64)
        u, g = start(x, data)
        self._state = (x, u, g)
        d = int(x.shape[0])
        self._scale = jnp.ones(d)
        self._jitter = step_jitter
        self.mass_diagonal = np.empty(0)
        self.warmup_acceptance = 0.0
        self.step_size = step_size
        if warmup:
            self._warm_up(step_size, warmup, target_acceptance, constants)
        zero = jnp.zeros(d)
        self._statistics = tuple(
            (0, zero, zero, zero, zero, zero) for _ in self._powers
        )

    def _keys(self, n: int) -> Any:
        jax = _jax()
        self._key, drawn = jax.random.split(self._key)
        return jax.random.split(drawn, n)

    def _warm_up(
        self,
        step_size: float,
        warmup: int,
        target: float,
        constants: tuple[float, float, float],
    ) -> None:
        jnp = self._jnp
        first = warmup // 2
        second = warmup - first

        def averaging(step: float) -> tuple[Any, ...]:
            return (jnp.log(10.0 * step), 0.0, jnp.log(step), 0.0, 0.0)

        state, dual, count, _, m2, _ = self._window(
            self._state,
            self._keys(first),
            averaging(step_size),
            self._jitter,
            self._scale,
            target,
            constants,
            self._data,
            length=first,
            record_from=first // 2,
        )
        variance = np.asarray(m2 / (count - 1.0))
        if not bool((variance > 0.0).all()):
            stuck = np.flatnonzero(~(variance > 0.0)).tolist()
            msg = (
                f"warm-up variance is zero on coordinate(s) {stuck}: the chain did "
                "not move there, so no mass can be estimated; lengthen the warm-up "
                "or start the step size smaller"
            )
            raise ValueError(msg)
        self._scale = jnp.sqrt(jnp.asarray(variance))
        averaged = float(np.exp(dual[3]))
        # `chain._warm_up`: window two restarts dual averaging from window
        # one's averaged step, and its first step is the new iteration's
        # averaged one, exp(0).
        restarted = (jnp.log(10.0 * averaged), 0.0, 0.0, 0.0, 0.0)
        state, dual, _, _, _, total = self._window(
            state,
            self._keys(second),
            restarted,
            self._jitter,
            self._scale,
            target,
            constants,
            self._data,
            length=second,
            record_from=second,
        )
        self._state = state
        self.step_size = float(np.exp(dual[3]))
        self.mass_diagonal = 1.0 / variance
        self.warmup_acceptance = float(total) / second

    def advance(
        self, n: int, store: bool, observe: bool
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """``n`` transitions: the draws flattened (empty unless ``store``), the count accepted and ``|dH|`` each."""
        if n == 0:
            return np.empty(0), 0, np.empty(0)
        step = self._jnp.asarray([self.step_size, self._jitter])
        (self._state, statistics), (draws, taken, errors) = self._block(
            self._state,
            self._keys(n),
            step,
            self._scale,
            self._statistics,
            self._data,
            store=store,
            observe=observe and bool(self._powers),
            powers=self._powers,
        )
        if observe:
            self._statistics = statistics
        # Copied: a device buffer is read-only, and torch will not wrap one.
        return (
            np.array(draws).reshape(-1),
            int(np.asarray(taken).sum()),
            np.asarray(errors),
        )

    def statistics(self, index: int) -> tuple[int, tuple[np.ndarray, ...]]:
        """Operator ``index``'s filter statistics, as ``oxisal``'s walks return them."""
        n, *sums = self._statistics[index]
        return int(n), tuple(np.array(value) for value in sums)
