"""JAX's value and gradient of the HMC objectives the adapter wrote (issue #991).

``target`` names the objective, written here in ``jax.numpy`` to mirror the
package's:

- ``"gaussian"``: ``0.5 x' P x`` with ``precision`` ``P``, dense or diagonal
  (:class:`snakes_and_ladders.validation.gaussian.GaussianTarget`);
- ``"mixture"``: the negative log-likelihood of ``observations`` under a
  one-dimensional mixture of ``n_components`` Gaussians, over ``theta =
  (k - 1 free weights, k means, k log scales)`` with the weights
  ``log_softmax([0, free])``
  (:class:`snakes_and_ladders.opt.mixture.GaussianMixtureObjective`).

At each row of ``points`` it returns ``values`` and ``gradients`` from
``jax.value_and_grad`` in float64. Three timings, each the median over the
points after one warm-up call: ``eager_seconds`` per point without
compilation, ``seconds`` per point under ``jit`` (the measured seconds), and
``batch_seconds`` for all points at once under ``jit(vmap)``. The peak
resident memory is the ``jit`` loop's.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from snakes_and_ladders.validation.protocol import dump, load, paths, peaked


def _per_point(call: Any, points: Any, jax: Any) -> float:
    """The median seconds of ``call`` over the rows of ``points``, after a warm-up."""
    jax.block_until_ready(call(points[0]))
    seconds = []
    for row in points:
        start = time.perf_counter()
        jax.block_until_ready(call(row))
        seconds.append(time.perf_counter() - start)
    return statistics.median(seconds)


def _gaussian(precision: Any, jnp: Any) -> Callable[[Any], Any]:
    """``0.5 x' P x``, with ``P`` diagonal when given as a vector."""

    def objective(x: Any) -> Any:
        if precision.ndim == 1:
            return 0.5 * jnp.sum(precision * x * x)
        return 0.5 * x @ (precision @ x)

    return objective


def _mixture(observations: Any, k: int, jax: Any) -> Callable[[Any], Any]:
    """The mixture's negative log-likelihood over ``(free weights, means, log scales)``."""
    jnp = jax.numpy

    def objective(theta: Any) -> Any:
        log_weight = jax.nn.log_softmax(jnp.concatenate([jnp.zeros(1), theta[: k - 1]]))
        mean = theta[k - 1 : 2 * k - 1]
        log_scale = theta[2 * k - 1 : 3 * k - 1]
        standard = (observations[:, None] - mean[None, :]) / jnp.exp(log_scale)
        log_density = (
            -0.5 * standard * standard - log_scale - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        return -jnp.sum(jax.nn.logsumexp(log_weight + log_density, axis=1))

    return objective


def main() -> None:
    """Differentiate the target at every point three ways and write the answers back."""
    import jax  # the framework, imported only in this interpreter

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    given, returned = paths()
    inputs = load(given)
    target = str(inputs["target"])

    objective = (
        _gaussian(jnp.asarray(inputs["precision"]), jnp)
        if target == "gaussian"
        else _mixture(
            jnp.asarray(inputs["observations"]), int(inputs["n_components"]), jax
        )
    )

    points = jnp.asarray(inputs["points"])
    eager = jax.value_and_grad(objective)
    compiled = jax.jit(eager)
    batched = jax.jit(jax.vmap(eager))

    with jax.disable_jit():
        eager_seconds = _per_point(eager, points, jax)

    seconds, peak_bytes = peaked(lambda: _per_point(compiled, points, jax))
    values, gradients = jax.block_until_ready(batched(points))
    start = time.perf_counter()
    jax.block_until_ready(batched(points))
    batch_seconds = time.perf_counter() - start

    dump(
        returned,
        {
            "values": np.asarray(values),
            "gradients": np.asarray(gradients),
            "eager_seconds": np.asarray(eager_seconds),
            "batch_seconds": np.asarray(batch_seconds),
        },
        seconds,
        peak_bytes,
    )


if __name__ == "__main__":
    main()
