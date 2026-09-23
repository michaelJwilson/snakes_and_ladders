"""BlackJAX's Hamiltonian dynamics on a Gaussian target the adapter wrote (issue #963).

The target is ``N(0, P^-1)``: ``precision`` is ``P``, a ``(d, d)`` matrix or
a ``(d,)`` diagonal. JAX runs in float64 (``jax_enable_x64``) at unit mass.

``mode`` 0 integrates: from ``position`` and ``momentum``, ``n_steps`` steps
of length ``step_size`` of the composition ``coefficients`` (alternating
momentum and position updates, palindromic), built by
``generate_euclidean_integrator``. Outputs ``position`` and ``momentum``.

``mode`` 1 samples: ``blackjax.hmc`` from ``position`` with ``step_size`` and
``n_steps`` leapfrog steps, ``n_draws`` transitions keyed from ``key``.
Outputs ``draws`` and ``acceptance``, the mean acceptance probability. With
``store_chain`` false the scan carries the state and emits only each
transition's acceptance, so no draw is stacked and ``draws`` is empty
(issue #997).

Each mode is compiled on one call first; the measured seconds are the second
call, to ``block_until_ready``, so compilation is not charged. It is reported
as ``compile_seconds``. The peak resident memory is the second call's too.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from snakes_and_ladders.validation.protocol import dump, load, paths, peaked

#: What ``mode`` selects.
INTEGRATE, SAMPLE = 0, 1


def main() -> None:
    """Build the target, run the requested mode twice, and write the second back."""
    import jax  # the framework, imported only in this interpreter

    jax.config.update("jax_enable_x64", True)
    import blackjax
    import jax.numpy as jnp
    from blackjax.mcmc import integrators, metrics

    given, returned = paths()
    inputs = load(given)
    precision = jnp.asarray(inputs["precision"])
    dimension = int(inputs["position"].size)

    def logdensity(x: Any) -> Any:
        if precision.ndim == 1:
            return -0.5 * jnp.sum(precision * x * x)
        return -0.5 * x @ (precision @ x)

    step_size = float(inputs["step_size"])
    n_steps = int(inputs["n_steps"])
    unit = jnp.ones(dimension)

    if int(inputs["mode"]) == INTEGRATE:
        metric = metrics.default_metric(unit)
        one_step = integrators.generate_euclidean_integrator(
            tuple(float(c) for c in inputs["coefficients"])
        )(logdensity, metric.kinetic_energy)

        @jax.jit
        def run(position: Any, momentum: Any) -> Any:
            state = integrators.new_integrator_state(logdensity, position, momentum)
            return jax.lax.fori_loop(
                0, n_steps, lambda _, s: one_step(s, step_size), state
            )

        arguments = (jnp.asarray(inputs["position"]), jnp.asarray(inputs["momentum"]))
    else:
        kernel = blackjax.hmc(
            logdensity,
            step_size=step_size,
            inverse_mass_matrix=unit,
            num_integration_steps=n_steps,
        )
        keys = jax.random.split(
            jax.random.key(int(inputs["key"])), int(inputs["n_draws"])
        )

        store_chain = bool(inputs.get("store_chain", np.asarray(True)))

        @jax.jit
        def run(position: Any, keys: Any) -> Any:
            def transition(state: Any, key: Any) -> tuple[Any, Any]:
                state, info = kernel.step(key, state)
                if store_chain:
                    return state, (state.position, info.acceptance_rate)
                return state, (jnp.zeros((0,)), info.acceptance_rate)

            return jax.lax.scan(transition, kernel.init(position), keys)[1]

        arguments = (jnp.asarray(inputs["position"]), keys)

    start = time.perf_counter()
    jax.block_until_ready(run(*arguments))
    compile_seconds = time.perf_counter() - start

    def timed_run() -> tuple[Any, float]:
        start = time.perf_counter()
        result = jax.block_until_ready(run(*arguments))
        return result, time.perf_counter() - start

    (result, seconds), peak_bytes = peaked(timed_run)

    if int(inputs["mode"]) == INTEGRATE:
        outputs = {
            "position": np.asarray(result.position),
            "momentum": np.asarray(result.momentum),
        }
    else:
        draws, acceptance = result
        outputs = {
            "draws": np.asarray(draws),
            "acceptance": np.asarray(np.mean(np.asarray(acceptance))),
        }
    outputs["compile_seconds"] = np.asarray(compile_seconds - seconds)
    dump(returned, outputs, seconds, peak_bytes)


if __name__ == "__main__":
    main()
