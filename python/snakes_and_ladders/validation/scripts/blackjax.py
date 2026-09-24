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

``mode`` 2 samples by ``blackjax.mala`` at ``step_size``, which is BlackJAX's
``epsilon`` in ``x + epsilon grad log p + sqrt(2 epsilon) xi``: the
package's Langevin step ``h`` is ``epsilon = h^2 / 2``. Outputs as mode 1
(issue #997).

``mode`` 3 samples by ``blackjax.additive_step_random_walk`` with a normal
step of standard deviation ``step_size`` per coordinate (a scalar or one per
coordinate), ``n_draws`` transitions keyed from ``key``; outputs as mode 1
(issue #1006). ``target`` 1 is Rosenbrock's function with ``constants``
``(a, b)`` in place of the Gaussian: ``log p = -U``.

``mode`` 4 replays: ``blackjax``'s ``build_rmh`` kernel with the increments
``increments`` supplied, one row per transition, and each transition's key
from ``key`` as mode 3 splits it. Outputs ``draws`` and ``uniforms``, the
uniform ``jax.random.bernoulli`` compared against on each transition's
acceptance key, so the package's :func:`~snakes_and_ladders.sample.metropolis.replay`
runs the same chain on the same randomness (issue #1006).

``mode`` 5 is mode 1 after ``blackjax.window_adaptation`` over ``warmup``
steps from ``position`` at ``target_acceptance``, the adapted step and
inverse mass then fixed for the draws; the warm-up and the draws run in one
compiled call. Outputs as mode 1, and ``step_size`` and
``inverse_mass_matrix``, the adapted values (issue #1008).

Each mode is compiled on one call first; the measured seconds are the second
call, to ``block_until_ready``, so compilation is not charged. It is reported
as ``compile_seconds``. The peak resident memory is the second call's too;
``first_peak_bytes`` is the first call's, compilation and the buffers XLA
keeps for later calls included (issue #997).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from snakes_and_ladders.validation.protocol import dump, load, paths, peaked

#: What ``mode`` selects.
INTEGRATE, SAMPLE, LANGEVIN, RANDOM_WALK, REPLAY, ADAPTED = 0, 1, 2, 3, 4, 5

#: What ``target`` selects.
GAUSSIAN, ROSENBROCK = 0, 1


def main() -> None:
    """Build the target, run the requested mode twice, and write the second back."""
    import jax  # the framework, imported only in this interpreter

    jax.config.update("jax_enable_x64", True)
    import blackjax
    import jax.numpy as jnp
    from blackjax.mcmc import integrators, metrics

    given, returned = paths()
    inputs = load(given)
    dimension = int(inputs["position"].size)
    mode = int(inputs["mode"])

    if int(inputs.get("target", np.asarray(GAUSSIAN))) == ROSENBROCK:
        a, b = (float(c) for c in inputs["constants"])

        def logdensity(x: Any) -> Any:
            head, tail = x[:-1], x[1:]
            return -jnp.sum(b * (tail - head**2) ** 2 + (a - head) ** 2)

    else:
        precision = jnp.asarray(inputs["precision"])

        def logdensity(x: Any) -> Any:
            if precision.ndim == 1:
                return -0.5 * jnp.sum(precision * x * x)
            return -0.5 * x @ (precision @ x)

    step_size = float(np.asarray(inputs["step_size"]).reshape(-1)[0])
    n_steps = int(inputs["n_steps"])
    unit = jnp.ones(dimension)

    if mode == INTEGRATE:
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

        arguments: tuple[Any, ...] = (
            jnp.asarray(inputs["position"]),
            jnp.asarray(inputs["momentum"]),
        )
    elif mode == REPLAY:
        from blackjax.mcmc import random_walk

        rmh = random_walk.build_rmh()
        keys = jax.random.split(
            jax.random.key(int(inputs["key"])), int(inputs["n_draws"])
        )

        @jax.jit
        def run(position: Any, steps: Any) -> Any:
            def transition(state: Any, step: Any) -> tuple[Any, Any]:
                key, increment = step
                state, _ = rmh(key, state, logdensity, lambda _, x: x + increment)
                # `rmh_proposal` splits the key into the proposal's and the
                # acceptance's, and `bernoulli` compares one uniform on the
                # second against the probability.
                accept_key = jax.random.split(key, 2)[1]
                return state, (state.position, jax.random.uniform(accept_key))

            return jax.lax.scan(
                transition, random_walk.init(position, logdensity), steps
            )[1]

        arguments = (
            jnp.asarray(inputs["position"]),
            (keys, jnp.asarray(inputs["increments"])),
        )
    elif mode == ADAPTED:
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            logdensity,
            num_integration_steps=n_steps,
            target_acceptance_rate=float(inputs["target_acceptance"]),
        )
        n_warmup, n_draws = int(inputs["warmup"]), int(inputs["n_draws"])
        store_chain = bool(inputs.get("store_chain", np.asarray(True)))

        @jax.jit
        def run(position: Any, key: Any) -> Any:
            warm_key, draw_key = jax.random.split(key)
            (state, parameters), _ = warmup.run(warm_key, position, num_steps=n_warmup)
            kernel = blackjax.hmc(logdensity, **parameters)

            def transition(state: Any, key: Any) -> tuple[Any, Any]:
                state, info = kernel.step(key, state)
                if store_chain:
                    return state, (state.position, info.acceptance_rate)
                return state, (jnp.zeros((0,)), info.acceptance_rate)

            draws = jax.lax.scan(
                transition, state, jax.random.split(draw_key, n_draws)
            )[1]
            return draws, parameters["step_size"], parameters["inverse_mass_matrix"]

        arguments = (jnp.asarray(inputs["position"]), jax.random.key(int(inputs["key"])))
    else:
        if mode == RANDOM_WALK:
            from blackjax.mcmc import random_walk

            kernel = blackjax.additive_step_random_walk(
                logdensity,
                random_walk.normal(
                    jnp.broadcast_to(jnp.asarray(inputs["step_size"]), (dimension,))
                ),
            )
        elif mode == LANGEVIN:
            kernel = blackjax.mala(logdensity, step_size=step_size)
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
    _, first_peak_bytes = peaked(lambda: jax.block_until_ready(run(*arguments)))
    compile_seconds = time.perf_counter() - start

    def timed_run() -> tuple[Any, float]:
        start = time.perf_counter()
        result = jax.block_until_ready(run(*arguments))
        return result, time.perf_counter() - start

    (result, seconds), peak_bytes = peaked(timed_run)

    if mode == INTEGRATE:
        outputs = {
            "position": np.asarray(result.position),
            "momentum": np.asarray(result.momentum),
        }
    elif mode == ADAPTED:
        (draws, acceptance), adapted_step, inverse_mass = result
        outputs = {
            "draws": np.asarray(draws),
            "acceptance": np.asarray(np.mean(np.asarray(acceptance))),
            "step_size": np.asarray(adapted_step),
            "inverse_mass_matrix": np.asarray(inverse_mass),
        }
    elif mode == REPLAY:
        draws, uniforms = result
        outputs = {"draws": np.asarray(draws), "uniforms": np.asarray(uniforms)}
    else:
        draws, acceptance = result
        outputs = {
            "draws": np.asarray(draws),
            "acceptance": np.asarray(np.mean(np.asarray(acceptance))),
        }
    outputs["compile_seconds"] = np.asarray(compile_seconds - seconds)
    outputs["first_peak_bytes"] = np.asarray(first_peak_bytes)
    dump(returned, outputs, seconds, peak_bytes)


if __name__ == "__main__":
    main()
