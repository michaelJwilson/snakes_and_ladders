"""BlackJAX's Hamiltonian dynamics on a Gaussian target the adapter wrote (issue #963).

The target is ``N(0, P^-1)``: ``precision`` is ``P``, a ``(d, d)`` matrix or
a ``(d,)`` diagonal. JAX runs in float64 (``jax_enable_x64``) at unit mass.

``mode`` 0 integrates: from ``position`` and ``momentum``, ``n_steps`` steps
of length ``step_size`` of the composition ``coefficients`` (alternating
momentum and position updates, palindromic), built by
``generate_euclidean_integrator``. Outputs ``position`` and ``momentum``.

``mode`` 1 samples: ``blackjax.hmc`` from ``position`` with ``step_size`` and
``n_steps`` leapfrog steps, ``n_draws`` transitions keyed from ``key``.
Outputs ``draws`` and ``acceptance``, the mean acceptance probability; and
for :mod:`sal.external.hmc` (issue #1282) ``accepted``, the fraction
accepted, ``energy_error``, ``|H(proposal) - H(current)|`` per transition, and
``potential``, ``-log p`` at each draw. With ``store_chain`` false the scan
carries the state and emits only each transition's acceptance and energy
error, so no draw is stacked and ``draws`` and ``potential`` are empty
(issue #997).

``mode`` 2 samples by ``blackjax.mala`` at ``step_size``, which is BlackJAX's
``epsilon`` in ``x + epsilon grad log p + sqrt(2 epsilon) xi``: the
package's Langevin step ``h`` is ``epsilon = h^2 / 2``. Outputs as mode 1
(issue #997).

``mode`` 3 samples by ``blackjax.additive_step_random_walk`` with a normal
step of standard deviation ``step_size`` per coordinate (a scalar or one per
coordinate), ``n_draws`` transitions keyed from ``key``; outputs as mode 1
(issue #1006). ``target`` 1 is Rosenbrock's function with ``constants``
``(a, b)`` in place of the Gaussian: ``log p = -U``; ``target`` 2 is a
Gaussian mixture's log-likelihood of ``values`` at ``n_components``, in
``GaussianMixtureObjective``'s ``theta``; ``target`` 3 a Gaussian HMM's of
the sequences ``values`` at ``n_states``, in a Gaussian ``EmissionHmmObjective``'s,
by the textbook log-space forward recursion (issue #1008).

``mode`` 4 replays: ``blackjax``'s ``build_rmh`` kernel with the increments
``increments`` supplied, one row per transition, and each transition's key
from ``key`` as mode 3 splits it. Outputs ``draws`` and ``uniforms``, the
uniform ``jax.random.bernoulli`` compared against on each transition's
acceptance key, so the package's :func:`~sal.sample.metropolis.replay`
runs the same chain on the same randomness (issue #1006).

``mode`` 5 is mode 1 after ``blackjax.window_adaptation`` over ``warmup``
steps from ``position`` at ``target_acceptance``, the adapted step and
inverse mass then fixed for the draws; the warm-up and the draws run in one
compiled call. Outputs as mode 1, and ``step_size`` and
``inverse_mass_matrix``, the adapted values (issue #1008). The warm-up starts
at ``initial_step_size`` where it is sent, and at BlackJAX's default, 1,
where it is not; ``warmup_acceptance``, ``flat`` and ``window`` are
:func:`warmup_outputs`'s (issue #1282).

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

from sal.external.hmc_inputs import MODES, TARGETS
from sal.external.protocol import dump, peaked, received

#: What ``mode`` selects, by :data:`~sal.external.hmc_inputs.MODES`'s codes.
INTEGRATE, SAMPLE, LANGEVIN, RANDOM_WALK, REPLAY, ADAPTED = range(len(MODES))

#: What ``target`` selects, by :data:`~sal.external.hmc_inputs.TARGETS`'s codes.
GAUSSIAN, ROSENBROCK, MIXTURE, GAUSSIAN_HMM = range(len(TARGETS))


def chain_outputs(
    positions: Any, acceptance: Any, accepted: Any, energy_error: Any, potential: Any
) -> dict[str, np.ndarray]:
    """A recorded chain's outputs: draws, mean acceptance probability and the rest per transition."""
    return {
        "draws": np.asarray(positions),
        "acceptance": np.asarray(np.mean(np.asarray(acceptance))),
        "accepted": np.asarray(np.mean(np.asarray(accepted, dtype=np.float64))),
        "energy_error": np.asarray(energy_error, dtype=np.float64),
        "potential": np.asarray(potential, dtype=np.float64),
    }


def warmup_outputs(
    n_warmup: int, acceptance: np.ndarray, positions: np.ndarray
) -> dict[str, np.ndarray]:
    """What the window adaptation's schedule says of its own warm-up.

    ``warmup_acceptance`` is the mean acceptance probability over the final
    fast window, the step size's last; ``flat`` the coordinates that did not
    move over the last slow window, the one the inverse mass is estimated
    from. Without a slow window (fewer than 20 steps) every step is fast and
    ``flat`` is empty.
    """
    from blackjax.adaptation.window_adaptation import build_schedule

    schedule = np.asarray(build_schedule(n_warmup))
    slow = np.flatnonzero(schedule[:, 0] == 1)
    if slow.size == 0:
        return {
            "warmup_acceptance": np.asarray(np.mean(acceptance)),
            "flat": np.zeros(0, dtype=np.int64),
            "window": np.asarray(0, dtype=np.int64),
        }
    ends = np.flatnonzero(schedule[:, 1])
    first = int(ends[-2]) + 1 if ends.size > 1 else int(slow[0])
    window = positions[first : int(ends[-1]) + 1]
    moved = np.ptp(window.reshape(window.shape[0], -1), axis=0)
    return {
        "warmup_acceptance": np.asarray(np.mean(acceptance[int(ends[-1]) + 1 :])),
        "flat": np.flatnonzero(moved == 0.0).astype(np.int64),
        "window": np.asarray(window.shape[0], dtype=np.int64),
    }


def main() -> None:
    """Build the target, run the requested mode twice, and write the second back."""
    import jax  # the framework, imported only in this interpreter

    jax.config.update("jax_enable_x64", True)
    import blackjax
    import jax.numpy as jnp
    from blackjax.mcmc import integrators, metrics

    inputs, returned = received()
    dimension = int(inputs["position"].size)
    mode = int(inputs["mode"])

    target = int(inputs.get("target", np.asarray(GAUSSIAN)))
    if target == GAUSSIAN_HMM:
        sequences = jnp.asarray(inputs["values"])
        m = int(inputs["n_states"])

        def logdensity(x: Any) -> Any:
            # A Gaussian `EmissionHmmObjective`'s theta: m - 1 free initial, m (m - 1)
            # free transition rows, m means, m log scales; the textbook
            # forward recursion in log space, one sequence per row.
            def simplex(free: Any) -> Any:
                pinned = jnp.concatenate(
                    [jnp.zeros(free.shape[:-1] + (1,)), free], axis=-1
                )
                return jax.nn.log_softmax(pinned, axis=-1)

            log_initial = simplex(x[: m - 1])
            log_transition = simplex(x[m - 1 : m * m - 1].reshape(m, m - 1))
            mean, log_scale = x[m * m - 1 : m * m - 1 + m], x[m * m - 1 + m :]

            def one(y: Any) -> Any:
                z = (y[:, None] - mean) * jnp.exp(-log_scale)
                emit = -0.5 * jnp.log(2 * jnp.pi) - log_scale - 0.5 * z * z

                def step(alpha: Any, row: Any) -> tuple[Any, None]:
                    alpha = jax.scipy.special.logsumexp(
                        alpha[:, None] + log_transition, axis=0
                    )
                    return alpha + row, None

                alpha, _ = jax.lax.scan(step, log_initial + emit[0], emit[1:])
                return jax.scipy.special.logsumexp(alpha)

            return jnp.sum(jax.vmap(one)(sequences))

    elif target == MIXTURE:
        values = jnp.asarray(inputs["values"])
        k = int(inputs["n_components"])

        def logdensity(x: Any) -> Any:
            # `GaussianMixtureObjective`'s theta: k - 1 free weights, k
            # means, k log scales, the weights a softmax pinned at zero.
            log_weight = jax.nn.log_softmax(jnp.concatenate([jnp.zeros(1), x[: k - 1]]))
            mean, log_scale = x[k - 1 : 2 * k - 1], x[2 * k - 1 :]
            z = (values[:, None] - mean) * jnp.exp(-log_scale)
            joint = log_weight - log_scale - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * z * z
            return jnp.sum(jax.scipy.special.logsumexp(joint, axis=1))

    elif target == ROSENBROCK:
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

    store_chain = bool(inputs.get("store_chain", np.asarray(True)))

    def recorded(kernel: Any, metric: Any) -> Any:
        # One HMC transition, and what `sal.external.hmc` reads off it: the
        # position, the acceptance probability, whether it was accepted,
        # |H(proposal) - H(current)| at the momentum drawn, and the
        # potential -log p at the position, which a caller checks its own
        # energy against.
        def transition(state: Any, key: Any) -> tuple[Any, Any]:
            new, info = kernel.step(key, state)
            current = -state.logdensity + metric.kinetic_energy(info.momentum)
            kept = (
                (new.position, -new.logdensity)
                if store_chain
                else (
                    jnp.zeros((0,)),
                    jnp.zeros(()),
                )
            )
            return new, (
                kept[0],
                info.acceptance_rate,
                info.is_accepted,
                jnp.abs(info.energy - current),
                kept[1],
            )

        return transition

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
        initial = (
            {"initial_step_size": float(inputs["initial_step_size"])}
            if "initial_step_size" in inputs
            else {}
        )
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            logdensity,
            num_integration_steps=n_steps,
            target_acceptance_rate=float(inputs["target_acceptance"]),
            **initial,
        )
        n_warmup, n_draws = int(inputs["warmup"]), int(inputs["n_draws"])

        @jax.jit
        def run(position: Any, key: Any) -> Any:
            warm_key, draw_key = jax.random.split(key)
            (state, parameters), adapting = warmup.run(
                warm_key, position, num_steps=n_warmup
            )
            kernel = blackjax.hmc(logdensity, **parameters)
            transition = recorded(
                kernel, metrics.default_metric(parameters["inverse_mass_matrix"])
            )
            draws = jax.lax.scan(
                transition, state, jax.random.split(draw_key, n_draws)
            )[1]
            return (
                draws,
                parameters["step_size"],
                parameters["inverse_mass_matrix"],
                adapting.info.acceptance_rate,
                adapting.state.position,
            )

        arguments = (
            jnp.asarray(inputs["position"]),
            jax.random.key(int(inputs["key"])),
        )
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

        @jax.jit
        def run(position: Any, keys: Any) -> Any:
            if mode == SAMPLE:
                transition = recorded(kernel, metrics.default_metric(unit))
            else:

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
        draws, adapted_step, inverse_mass, warm_acceptance, warm_positions = result
        outputs = {
            **chain_outputs(*draws),
            "step_size": np.asarray(adapted_step),
            "inverse_mass_matrix": np.asarray(inverse_mass),
            **warmup_outputs(
                int(inputs["warmup"]),
                np.asarray(warm_acceptance),
                np.asarray(warm_positions),
            ),
        }
    elif mode == SAMPLE:
        outputs = chain_outputs(*result)
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
