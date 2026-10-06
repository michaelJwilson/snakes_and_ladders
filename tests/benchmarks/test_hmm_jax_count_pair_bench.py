"""A count-pair HMM's value and gradient: the RUST route, the JAX twin and a dense jitted forward (issue #1206).

The instance is the shape #1204 reports: 7,644 rows in 88 segments of
unequal length, K = 7, a negative-binomial total and beta-binomial
successes, drawn by ``sim.hmm.simulate_sequences``. Two forms: the joint
pair, and the independent pair with an exposure and a trial count per
position. Three routes per form:

* ``rust``: ``EmissionHmmObjective(backend=Backend.RUST)``, the compiled
  E step and one torch backward pass;
* ``jax``: the same objective under ``Backend.JAX``, the twin of
  :mod:`sal.opt.hmm.jax`, padded and masked;
* ``dense``: the reference written here --- the emission in ``jnp`` from
  ``gammaln`` differences, a log-space forward over the padded segments
  under ``lax.scan``, and ``jit(value_and_grad)`` of it by autodiff --- the
  forward a user writes by hand.

**Threads.** ``cores`` pins the process to the first 1 or 4 CPUs with
``os.sched_setaffinity`` and sets ``torch.set_num_threads`` to match; XLA's
and Rayon's pools are sized at start-up, and the affinity confines them.
Measured at the stress size alone, marked ``release`` (root ``CLAUDE.md``,
Measurement). The three routes are asserted to agree before they are timed.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import CountPairEmission
from sal.opt.hmm import EmissionHmmObjective
from sal.ragged import Ragged
from sal.sim.hmm import HmmParams, simulate_sequences

SEED = 1206
ROWS = 7_644
SEGMENTS = 88
K = 7

pytest.importorskip("jax")


def _lengths() -> tuple[int, ...]:
    """88 unequal lengths summing to 7,644, seeded."""
    rng = np.random.default_rng(SEED)
    raw = rng.integers(20, 160, SEGMENTS).astype(np.float64)
    lengths = np.maximum(np.floor(raw * ROWS / raw.sum()).astype(np.int64), 2)
    lengths[0] += ROWS - int(lengths.sum())
    return tuple(int(length) for length in lengths)


def _family(*, joint: bool) -> CountPairEmission:
    """Seven states: depth from 10 to 200, allele rate from 0.1 to 0.9."""
    return CountPairEmission(
        np.full(K, 20.0),
        np.geomspace(10.0, 200.0, K),
        30.0 * np.linspace(0.1, 0.9, K),
        30.0 * (1.0 - np.linspace(0.1, 0.9, K)),
        None if joint else np.full(K, 300),
        joint=joint,
    )


def _instance(form: str) -> tuple[Ragged, CountPairEmission, np.ndarray | None]:
    joint = form == "joint"
    family = _family(joint=joint)
    transition = np.full((K, K), 0.02 / (K - 1)) + np.eye(K) * (
        1.0 - 0.02 - 0.02 / (K - 1)
    )
    lengths = _lengths()
    params = HmmParams(
        n_states=K,
        lengths=lengths,
        initial=np.full(K, 1.0 / K),
        transition=transition,
        emissions=family,
        seed=SEED,
        tolerance=0.0,
    )
    values = np.asarray(simulate_sequences(params).observations, dtype=np.float64)
    data = Ragged(values.reshape(ROWS, 2), lengths)
    if joint:
        return data, family, None
    rng = np.random.default_rng(SEED + 1)
    covariate = np.stack([rng.uniform(0.5, 2.0, ROWS), np.full(ROWS, 300.0)], axis=1)
    return data, family, covariate


def _dense(
    objective: EmissionHmmObjective, covariate: np.ndarray | None
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """The hand-written reference: a dense emission and a log-space forward, differentiated by autodiff."""
    import jax

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    jnp = jax.numpy
    gammaln = jax.scipy.special.gammaln
    logsumexp = jax.scipy.special.logsumexp
    lengths = np.asarray(objective.lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    steps = np.arange(lengths.max())
    mask = steps[None, :] < lengths[:, None]
    rows = offsets[:, None] + np.where(mask, steps[None, :], 0)
    values = objective.observations.numpy()[rows]
    y, s = values[..., 0:1], values[..., 1:2]
    if covariate is None:
        exposure, n = np.ones_like(y), y
    else:
        exposure, n = covariate[rows][..., 0:1], covariate[rows][..., 1:2]
    blocks = objective.blocks
    names = ("dispersion", "mean", "alpha", "beta")
    data = {
        "y": y,
        "s": s,
        "n": n,
        "exposure": exposure,
        "mask": mask,
    }

    def simplex(free: Any) -> Any:
        pinned = jnp.concatenate([jnp.zeros(free.shape[:-1] + (1,)), free], axis=-1)
        return jax.nn.log_softmax(pinned, axis=-1)

    def nll(theta: Any, data: dict[str, Any]) -> Any:
        log_initial = simplex(theta[blocks["log_initial"]])
        log_transition = simplex(theta[blocks["log_transition"]].reshape(K, K - 1))
        r, mu, a, b = (jnp.exp(theta[blocks[name]]) for name in names)
        y, s, n = data["y"], data["s"], data["n"]
        mu = data["exposure"] * mu
        emit = (
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1.0)
            + r * jnp.log(r / (r + mu))
            + y * jnp.log(mu / (r + mu))
            + gammaln(n + 1.0)
            - gammaln(s + 1.0)
            - gammaln(n - s + 1.0)
            + gammaln(s + a)
            + gammaln(n - s + b)
            - gammaln(n + a + b)
            + gammaln(a + b)
            - gammaln(a)
            - gammaln(b)
        )
        keep = jnp.moveaxis(data["mask"], 1, 0)
        columns = jnp.moveaxis(emit, 1, 0)

        def step(carry: Any, xs: Any) -> tuple[Any, None]:
            column, real = xs
            onward = (
                logsumexp(carry[:, :, None] + log_transition[None], axis=1) + column
            )
            return jnp.where(real[:, None], onward, carry), None

        last, _ = jax.lax.scan(step, log_initial + columns[0], (columns[1:], keep[1:]))
        return -logsumexp(last, axis=-1).sum()

    compiled = jax.jit(jax.value_and_grad(nll))
    placed = {key: jax.device_put(value) for key, value in data.items()}

    def call(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = compiled(theta, placed)
        return float(value), np.asarray(gradient)

    return call


@pytest.fixture(scope="module", params=["joint", "independent covariate"])
def routes(
    request: pytest.FixtureRequest,
) -> tuple[dict[str, Callable[[], tuple[float, np.ndarray]]], str]:
    data, family, covariate = _instance(request.param)
    rust, twin = (
        EmissionHmmObjective(data, family, covariate=covariate, backend=backend)
        for backend in (Backend.RUST, Backend.JAX)
    )
    theta = rust.initial() + 0.01 * torch.arange(rust.n_parameters, dtype=torch.float64)
    dense = _dense(rust, covariate)

    def tensor_route(
        objective: EmissionHmmObjective,
    ) -> Callable[[], tuple[float, np.ndarray]]:
        def call() -> tuple[float, np.ndarray]:
            value, gradient = objective.value_and_gradient(theta)
            return float(value), gradient.numpy()

        return call

    table: dict[str, Callable[[], tuple[float, np.ndarray]]] = {
        "rust": tensor_route(rust),
        "jax": tensor_route(twin),
        "dense": lambda: dense(theta.numpy()),
    }
    # The three agree before any is timed: value to 1e-12 relative,
    # gradient to 1e-10 of its largest entry.
    want_value, want_gradient = table["rust"]()
    for name in ("jax", "dense"):
        value, gradient = table[name]()
        assert math.isclose(value, want_value, rel_tol=1e-12), (name, value, want_value)
        scale = float(np.abs(want_gradient).max())
        assert float(np.abs(gradient - want_gradient).max()) <= 1e-10 * scale, name
    return table, request.param


@pytest.fixture(params=[1, 4], ids=["1core", "4cores"])
def cores(request: pytest.FixtureRequest) -> Iterator[int]:
    available = os.sched_getaffinity(0)
    if len(available) < request.param:
        pytest.skip(f"{request.param} cores asked, {len(available)} available")
    threads = torch.get_num_threads()
    os.sched_setaffinity(0, sorted(available)[: request.param])
    torch.set_num_threads(request.param)
    yield request.param
    os.sched_setaffinity(0, available)
    torch.set_num_threads(threads)


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("route", ["rust", "jax", "dense"])
def test_count_pair_value_and_gradient_at_stress_size(
    benchmark: Any,
    routes: tuple[dict[str, Callable[[], tuple[float, np.ndarray]]], str],
    cores: int,
    route: str,
) -> None:
    table, form = routes
    benchmark.extra_info.update({"form": form, "cores": cores, "route": route})
    table[route]()  # warm: compiled before it is timed
    value, _ = benchmark.pedantic(
        table[route], rounds=30, iterations=1, warmup_rounds=3
    )
    assert math.isfinite(value)
