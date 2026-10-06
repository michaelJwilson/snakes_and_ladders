"""`EmissionHmmObjective` value and gradient: the compiled E step against autograd (issue #1169).

The declared route runs ``oxisal.ragged_posteriors`` and one backward pass of
the Fisher-identity surrogate; the autograd route differentiates the torch
forward recursion. `tests/regression/opt/test_opt_hmm_emission_objective.py`
pins the two at 1e-10; this measures the ratio. Four-state sticky
negative-binomial chains, the fixture of `test_hmm_ragged_estep_bench.py`.
The ratio is read at the stress size, 200 chains of 100-3,000 positions,
marked ``release``; the gate size times the same two routes per PR and
decides nothing (root ``CLAUDE.md``, Measurement). A four-state Gaussian HMM
of 200 sequences of 60, the fixture of issue #1248, times the supported
kernel's route against autograd's; on ragged segments (issue #1254) at the
gate size per PR and at the stress size for a release. Each count family the
``count_hmm`` kernel takes (issue #1255) times the kernel's call against the
E step and backward pass it replaces, at 200 sequences of 60 per PR and on
the ragged stress segments for a release; the independent pair with a
covariate per channel since issue #1265.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
import torch
from sal.emissions import (
    BetaBinomialEmission,
    CountPairEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.opt.hmm import EmissionHmmObjective, family_start
from sal.opt.objective import autograd_value_and_gradient
from sal.ragged import Ragged

ROUTES: dict[
    str,
    Callable[[EmissionHmmObjective, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
] = {
    "compiled": lambda objective, theta: objective.value_and_gradient(theta),
    "autograd": autograd_value_and_gradient,
}


def _objective(
    n_chains: int, shortest: int, longest: int
) -> tuple[EmissionHmmObjective, torch.Tensor]:
    """Sticky negative-binomial chains over four states, seeded, and a point off the start."""
    rng = np.random.default_rng(1169)
    lengths = rng.integers(shortest, longest, n_chains)
    means = np.geomspace(2.0, 60.0, 4)
    chains = []
    for length in lengths:
        states = np.zeros(length, dtype=np.int64)
        for t in range(1, length):
            states[t] = states[t - 1] if rng.random() < 0.97 else rng.integers(0, 4)
        chains.append(rng.negative_binomial(8.0, 8.0 / (8.0 + means[states])))
    batch = Ragged(np.concatenate(chains).astype(float), tuple(int(x) for x in lengths))
    objective = EmissionHmmObjective(
        batch, NegativeBinomialEmission([3.0] * 4, means * 1.3)
    )
    theta = objective.initial()
    return objective, theta + 0.01 * torch.arange(theta.numel(), dtype=torch.float64)


@pytest.fixture(scope="module")
def gate() -> tuple[EmissionHmmObjective, torch.Tensor]:
    # The gate size: thirty chains of 40-300 positions.
    return _objective(30, 40, 300)


@pytest.mark.benchmark
@pytest.mark.parametrize("route", list(ROUTES))
def test_value_and_gradient_at_gate_size(
    benchmark: object, gate: tuple[EmissionHmmObjective, torch.Tensor], route: str
) -> None:
    objective, theta = gate
    value, _ = benchmark(lambda: ROUTES[route](objective, theta))  # type: ignore[operator]
    assert math.isfinite(float(value))


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("route", list(ROUTES))
def test_value_and_gradient_at_stress_size(benchmark: object, route: str) -> None:
    # 200 chains of 100-3,000 positions, the size the ragged E step was kept for.
    objective, theta = _objective(200, 100, 3000)
    value, _ = benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: ROUTES[route](objective, theta), rounds=5, iterations=1
    )
    assert math.isfinite(float(value))


def _gaussian() -> tuple[EmissionHmmObjective, torch.Tensor]:
    """Four sticky Gaussian states, 200 sequences of 60, seeded, and a point off the start."""
    rng = np.random.default_rng(1248)
    states = np.zeros((200, 60), dtype=np.int64)
    for t in range(1, 60):
        stay = rng.random(200) < 0.9
        states[:, t] = np.where(stay, states[:, t - 1], rng.integers(0, 4, 200))
    observations = np.array([-3.0, -1.0, 1.0, 3.0])[states] + 0.5 * rng.normal(
        size=states.shape
    )
    objective = EmissionHmmObjective(
        observations, family_start(GaussianEmission, observations, 4)
    )
    theta = objective.initial()
    return objective, theta + 0.01 * torch.arange(theta.numel(), dtype=torch.float64)


@pytest.mark.benchmark
@pytest.mark.parametrize("route", list(ROUTES))
def test_supported_gaussian_value_and_gradient(benchmark: object, route: str) -> None:
    # Issue #1248: "compiled" is the supported kernel's one call here.
    objective, theta = _gaussian()
    value, _ = benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: ROUTES[route](objective, theta), rounds=3, iterations=5
    )
    assert math.isfinite(float(value))


def _ragged_gaussian(
    n_chains: int, shortest: int, longest: int
) -> tuple[EmissionHmmObjective, torch.Tensor]:
    """Four sticky Gaussian states on ragged segments, seeded, and a point off the start."""
    rng = np.random.default_rng(1254)
    lengths = rng.integers(shortest, longest, n_chains)
    chains = []
    for length in lengths:
        states = np.zeros(length, dtype=np.int64)
        for t in range(1, length):
            states[t] = states[t - 1] if rng.random() < 0.9 else rng.integers(0, 4)
        chains.append(np.array([-3.0, -1.0, 1.0, 3.0])[states])
    values = np.concatenate(chains) + 0.5 * rng.normal(size=int(lengths.sum()))
    batch = Ragged(values, tuple(int(x) for x in lengths))
    objective = EmissionHmmObjective(batch, family_start(GaussianEmission, values, 4))
    theta = objective.initial()
    return objective, theta + 0.01 * torch.arange(theta.numel(), dtype=torch.float64)


@pytest.mark.benchmark
@pytest.mark.parametrize("route", list(ROUTES))
def test_ragged_gaussian_at_gate_size(benchmark: object, route: str) -> None:
    # Issue #1254: thirty segments of 40-300; "compiled" is the kernel's call.
    objective, theta = _ragged_gaussian(30, 40, 300)
    value, _ = benchmark(lambda: ROUTES[route](objective, theta))  # type: ignore[operator]
    assert math.isfinite(float(value))


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("route", list(ROUTES))
def test_ragged_gaussian_at_stress_size(benchmark: object, route: str) -> None:
    # Issue #1254: 200 segments of 100-3,000, the stress size above.
    objective, theta = _ragged_gaussian(200, 100, 3000)
    value, _ = benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: ROUTES[route](objective, theta), rounds=3, iterations=1
    )
    assert math.isfinite(float(value))


#: Per-state means of the count families (issue #1255).
_MEANS = np.geomspace(2.0, 60.0, 4)

#: Each count family the kernel takes: the truth drawn from, the start
#: fitted from, and its covariate: ``"exposure"`` per position scaling the
#: mean, ``"pair"`` an exposure and a trial count per position, one per
#: channel (issue #1265), or ``""``.
COUNT_FAMILIES: dict[str, tuple[EmissionFamily, EmissionFamily, str]] = {
    "poisson": (PoissonEmission(_MEANS), PoissonEmission(_MEANS * 1.3), ""),
    "negative-binomial": (
        NegativeBinomialEmission([8.0] * 4, _MEANS),
        NegativeBinomialEmission([3.0] * 4, _MEANS * 1.3),
        "",
    ),
    "negative-binomial-exposure": (
        NegativeBinomialEmission([8.0] * 4, _MEANS),
        NegativeBinomialEmission([3.0] * 4, _MEANS * 1.3),
        "exposure",
    ),
    "beta-binomial": (
        BetaBinomialEmission([40.0] * 4, [1.0, 3.0, 6.0, 9.0], [9.0, 6.0, 3.0, 1.0]),
        BetaBinomialEmission([40.0] * 4, [1.5, 3.0, 5.0, 8.0], [8.0, 5.0, 3.0, 1.5]),
        "",
    ),
    "pair": (
        CountPairEmission(
            [8.0] * 4,
            _MEANS * 2,
            [1.0, 3.0, 6.0, 9.0],
            [9.0, 6.0, 3.0, 1.0],
            joint=True,
        ),
        CountPairEmission(
            [3.0] * 4,
            _MEANS * 2.6,
            [1.5, 3.0, 5.0, 8.0],
            [8.0, 5.0, 3.0, 1.5],
            joint=True,
        ),
        "",
    ),
    "pair-covariate": (
        CountPairEmission(
            [8.0] * 4,
            _MEANS * 2,
            [1.0, 3.0, 6.0, 9.0],
            [9.0, 6.0, 3.0, 1.0],
            [40.0] * 4,
            joint=False,
        ),
        CountPairEmission(
            [3.0] * 4,
            _MEANS * 2.6,
            [1.5, 3.0, 5.0, 8.0],
            [8.0, 5.0, 3.0, 1.5],
            [40.0] * 4,
            joint=False,
        ),
        "pair",
    ),
}

#: The kernel's call, and the route before issue #1255.
COUNT_ROUTES: dict[
    str,
    Callable[[EmissionHmmObjective, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
] = {
    "kernel": lambda objective, theta: objective.value_and_gradient(theta),
    "e-step": lambda objective, theta: objective._e_step_route(theta),
}


def _count_family(
    family: str, lengths: np.ndarray
) -> tuple[EmissionHmmObjective, torch.Tensor]:
    """Four sticky states of ``family`` on segments of ``lengths``, seeded, and a point off the start."""
    rng = np.random.default_rng(1255)
    truth, start, kind = COUNT_FAMILIES[family]
    chains = []
    for length in lengths:
        states = np.zeros(length, dtype=np.int64)
        for t in range(1, length):
            states[t] = states[t - 1] if rng.random() < 0.97 else rng.integers(0, 4)
        chains.append(states)
    states = np.concatenate(chains)
    covariate = None
    if kind == "exposure":
        covariate = rng.uniform(0.5, 2.0, (states.size, 1))
    if kind == "pair":
        covariate = np.stack(
            [
                rng.uniform(0.5, 2.0, states.size),
                rng.integers(20, 61, states.size).astype(float),
            ],
            axis=1,
        )
    observations = np.asarray(
        truth.sample(states, rng, covariate=covariate), dtype=np.float64
    )
    objective = EmissionHmmObjective(
        Ragged(observations, tuple(int(x) for x in lengths)), start, covariate=covariate
    )
    assert objective.supported_gradient() is not None
    theta = objective.initial()
    return objective, theta + 0.01 * torch.arange(theta.numel(), dtype=torch.float64)


@pytest.mark.benchmark
@pytest.mark.parametrize("route", list(COUNT_ROUTES))
@pytest.mark.parametrize("family", list(COUNT_FAMILIES))
def test_count_family_value_and_gradient(
    benchmark: object, family: str, route: str
) -> None:
    # Issue #1255: 200 sequences of 60, the gate size.
    objective, theta = _count_family(family, np.full(200, 60))
    value, _ = benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: COUNT_ROUTES[route](objective, theta), rounds=3, iterations=5
    )
    assert math.isfinite(float(value))


@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.parametrize("route", list(COUNT_ROUTES))
@pytest.mark.parametrize("family", list(COUNT_FAMILIES))
def test_count_family_at_stress_size(
    benchmark: object, family: str, route: str
) -> None:
    # Issue #1255: 200 segments of 100-3,000 (291,142 positions), where the
    # ratio is read.
    lengths = np.random.default_rng(1255).integers(100, 3000, 200)
    objective, theta = _count_family(family, lengths)
    value, _ = benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: COUNT_ROUTES[route](objective, theta), rounds=3, iterations=1
    )
    assert math.isfinite(float(value))
