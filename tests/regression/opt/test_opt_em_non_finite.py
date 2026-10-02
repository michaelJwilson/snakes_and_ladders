"""Every EM entry point stops at the step whose log-likelihood is not finite (issue #1179).

The streamed count Baum-Welch step returned a ``nan`` log-likelihood once an
emptied state's mass underflowed, and the run spent its 500 iterations on it
and reported ``Stop.BUDGET``. #1180 removed that cause; this pins the other
half: :func:`sal.opt.em.em_loop` raises at the first non-finite value, and
every entry point loops through it, on the torch and the streamed Rust route.

Referee: the step is wrapped to return ``nan`` at iteration ``K``; each route
raises naming iteration ``K`` after exactly ``K`` calls. A route that looped
outside ``em_loop`` would never call the wrapper and would return instead.
Finite fits are pinned bitwise by the existing oracle tests, which pass
unchanged: the guard reads the value and returns nothing new.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
from sal import oxisal
from sal.backend import Backend
from sal.emissions import GaussianEmission, PoissonEmission
from sal.opt import em
from sal.opt import emission_mixture as emission_mixture_module
from sal.opt import mixture as mixture_module
from sal.opt.hmm import estimation as estimation_module
from sal.opt.hmm.estimation import baum_welch, baum_welch_family

K = 3
"""The iteration whose log-likelihood the wrapped step reports as ``nan``."""

SEED = 1179
BACKENDS = [Backend.PYTHON, Backend.RUST]


@pytest.fixture
def poisoned(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap every module's ``em_loop`` so its step returns ``nan`` at iteration ``K``."""
    calls: list[int] = []

    def wrapped(step: Callable[[Any], tuple[Any, float]], start: Any, **kw: Any) -> Any:
        def poisoned_step(state: Any) -> tuple[Any, float]:
            calls.append(len(calls) + 1)
            advanced, log_likelihood = step(state)
            return advanced, math.nan if len(calls) == K else log_likelihood

        return em.em_loop(poisoned_step, start, **kw)

    module: ModuleType
    for module in (estimation_module, mixture_module, emission_mixture_module):
        monkeypatch.setattr(module, "em_loop", wrapped)
    return calls


def _uniform(m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A uniform log initial and log transition over ``m`` states."""
    log_initial = torch.full((m,), -math.log(m), dtype=torch.float64)
    return log_initial, log_initial.repeat(m, 1)


def _raises_at_k(calls: list[int], fit: Callable[[], object]) -> None:
    """``fit`` raises naming iteration ``K`` after exactly ``K`` steps."""
    with pytest.raises(ValueError, match=rf"is nan at iteration {K},"):
        fit()
    assert calls == list(range(1, K + 1))


@pytest.mark.smoke
@pytest.mark.parametrize("backend", BACKENDS)
def test_categorical_baum_welch_stops_at_a_nan(
    poisoned: list[int], backend: Backend
) -> None:
    rng = np.random.default_rng(SEED)
    observations = rng.integers(0, 3, (4, 50))
    log_initial, log_transition = _uniform(2)
    log_emission = torch.log(
        torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float64)
    )
    _raises_at_k(
        poisoned,
        lambda: baum_welch(
            observations, log_initial, log_transition, log_emission, backend=backend
        ),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian_baum_welch_family_stops_at_a_nan(
    poisoned: list[int], backend: Backend
) -> None:
    rng = np.random.default_rng(SEED)
    observations = rng.normal(0.0, 1.0, (4, 50)) + 5.0 * rng.integers(0, 2, (4, 50))
    start = GaussianEmission(np.array([0.0, 5.0]), np.array([1.0, 1.0]), 1e-6)
    _raises_at_k(
        poisoned,
        lambda: baum_welch_family(observations, *_uniform(2), start, backend=backend),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("backend", BACKENDS)
def test_count_baum_welch_family_stops_at_a_nan(
    poisoned: list[int], backend: Backend
) -> None:
    rng = np.random.default_rng(SEED)
    observations = rng.poisson(rng.choice([3.0, 30.0], (4, 50))).astype(np.int64)
    start = PoissonEmission(np.array([2.0, 20.0]))
    _raises_at_k(
        poisoned,
        lambda: baum_welch_family(observations, *_uniform(2), start, backend=backend),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian_mixture_stops_at_a_nan(poisoned: list[int], backend: Backend) -> None:
    rng = np.random.default_rng(SEED)
    values = np.concatenate([rng.normal(0.0, 1.0, 100), rng.normal(5.0, 1.0, 100)])
    start = GaussianEmission(np.array([-1.0, 6.0]), np.array([1.0, 1.0]), 1e-6)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    _raises_at_k(
        poisoned,
        lambda: mixture_module.expectation_maximization(
            values, weights, start, backend=backend
        ),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [np.int64, np.float64], ids=["cells", "draws"])
def test_count_mixture_stops_at_a_nan(
    poisoned: list[int], dtype: type[np.generic]
) -> None:
    # Integer counts take the distinct-count route (#997), floats the
    # per-observation one: both loop through `em_loop`.
    rng = np.random.default_rng(SEED)
    values = rng.poisson(rng.choice([3.0, 30.0], 200)).astype(dtype)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    _raises_at_k(
        poisoned,
        lambda: emission_mixture_module.expectation_maximization(
            values, weights, PoissonEmission(np.array([2.0, 20.0]))
        ),
    )


@pytest.mark.smoke
def test_a_nan_from_the_compiled_count_step_stops_the_streamed_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The defect's own path, unwrapped: `oxisal.count_em_step` hands back
    # `nan` at iteration K, as it did on #1179's reproduction before #1180.
    calls: list[int] = []

    def count_em_step(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        calls.append(len(calls) + 1)
        *returned, log_likelihood = oxisal.count_em_step(*args, **kwargs)
        return (*returned, math.nan if len(calls) == K else log_likelihood)

    class Kernel:
        """``oxisal`` with ``count_em_step`` replaced."""

        def __getattr__(self, name: str) -> Any:
            return count_em_step if name == "count_em_step" else getattr(oxisal, name)

    monkeypatch.setattr(estimation_module, "oxisal", Kernel())
    rng = np.random.default_rng(SEED)
    observations = rng.poisson(rng.choice([3.0, 30.0], (4, 50))).astype(np.int64)
    start = PoissonEmission(np.array([2.0, 20.0]))
    _raises_at_k(
        calls,
        lambda: baum_welch_family(
            observations, *_uniform(2), start, backend=Backend.RUST
        ),
    )
