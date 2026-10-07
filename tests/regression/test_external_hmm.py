"""`external.hmm` takes `opt.hmm`'s arguments and refuses before it spawns (issue #1282, step 5).

Read against the siblings' signatures and the `Capability` declarations:
`fit` is `opt.hmm.baum_welch_family`'s first five arguments with the solver
after the family, `viterbi` `likelihood.hmm.viterbi`'s four, and
`forward_log_likelihood` `opt.hmm.forward_log_likelihood`'s four; every family
hmmlearn has no model for, a kernel per step, an empty budget and a fit of
one-position segments are refused with no subprocess started. The answers are
pinned to the adapter's bitwise, and the log-likelihood to sal's forward
recursion, in `tests/validation/test_hmmlearn.py`.
"""

from __future__ import annotations

import inspect
import subprocess
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from sal import external
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.external import CapabilityRefused, Solver, hmm
from sal.external.hmm import (
    ExternalFit,
    ExternalLogLikelihood,
    ExternalPaths,
    family_capability,
)
from sal.likelihood import hmm as evaluators
from sal.opt import hmm as siblings
from sal.opt.em import EmConfig
from sal.opt.hmm import EmFit
from sal.ragged import Ragged

LOG_INITIAL = torch.log(torch.tensor([0.4, 0.6], dtype=torch.float64))
LOG_TRANSITION = torch.log(torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float64))
OBSERVATIONS = np.random.default_rng(1282).integers(0, 3, size=(4, 6))
CATEGORICAL = CategoricalEmission(np.array([[0.5, 0.3, 0.2], [0.1, 0.3, 0.6]]))


def _spawns(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every process the package would start, starting none."""
    started: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        started.append((args, kwargs))
        message = "no subprocess may start"
        raise AssertionError(message)

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    return started


def _names(call: Callable[..., Any]) -> list[str]:
    return list(inspect.signature(call).parameters)


@pytest.mark.critical
@pytest.mark.smoke
def test_each_signature_is_its_siblings_and_each_result_its_type() -> None:
    # The sibling's leading arguments, in its order and kind, then the
    # solver; `session` and `timeout` are keyword-only additions.
    family = _names(siblings.baum_welch_family)
    assert _names(hmm.fit) == [*family[:4], "solver", "config", "timeout", "session"]
    assert _names(hmm.viterbi) == [
        *_names(evaluators.viterbi)[:4],
        "solver",
        "timeout",
        "session",
    ]
    assert _names(hmm.forward_log_likelihood) == [
        *_names(siblings.forward_log_likelihood),
        "solver",
        "lengths",
        "timeout",
        "session",
    ]
    assert "lengths" in _names(siblings.forward_log_likelihood_ragged)
    for call in (hmm.fit, hmm.viterbi, hmm.forward_log_likelihood):
        parameters = list(inspect.signature(call).parameters.values())
        solver = _names(call).index("solver")
        assert all(
            p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for p in parameters[: solver + 1]
        )
        assert parameters[-1].kind is inspect.Parameter.KEYWORD_ONLY
    assert issubclass(ExternalFit, EmFit)
    # The sibling's pair unpacks from the path; the scalar reads as a float.
    assert list(ExternalPaths.__dataclass_fields__)[:2] == [
        "states",
        "log_probability",
    ]
    assert "log_likelihood" in ExternalLogLikelihood.__dataclass_fields__


@pytest.mark.critical
@pytest.mark.analytic
def test_the_root_holds_no_hmm_call() -> None:
    for task in ("fit", "viterbi", "forward_log_likelihood"):
        assert not hasattr(external, task), task
        assert getattr(hmm, task).__module__ == "sal.external.hmm"


#: One family per capability hmmlearn does not declare.
REFUSED: list[tuple[EmissionFamily, str]] = [
    (
        GaussianEmission(np.zeros((2, 2)), np.ones((2, 2)), 1e-12),
        "multi_channel_emissions",
    ),
    (
        BinomialEmission(np.array([10.0, 10.0]), np.array([0.2, 0.7])),
        "binomial_emissions",
    ),
    (
        NegativeBinomialEmission(np.array([2.0, 5.0]), np.array([1.0, 3.0])),
        "negative_binomial_emissions",
    ),
    (
        BetaBinomialEmission(
            np.array([10.0, 10.0]), np.array([1.0, 2.0]), np.array([2.0, 1.0])
        ),
        "beta_binomial_emissions",
    ),
]


@pytest.mark.analytic
@pytest.mark.parametrize(("components", "missing"), REFUSED, ids=lambda x: str(x)[:12])
@pytest.mark.parametrize("call", ["fit", "viterbi", "forward_log_likelihood"])
def test_a_family_hmmlearn_lacks_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    components: EmissionFamily,
    missing: str,
    call: str,
) -> None:
    started = _spawns(monkeypatch)
    assert family_capability(components) == missing
    with pytest.raises(CapabilityRefused, match=f"does not offer: {missing}$") as no:
        getattr(hmm, call)(
            OBSERVATIONS, LOG_INITIAL, LOG_TRANSITION, components, Solver.HMMLEARN
        )
    assert no.value.missing == frozenset({external.Capability(missing)})
    assert started == []


@pytest.mark.analytic
def test_the_named_families_and_an_unnamed_one_map_to_their_capabilities() -> None:
    class Own(PoissonEmission):
        """A caller's own family: a subclass is not the family hmmlearn models."""

    assert family_capability(CATEGORICAL) == "categorical_emissions"
    assert family_capability(PoissonEmission(np.array([1.0, 2.0]))) == (
        "poisson_emissions"
    )
    assert family_capability(GaussianEmission(np.zeros(2), np.ones(2), 1e-12)) == (
        "gaussian_emissions"
    )
    assert family_capability(Own(np.array([1.0, 2.0]))) == "unnamed_emissions"
    assert external.Capability.UNNAMED_EMISSIONS not in Solver.HMMLEARN.capabilities


@pytest.mark.analytic
def test_a_solver_without_the_task_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = _spawns(monkeypatch)
    with pytest.raises(CapabilityRefused, match="hmm_fit"):
        hmm.fit(OBSERVATIONS, LOG_INITIAL, LOG_TRANSITION, CATEGORICAL, Solver.GCO_SWAP)
    assert started == []


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("call", "arguments", "keywords", "message"),
    [
        (
            "fit",
            (OBSERVATIONS, LOG_INITIAL, LOG_TRANSITION.expand(5, 2, 2), CATEGORICAL),
            {},
            "one \\(2, 2\\) kernel",
        ),
        (
            "fit",
            (OBSERVATIONS, LOG_INITIAL, LOG_TRANSITION, CATEGORICAL),
            {"config": EmConfig(max_iterations=0)},
            "after an iteration",
        ),
        (
            "fit",
            (
                Ragged(OBSERVATIONS.reshape(-1)[:3], (1, 1, 1)),
                LOG_INITIAL,
                LOG_TRANSITION,
                CATEGORICAL,
            ),
            {},
            "one position",
        ),
        (
            "viterbi",
            (OBSERVATIONS.reshape(-1), LOG_INITIAL, LOG_TRANSITION, CATEGORICAL),
            {},
            "n_sequences, length",
        ),
        (
            "forward_log_likelihood",
            (OBSERVATIONS.reshape(-1), LOG_INITIAL, LOG_TRANSITION, CATEGORICAL),
            {"lengths": (5, 5)},
            "tile",
        ),
    ],
    ids=["kernel-per-step", "no-iteration", "one-position", "shape", "lengths"],
)
def test_a_malformed_call_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    call: str,
    arguments: tuple[Any, ...],
    keywords: dict[str, Any],
    message: str,
) -> None:
    started = _spawns(monkeypatch)
    with pytest.raises(ValueError, match=message):
        getattr(hmm, call)(*arguments, Solver.HMMLEARN, **keywords)
    assert started == []
