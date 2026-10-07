"""`external.hmc` takes `sample.hmc.sample`'s arguments and refuses before it spawns (issue #1282, step 6).

Read against the sibling's signature and the `Capability` declarations:
`sample` is `sample.hmc.sample`'s three positional arguments with the solver
after them. A target the script does not rebuild in JAX --- an objective with
no kernel, a count mixture, a count HMM, a Gaussian HMM of unequal segments
--- a jittered warm-up step, a warm-up below BlackJAX's 20 steps and an
`"auto"` step are refused with no subprocess started and no draw taken from
`rng`. The chains are pinned to the adapter's bitwise, the rebuilt targets to
the objectives' values, and the moments to the Gaussian's, in
`tests/validation/test_blackjax.py`.
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
from sal.emissions import GaussianEmission
from sal.external import CapabilityRefused, Solver, hmc
from sal.external.hmc import ExternalChain, target_capabilities
from sal.opt.hmm.objectives import EmissionHmmObjective
from sal.opt.mixture import GaussianMixtureObjective
from sal.opt.testfunctions import Rosenbrock
from sal.ragged import Ragged
from sal.sample import hmc as sibling
from sal.sample.chain import Adaptation
from sal.validation.gaussian import GaussianTarget

GAUSSIAN = GaussianTarget(np.array([1.0, 2.0, 4.0]))


class Declared:
    """A target declaring the kernel it is handed; refused before it is read further."""

    def __init__(self, kernel: str | None) -> None:
        self.kernel = kernel

    def supported_gradient(self) -> tuple[str, dict[str, float]] | None:
        """``(kernel, {})``, or ``None`` for no kernel."""
        return None if self.kernel is None else (self.kernel, {})


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
def test_the_signature_is_the_siblings_and_the_result_its_type() -> None:
    # The sibling's positional arguments in its order, then the solver; of its
    # keywords those BlackJAX honours, in its order; `timeout` and `session`
    # are keyword-only additions.
    ours, theirs = _names(hmc.sample), _names(sibling.sample)
    assert ours[:4] == [*theirs[:3], "solver"]
    shared = [name for name in theirs if name in ours[4:]]
    assert ours[4:] == [*shared, "timeout", "session"]
    assert shared == ["step_size", "n_steps", "start", "adaptation"]
    parameters = inspect.signature(hmc.sample).parameters
    assert inspect.signature(sibling.sample).parameters["n_steps"].default == (
        parameters["n_steps"].default
    )
    assert all(
        parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in ours[4:]
    )
    assert issubclass(ExternalChain, sibling.HmcChain)
    assert {"termination", "provenance", "seconds"} <= set(
        ExternalChain.__dataclass_fields__
    )


@pytest.mark.critical
@pytest.mark.analytic
def test_the_root_holds_no_hmc_call() -> None:
    assert not hasattr(external, "sample")
    assert hmc.sample.__module__ == "sal.external.hmc"


@pytest.mark.analytic
def test_each_target_maps_to_its_capability() -> None:
    rng = np.random.default_rng(1282)
    values = rng.normal(size=(3, 8))
    equal = EmissionHmmObjective(
        values, GaussianEmission(np.array([-1.0, 1.0]), np.ones(2), 1e-6)
    )
    ragged = EmissionHmmObjective(
        Ragged(values.reshape(-1)[:20], (8, 12)),
        GaussianEmission(np.array([-1.0, 1.0]), np.ones(2), 1e-6),
    )
    assert target_capabilities(GAUSSIAN) == {"gaussian_target"}
    assert target_capabilities(Rosenbrock(3)) == {"rosenbrock_target"}
    assert target_capabilities(GaussianMixtureObjective(values.reshape(-1), 2)) == {
        "gaussian_mixture_target"
    }
    assert target_capabilities(equal) == {"gaussian_hmm_target"}
    assert target_capabilities(ragged) == {"gaussian_hmm_target", "ragged_segments"}
    assert target_capabilities(Declared(None)) == {"undeclared_target"}
    assert target_capabilities(Declared("count_hmm")) == {"count_hmm_target"}
    assert target_capabilities(Declared("count_mixture")) == {"count_mixture_target"}
    assert target_capabilities(Declared("a_kernel_to_come")) == {"undeclared_target"}


#: One target per capability BlackJAX does not declare.
REFUSED: list[tuple[Any, str]] = [
    (Declared(None), "undeclared_target"),
    (Declared("count_hmm"), "count_hmm_target"),
    (Declared("count_mixture"), "count_mixture_target"),
    (
        EmissionHmmObjective(
            Ragged(np.linspace(-1.0, 1.0, 20), (8, 12)),
            GaussianEmission(np.array([-1.0, 1.0]), np.ones(2), 1e-6),
        ),
        "ragged_segments",
    ),
]


@pytest.mark.analytic
@pytest.mark.parametrize(("objective", "missing"), REFUSED, ids=lambda x: str(x)[:16])
def test_a_target_blackjax_lacks_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch, objective: Any, missing: str
) -> None:
    started = _spawns(monkeypatch)
    rng = np.random.default_rng(1282)
    before = rng.bit_generator.state
    with pytest.raises(CapabilityRefused, match=f"does not offer: {missing}$") as no:
        hmc.sample(objective, rng, 10, Solver.BLACKJAX_HMC, step_size=0.1)
    assert no.value.missing == frozenset({external.Capability(missing)})
    assert started == []
    assert rng.bit_generator.state == before


@pytest.mark.analytic
def test_a_jittered_step_and_a_solver_that_does_not_sample_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = _spawns(monkeypatch)
    with pytest.raises(CapabilityRefused, match="does not offer: step_jitter$"):
        hmc.sample(
            GAUSSIAN,
            np.random.default_rng(0),
            10,
            Solver.BLACKJAX_HMC,
            step_size=0.1,
            adaptation=Adaptation(100, 0.8, 0.1),
        )
    with pytest.raises(CapabilityRefused, match="hmc_sample"):
        hmc.sample(
            GAUSSIAN, np.random.default_rng(0), 10, Solver.GCO_SWAP, step_size=0.1
        )
    assert started == []


@pytest.mark.analytic
@pytest.mark.parametrize(
    ("keywords", "n_samples", "message"),
    [
        ({"step_size": "auto"}, 10, "given step"),
        ({"step_size": 0.0}, 10, "positive"),
        ({"step_size": 0.1, "n_steps": 0}, 10, "n_steps"),
        ({"step_size": 0.1}, 0, "n_samples"),
        (
            {"step_size": 0.1, "adaptation": Adaptation(19, 0.8, 0.0)},
            10,
            "below 20 warm-up steps",
        ),
    ],
    ids=["auto", "step", "steps", "samples", "warmup"],
)
def test_a_malformed_call_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    keywords: dict[str, Any],
    n_samples: int,
    message: str,
) -> None:
    started = _spawns(monkeypatch)
    generator = torch.Generator().manual_seed(0)
    before = generator.get_state().clone()
    with pytest.raises(ValueError, match=message):
        hmc.sample(GAUSSIAN, generator, n_samples, Solver.BLACKJAX_HMC, **keywords)
    assert started == []
    assert torch.equal(generator.get_state(), before)


@pytest.mark.analytic
def test_the_seed_is_one_draw_from_either_generator() -> None:
    numpy_rng, torch_rng = np.random.default_rng(7), torch.Generator().manual_seed(7)
    assert hmc.seed_of(numpy_rng) == int(
        np.random.default_rng(7).integers(0, 2**31 - 1)
    )
    assert hmc.seed_of(torch_rng) == int(
        torch.randint(0, 2**31 - 1, (), generator=torch.Generator().manual_seed(7))
    )
