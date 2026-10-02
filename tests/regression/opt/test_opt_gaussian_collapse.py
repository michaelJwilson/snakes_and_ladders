"""A collapsed Gaussian state is held, clamped or refused, and the fit names it (issue #1160).

The planted instance is #1136's under Gaussian emissions: two generating
components at means 30 and 200, scales 5 and 20, 4,000 draws, and a third
component started where it collapses. Two collapses are planted:

- **narrow**: the third component's mean on one draw and its scale a hundredth
  of the floor's, so its first re-estimated variance is ``1.5e-7``, under the
  floor ``4.7e-4``;
- **emptied**: the third component's mean at 1e12, so the first E step leaves
  it no posterior mass.

Referees, each on both backends:

- ``HOLD`` returns the collapsed state's start parameters bitwise and lists it
  in ``frozen``; ``CLAMP`` returns the floor's square root as the scale
  bitwise at the step that clamps; ``REFUSE`` raises #122's message.
- The two surviving components land within 2% of their planted means.
- The log-likelihood never decreases across EM iterations under ``HOLD`` or
  ``CLAMP``, to a relative 1e-13 (rounding; measured at most 1.7e-15). A
  clamp is the M step constrained to ``variance >= floor``, whose maximizer
  is the floor whenever the free one is below it, so the bound the issue
  allows for a clamp is zero here.
- The Rust and torch routes agree on ``frozen`` and to 1e-12 relative on every
  parameter; a fit that does not collapse is pinned bitwise by the existing
  Gaussian mixture and HMM tests, which this change leaves passing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sal.backend import Backend
from sal.emissions import (
    Collapse,
    GaussianEmission,
    NegativeBinomialEmission,
    pooled_variance_floor,
)
from sal.opt.em import EmConfig
from sal.opt.hmm.estimation import baum_welch_family
from sal.opt.mixture import (
    MixtureFit,
    expectation_maximization,
    responsibilities_torch,
)

SEED = 1136

#: The planted means and scales, #1136's count-pair means as Gaussians.
MEANS = (30.0, 200.0)
SCALES = (5.0, 20.0)

#: A relative decrease of the log-likelihood across one EM iteration below
#: which it is rounding: the largest measured is 1.7e-15.
MONOTONE_RTOL = 1e-13

BACKENDS = [Backend.PYTHON, Backend.RUST]


def _planted() -> tuple[np.ndarray, np.ndarray, float]:
    """Draws from the two planted components, their labels, and the pooled floor."""
    rng = np.random.default_rng(SEED)
    labels = rng.integers(0, 2, 4_000)
    draws = rng.normal(np.asarray(MEANS)[labels], np.asarray(SCALES)[labels])
    return draws, labels, pooled_variance_floor(draws)


def _start(
    draws: np.ndarray, floor: float, mode: Collapse, *, emptied: bool
) -> GaussianEmission:
    """Two components near the planted ones and a third that collapses."""
    third = (1e12, 5.0) if emptied else (float(draws[0]), math.sqrt(floor) / 100.0)
    return GaussianEmission(
        [25.0, 180.0, third[0]],
        [5.0, 20.0, third[1]],
        floor,
        on_collapse=mode,
    )


def _fit(
    mode: Collapse, backend: Backend, *, emptied: bool, iterations: int = 500
) -> tuple[MixtureFit, GaussianEmission, np.ndarray, np.ndarray]:
    draws, labels, floor = _planted()
    start = _start(draws, floor, mode, emptied=emptied)
    fit = expectation_maximization(
        draws,
        torch.full((3,), 1.0 / 3.0, dtype=torch.float64),
        start,
        EmConfig(max_iterations=iterations),
        backend=backend,
    )
    return fit, start, draws, labels


def _assert_survivors_recover(
    fit: MixtureFit, draws: np.ndarray, labels: np.ndarray
) -> None:
    for k, planted in enumerate(MEANS):
        fitted = float(fit.components.mean[k])
        assert fitted == pytest.approx(planted, rel=0.02)
        assert fitted == pytest.approx(float(draws[labels == k].mean()), rel=0.02)


@pytest.mark.oracle
@pytest.mark.parametrize("emptied", [False, True], ids=["narrow", "emptied"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian_collapse_hold_returns_the_last_parameters_bitwise(
    backend: Backend, emptied: bool
) -> None:
    fit, start, draws, labels = _fit(Collapse.HOLD, backend, emptied=emptied)

    assert fit.frozen == (2,)
    assert fit.termination.converged
    assert torch.equal(fit.components.mean[2], start.mean[2])
    assert torch.equal(fit.components.scale[2], start.scale[2])
    assert fit.components.on_collapse is Collapse.HOLD
    _assert_survivors_recover(fit, draws, labels)


@pytest.mark.oracle
@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian_collapse_clamp_sets_the_variance_to_the_floor(
    backend: Backend,
) -> None:
    # One step: the clamp is read where it is applied. The scale is the
    # floor's square root exactly, and the mean is the posterior-weighted one
    # at the start, recomputed here from the responsibilities.
    fit, start, draws, _ = _fit(Collapse.CLAMP, backend, emptied=False, iterations=1)
    values = torch.as_tensor(draws, dtype=torch.float64)
    owned = responsibilities_torch(
        values, torch.full((3,), -math.log(3.0), dtype=torch.float64), start
    )[:, 2]

    assert fit.frozen == (2,)
    assert float(fit.components.scale[2]) == math.sqrt(start.variance_floor)
    assert float(fit.components.mean[2]) == pytest.approx(
        float((owned * values).sum() / owned.sum()), rel=1e-12
    )


@pytest.mark.oracle
@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian_collapse_clamp_holds_an_emptied_mean_at_the_floor(
    backend: Backend,
) -> None:
    fit, start, draws, labels = _fit(Collapse.CLAMP, backend, emptied=True)

    assert fit.frozen == (2,)
    assert torch.equal(fit.components.mean[2], start.mean[2])
    assert float(fit.components.scale[2]) == math.sqrt(start.variance_floor)
    _assert_survivors_recover(fit, draws, labels)


@pytest.mark.oracle
@pytest.mark.parametrize("emptied", [False, True], ids=["narrow", "emptied"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian_collapse_refuse_raises_the_issue_122_message(
    backend: Backend, emptied: bool
) -> None:
    with pytest.raises(ValueError, match="unbounded as a variance goes to zero"):
        _fit(Collapse.REFUSE, backend, emptied=emptied)


@pytest.mark.oracle
@pytest.mark.parametrize("mode", [Collapse.HOLD, Collapse.CLAMP])
@pytest.mark.parametrize("emptied", [False, True], ids=["narrow", "emptied"])
def test_gaussian_collapse_routes_agree_on_frozen_and_parameters(
    mode: Collapse, emptied: bool
) -> None:
    torch_fit = _fit(mode, Backend.PYTHON, emptied=emptied)[0]
    rust_fit = _fit(mode, Backend.RUST, emptied=emptied)[0]

    assert torch_fit.frozen == rust_fit.frozen == (2,)
    assert torch_fit.termination == rust_fit.termination
    for name in ("mean", "scale"):
        assert_allclose(
            getattr(rust_fit.components, name).numpy(),
            getattr(torch_fit.components, name).numpy(),
            rtol=1e-12,
        )
    assert_allclose(rust_fit.weights.numpy(), torch_fit.weights.numpy(), rtol=1e-12)


@pytest.mark.oracle
@pytest.mark.parametrize("mode", [Collapse.HOLD, Collapse.CLAMP])
@pytest.mark.parametrize("emptied", [False, True], ids=["narrow", "emptied"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian_collapse_never_decreases_the_log_likelihood(
    backend: Backend, mode: Collapse, emptied: bool
) -> None:
    draws, _, floor = _planted()
    weights = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
    components = _start(draws, floor, mode, emptied=emptied)
    trace = []
    for _ in range(80):
        step = expectation_maximization(
            draws,
            weights,
            components,
            EmConfig(max_iterations=1, tolerance=0.0),
            backend=backend,
        )
        trace.append(step.log_likelihood)
        weights, components = step.weights, step.components

    values = np.asarray(trace)
    assert bool((np.diff(values) >= -MONOTONE_RTOL * np.abs(values[1:])).all())


def _chains(
    mode: Collapse, *, emptied: bool
) -> tuple[np.ndarray, np.ndarray, GaussianEmission]:
    """The planted draws as 40 chains of 100, their labels, and a three-state start."""
    draws, labels, floor = _planted()
    return (
        draws.reshape(40, 100),
        labels.reshape(40, 100),
        _start(draws, floor, mode, emptied=emptied),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("emptied", [False, True], ids=["narrow", "emptied"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian_collapse_hold_in_baum_welch_names_the_held_state(
    backend: Backend, emptied: bool
) -> None:
    observations, labels, start = _chains(Collapse.HOLD, emptied=emptied)
    uniform = torch.full((3,), -math.log(3.0), dtype=torch.float64)

    fit = baum_welch_family(
        observations,
        uniform,
        uniform.repeat(3, 1),
        start,
        backend=backend,
    )

    assert isinstance(fit.components, GaussianEmission)
    assert fit.frozen == (2,)
    assert fit.termination.converged
    assert torch.equal(fit.components.mean[2], start.mean[2])
    assert torch.equal(fit.components.scale[2], start.scale[2])
    for k, planted in enumerate(MEANS):
        assert float(fit.components.mean[k]) == pytest.approx(planted, rel=0.02)
        assert float(fit.components.mean[k]) == pytest.approx(
            float(observations[labels == k].mean()), rel=0.02
        )


@pytest.mark.oracle
@pytest.mark.parametrize("mode", [Collapse.CLAMP, Collapse.REFUSE])
def test_gaussian_collapse_baum_welch_routes_agree(mode: Collapse) -> None:
    observations, _, start = _chains(mode, emptied=False)
    uniform = torch.full((3,), -math.log(3.0), dtype=torch.float64)

    def run(backend: Backend) -> tuple[tuple[int, ...], np.ndarray, np.ndarray] | str:
        try:
            fit = baum_welch_family(
                observations,
                uniform,
                uniform.repeat(3, 1),
                start,
                EmConfig(max_iterations=1),
                backend=backend,
            )
        except ValueError as refusal:
            return str(refusal).split(":")[0]
        assert isinstance(fit.components, GaussianEmission)
        return fit.frozen, fit.components.mean.numpy(), fit.components.scale.numpy()

    torch_run, rust_run = run(Backend.PYTHON), run(Backend.RUST)

    if mode is Collapse.REFUSE:
        assert isinstance(torch_run, str)
        assert isinstance(rust_run, str)
        assert torch_run.startswith("state(s) [2] re-estimated to variance")
        assert rust_run.startswith("state(s) [2] re-estimated to variance")
        return
    assert not isinstance(torch_run, str)
    assert not isinstance(rust_run, str)
    assert torch_run[0] == rust_run[0] == (2,)
    assert (
        float(rust_run[2][2])
        == float(torch_run[2][2])
        == math.sqrt(start.variance_floor)
    )
    assert_allclose(rust_run[1], torch_run[1], rtol=1e-12)
    assert_allclose(rust_run[2], torch_run[2], rtol=1e-12)


@pytest.mark.oracle
def test_an_emptied_state_s_transition_row_is_held_on_the_compiled_count_route() -> (
    None
):
    # The row hold this change makes for the Gaussian serves the streamed
    # count step too, which normalizes the same linear-space pair counts: on
    # #1179's reproduction it returned a NaN fit after 500 iterations, and
    # now matches the log-space torch route within 1e-10.
    rng = np.random.default_rng(SEED)
    truth = NegativeBinomialEmission([5.0, 20.0], [30.0, 200.0])
    labels = rng.integers(0, 2, 4_000)
    counts = truth.sample(labels, rng).reshape(40, 100).astype(np.int64)
    start = NegativeBinomialEmission([5.0, 20.0, 5.0], [25.0, 180.0, 1e12])
    uniform = torch.full((3,), -math.log(3.0), dtype=torch.float64)

    fits = [
        baum_welch_family(counts, uniform, uniform.repeat(3, 1), start, backend=b)
        for b in BACKENDS
    ]

    for fit in fits:
        assert fit.frozen == (2,)
        assert fit.termination.converged
    assert fits[1].log_likelihood == pytest.approx(fits[0].log_likelihood, rel=1e-10)
    # Between the two live states only: into the emptied one the linear-space
    # route reads exactly zero (-inf) where the log-space one carries -1e3,
    # and out of it the held row against a row of no data.
    assert_allclose(
        fits[1].log_transition.numpy()[:2, :2],
        fits[0].log_transition.numpy()[:2, :2],
        rtol=1e-10,
    )
    rust, torch_route = fits[1].components, fits[0].components
    assert isinstance(rust, NegativeBinomialEmission)
    assert isinstance(torch_route, NegativeBinomialEmission)
    assert_allclose(rust.mean.numpy(), torch_route.mean.numpy(), rtol=1e-10)
