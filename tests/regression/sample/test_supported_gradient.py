"""Each kernel an objective's ``supported_gradient`` names, against autograd through the objective (issues #1220, #1248, #1254, #1255).

A compiled chain runs ``oxisal.SupportedEnergy`` on ``(kernel, data)``, and so
does the objective's own gradient; autograd through ``__call__`` is the
independent reference. At five seeded points per objective the value is
within 1e-10 of autograd's, relative, and the gradient within 1e-10 of its
largest coordinate. Measured: 3.7e-15 in the value and 9.4e-14 in the
gradient at most (the negative binomial mixture), over the seven objectives;
4.4e-15 and 1.7e-13 over the seven count HMMs of issue #1255, and 1.7e-15
and 1.1e-13 at a dispersion up to 1e15 and a concentration up to 1e12.
A Gaussian or count HMM's ``value_and_gradient`` is its kernel's call where the kernel
is supported, on segments of any length since #1254, and the E step and
backward pass, unchanged, where it is not.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
import torch
from sal import oxisal
from sal.backend import Backend
from sal.emissions import (
    BetaBinomialEmission,
    BinomialEmission,
    CountPairEmission,
    EmissionFamily,
    GaussianEmission,
    NegativeBinomialEmission,
    PoissonEmission,
)
from sal.opt.emission_mixture import EmissionMixtureObjective
from sal.opt.hmm import EmissionHmmObjective, family_start
from sal.opt.mixture import GaussianMixtureObjective
from sal.opt.objective import Objective, autograd_value_and_gradient
from sal.opt.testfunctions import Rosenbrock
from sal.ragged import Ragged
from sal.sample import declared
from sal.validation.gaussian import GaussianTarget, dense_precision, diagonal_precision

SEED = 1220
TOLERANCE = 1e-10
#: The ragged segments of issue #1254: one of length 1 (#1240), and one long.
RAGGED = (1, 2, 7, 60)


def _hmm_observations() -> np.ndarray:
    """Six sticky two-state Gaussian sequences of 40, seeded."""
    rng = np.random.default_rng(SEED)
    states = np.zeros((6, 40), dtype=int)
    for t in range(1, 40):
        stay = rng.random(6) < 0.9
        states[:, t] = np.where(stay, states[:, t - 1], 1 - states[:, t - 1])
    return np.array([-1.0, 1.0])[states] + 0.6 * rng.normal(size=states.shape)


def _hmm(
    backend: Backend = Backend.TORCH, ragged: bool = False
) -> EmissionHmmObjective:
    """The six sequences of 40, or their first 70 values as the segments :data:`RAGGED`."""
    observations = _hmm_observations()
    data: np.ndarray | Ragged = observations
    if ragged:
        data = Ragged(observations.reshape(-1)[: sum(RAGGED)], RAGGED)
    return EmissionHmmObjective(
        data, family_start(GaussianEmission, observations, 2), backend=backend
    )


def _mixture() -> Objective:
    rng = np.random.default_rng(SEED)
    labels = rng.choice(3, size=1_500, p=[0.3, 0.3, 0.4])
    return GaussianMixtureObjective(
        np.array([-4.0, 0.0, 5.0])[labels] + rng.normal(size=1_500), 3
    )


def _counts(family: NegativeBinomialEmission | CountPairEmission) -> Objective:
    rng = np.random.default_rng(SEED)
    labels = rng.integers(0, family.n_states, 2_000)
    observations = np.asarray(family.sample(labels, rng), dtype=np.float64)
    return EmissionMixtureObjective(observations, family)


#: Per-state means of the count HMMs: four sticky states (issue #1255).
COUNT_MEANS = np.array([2.0, 6.0, 18.0, 50.0])


def _count_hmm(
    truth: EmissionFamily,
    start: EmissionFamily,
    *,
    exposure: bool = False,
    backend: Backend = Backend.RUST,
) -> EmissionHmmObjective:
    """Draws of ``truth`` on the segments :data:`RAGGED` and three more, fitted from ``start``.

    With ``exposure``, one per position in ``[0.5, 2)`` and one of zero, the
    unobserved count of issue #933.
    """
    rng = np.random.default_rng([SEED, 1255])
    lengths = (*RAGGED, 30, 45, 90)
    states = np.zeros(sum(lengths), dtype=np.int64)
    for t in range(1, states.size):
        stay = rng.random() < 0.9
        states[t] = states[t - 1] if stay else rng.integers(0, truth.n_states)
    covariate = None
    if exposure:
        covariate = rng.uniform(0.5, 2.0, (states.size, 1))
        covariate[3] = 0.0
    observations = np.asarray(
        truth.sample(states, rng, covariate=covariate), dtype=np.float64
    )
    return EmissionHmmObjective(
        Ragged(observations, lengths), start, covariate=covariate, backend=backend
    )


def _beta_binomial(alpha: list[float], beta: list[float]) -> BetaBinomialEmission:
    return BetaBinomialEmission([40.0] * len(alpha), alpha, beta)


def _pair(
    dispersion: list[float], alpha: list[float], beta: list[float]
) -> CountPairEmission:
    mean = 2.0 * COUNT_MEANS[: len(alpha)]
    return CountPairEmission(dispersion, mean, alpha, beta, joint=True)


#: Each count family the ``count_hmm`` kernel takes: the truth drawn from,
#: the start fitted from, and whether an exposure scales the mean.
COUNT_HMMS: dict[str, tuple[EmissionFamily, EmissionFamily, bool]] = {
    "poisson": (
        PoissonEmission(COUNT_MEANS),
        PoissonEmission(1.3 * COUNT_MEANS),
        False,
    ),
    "negative-binomial": (
        NegativeBinomialEmission([8.0] * 4, COUNT_MEANS),
        NegativeBinomialEmission([3.0] * 4, 1.3 * COUNT_MEANS),
        False,
    ),
    "negative-binomial-exposure": (
        NegativeBinomialEmission([8.0] * 4, COUNT_MEANS),
        NegativeBinomialEmission([3.0] * 4, 1.3 * COUNT_MEANS),
        True,
    ),
    "beta-binomial": (
        _beta_binomial([1.0, 3.0, 6.0, 9.0], [9.0, 6.0, 3.0, 1.0]),
        _beta_binomial([1.5, 3.0, 5.0, 8.0], [8.0, 5.0, 3.0, 1.5]),
        False,
    ),
    "beta-binomial-rate": (
        _beta_binomial([1.0, 3.0, 6.0, 9.0], [9.0, 6.0, 3.0, 1.0]),
        _beta_binomial([1.5, 3.0, 5.0, 8.0], [8.0, 5.0, 3.0, 1.5]).rate_concentration(),
        False,
    ),
    "pair": (
        _pair([8.0] * 4, [1.0, 3.0, 6.0, 9.0], [9.0, 6.0, 3.0, 1.0]),
        _pair([3.0] * 4, [1.5, 3.0, 5.0, 8.0], [8.0, 5.0, 3.0, 1.5]),
        False,
    ),
    "pair-rate": (
        _pair([8.0] * 4, [1.0, 3.0, 6.0, 9.0], [9.0, 6.0, 3.0, 1.0]),
        _pair(
            [3.0] * 4, [1.5, 3.0, 5.0, 8.0], [8.0, 5.0, 3.0, 1.5]
        ).rate_concentration(),
        False,
    ),
}


def _count(name: str, backend: Backend = Backend.RUST) -> EmissionHmmObjective:
    truth, start, exposure = COUNT_HMMS[name]
    return _count_hmm(truth, start, exposure=exposure, backend=backend)


OBJECTIVES: dict[str, Callable[[], Objective]] = {
    "gaussian-diagonal": lambda: GaussianTarget(diagonal_precision(10)),
    "gaussian-dense": lambda: GaussianTarget(
        dense_precision(10, np.random.default_rng(SEED))
    ),
    "rosenbrock": lambda: Rosenbrock(dimension=5, b=10.0),
    "gaussian-mixture": _mixture,
    "gaussian-hmm": _hmm,
    "gaussian-hmm-ragged": lambda: _hmm(ragged=True),
    "count-mixture-negative-binomial": lambda: _counts(
        NegativeBinomialEmission([3.0, 8.0, 20.0], [5.0, 20.0, 60.0])
    ),
    "count-mixture-pair": lambda: _counts(
        CountPairEmission(
            [6.0, 12.0], [20.0, 60.0], [2.0, 9.0], [8.0, 3.0], joint=True
        ).rate_concentration()
    ),
    **{f"count-hmm-{name}": partial(_count, name) for name in COUNT_HMMS},
}


@pytest.mark.oracle
@pytest.mark.parametrize("name", list(OBJECTIVES))
def test_each_supported_kernel_is_autograd_through_the_objective(name: str) -> None:
    objective = OBJECTIVES[name]()
    supported = declared.declared_energy(objective)
    assert supported is not None
    n = int(objective.initial().shape[0])
    kernel = oxisal.SupportedEnergy(*supported, n)
    rng = np.random.default_rng([SEED, len(name)])
    for _ in range(5):
        theta = objective.initial() + 0.1 * torch.as_tensor(rng.normal(size=n))
        value, gradient = kernel.value_and_gradient(theta.numpy())
        want_value, want_gradient = autograd_value_and_gradient(objective, theta)
        reference = want_gradient.numpy()
        assert abs(value - float(want_value)) <= TOLERANCE * abs(float(want_value))
        assert np.abs(gradient - reference).max() <= TOLERANCE * np.abs(reference).max()


@pytest.mark.smoke
def test_every_named_kernel_builds_and_an_unknown_one_is_refused() -> None:
    # The names `sample.declared` lists are the ones `src/energy.rs` builds.
    named = {
        declared.GAUSSIAN,
        declared.ROSENBROCK,
        declared.GAUSSIAN_MIXTURE,
        declared.GAUSSIAN_HMM,
        declared.COUNT_MIXTURE,
        declared.COUNT_HMM,
    }
    seen = set()
    for make in OBJECTIVES.values():
        supported = declared.declared_energy(make())
        assert supported is not None
        seen.add(supported[0])
    assert seen == named
    with pytest.raises(ValueError, match="no supported gradient kernel"):
        oxisal.SupportedEnergy("quadratic", {}, 2)
    with pytest.raises(ValueError, match="neither d"):
        oxisal.SupportedEnergy(declared.GAUSSIAN, {"precision": np.ones(3)}, 2)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "case", ["rows", "ragged", *(f"count-{name}" for name in COUNT_HMMS)]
)
def test_a_supported_hmm_s_value_and_gradient_is_its_kernel_s_and_autograd_s(
    case: str,
) -> None:
    # Issue #1248: on the default backend, the supported Gaussian HMM's
    # value and gradient are one call of its kernel, bitwise, and within
    # 1e-10 of autograd through `__call__`, relative, at five seeded points;
    # on the segments `RAGGED` too since #1254, and each count family since
    # #1255.
    objective = (
        _count(case.removeprefix("count-"))
        if case.startswith("count-")
        else _hmm(Backend.RUST, ragged=case == "ragged")
    )
    supported = objective.supported_gradient()
    assert supported is not None
    n = objective.n_parameters
    kernel = oxisal.SupportedEnergy(*supported, n)
    rng = np.random.default_rng([SEED, 1248])
    for _ in range(5):
        theta = objective.initial() + 0.1 * torch.as_tensor(rng.normal(size=n))
        value, gradient = objective.value_and_gradient(theta)
        want_value, want_gradient = kernel.value_and_gradient(theta.numpy())
        assert float(value) == want_value
        assert np.array_equal(gradient.numpy(), want_gradient)
        assert torch.equal(objective.gradient(theta), gradient)
        reference_value, reference = autograd_value_and_gradient(objective, theta)
        assert abs(float(value) - float(reference_value)) <= TOLERANCE * abs(
            float(reference_value)
        )
        assert (
            np.abs(gradient.numpy() - reference.numpy()).max()
            <= TOLERANCE * np.abs(reference.numpy()).max()
        )


def _binomial_hmm() -> EmissionHmmObjective:
    """Binomial counts on the segments :data:`RAGGED`: a family no kernel supports."""
    rng = np.random.default_rng([SEED, 1254])
    counts = rng.binomial(10, np.where(_hmm_observations() > 0, 0.6, 0.2)).astype(float)
    flat = counts.reshape(-1)[: sum(RAGGED)]
    trials = np.full(2, 10.0)
    return EmissionHmmObjective(
        Ragged(flat, RAGGED), family_start(BinomialEmission, flat, 2, trials=trials)
    )


@pytest.mark.oracle
@pytest.mark.patch
@pytest.mark.parametrize("backend", [Backend.RUST, Backend.TORCH])
def test_an_unsupported_hmm_route_is_unchanged(backend: Backend) -> None:
    # A family no kernel supports, or the TORCH backend, keeps the route
    # before #1248: the TORCH backend is autograd's, bitwise, and the RUST
    # route the compiled E step and backward pass, within 1e-10 of autograd.
    objective = _binomial_hmm() if backend is Backend.RUST else _hmm(backend)
    theta = objective.initial() + torch.linspace(
        -0.3, 0.2, objective.n_parameters, dtype=torch.float64
    )
    value, gradient = objective.value_and_gradient(theta)
    want_value, want_gradient = autograd_value_and_gradient(objective, theta)
    if backend is Backend.TORCH:
        assert float(value) == float(want_value)
        assert torch.equal(gradient, want_gradient)
        return
    assert objective.supported_gradient() is None
    assert abs(float(value) - float(want_value)) <= TOLERANCE * abs(float(want_value))
    assert (
        gradient - want_gradient
    ).abs().max() <= TOLERANCE * want_gradient.abs().max()


@pytest.mark.oracle
@pytest.mark.patch
def test_equal_lengths_end_to_end_are_the_rows_bitwise() -> None:
    # Issue #1254: the kernel on segments end to end beside equal lengths is
    # the kernel on `(n_sequences, length)` rows, bitwise, at five seeded
    # points; the rows are the route before #1254.
    observations = _hmm_observations()
    objective = _hmm(Backend.RUST)
    supported = objective.supported_gradient()
    assert supported is not None
    name, data = supported
    n = objective.n_parameters
    flat = oxisal.SupportedEnergy(name, data, n)
    rows = oxisal.SupportedEnergy(name, {"m": 2, "observations": observations}, n)
    rng = np.random.default_rng([SEED, 1254])
    for _ in range(5):
        theta = (
            objective.initial() + 0.1 * torch.as_tensor(rng.normal(size=n))
        ).numpy()
        value, gradient = flat.value_and_gradient(theta)
        want_value, want_gradient = rows.value_and_gradient(theta)
        assert value == want_value
        assert np.array_equal(gradient, want_gradient)


@pytest.mark.smoke
def test_a_zero_length_or_a_short_sum_is_refused() -> None:
    # The kernel refuses a segment of length 0 and lengths that do not sum
    # to the observations; a segment of length 1 is admitted (#1240).
    values = np.array([0.1, 0.2, 0.3])
    for lengths, ok in [((1, 0, 2), False), ((1, 1), False), ((1, 2), True)]:
        data = {"m": 2, "observations": values, "lengths": np.array(lengths)}
        if ok:
            oxisal.SupportedEnergy(declared.GAUSSIAN_HMM, data, 7)
            continue
        with pytest.raises(ValueError, match="length 0|sum to"):
            oxisal.SupportedEnergy(declared.GAUSSIAN_HMM, data, 7)


def _large_shape(family: str, reading: str, shape: float) -> EmissionHmmObjective:
    """Three states, the last at dispersion or concentration ``shape``, in the reading named."""
    if family in ("negative-binomial", "exposure"):
        truth: EmissionFamily = NegativeBinomialEmission(
            [5.0, 1e6, 1e12], [3.0, 10.0, 30.0]
        )
        start: EmissionFamily = NegativeBinomialEmission(
            [4.0, 1e5, shape], [3.5, 9.0, 28.0]
        )
    else:
        dispersion = [5.0, 1e6, shape]
        truth_ab = ([2.0, 5e5, 3e11], [6.0, 5e5, 7e11])
        start_ab = ([2.0, 4e5, 0.3 * shape], [6.0, 6e5, 0.7 * shape])
        if family == "beta-binomial":
            truth = BetaBinomialEmission([40.0] * 3, *truth_ab)
            start = BetaBinomialEmission([40.0] * 3, *start_ab)
        else:
            truth = CountPairEmission(
                dispersion, [20.0, 40.0, 60.0], *truth_ab, joint=True
            )
            start = CountPairEmission(
                dispersion, [20.0, 40.0, 60.0], *start_ab, joint=True
            )
        if reading == "rate":
            start = start.rate_concentration()
    return _count_hmm(truth, start, exposure=family == "exposure")


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("family", "reading", "shape"),
    [
        *(
            (f, "natural", r)
            for f in ("negative-binomial", "exposure")
            for r in (1e8, 1e12, 1e15)
        ),
        *(
            (f, reading, tau)
            for f in ("beta-binomial", "pair")
            for reading in ("natural", "rate")
            for tau in (1e6, 1e9, 1e12)
        ),
    ],
)
def test_the_count_kernel_holds_at_large_shapes(
    family: str, reading: str, shape: float
) -> None:
    # Issue #1255: a dispersion `r = 1 / alpha` up to 1e15 (alpha -> 0, the
    # Poisson limit) and a concentration `a + b` up to 1e12 (the binomial
    # limit), where each lgamma and digamma difference cancels if formed
    # directly. The kernel's are prefix sums over integers (#1136); autograd
    # through `__call__` takes the large-shape series of `sal.emissions.rising`.
    # Measured: 1.7e-15 in the value and 1.1e-13 of the largest coordinate in
    # the gradient, at most.
    objective = _large_shape(family, reading, shape)
    assert objective.supported_gradient() is not None
    theta = objective.initial()
    value, gradient = objective.value_and_gradient(theta)
    want_value, want_gradient = autograd_value_and_gradient(objective, theta)
    assert abs(float(value) - float(want_value)) <= TOLERANCE * abs(float(want_value))
    assert (
        gradient - want_gradient
    ).abs().max() <= TOLERANCE * want_gradient.abs().max()


@pytest.mark.oracle
@pytest.mark.patch
def test_a_count_hmm_s_torch_backend_is_autograd_bitwise() -> None:
    # Issue #1255: the TORCH backend names no kernel and is autograd through
    # `__call__`, bitwise, for a family the kernel takes on the default.
    objective = _count("pair", Backend.TORCH)
    assert objective._supported_kernel() is None
    theta = objective.initial() + 0.05
    value, gradient = objective.value_and_gradient(theta)
    want_value, want_gradient = autograd_value_and_gradient(objective, theta)
    assert float(value) == float(want_value)
    assert torch.equal(gradient, want_gradient)
    assert torch.equal(objective.gradient(theta), want_gradient)


@pytest.mark.smoke
@pytest.mark.patch
def test_where_the_count_kernel_is_nan_the_e_step_route_answers() -> None:
    # A mean overflowing to inf is outside the kernel's domain: it returns
    # nan, and `value_and_gradient` and `gradient` take the E step and
    # backward pass there, as they do where no kernel holds.
    objective = _count("poisson")
    theta = objective.initial()
    theta[objective.blocks["mean"]] = 800.0
    kernel_value, _ = objective._supported_kernel().value_and_gradient(  # type: ignore[union-attr]
        theta.numpy()
    )
    assert np.isnan(kernel_value)
    value, gradient = objective.value_and_gradient(theta)
    want_value, want_gradient = objective._e_step_route(theta)
    assert torch.equal(value, want_value) or (value.isnan() and want_value.isnan())
    assert torch.allclose(gradient, want_gradient, equal_nan=True, rtol=0, atol=0)
    assert torch.allclose(
        objective.gradient(theta), want_gradient, equal_nan=True, rtol=0, atol=0
    )
