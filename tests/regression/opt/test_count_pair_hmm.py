"""The compiled count-pair HMM against its oracles on `count_hmm_reference/ci` (issue #1412).

`sal.opt.hmm.CountPairHmm` runs Baum-Welch in Rust in one crossing. Each
path is pinned to the Python one it replaces: per-state to
`baum_welch_family` (87 iterations to 1e-10, the same count, log-likelihood
within 1e-14 relative and parameters within 1.3e-11 measured); tied to
`baum_welch_family` through `hmm_fit_semantics.tied_m_step` (10 iterations,
log-likelihood 1.5e-14, parameters 2.8e-13); the posteriors and the decode to
`likelihood.ragged`; the joint L-BFGS M step to `scipy.optimize` on the same
expected log-likelihood. The declared tolerance is 1e-9 relative throughout:
the dispersion solves stop within 1e-12 in `log r` and the bisections within
1e-10, so the two routes agree to those and not bitwise.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import scipy.optimize
import torch
from sal.emissions import CountPairEmission
from sal.likelihood.ragged import posteriors, viterbi
from sal.opt.em import EmConfig
from sal.opt.hmm import CountPairHmm, MStepSolver, baum_welch_family
from sal.opt.termination import Stop
from sal.qa import hmm_fit_semantics as study
from sal.ragged import Ragged
from sal.sim.count_hmm_cell import CountHmm
from sal.sim.fixtures import fixture

#: Relative agreement declared between the compiled and Python routes.
TOLERANCE = 1e-9


@pytest.fixture(autouse=True)
def _single_thread() -> Iterator[None]:
    # The suite's one thread per process; restored after.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def cell() -> tuple[CountHmm, int, float]:
    params = fixture("count_hmm_reference", "ci").params
    return params.instance(), params.n_fit_states, params.stay


def _batches(cell: CountHmm) -> tuple[Ragged, Ragged]:
    return (
        Ragged(np.ascontiguousarray(cell.observations.astype(np.int64)), cell.lengths),
        Ragged(np.ascontiguousarray(cell.covariate), cell.lengths),
    )


def _model(
    cell: CountHmm, k: int, stay: float, semantics: study.Semantics, **options: bool
) -> CountPairHmm:
    log_initial, log_transition, family = study.start(cell, k, stay)
    observations, covariate = _batches(cell)
    return CountPairHmm(
        observations,
        log_initial.numpy(),
        log_transition.numpy(),
        family,
        covariate=covariate,
        tied=semantics.tied,
        fit_transition=semantics.fit_transition,
        **options,
    )


def _close(found: CountPairEmission, expected: CountPairEmission) -> None:
    for mine, theirs in (
        (found.total.dispersion, expected.total.dispersion),
        (found.total.mean, expected.total.mean),
        (found.alpha, expected.alpha),
        (found.beta, expected.beta),
    ):
        np.testing.assert_allclose(mine.numpy(), theirs.numpy(), rtol=TOLERANCE)


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("semantics", "iterations"), [(study.SAL, 500), (study.DOWNSTREAM, 10)]
)
def test_the_fit_is_baum_welch_familys(
    cell: tuple[CountHmm, int, float], semantics: study.Semantics, iterations: int
) -> None:
    data, k, stay = cell
    log_initial, log_transition, family = study.start(data, k, stay)
    observations, covariate = _batches(data)
    config = EmConfig(max_iterations=iterations, tolerance=semantics.tolerance)
    found = _model(data, k, stay, semantics).fit(config)
    expected = baum_welch_family(
        observations,
        log_initial,
        log_transition,
        family,
        config,
        covariate=covariate,
        fit_transition=semantics.fit_transition,
        m_step=study.tied_m_step if semantics.tied else None,
    )
    assert found.spent == expected.spent
    assert found.termination == expected.termination
    assert found.log_likelihood == pytest.approx(expected.log_likelihood, rel=TOLERANCE)
    assert isinstance(found.components, CountPairEmission)
    assert isinstance(expected.components, CountPairEmission)
    _close(found.components, expected.components)
    np.testing.assert_allclose(
        found.log_transition.exp().numpy(),
        expected.log_transition.exp().numpy(),
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        found.log_initial.exp().numpy(),
        expected.log_initial.exp().numpy(),
        atol=TOLERANCE,
    )


@pytest.mark.analytic
def test_a_held_chain_is_held_bitwise(cell: tuple[CountHmm, int, float]) -> None:
    data, k, stay = cell
    model = _model(data, k, stay, study.DOWNSTREAM, fit_initial=False)
    log_initial, log_transition = model.log_initial, model.log_transition
    fitted = model.fit(EmConfig(max_iterations=5, tolerance=0.0))
    assert fitted.termination.reason is Stop.BUDGET
    np.testing.assert_array_equal(fitted.log_initial.numpy(), log_initial)
    np.testing.assert_array_equal(fitted.log_transition.numpy(), log_transition)
    # NB the tie holds across states after the solve
    dispersion = model.components.total.dispersion.numpy()
    assert np.all(dispersion == dispersion[0])


@pytest.mark.oracle
def test_posteriors_and_decode_are_the_ragged_kernels(
    cell: tuple[CountHmm, int, float],
) -> None:
    data, k, stay = cell
    model = _model(data, k, stay, study.SAL)
    emit = model.components.log_density(
        torch.as_tensor(data.observations), covariate=torch.as_tensor(data.covariate)
    )
    density = Ragged(np.ascontiguousarray(emit.numpy()), data.lengths)
    expected = posteriors(density, model.log_initial, model.log_transition)
    found = model.posteriors()
    np.testing.assert_allclose(
        np.exp(found.log_posterior), np.exp(expected.log_posterior), atol=1e-12
    )
    assert model.log_likelihood() == pytest.approx(
        float(np.sum(expected.log_evidence)), rel=1e-12
    )
    path, log_joint = model.viterbi()
    decoded = viterbi(density, model.log_initial, model.log_transition)
    np.testing.assert_array_equal(path, decoded.path)
    np.testing.assert_allclose(log_joint, decoded.log_joint, rtol=1e-12)


def _expected_log_likelihood(
    family: CountPairEmission, data: CountHmm, weights: np.ndarray
) -> float:
    emit = family.log_density(
        torch.as_tensor(data.observations), covariate=torch.as_tensor(data.covariate)
    )
    return float((torch.as_tensor(weights) * emit).sum())


@pytest.mark.oracle
@pytest.mark.parametrize(
    "tied",
    # NB scipy's finite differences over 28 coordinates take 71 s; the tied 16 take 8 s
    [pytest.param(False, marks=pytest.mark.release), True],
)
def test_the_joint_m_step_is_scipys_maximum(
    cell: tuple[CountHmm, int, float], tied: bool
) -> None:
    data, k, stay = cell
    semantics = study.DOWNSTREAM if tied else study.SAL
    model = _model(data, k, stay, semantics)
    weights = np.exp(model.posteriors().log_posterior)
    start = model.components
    newton = _model(data, k, stay, semantics).m_step(weights)
    joint = model.m_step(weights, solver=MStepSolver.LBFGS)
    shared = 1 if tied else k

    def family(theta: np.ndarray) -> CountPairEmission:
        rate = 1.0 / (1.0 + np.exp(-theta[k : 2 * k]))
        tau = np.broadcast_to(np.exp(theta[2 * k + shared :]), (k,)).copy()
        return CountPairEmission(
            np.broadcast_to(np.exp(theta[2 * k : 2 * k + shared]), (k,)).copy(),
            np.exp(theta[:k]),
            rate * tau,
            (1.0 - rate) * tau,
            np.ones(k, dtype=np.int64),
            joint=False,
        )

    tau0 = (start.alpha + start.beta).numpy()
    theta = np.concatenate(
        [
            np.log(start.total.mean.numpy()),
            np.log(start.alpha.numpy() / start.beta.numpy()),
            np.log(start.total.dispersion.numpy()[:shared]),
            np.log(tau0[:shared]),
        ]
    )
    scale = float(weights.sum())
    solved = scipy.optimize.minimize(
        lambda x: -_expected_log_likelihood(family(x), data, weights) / scale,
        theta,
        method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 2000},
    )
    best = _expected_log_likelihood(family(solved.x), data, weights)
    mine = _expected_log_likelihood(joint, data, weights)
    # NB the joint maximum is at least scipy's, and at least the per-block step's
    assert mine >= best - 1e-6 * abs(best)
    assert mine >= _expected_log_likelihood(newton, data, weights) - 1e-9 * abs(mine)


@pytest.mark.oracle
def test_the_newton_m_step_is_the_familys(cell: tuple[CountHmm, int, float]) -> None:
    data, k, stay = cell
    model = _model(data, k, stay, study.SAL)
    weights = np.exp(model.posteriors().log_posterior)
    start = model.components
    expected = start.reestimate(
        torch.as_tensor(data.observations),
        torch.as_tensor(weights),
        covariate=torch.as_tensor(data.covariate),
    ).components
    _close(model.m_step(weights), expected)


@pytest.mark.analytic
def test_a_parameter_tolerance_stops_sooner(cell: tuple[CountHmm, int, float]) -> None:
    data, k, stay = cell
    config = EmConfig(max_iterations=500, tolerance=1e-10)
    plain = _model(data, k, stay, study.SAL).fit(config)
    loose = _model(data, k, stay, study.SAL).fit(config, parameter_tolerance=1e-3)
    assert loose.termination.reason is Stop.CONVERGED
    assert loose.spent < plain.spent


@pytest.mark.bug
def test_a_tied_start_must_hold_one_value(cell: tuple[CountHmm, int, float]) -> None:
    data, k, stay = cell
    log_initial, log_transition, family = study.start(data, k, stay)
    observations, covariate = _batches(data)
    varied = CountPairEmission(
        np.linspace(1.0, 2.0, k),
        family.total.mean,
        family.alpha,
        family.beta,
        np.ones(k, dtype=np.int64),
        joint=False,
    )
    with pytest.raises(ValueError, match="tied"):
        CountPairHmm(
            observations,
            log_initial.numpy(),
            log_transition.numpy(),
            varied,
            covariate=covariate,
            tied=True,
        )
