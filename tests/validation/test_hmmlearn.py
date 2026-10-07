"""`opt.hmm.baum_welch` against hmmlearn's, run in a subprocess (issue #975).

`CategoricalHMM` on `hmm/ci.yaml` (3 states, 4 symbols, 600 x 15): ten
iterations and one alone within 1e-11. Streamed Gaussian and Poisson steps
against `GaussianHMM`, `PoissonHMM` (no priors or floors): within 1e-9
relative (#997). Compiled Viterbi against `decode` (every position,
log-probability 1e-11) and forward against `score` (1e-11; #997). Runtime
goal: `test_goals.py`. `sal.external.hmm` (#1282): the categorical fit
bitwise the adapter's, its `score` sal's forward recursion within 1e-11,
ragged and with one-position segments included. Every comparison above calls
`external.hmm`, served by one session per module (step 7); the adapter's
`baum_welch` stays for that pin and for
`tests/benchmarks/test_opt_external_em_bench.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from sal.emissions import (
    CategoricalEmission,
    EmissionFamily,
    GaussianEmission,
    PoissonEmission,
)
from sal.external import Session, Solver, provenance, session
from sal.external import hmm as external_hmm
from sal.external.hmm import probabilities
from sal.fixtures import load_params
from sal.likelihood.hmm import hmm_log_likelihood, viterbi
from sal.opt.em import EmConfig
from sal.opt.hmm import (
    baum_welch,
    baum_welch_family,
    forward_log_likelihood,
    forward_log_likelihood_ragged,
)
from sal.opt.termination import Stop
from sal.ragged import Ragged
from sal.sim.hmm import HmmParams, simulate_sequences
from sal.validation import hmmlearn

from tests._fixtures import FIXTURES_DIR
from tests._frameworks import requires

pytestmark = [
    pytest.mark.validation,
    requires("hmmlearn"),
]

FIXTURE: Path = FIXTURES_DIR / "hmm" / "ci.yaml"

#: The float64 tolerance two implementations of one recursion are held to.
ATOL = 1e-11


@pytest.fixture(scope="module")
def served() -> Iterator[Session]:
    """One hmmlearn worker for the module's calls, so each skips interpreter and import start-up."""
    with session(Solver.HMMLEARN) as opened:
        yield opened


def _logs(*probabilities: np.ndarray) -> tuple[torch.Tensor, ...]:
    """Each probability array as the log tensor `external.hmm` and `opt.hmm` take."""
    return tuple(torch.log(torch.as_tensor(p)) for p in probabilities)


def _fixed(n_iter: int) -> EmConfig:
    """Exactly ``n_iter`` iterations: no tolerance stops the loop first."""
    return EmConfig(max_iterations=n_iter, tolerance=-np.inf)


def _start(params: HmmParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A start away from the truth: each row a perturbed uniform, seeded."""
    rng = np.random.default_rng(975)

    def simplex(*shape: int) -> np.ndarray:
        draw = rng.random(shape) + 0.5
        return np.asarray(draw / draw.sum(axis=-1, keepdims=True))

    return (
        simplex(params.n_states),
        simplex(params.n_states, params.n_states),
        simplex(params.n_states, params.n_symbols),
    )


def _ours(
    observations: np.ndarray, start: tuple[np.ndarray, ...], n_iter: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial, transition, emission = _logs(*start)
    fit = baum_welch(observations, initial, transition, emission, config=_fixed(n_iter))
    return (
        fit.log_initial.exp().numpy(),
        fit.log_transition.exp().numpy(),
        fit.log_emission.exp().numpy(),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("n_iter", [1, 10])
def test_baum_welch_is_hmmlearns_iteration_for_iteration(
    n_iter: int, served: Session
) -> None:
    params = load_params(FIXTURE, HmmParams)
    observations = simulate_sequences(params).observations
    start = _start(params)
    log_initial, log_transition, log_emission = _logs(*start)
    theirs = external_hmm.fit(
        observations,
        log_initial,
        log_transition,
        CategoricalEmission.from_log(log_emission),
        Solver.HMMLEARN,
        _fixed(n_iter),
        session=served,
    )
    assert isinstance(theirs.components, CategoricalEmission)
    fitted = (
        theirs.log_initial.exp().numpy(),
        theirs.log_transition.exp().numpy(),
        theirs.components.log_matrix.exp().numpy(),
    )
    ours = _ours(observations, start, n_iter)
    assert theirs.spent == n_iter
    for mine, other in zip(ours, fitted, strict=True):
        np.testing.assert_allclose(mine, other, rtol=0.0, atol=ATOL)
    # The start is not a fixed point: ten iterations moved the emissions.
    assert np.abs(fitted[2] - start[2]).max() > 1e-3


def _family_sequences(family: str) -> np.ndarray:
    """200 sequences of 30 from a sticky three-state chain, seed 997."""
    rng = np.random.default_rng(997)
    cumulative = np.array(
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]]
    ).cumsum(axis=1)
    states = np.empty((200, 30), dtype=np.int64)
    states[:, 0] = rng.choice(3, size=200)
    for t in range(1, 30):
        above = rng.random(200)[:, None] > cumulative[states[:, t - 1]]
        states[:, t] = above.sum(axis=1)
    if family == "gaussian":
        return np.asarray(rng.normal(np.array([-2.0, 0.0, 3.0])[states], 1.0))
    return np.asarray(rng.poisson(np.array([1.0, 5.0, 12.0])[states]))


#: The families' start: initial distribution and kernel, as probabilities.
FAMILY_INITIAL = np.array([0.4, 0.3, 0.3])
FAMILY_TRANSITION = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])


def _family_start(family: str) -> EmissionFamily:
    """The family's start: means and variances, or rates, away from the truth."""
    if family == "gaussian":
        return GaussianEmission(
            np.array([-1.5, 0.5, 2.5]), np.sqrt(np.array([1.2, 1.0, 1.8])), 1e-12
        )
    return PoissonEmission(np.array([2.0, 4.0, 10.0]))


def _family_problem(
    family: str,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, EmissionFamily]:
    """The family's sequences and start, as `opt.hmm` and `external.hmm` both take them."""
    log_initial, log_transition = _logs(FAMILY_INITIAL, FAMILY_TRANSITION)
    return (
        _family_sequences(family),
        log_initial,
        log_transition,
        _family_start(family),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_the_streamed_family_fit_is_hmmlearns(family: str, served: Session) -> None:
    # Issue #997: the streamed Gaussian and Poisson steps against hmmlearn's
    # `GaussianHMM` and `PoissonHMM` with every prior and floor zero, ten
    # iterations from one start: each parameter within 1e-9 relative.
    problem = _family_problem(family)
    ours = baum_welch_family(*problem, config=_fixed(10))
    theirs = external_hmm.fit(*problem, Solver.HMMLEARN, _fixed(10), session=served)
    np.testing.assert_allclose(
        ours.log_initial.exp().numpy(), theirs.log_initial.exp().numpy(), rtol=1e-9
    )
    np.testing.assert_allclose(
        ours.log_transition.exp().numpy(),
        theirs.log_transition.exp().numpy(),
        rtol=1e-9,
    )
    fitted = ours.components.named_parameters()
    other = theirs.components.named_parameters()
    np.testing.assert_allclose(fitted["mean"].numpy(), other["mean"].numpy(), rtol=1e-9)
    if family == "gaussian":
        np.testing.assert_allclose(
            fitted["scale"].numpy() ** 2, other["scale"].numpy() ** 2, rtol=1e-9
        )


@pytest.mark.oracle
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_viterbi_is_hmmlearns(family: str, served: Session) -> None:
    # Issue #997: the compiled Viterbi against hmmlearn's `decode`: every
    # path position equal and the total joint log-probability within 1e-11.
    problem = _family_problem(family)
    states, log_probability = viterbi(*problem)
    theirs = external_hmm.viterbi(*problem, Solver.HMMLEARN, session=served)
    np.testing.assert_array_equal(states, theirs.states)
    np.testing.assert_allclose(log_probability, theirs.log_probability, rtol=1e-11)


@pytest.mark.oracle
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_the_hmm_log_likelihood_is_hmmlearns_score(
    family: str, served: Session
) -> None:
    # Issue #997: the compiled forward pass against hmmlearn's `score`.
    problem = _family_problem(family)
    ours = hmm_log_likelihood(*problem)
    theirs = external_hmm.forward_log_likelihood(
        *problem, Solver.HMMLEARN, session=served
    )
    np.testing.assert_allclose(ours, float(theirs), rtol=1e-11)
    assert theirs.spent == 1
    assert theirs.termination.reason is Stop.CONVERGED


# `sal.external.hmm` (issue #1282, step 5): the categorical fit sends the
# adapter's bytes, so its answer is the adapter's bitwise; a probability is
# compared through the `np.log` the call reads it back with.


@pytest.mark.oracle
@pytest.mark.parametrize("n_iter", [1, 10])
def test_the_external_categorical_fit_is_the_adapters_bitwise(
    n_iter: int, served: Session
) -> None:
    params = load_params(FIXTURE, HmmParams)
    observations = simulate_sequences(params).observations
    log_initial, log_transition, log_emission = _logs(*_start(params))
    ours = external_hmm.fit(
        observations,
        log_initial,
        log_transition,
        CategoricalEmission.from_log(log_emission),
        Solver.HMMLEARN,
        _fixed(n_iter),
        session=served,
    )
    theirs = hmmlearn.baum_welch(
        observations,
        probabilities(log_initial),
        probabilities(log_transition),
        probabilities(log_emission),
        n_iter,
    )
    assert isinstance(ours.components, CategoricalEmission)
    assert np.array_equal(ours.log_initial.numpy(), np.log(theirs.initial))
    assert np.array_equal(ours.log_transition.numpy(), np.log(theirs.transition))
    assert np.array_equal(ours.components.log_matrix.numpy(), np.log(theirs.emission))
    assert ours.spent == ours.termination.iterations == theirs.iterations == n_iter
    assert ours.termination.reason is Stop.BUDGET
    assert ours.provenance == provenance(Solver.HMMLEARN)


@pytest.mark.oracle
def test_the_external_log_likelihood_is_sals_forward_recursion(
    served: Session,
) -> None:
    # The oracle: hmmlearn's `score` against `opt.hmm.forward_log_likelihood`
    # at the fixture's start, and the ragged form, three segments of one
    # position among them, against `forward_log_likelihood_ragged`; within
    # the adapter's 1e-11 relative.
    params = load_params(FIXTURE, HmmParams)
    observations = simulate_sequences(params).observations
    log_initial, log_transition, log_emission = _logs(*_start(params))
    theirs = external_hmm.forward_log_likelihood(
        observations,
        log_initial,
        log_transition,
        log_emission,
        Solver.HMMLEARN,
        session=served,
    )
    ours = forward_log_likelihood(
        torch.as_tensor(observations), log_initial, log_transition, log_emission
    )
    np.testing.assert_allclose(float(theirs), float(ours), rtol=1e-11)

    lengths = (1, 14, 1, 30, 1, 53)
    values = observations.reshape(-1)[: sum(lengths)]
    ragged = external_hmm.forward_log_likelihood(
        values,
        log_initial,
        log_transition,
        log_emission,
        Solver.HMMLEARN,
        lengths=lengths,
        session=served,
    )
    density = CategoricalEmission.from_log(log_emission).log_density(
        torch.as_tensor(values)
    )
    expected = forward_log_likelihood_ragged(
        density, lengths, log_initial, log_transition
    )
    np.testing.assert_allclose(float(ragged), float(expected), rtol=1e-11)


@pytest.mark.oracle
def test_the_external_fit_stops_where_sals_does(served: Session) -> None:
    # EmConfig's relative test, run inside hmmlearn's loop: from one start,
    # the default budget and tolerance stop both fits at the same iteration,
    # and the last E step's log-likelihoods agree within 1e-11 relative.
    observations = _family_sequences("gaussian")
    log_initial, log_transition = _logs(FAMILY_INITIAL, FAMILY_TRANSITION)
    start = _family_start("gaussian")
    theirs = external_hmm.fit(
        observations,
        log_initial,
        log_transition,
        start,
        Solver.HMMLEARN,
        session=served,
    )
    ours = baum_welch_family(observations, log_initial, log_transition, start)
    assert theirs.termination == ours.termination
    assert theirs.termination.reason is Stop.CONVERGED
    np.testing.assert_allclose(theirs.log_likelihood, ours.log_likelihood, rtol=1e-11)


@pytest.mark.oracle
def test_a_ragged_external_call_through_a_session_is_the_one_shot_call() -> None:
    # A `Ragged` batch, a one-position segment first, through a session and
    # through a fresh subprocess: the same bytes reach hmmlearn either way.
    observations = _family_sequences("poisson").reshape(-1)[:70]
    batch = Ragged(observations, (1, 29, 40))
    log_initial, log_transition = _logs(FAMILY_INITIAL, FAMILY_TRANSITION)
    start = _family_start("poisson")
    arguments = (batch, log_initial, log_transition, start, Solver.HMMLEARN)
    alone = external_hmm.viterbi(*arguments)
    with session(Solver.HMMLEARN) as opened:
        served = external_hmm.viterbi(*arguments, session=opened)
        fitted = external_hmm.fit(
            *arguments, EmConfig(max_iterations=3), session=opened
        )
    assert alone.states.shape == (70,)
    assert np.array_equal(alone.states, served.states)
    assert alone.log_probability == served.log_probability
    assert fitted.spent == 3
