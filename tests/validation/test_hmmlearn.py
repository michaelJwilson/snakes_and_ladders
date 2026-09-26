"""`opt.hmm.baum_welch` against hmmlearn's, run in a subprocess (issue #975).

`CategoricalHMM` on `hmm/ci.yaml` (3 states, 4 symbols, 600 x 15): ten
iterations and one alone within 1e-11. Streamed Gaussian and Poisson steps
against `GaussianHMM`, `PoissonHMM` (no priors or floors): within 1e-9
relative (#997). Compiled Viterbi against `decode` (every position,
log-probability 1e-11) and forward against `score` (1e-11; #997). Runtime
goal: `test_goals.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sal.emissions import GaussianEmission, PoissonEmission
from sal.fixtures import load_params
from sal.likelihood.hmm import hmm_log_likelihood, viterbi
from sal.opt.em import EmConfig
from sal.opt.hmm import baum_welch, baum_welch_family
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
    initial, transition, emission = (torch.log(torch.as_tensor(p)) for p in start)
    fit = baum_welch(
        observations,
        initial,
        transition,
        emission,
        config=EmConfig(max_iterations=n_iter, tolerance=-np.inf),
    )
    return (
        fit.log_initial.exp().numpy(),
        fit.log_transition.exp().numpy(),
        fit.log_emission.exp().numpy(),
    )


@pytest.mark.oracle
@pytest.mark.parametrize("n_iter", [1, 10])
def test_baum_welch_is_hmmlearns_iteration_for_iteration(n_iter: int) -> None:
    params = load_params(FIXTURE, HmmParams)
    observations = simulate_sequences(params).observations
    start = _start(params)
    theirs = hmmlearn.baum_welch(observations, *start, n_iter)
    ours = _ours(observations, start, n_iter)
    assert theirs.iterations == n_iter
    for mine, other in zip(
        ours, (theirs.initial, theirs.transition, theirs.emission), strict=False
    ):
        np.testing.assert_allclose(mine, other, rtol=0.0, atol=ATOL)
    # The start is not a fixed point: ten iterations moved the emissions.
    assert np.abs(theirs.emission - start[2]).max() > 1e-3


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


@pytest.mark.oracle
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_the_streamed_family_fit_is_hmmlearns(family: str) -> None:
    # Issue #997: the streamed Gaussian and Poisson steps against hmmlearn's
    # `GaussianHMM` and `PoissonHMM` with every prior and floor zero, ten
    # iterations from one start: each parameter within 1e-9 relative.
    observations = _family_sequences(family)
    initial = np.array([0.4, 0.3, 0.3])
    transition = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    start = (
        {"mean": np.array([-1.5, 0.5, 2.5]), "variance": np.array([1.2, 1.0, 1.8])}
        if family == "gaussian"
        else {"rate": np.array([2.0, 4.0, 10.0])}
    )
    family_start = (
        GaussianEmission(start["mean"], np.sqrt(start["variance"]), 1e-12)
        if family == "gaussian"
        else PoissonEmission(start["rate"])
    )
    ours = baum_welch_family(
        observations,
        torch.log(torch.as_tensor(initial)),
        torch.log(torch.as_tensor(transition)),
        family_start,
        config=EmConfig(max_iterations=10, tolerance=-np.inf),
    )
    theirs = hmmlearn.family_baum_welch(observations, initial, transition, start, 10)
    np.testing.assert_allclose(
        ours.log_initial.exp().numpy(), theirs.initial, rtol=1e-9
    )
    np.testing.assert_allclose(
        ours.log_transition.exp().numpy(), theirs.transition, rtol=1e-9
    )
    fitted = ours.components.named_parameters()
    if family == "gaussian":
        np.testing.assert_allclose(
            fitted["mean"].numpy(), theirs.emission["mean"], rtol=1e-9
        )
        np.testing.assert_allclose(
            fitted["scale"].numpy() ** 2, theirs.emission["variance"], rtol=1e-9
        )
    else:
        np.testing.assert_allclose(
            fitted["mean"].numpy(), theirs.emission["rate"], rtol=1e-9
        )


@pytest.mark.oracle
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_viterbi_is_hmmlearns(family: str) -> None:
    # Issue #997: the compiled Viterbi against hmmlearn's `decode`: every
    # path position equal and the total joint log-probability within 1e-11.
    observations = _family_sequences(family)
    initial = np.array([0.4, 0.3, 0.3])
    transition = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    start = (
        {"mean": np.array([-1.5, 0.5, 2.5]), "variance": np.array([1.2, 1.0, 1.8])}
        if family == "gaussian"
        else {"rate": np.array([2.0, 4.0, 10.0])}
    )
    family_start = (
        GaussianEmission(start["mean"], np.sqrt(start["variance"]), 1e-12)
        if family == "gaussian"
        else PoissonEmission(start["rate"])
    )
    states, log_probability = viterbi(
        observations,
        torch.log(torch.as_tensor(initial)),
        torch.log(torch.as_tensor(transition)),
        family_start,
    )
    theirs = hmmlearn.viterbi(observations, initial, transition, start)
    np.testing.assert_array_equal(states, theirs.states)
    np.testing.assert_allclose(log_probability, theirs.log_probability, rtol=1e-11)


@pytest.mark.oracle
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_the_hmm_log_likelihood_is_hmmlearns_score(family: str) -> None:
    # Issue #997: the compiled forward pass against hmmlearn's `score`.
    observations = _family_sequences(family)
    initial = np.array([0.4, 0.3, 0.3])
    transition = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    start = (
        {"mean": np.array([-1.5, 0.5, 2.5]), "variance": np.array([1.2, 1.0, 1.8])}
        if family == "gaussian"
        else {"rate": np.array([2.0, 4.0, 10.0])}
    )
    family_start = (
        GaussianEmission(start["mean"], np.sqrt(start["variance"]), 1e-12)
        if family == "gaussian"
        else PoissonEmission(start["rate"])
    )
    ours = hmm_log_likelihood(
        observations,
        torch.log(torch.as_tensor(initial)),
        torch.log(torch.as_tensor(transition)),
        family_start,
    )
    theirs = hmmlearn.score(observations, initial, transition, start)
    np.testing.assert_allclose(ours, theirs.log_likelihood, rtol=1e-11)
