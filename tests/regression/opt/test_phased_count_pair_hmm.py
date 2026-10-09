"""The phased count-pair HMM and the decode shift against their oracles (issue #1412).

`sal.opt.hmm.PhasedCountPairHmm` is `CountPairHmm`'s Rust model on the `2 K`
phased chain. Its posteriors and decode are pinned to `likelihood.ragged`
under each Kronecker kind, on a density built here from the family's own
`log_density` on the successes and on the failures; its fit to a reference
Baum-Welch written below on the Python E step and the family's own
`reestimate` on the positions unfolded (phase 1 a copy with `z -> n - z`).
The declared tolerance is `test_count_pair_hmm`'s, 1e-9 relative. The
`DecodeShift` is pinned analytically: after one iteration each group's
exposures are `e exp(-S_g)` with `S_g` read off the starting decode and means.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import CountPairEmission
from sal.likelihood.ragged import posteriors, viterbi
from sal.opt.em import EmConfig
from sal.opt.hmm import CountPairHmm, DecodeShift, PhasedCountPairHmm, SwitchKind
from sal.qa import hmm_fit_semantics as study
from sal.ragged import Ragged
from sal.sim.count_hmm_cell import CountHmm
from sal.sim.fixtures import fixture
from scipy.special import logsumexp

#: Relative agreement declared between the compiled and Python routes.
TOLERANCE = 1e-9
KINDS = (SwitchKind.KRONECKER, SwitchKind.KRONECKER_DIAGONAL)


@pytest.fixture(autouse=True)
def _single_thread() -> Iterator[None]:
    # The suite's one thread per process; restored after.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def cell() -> tuple[CountHmm, int, float]:
    # count_hmm_reference/ci: the unphased pin's instance, read here by the phased chain.
    params = fixture("count_hmm_reference", "ci").params
    return params.instance(), params.n_fit_states, params.stay


def _switch(n: int) -> np.ndarray:
    # A switch probability varying per position, 0.02 to 0.3: data the chain holds.
    return 0.02 + 0.28 * (np.arange(n) % 7) / 6.0


def _flipped(observations: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    # The successes as failures, n - z: what phase 1 reads.
    out = observations.copy()
    out[:, 1] = covariate[:, 1].astype(observations.dtype) - observations[:, 1]
    return out


def _phased_density(family: CountPairEmission, data: CountHmm) -> np.ndarray:
    # (n, 2K) in the Kronecker order 2k + a, from the family's own density on each phase.
    cov = torch.as_tensor(data.covariate)
    zero = family.log_density(torch.as_tensor(data.observations), covariate=cov).numpy()
    one = family.log_density(
        torch.as_tensor(_flipped(data.observations, data.covariate)), covariate=cov
    ).numpy()
    return np.stack([zero, one], axis=2).reshape(zero.shape[0], -1)


def _model(data: CountHmm, k: int, stay: float, kind: SwitchKind) -> PhasedCountPairHmm:
    log_initial, log_transition, family = study.start(data, k, stay)
    return PhasedCountPairHmm(
        Ragged(np.ascontiguousarray(data.observations.astype(np.int64)), data.lengths),
        np.repeat(log_initial.numpy(), 2) - np.log(2.0),
        log_transition.numpy(),
        family,
        covariate=Ragged(np.ascontiguousarray(data.covariate), data.lengths),
        switch=_switch(len(data.observations)),
        switch_kind=kind,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("kind", KINDS)
def test_phased_posteriors_and_decode_are_the_ragged_kernels(
    cell: tuple[CountHmm, int, float], kind: SwitchKind
) -> None:
    data, k, stay = cell
    model = _model(data, k, stay, kind)
    density = Ragged(
        np.ascontiguousarray(_phased_density(model.components, data)), data.lengths
    )
    switch = _switch(len(data.observations))
    # The Python oracle E step under the same switch, not the kernel the model calls.
    expected = posteriors(
        density,
        model.log_initial,
        model.log_transition,
        switch=switch,
        switch_kind=kind,
        backend=Backend.PYTHON,
    )
    found = model.posteriors()
    assert model.n_hidden == 2 * k
    np.testing.assert_allclose(
        np.exp(found.log_posterior), np.exp(expected.log_posterior), atol=1e-12
    )
    np.testing.assert_allclose(
        np.exp(found.log_counts), np.exp(expected.log_counts), rtol=1e-10
    )
    assert model.log_likelihood() == pytest.approx(
        float(np.sum(expected.log_evidence)), rel=1e-12
    )
    path, log_joint = model.viterbi()
    decoded = viterbi(
        density,
        model.log_initial,
        model.log_transition,
        switch=switch,
        switch_kind=kind,
    )
    np.testing.assert_array_equal(path, decoded.path)
    np.testing.assert_allclose(log_joint, decoded.log_joint, rtol=1e-12)


def _reference_fit(
    data: CountHmm, k: int, stay: float, kind: SwitchKind, iterations: int
) -> tuple[CountPairEmission, np.ndarray, np.ndarray, float]:
    # Baum-Welch on the phased chain in Python: the oracle E step, then the family's own
    # reestimate on the positions unfolded, then the initial and the copy-state transition.
    log_initial, log_transition_t, family = study.start(data, k, stay)
    initial = np.repeat(log_initial.numpy(), 2) - np.log(2.0)
    transition = log_transition_t.numpy()
    switch = _switch(len(data.observations))
    unfolded = torch.as_tensor(
        np.concatenate([data.observations, _flipped(data.observations, data.covariate)])
    )
    covariate = torch.as_tensor(np.concatenate([data.covariate, data.covariate]))
    starts = np.concatenate([[0], np.cumsum(data.lengths)[:-1]])
    log_likelihood = -np.inf
    for _ in range(iterations):
        density = Ragged(
            np.ascontiguousarray(_phased_density(family, data)), data.lengths
        )
        post = posteriors(
            density,
            initial,
            transition,
            switch=switch,
            switch_kind=kind,
            backend=Backend.PYTHON,
        )
        log_likelihood = float(np.sum(post.log_evidence))
        weights = np.exp(post.log_posterior)
        stacked = np.concatenate([weights[:, 0::2], weights[:, 1::2]])
        family = family.reestimate(
            unfolded, torch.as_tensor(stacked), covariate=covariate
        ).components
        initial = logsumexp(post.log_posterior[starts], axis=0) - np.log(len(starts))
        blocks = logsumexp(post.log_counts.reshape(k, 2, k, 2), axis=(1, 3))
        transition = blocks - logsumexp(blocks, axis=1, keepdims=True)
    return family, initial, transition, log_likelihood


@pytest.mark.oracle
@pytest.mark.parametrize("kind", KINDS)
def test_the_phased_fit_is_the_reference_baum_welch(
    cell: tuple[CountHmm, int, float], kind: SwitchKind
) -> None:
    data, k, stay = cell
    iterations = 8
    fitted = _model(data, k, stay, kind).fit(
        EmConfig(max_iterations=iterations, tolerance=0.0)
    )
    family, initial, transition, log_likelihood = _reference_fit(
        data, k, stay, kind, iterations
    )
    assert fitted.spent == iterations
    assert fitted.log_likelihood == pytest.approx(log_likelihood, rel=TOLERANCE)
    found = fitted.components
    assert isinstance(found, CountPairEmission)
    for mine, theirs in (
        (found.total.dispersion, family.total.dispersion),
        (found.total.mean, family.total.mean),
        (found.alpha, family.alpha),
        (found.beta, family.beta),
    ):
        np.testing.assert_allclose(mine.numpy(), theirs.numpy(), rtol=TOLERANCE)
    np.testing.assert_allclose(
        fitted.log_initial.exp().numpy(), np.exp(initial), atol=TOLERANCE
    )
    np.testing.assert_allclose(
        fitted.log_transition.exp().numpy(), np.exp(transition), atol=TOLERANCE
    )


@pytest.mark.analytic
def test_a_switch_of_zero_holds_each_phase(cell: tuple[CountHmm, int, float]) -> None:
    # With s = 0 and all mass on phase 0 at the start the phase never moves: the phased
    # likelihood is the unphased one.
    data, k, stay = cell
    log_initial, log_transition, family = study.start(data, k, stay)
    observations = Ragged(
        np.ascontiguousarray(data.observations.astype(np.int64)), data.lengths
    )
    covariate = Ragged(np.ascontiguousarray(data.covariate), data.lengths)
    initial = np.stack([log_initial.numpy(), np.full(k, -np.inf)], axis=1).ravel()
    phased = PhasedCountPairHmm(
        observations,
        initial,
        log_transition.numpy(),
        family,
        covariate=covariate,
        switch=np.zeros(len(data.observations)),
    )
    unphased = CountPairHmm(
        observations,
        log_initial.numpy(),
        log_transition.numpy(),
        family,
        covariate=covariate,
    )
    assert phased.log_likelihood() == pytest.approx(
        unphased.log_likelihood(), rel=1e-12
    )


@pytest.mark.analytic
def test_the_decode_shift_rescales_each_group(
    cell: tuple[CountHmm, int, float],
) -> None:
    data, k, stay = cell
    n = len(data.observations)
    groups = (np.arange(n) * 3) // n
    weights = 1.0 + (np.arange(n) % 5) / 10.0
    log_initial, log_transition, family = study.start(data, k, stay)
    observations = Ragged(
        np.ascontiguousarray(data.observations.astype(np.int64)), data.lengths
    )
    covariate = Ragged(np.ascontiguousarray(data.covariate), data.lengths)
    model = CountPairHmm(
        observations,
        log_initial.numpy(),
        log_transition.numpy(),
        family,
        covariate=covariate,
        shift=DecodeShift(groups, weights),
    )
    # The shift reads the first E step's decode and the starting means.
    decode = model.posteriors().log_posterior.argmax(axis=1)
    mean = family.total.mean.numpy()
    model.fit(EmConfig(max_iterations=1, tolerance=0.0))
    terms = weights * mean[decode]
    expected = data.covariate[:, 0].copy()
    for g in range(3):
        inside = groups == g
        observed = inside & (expected > 0)
        expected[observed] = expected[observed] / terms[inside].sum()
    np.testing.assert_allclose(model.exposures, expected, rtol=1e-13)


@pytest.mark.bug
def test_a_phased_chain_refuses_a_bad_switch(cell: tuple[CountHmm, int, float]) -> None:
    data, k, stay = cell
    with pytest.raises(ValueError, match="Kronecker kind"):
        _model(data, k, stay, SwitchKind.STAY_OR_MOVE)
