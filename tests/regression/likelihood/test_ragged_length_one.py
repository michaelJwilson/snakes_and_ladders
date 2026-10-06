"""Segments of one position against enumeration of every state path, on every route.

Issue #1233. A segment of one position reads no transition: its posterior is
``initial * emission`` normalised, its Viterbi state that product's arg max and
its log evidence the log-sum-exp of the same. The batch mixes lengths 1, 2 and
7, so the length-1 segments sit at the start, between longer ones and at the
end. Referee: brute-force enumeration of every path of every segment, within
1e-12 on the NumPy, Rust, torch and JAX routes.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import torch
from sal.backend import Backend
from sal.emissions import GaussianEmission
from sal.likelihood.ragged import (
    SwitchKind,
    posteriors,
    posteriors_oracle,
    sample_paths,
    viterbi,
)
from sal.numerics import logsumexp as normaliser
from sal.opt.hmm import EmissionHmmObjective, family_start
from sal.opt.hmm.forward import forward_log_likelihood_ragged
from sal.ragged import Ragged
from scipy.special import logsumexp

#: Length 1 first, between and last; 2 and 7 beside them.
LENGTHS = (1, 2, 7, 1, 1, 2)

#: Enumeration against a recursion: summation order alone separates them.
TOLERANCE = 1e-12

BACKENDS = [Backend.PYTHON, Backend.RUST]


def _model(seed: int, n_states: int = 3) -> tuple[Ragged, np.ndarray, np.ndarray]:
    """Scores, a log prior and a log kernel, drawn from `seed`."""
    rng = np.random.default_rng(seed)
    scores = Ragged(rng.normal(size=(sum(LENGTHS), n_states)), LENGTHS)
    log_initial = np.log(rng.dirichlet(np.ones(n_states)))
    log_transition = np.log(rng.dirichlet(np.ones(n_states), size=n_states))
    return scores, log_initial, log_transition


def _enumerated(
    scores: Ragged, log_initial: np.ndarray, log_transition: np.ndarray
) -> dict[str, np.ndarray]:
    """Every path of every segment scored: marginals, counts, evidence, best path."""
    n_states = log_initial.shape[0]
    gamma, counts = [], np.zeros((n_states, n_states))
    evidence, best, best_joint = [], [], []
    for segment in scores.segments():
        paths = list(product(range(n_states), repeat=len(segment)))
        joint = np.array(
            [
                log_initial[path[0]]
                + segment[0, path[0]]
                + sum(
                    log_transition[path[t - 1], path[t]] + segment[t, path[t]]
                    for t in range(1, len(segment))
                )
                for path in paths
            ]
        )
        total = float(logsumexp(joint))
        weight = np.exp(joint - total)
        marginal = np.zeros((len(segment), n_states))
        for w, path in zip(weight, paths, strict=True):
            marginal[np.arange(len(segment)), path] += w
            for t in range(1, len(segment)):
                counts[path[t - 1], path[t]] += w
        gamma.append(marginal)
        evidence.append(total)
        # `np.argmax` takes the first maximum, the lowest path in product order.
        best.extend(paths[int(np.argmax(joint))])
        best_joint.append(float(joint.max()))
    return {
        "gamma": np.concatenate(gamma),
        "counts": counts,
        "evidence": np.asarray(evidence),
        "path": np.asarray(best, dtype=np.int64),
        "log_joint": np.asarray(best_joint),
    }


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("backend", BACKENDS, ids=str)
@pytest.mark.parametrize("seed", range(3))
def test_posteriors_equal_enumeration(backend: Backend, seed: int) -> None:
    """Marginals, transition counts and evidence, against every path."""
    scores, log_initial, log_transition = _model(seed)
    truth = _enumerated(scores, log_initial, log_transition)
    got = posteriors(scores, log_initial, log_transition, backend=backend)
    np.testing.assert_allclose(
        np.exp(got.log_posterior), truth["gamma"], rtol=0.0, atol=TOLERANCE
    )
    np.testing.assert_allclose(
        np.exp(got.log_counts), truth["counts"], rtol=0.0, atol=TOLERANCE
    )
    np.testing.assert_allclose(
        got.log_evidence, truth["evidence"], rtol=0.0, atol=TOLERANCE
    )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("backend", BACKENDS, ids=str)
@pytest.mark.parametrize("seed", range(3))
def test_viterbi_equals_enumeration(backend: Backend, seed: int) -> None:
    """The best path of every segment and its joint, against every path."""
    scores, log_initial, log_transition = _model(seed)
    truth = _enumerated(scores, log_initial, log_transition)
    got = viterbi(scores, log_initial, log_transition, backend=backend)
    np.testing.assert_array_equal(got.path, truth["path"])
    np.testing.assert_allclose(
        got.log_joint, truth["log_joint"], rtol=0.0, atol=TOLERANCE
    )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(3))
def test_ffbs_draws_score_as_enumeration_scores_them(seed: int) -> None:
    """A drawn path's joint and evidence, against enumeration; Rust draws NumPy's path.

    The two routes invert the same uniforms, so the paths agree exactly.
    """
    scores, log_initial, log_transition = _model(seed)
    truth = _enumerated(scores, log_initial, log_transition)
    drawn = {
        backend: sample_paths(
            scores,
            log_initial,
            log_transition,
            np.random.default_rng(seed),
            backend=backend,
        )
        for backend in BACKENDS
    }
    np.testing.assert_array_equal(drawn[Backend.RUST].path, drawn[Backend.PYTHON].path)
    for one in drawn.values():
        np.testing.assert_allclose(
            one.log_evidence, truth["evidence"], rtol=0.0, atol=TOLERANCE
        )
        expected = []
        for segment, path in zip(
            scores.segments(), Ragged(one.path, LENGTHS).segments(), strict=True
        ):
            expected.append(
                log_initial[path[0]]
                + segment[0, path[0]]
                + sum(
                    log_transition[path[t - 1], path[t]] + segment[t, path[t]]
                    for t in range(1, len(path))
                )
            )
        np.testing.assert_allclose(one.log_joint, expected, rtol=0.0, atol=TOLERANCE)


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("backend", BACKENDS, ids=str)
@pytest.mark.parametrize("kind", list(SwitchKind), ids=str)
def test_a_one_position_segment_reads_no_switch(
    backend: Backend, kind: SwitchKind
) -> None:
    """Under every switch kind a length-1 segment's evidence is ``logsumexp(initial + scores)``.

    Its switch entry is never read, so the evidence is the switchless one.
    """
    rng = np.random.default_rng(1233)
    width = 3 if kind is SwitchKind.STAY_OR_MOVE else 6
    scores = Ragged(rng.normal(size=(sum(LENGTHS), width)), LENGTHS)
    log_initial = np.log(rng.dirichlet(np.ones(width)))
    log_transition = np.log(rng.dirichlet(np.ones(3), size=3))
    switch = rng.uniform(size=sum(LENGTHS))
    got = posteriors(
        scores,
        log_initial,
        log_transition,
        switch=switch,
        switch_kind=kind,
        backend=backend,
    )
    starts = np.asarray(scores.offsets[:-1])
    ones = np.flatnonzero(np.asarray(LENGTHS) == 1)
    expected = logsumexp(log_initial + scores.values[starts[ones]], axis=1)
    np.testing.assert_allclose(
        got.log_evidence[ones], expected, rtol=0.0, atol=TOLERANCE
    )


@pytest.mark.critical
@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(3))
def test_the_torch_forward_equals_enumeration(seed: int) -> None:
    """`forward_log_likelihood_ragged`'s total, against the summed enumerated evidence."""
    scores, log_initial, log_transition = _model(seed)
    truth = _enumerated(scores, log_initial, log_transition)
    total = forward_log_likelihood_ragged(
        torch.as_tensor(scores.values),
        LENGTHS,
        torch.as_tensor(log_initial),
        torch.as_tensor(log_transition),
    )
    assert abs(float(total) - float(truth["evidence"].sum())) <= TOLERANCE


@pytest.mark.critical
@pytest.mark.oracle
def test_the_jax_twin_equals_enumeration() -> None:
    """The JAX objective's value, against enumeration at the objective's own chain."""
    pytest.importorskip("jax")
    from sal.opt.hmm import jax as hmm_jax

    rng = np.random.default_rng(1233)
    data = Ragged(rng.normal(size=sum(LENGTHS)), LENGTHS)
    objective = EmissionHmmObjective(
        data, family_start(GaussianEmission, data.values, 3), backend=Backend.JAX
    )
    assert hmm_jax.twinned(objective)
    theta = objective.initial() + 0.2 * torch.as_tensor(
        rng.normal(size=objective.n_parameters)
    )
    log_initial, log_transition = (
        part.detach().numpy() for part in objective._chain(theta)
    )
    scores = Ragged(objective._log_density(theta).detach().numpy(), LENGTHS)
    truth = _enumerated(scores, log_initial, log_transition)
    value, _ = hmm_jax.value_and_grad(objective)(theta.numpy())
    assert abs(value + float(truth["evidence"].sum())) <= TOLERANCE


@pytest.mark.critical
@pytest.mark.analytic
def test_a_one_position_posterior_is_the_prior_times_the_emission() -> None:
    """``start * emission / sum``, bitwise on the NumPy oracle.

    The oracle normalises in log space with `sal.numerics.logsumexp` and
    exponentiates, as `forward_backward` does; written that way the identity
    is exact. Against SciPy's normaliser in probability space it holds within
    1e-15 relative: the two log-sum-exps differ in the last bit.
    """
    scores, log_initial, log_transition = _model(0)
    got = posteriors_oracle(scores, log_initial, log_transition)
    for start in np.asarray(scores.offsets[:-1])[np.asarray(LENGTHS) == 1]:
        joint = log_initial + scores.values[start]
        expected = np.log(np.exp(joint - float(normaliser(joint, axis=0))))
        np.testing.assert_array_equal(got.log_posterior[start], expected)
        product_ = np.exp(log_initial) * np.exp(scores.values[start])
        np.testing.assert_allclose(
            np.exp(got.log_posterior[start]),
            product_ / product_.sum(),
            rtol=1e-15,
            atol=0.0,
        )
