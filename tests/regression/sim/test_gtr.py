"""Regression tests for the general time-reversible model.

Per root ``CLAUDE.md`` ("Pin to Independent Sources"), GTR is checked against
Jukes-Cantor, which this repository already validates against its own
closed form and against simulated frequencies: equal exchangeabilities and a
uniform ``pi`` must reproduce ``jc_rate_matrix`` and
``jc_transition_probabilities`` exactly, not approximately. The three model
invariants ``sim/CLAUDE.md`` cares about -- zero row sums, detailed balance,
and the rate normalization -- are asserted directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from phylo.sim.gtr import (
    exchangeabilities_from_free,
    exchangeability_matrix,
    gtr_rate_matrix,
    n_exchangeabilities,
    reversible_transition_probabilities,
)
from phylo.sim.jc import jc_rate_matrix, jc_transition_probabilities
from phylo.sim.simulate import simulate_alignment

from tests._fixtures import SMALL_SITES, load_fixture

# A deliberately asymmetric truth: no two exchangeabilities equal, no two
# frequencies equal, so a bug that collapsed either would show.
TRUE_EXCHANGEABILITIES = np.array([1.6, 0.4, 0.9, 0.7, 2.1, 1.0])
TRUE_PI = np.array([0.35, 0.15, 0.30, 0.20])


@pytest.mark.oracle
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_equal_rates_and_uniform_pi_reproduce_jukes_cantor(k: int) -> None:
    rate = gtr_rate_matrix(np.ones(n_exchangeabilities(k)), np.full(k, 1.0 / k))
    assert_allclose(rate, jc_rate_matrix(k), atol=1e-15)


@pytest.mark.oracle
@pytest.mark.parametrize("k", [2, 4, 5])
@pytest.mark.parametrize("t", [0.0, 0.05, 0.5, 2.0])
def test_transition_probabilities_reproduce_the_jc_closed_form(
    k: int, t: float
) -> None:
    pi = np.full(k, 1.0 / k)
    rate = gtr_rate_matrix(np.ones(n_exchangeabilities(k)), pi)
    assert_allclose(
        reversible_transition_probabilities(rate, pi, t),
        jc_transition_probabilities(t, k=k),
        atol=1e-15,
    )


@pytest.mark.mathematical
def test_rows_of_the_rate_matrix_sum_to_zero() -> None:
    rate = gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI)
    assert_allclose(rate.sum(axis=1), np.zeros(4), atol=1e-15)


@pytest.mark.mathematical
def test_the_model_satisfies_detailed_balance() -> None:
    # Reversibility is the property everything else here rests on: it is what
    # makes the symmetric eigendecomposition exact, and what makes the two
    # branches below a rooted root confounded.
    rate = gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI)
    flux = TRUE_PI[:, np.newaxis] * rate
    assert_allclose(flux, flux.T, atol=1e-15)


@pytest.mark.mathematical
def test_pi_is_stationary() -> None:
    rate = gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI)
    assert_allclose(TRUE_PI @ rate, np.zeros(4), atol=1e-15)


@pytest.mark.mathematical
def test_the_rate_is_normalized_to_one_substitution_per_unit_time() -> None:
    rate = gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI)
    assert_allclose(-(TRUE_PI * np.diag(rate)).sum(), 1.0, rtol=1e-14)


@pytest.mark.mathematical
@pytest.mark.parametrize("scale", [0.1, 3.7, 100.0])
def test_scaling_every_exchangeability_changes_nothing(scale: float) -> None:
    # Why one exchangeability has to be pinned. This is an exact invariance,
    # not an approximate one: the rate normalization divides the scale
    # straight back out, so without a gauge the likelihood is flat along this
    # direction and no parameter has a confidence interval.
    assert_allclose(
        gtr_rate_matrix(scale * TRUE_EXCHANGEABILITIES, TRUE_PI),
        gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI),
        atol=1e-15,
    )


@pytest.mark.mathematical
@pytest.mark.parametrize("t", [0.05, 0.4, 1.5])
def test_transition_probabilities_are_a_stochastic_matrix(t: float) -> None:
    rate = gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI)
    probabilities = reversible_transition_probabilities(rate, TRUE_PI, t)
    assert_allclose(probabilities.sum(axis=1), np.ones(4), atol=1e-14)
    assert bool((probabilities > 0.0).all())


@pytest.mark.edge_case
@pytest.mark.oracle
def test_zero_branch_length_gives_the_identity() -> None:
    rate = gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI)
    assert_allclose(
        reversible_transition_probabilities(rate, TRUE_PI, 0.0), np.eye(4), atol=1e-15
    )


@pytest.mark.mathematical
def test_a_long_branch_converges_to_the_stationary_distribution() -> None:
    # An analytic limit, so this pins the eigendecomposition against the
    # model rather than against another implementation of it.
    rate = gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI)
    probabilities = reversible_transition_probabilities(rate, TRUE_PI, 200.0)
    for row in probabilities:
        assert_allclose(row, TRUE_PI, atol=1e-12)


@pytest.mark.mathematical
def test_transition_probabilities_compose_along_a_branch() -> None:
    # P(s) P(t) == P(s + t), the defining property of a Markov semigroup.
    rate = gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI)
    first = reversible_transition_probabilities(rate, TRUE_PI, 0.3)
    second = reversible_transition_probabilities(rate, TRUE_PI, 0.45)
    assert_allclose(
        first @ second,
        reversible_transition_probabilities(rate, TRUE_PI, 0.75),
        atol=1e-14,
    )


@pytest.mark.structural
def test_free_exchangeabilities_are_completed_with_a_pinned_one() -> None:
    free = np.array([1.6, 0.4, 0.9, 0.7, 2.1])
    assert_allclose(exchangeabilities_from_free(free, 4), TRUE_EXCHANGEABILITIES)


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: n_exchangeabilities(1), "k must be >= 2"),
        (lambda: exchangeability_matrix(np.ones(5), 4), "expected 6 exchangeabilities"),
        (lambda: exchangeability_matrix(np.zeros(6), 4), "strictly positive"),
        (lambda: exchangeabilities_from_free(np.ones(3), 4), "expected 5 free"),
        (
            lambda: gtr_rate_matrix(np.ones(6), np.array([0.5, 0.2, 0.2, 0.2])),
            "sums to",
        ),
        (
            lambda: gtr_rate_matrix(np.ones(6), np.array([0.5, 0.5, 0.0, 0.0])),
            "strictly positive",
        ),
        (
            lambda: reversible_transition_probabilities(
                jc_rate_matrix(4), np.full(4, 0.25), -0.1
            ),
            "non-negative",
        ),
    ],
)
def test_malformed_input_is_refused(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]


# --- the simulator's new path --------------------------------------------


@pytest.mark.structural
def test_the_default_simulator_path_is_unchanged() -> None:
    # Root CLAUDE.md forbids silent behaviour changes, and a substitution
    # model is the last thing worth changing silently. Omitting rate_matrix
    # must still be Jukes-Cantor, bit for bit.
    params = load_fixture(SMALL_SITES)
    common = {
        "tau": params.tau,
        "k": params.k,
        "pi": params.pi,
        "seed": params.seed,
        "n_sites": 2000,
    }
    baseline = simulate_alignment(**common)  # type: ignore[arg-type]
    explicit = simulate_alignment(**common, rate_matrix=None)  # type: ignore[arg-type]
    for name, states in baseline.alignment.items():
        assert np.array_equal(states, explicit.alignment[name])


@pytest.mark.oracle
def test_simulating_under_a_jc_equivalent_gtr_matches_the_jc_path() -> None:
    # End-to-end: the two code paths reach the same alignment when the model
    # is the same one. Not asserted bit-for-bit -- the eigendecomposition and
    # the closed form differ by ~3e-16, which can flip an occasional
    # inverse-CDF draw -- but the disagreement must be at that scale, not at
    # the scale of a different model.
    params = load_fixture(SMALL_SITES)
    uniform = np.full(params.k, 1.0 / params.k)
    equivalent = gtr_rate_matrix(np.ones(n_exchangeabilities(params.k)), uniform)
    common = {
        "tau": params.tau,
        "k": params.k,
        "pi": uniform,
        "seed": params.seed,
        "n_sites": 20000,
    }
    baseline = simulate_alignment(**common)  # type: ignore[arg-type]
    through_gtr = simulate_alignment(**common, rate_matrix=equivalent)  # type: ignore[arg-type]

    for name, states in baseline.alignment.items():
        differing = int(np.count_nonzero(states != through_gtr.alignment[name]))
        assert differing / states.size < 1e-3, f"{name}: {differing} sites differ"


@pytest.mark.simulated_truth
def test_simulated_frequencies_match_the_generating_stationary_distribution() -> None:
    # The generative check for the new path: over a tree whose branches are
    # long relative to the rate, leaf frequencies approach pi. Compared
    # against the analytic pi, not against a second simulation.
    params = load_fixture(SMALL_SITES)
    rate = gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=TRUE_PI,
        seed=params.seed,
        n_sites=200000,
        rate_matrix=rate,
    )
    for states in dataset.alignment.values():
        observed = np.bincount(states, minlength=params.k) / states.size
        # Monte Carlo standard error at 2e5 draws is ~0.001; 0.01 is well
        # outside that and well inside a wrong stationary distribution.
        assert_allclose(observed, TRUE_PI, atol=0.01)
