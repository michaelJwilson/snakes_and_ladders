"""Pairwise distances invert the model exactly, converge on the truth, and state their error.

Issue #364. The Jukes--Cantor distance is pinned to the closed form it
inverts, the log-det distance to the branch length it equals under a
trace ``-k`` rate matrix and to ``-tr(Q)/k`` times it under a general one,
and both to their delta-method variance by the coverage of the interval it
gives over seeds --- the number a caller compares to Atteson's radius.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from snakes_and_ladders.likelihood.distance import (
    DistanceKind,
    distance_matrix,
    jukes_cantor_distance,
    log_det_distance,
    multinomial_delta_variance,
    pair_counts,
    tree_distances,
)
from snakes_and_ladders.sim.gtr import (
    gtr_rate_matrix,
    reversible_transition_probabilities,
)
from snakes_and_ladders.sim.jc import jc_transition_probabilities
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

from tests._fixtures import EIGHT_TAXA, FOUR_TAXA, load_fixture

#: Branch lengths the closed forms are inverted at, spanning short to near
#: saturation for four states (the saturation is at 0.75 of sites differing,
#: reached only as ``t`` grows without bound).
LENGTHS = (0.01, 0.1, 0.25, 0.5, 1.0, 2.0)


def _pair_from_frequencies(
    frequencies: np.ndarray, n_sites: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two sequences whose joint counts are ``round(n_sites * frequencies)``."""
    k = frequencies.shape[0]
    counts = np.rint(frequencies * n_sites).astype(np.int64)
    first = np.repeat(np.arange(k), counts.sum(axis=1))
    second = np.concatenate([np.repeat(np.arange(k), counts[row]) for row in range(k)])
    return first, second


@pytest.mark.mathematical
@pytest.mark.parametrize("t", LENGTHS)
@pytest.mark.parametrize("k", [2, 4])
def test_the_jukes_cantor_distance_inverts_the_transition_probabilities(
    t: float, k: int
) -> None:
    """At the exact pair frequencies ``P(t) / k`` the closed form returns ``t``.

    The pair frequency matrix of two taxa joined by a branch ``t`` is
    ``diag(pi) P(t)``; the distance read from it must be ``t`` to floating
    point, which is what makes the estimator an inversion rather than an
    approximation.
    """
    n_sites = 10**6
    frequencies = jc_transition_probabilities(t, k=k) / k
    first, second = _pair_from_frequencies(frequencies, n_sites)
    # Rounding to integer counts moves the total by at most k^2 sites and p
    # by at most that fraction; the closed form is held at the realized p.
    realized = 1.0 - np.trace(pair_counts(first, second, k)) / first.shape[0]
    expected = -((k - 1) / k) * np.log(1.0 - realized * k / (k - 1))

    estimate = jukes_cantor_distance(first, second, k)

    assert estimate.value == pytest.approx(expected, rel=1e-12)
    assert estimate.value == pytest.approx(t, abs=1e-4)


@pytest.mark.mathematical
@pytest.mark.parametrize("t", LENGTHS)
def test_the_log_det_distance_equals_the_branch_length_under_jukes_cantor(
    t: float,
) -> None:
    """``-tr(Q)/k`` is 1 for the normalized Jukes--Cantor ``Q``, so log-det returns ``t``."""
    k = 4
    frequencies = jc_transition_probabilities(t, k=k) / k
    first, second = _pair_from_frequencies(frequencies, 10**6)

    estimate = log_det_distance(first, second, k)

    assert estimate.value == pytest.approx(t, abs=2e-4)


@pytest.mark.mathematical
def test_the_log_det_distance_is_proportional_to_the_branch_length_under_gtr() -> None:
    """Under a reversible ``Q`` the log-det distance is ``-tr(Q)/k`` times ``t``.

    Additivity is what neighbor joining needs, and a constant factor keeps
    it; the factor is stated so a caller can undo it.
    """
    k = 4
    pi = np.array([0.1, 0.2, 0.3, 0.4])
    rate = gtr_rate_matrix(np.array([1.0, 2.0, 0.5, 3.0, 1.5, 1.0]), pi)
    factor = -np.trace(rate) / k
    for t in (0.1, 0.5, 1.0):
        frequencies = pi[:, np.newaxis] * reversible_transition_probabilities(
            rate, pi, t
        )
        first, second = _pair_from_frequencies(frequencies, 10**7)

        estimate = log_det_distance(first, second, k)

        assert estimate.value == pytest.approx(factor * t, rel=2e-3)
    assert factor != pytest.approx(1.0)


@pytest.mark.mathematical
def test_the_delta_variance_reduces_to_the_binomial_one() -> None:
    """A statistic linear in one proportion has variance ``p (1 - p) / L``."""
    frequencies = np.array([0.7, 0.3])
    gradient = np.array([0.0, 1.0])

    assert multinomial_delta_variance(frequencies, gradient, 1000) == pytest.approx(
        0.3 * 0.7 / 1000
    )


@pytest.mark.simulated_truth
@pytest.mark.parametrize("kind", list(DistanceKind))
def test_the_distance_converges_on_the_path_length_with_the_site_count(
    kind: DistanceKind,
) -> None:
    """The error against the true path length falls as ``1/sqrt(L)``.

    Two taxa ``A`` and ``B`` at branch lengths 0.1 and 0.25 below one root,
    path length 0.35, simulated at 1,000 to 100,000 sites over 100 seeds
    each: the root-mean-square error at each size, and that it falls by the
    factor the site ratio predicts. Realized: 0.0239, 0.00735 and 0.00221
    for the Jukes--Cantor distance, ratios 3.25 and 3.32 against
    ``sqrt(10) = 3.16``; 0.0240, 0.00735 and 0.00221 for log-det.
    """
    tau = Node("root", None, (Node("A", 0.1), Node("B", 0.25)))
    pi = np.full(4, 0.25)
    rms: list[float] = []
    for n_sites in (1000, 10_000, 100_000):
        errors = []
        for seed in range(100):
            dataset = simulate_alignment(
                tau, 4, pi, np.random.default_rng([20260908, seed]), n_sites
            )
            _, distances, _ = distance_matrix(dataset.alignment, 4, kind)
            errors.append(distances[0, 1] - 0.35)
        rms.append(float(np.sqrt(np.mean(np.square(errors)))))

    assert rms[0] < 0.03
    assert rms[2] < 0.003
    assert 2.5 < rms[0] / rms[1] < 4.0
    assert 2.5 < rms[1] / rms[2] < 4.0


@pytest.mark.simulated_truth
@pytest.mark.parametrize("kind", list(DistanceKind))
def test_the_stated_variance_covers_the_truth_at_the_nominal_rate(
    kind: DistanceKind,
) -> None:
    """``value +- 1.96 sqrt(variance)`` covers the true path length 95% of the time.

    The same pair at 2,000 sites over 400 seeds. A coverage inside the
    binomial 99% band around 0.95, ``[0.922, 0.978]``, is the delta method
    doing its job; outside it the variance would be wrong by more than
    sampling can explain. Realized: 0.940 (Jukes--Cantor) and 0.9425
    (log-det).
    """
    tau = Node("root", None, (Node("A", 0.1), Node("B", 0.25)))
    pi = np.full(4, 0.25)
    covered = 0
    n_seeds = 400
    for seed in range(n_seeds):
        dataset = simulate_alignment(
            tau, 4, pi, np.random.default_rng([20260909, seed]), 2000
        )
        _, distances, variances = distance_matrix(dataset.alignment, 4, kind)
        half_width = 1.96 * np.sqrt(variances[0, 1])
        covered += int(abs(distances[0, 1] - 0.35) <= half_width)
    rate = covered / n_seeds

    band = 2.576 * np.sqrt(0.95 * 0.05 / n_seeds)
    assert abs(rate - 0.95) <= band, rate


@pytest.mark.mathematical
@pytest.mark.parametrize("name", [FOUR_TAXA, EIGHT_TAXA])
def test_tree_distances_are_the_path_lengths(name: str) -> None:
    """Path length between two leaves is the sum of the branches between them.

    On the four-taxon fixture ``A`` to ``D`` crosses ``0.10 + 0.05 + 0.40``;
    on the eight-taxon one ``A`` to ``H`` crosses every level. Both are
    written out rather than computed, so the function is held to arithmetic
    it does not share.
    """
    params = load_fixture(name)
    names, matrix = tree_distances(params.tau)

    assert names == sorted(names)
    assert_allclose(matrix, matrix.T)
    assert_allclose(np.diag(matrix), 0.0)
    if name == FOUR_TAXA:
        assert matrix[names.index("A"), names.index("D")] == pytest.approx(0.55)
        assert matrix[names.index("C"), names.index("D")] == pytest.approx(0.55)
        assert matrix[names.index("A"), names.index("B")] == pytest.approx(0.35)
    else:
        assert matrix[names.index("A"), names.index("H")] == pytest.approx(
            0.10 + 0.05 + 0.05 + 0.05 + 0.05 + 0.40
        )
        assert matrix[names.index("A"), names.index("B")] == pytest.approx(0.25)


@pytest.mark.edge_case
def test_a_saturated_pair_is_refused() -> None:
    """Three quarters of sites differing has no finite four-state distance."""
    first = np.array([0, 1, 2, 3] * 25)
    second = np.array([1, 2, 3, 0] * 25)

    with pytest.raises(ValueError, match="saturation"):
        jukes_cantor_distance(first, second, 4)


@pytest.mark.edge_case
def test_a_singular_pair_frequency_matrix_is_refused() -> None:
    """An unobserved state makes ``det F`` zero and the log-det undefined."""
    first = np.array([0, 1, 0, 1, 2, 2])
    second = np.array([0, 1, 1, 0, 2, 2])

    with pytest.raises(ValueError, match="singular"):
        log_det_distance(first, second, 4)


@pytest.mark.edge_case
def test_mismatched_or_empty_sequences_are_refused() -> None:
    with pytest.raises(ValueError, match="shapes"):
        pair_counts(np.array([0, 1]), np.array([0]), 4)
    with pytest.raises(ValueError, match="at least one site"):
        pair_counts(np.array([], dtype=int), np.array([], dtype=int), 4)
    with pytest.raises(ValueError, match="at least 2 taxa"):
        distance_matrix({"A": np.array([0, 1])}, 4)
