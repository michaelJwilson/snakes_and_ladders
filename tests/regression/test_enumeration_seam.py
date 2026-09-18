"""The one enumeration behind the oracles, pinned to the loops it replaced.

Issue #387: five modules built the product space with their own
``itertools.product`` and three took the argmax with their own loop. The
seam is checked against ``itertools.product`` itself --- order, count and
values --- and its reductions against the arithmetic the adapters carried,
so an adapter that calls it returns what it returned before.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from snakes_and_ladders.enumeration import (
    MAX_ENUMERABLE_CONFIGURATIONS,
    argmax,
    configurations,
    posterior,
    site_marginals,
)
from snakes_and_ladders.learn.hmm import enumerate_paths
from snakes_and_ladders.learn.potts import enumerate_configurations
from snakes_and_ladders.numerics import logsumexp


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize(("n_states", "n_sites"), [(2, 1), (2, 5), (3, 4), (5, 3)])
def test_the_product_space_is_itertools_product_in_its_order(
    n_states: int, n_sites: int
) -> None:
    expected = np.array(
        list(itertools.product(range(n_states), repeat=n_sites)), dtype=np.int64
    )

    realized = configurations(n_states, n_sites)

    assert realized.dtype == np.int64
    assert realized.flags.c_contiguous
    assert np.array_equal(realized, expected)


@pytest.mark.smoke
def test_no_sites_is_the_one_empty_configuration() -> None:
    assert configurations(3, 0).shape == (1, 0)


@pytest.mark.smoke
def test_the_cap_refuses_with_the_callers_name_for_the_space() -> None:
    with pytest.raises(ValueError, match="3\\*\\*12 hidden paths is past the limit"):
        configurations(3, 12, what="3**12 hidden paths", limit=1000)
    with pytest.raises(ValueError, match="2\\*\\*20 configurations"):
        configurations(2, 20, limit=MAX_ENUMERABLE_CONFIGURATIONS)


@pytest.mark.oracle
def test_the_posterior_is_the_shifted_exponential_and_its_log_sum() -> None:
    # The arithmetic `enumerate_potts` carried, term for term.
    log_weights = np.random.default_rng(387).normal(size=27) * 3.0
    peak = log_weights.max()
    unnormalized = np.exp(log_weights - peak)

    log_partition, probability = posterior(log_weights)

    assert log_partition == float(np.log(unnormalized.sum()) + peak)
    assert np.array_equal(probability, unnormalized / unnormalized.sum())
    assert log_partition == pytest.approx(float(logsumexp(log_weights, axis=0)))


@pytest.mark.oracle
def test_site_marginals_add_in_configuration_order() -> None:
    states = configurations(3, 4)
    weights = np.asarray(np.random.default_rng(1).random(states.shape[0]))
    expected = np.zeros((4, 3))
    for configuration, weight in zip(states, weights, strict=True):
        expected[np.arange(4), configuration] += weight

    assert np.array_equal(site_marginals(states, weights, 3), expected)


@pytest.mark.analytic
def test_the_argmax_is_the_first_maximizer() -> None:
    assert argmax(np.array([1.0, 3.0, 3.0, 2.0])) == 1
    assert argmax(np.array([-np.inf, -np.inf])) == 0


@pytest.mark.smoke
def test_the_learn_enumerators_yield_the_same_tuples_under_the_cap() -> None:
    assert list(enumerate_configurations(3, 3)) == list(
        itertools.product(range(3), repeat=3)
    )
    assert list(enumerate_paths(2, 4)) == list(itertools.product(range(2), repeat=4))
    assert all(
        isinstance(state, int) for state in next(iter(enumerate_configurations(2, 2)))
    )
    with pytest.raises(ValueError, match="past the limit"):
        next(iter(enumerate_paths(2, 30)))
