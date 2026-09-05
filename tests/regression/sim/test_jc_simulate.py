"""Regression tests for the k-state Jukes-Cantor simulator.

Per ``python/phylo/sim/CLAUDE.md``, simulated substitution frequencies are
validated against the closed-form Jukes-Cantor transition probabilities
(docs/tex/main.tex, eq. jc) -- never against the likelihood/pruning code,
which does not exist yet -- within a stated Monte Carlo tolerance. Math
invariants (rows summing to 1, detailed balance) are pinned independently
of any simulation.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from phylo.sim.jc import jc_rate_matrix, jc_transition_probabilities
from phylo.sim.params import load_simulation_params
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import edges

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "simulation_params.yaml"

# A variety of site and taxa sizes, per sim/CLAUDE.md's local rule, sharing
# one validation body below.
SITE_AND_TAXA_FIXTURES = (
    "simulation_params.yaml",  # 4 taxa, 200_000 sites
    "simulation_params_small_sites.yaml",  # 4 taxa, 20_000 sites
    "simulation_params_8taxa.yaml",  # 8 taxa, 200_000 sites
)


@pytest.mark.mathematical
def test_jc_transition_probabilities_rows_sum_to_one() -> None:
    p = jc_transition_probabilities(0.3, k=4)
    assert_allclose(p.sum(axis=1), np.ones(4), rtol=1e-12)


@pytest.mark.edge_case
@pytest.mark.oracle
def test_jc_transition_probabilities_at_zero_is_identity() -> None:
    p = jc_transition_probabilities(0.0, k=4)
    assert_allclose(p, np.eye(4), atol=1e-12)


@pytest.mark.mathematical
def test_jc_transition_probabilities_at_infinity_is_stationary() -> None:
    # k*t/(k-1) = 40 drives exp(...) to ~4e-18, well past float64 precision.
    p = jc_transition_probabilities(30.0, k=4)
    assert_allclose(p, np.full((4, 4), 0.25), atol=1e-12)


@pytest.mark.mathematical
def test_jc_rate_matrix_is_normalised() -> None:
    k = 4
    q = jc_rate_matrix(k)
    pi = np.full(k, 1.0 / k)

    # Rows of a rate matrix sum to zero.
    assert_allclose(q.sum(axis=1), np.zeros(k), atol=1e-12)
    # Branch-length normalization: -sum_i pi_i q_ii = 1 (eq. normalisation).
    assert np.isclose(-np.sum(pi * np.diagonal(q)), 1.0)


@pytest.mark.mathematical
def test_jc_detailed_balance_under_uniform_stationary_distribution() -> None:
    k = 4
    pi = np.full(k, 1.0 / k)
    p = jc_transition_probabilities(0.4, k=k)

    lhs = pi[:, np.newaxis] * p
    rhs = pi[np.newaxis, :] * p.T
    assert_allclose(lhs, rhs, atol=1e-12)


@pytest.mark.oracle
@pytest.mark.simulated_truth
@pytest.mark.parametrize("fixture_name", SITE_AND_TAXA_FIXTURES)
def test_simulated_substitution_frequencies_match_analytic_jc(
    fixture_name: str,
) -> None:
    params = load_simulation_params(FIXTURES_DIR / fixture_name)
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )

    for parent, child in edges(dataset.tau):
        assert child.branch_length is not None  # every non-root node has one
        parent_states = dataset.node_states[parent.name]
        child_states = dataset.node_states[child.name]

        expected = jc_transition_probabilities(child.branch_length, k=params.k)
        observed = np.zeros((params.k, params.k))
        for i in range(params.k):
            from_i = parent_states == i
            observed[i] = np.bincount(
                child_states[from_i], minlength=params.k
            ) / np.sum(from_i)

        assert_allclose(
            observed,
            expected,
            atol=params.tolerance,
            err_msg=f"branch {parent.name}->{child.name} (t={child.branch_length})",
        )


@pytest.mark.structural
def test_simulation_is_reproducible_given_seed() -> None:
    params = load_simulation_params(FIXTURE)
    first = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=1000
    )
    second = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=1000
    )

    for name, states in first.node_states.items():
        assert_allclose(states, second.node_states[name])


@pytest.mark.structural
def test_alignment_holds_exactly_the_leaf_states() -> None:
    params = load_simulation_params(FIXTURE)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=10
    )

    assert set(dataset.alignment) == {"A", "B", "C", "D"}
    for name, states in dataset.alignment.items():
        assert_allclose(states, dataset.node_states[name])


@pytest.mark.structural
def test_newick_carries_every_leaf_and_terminates() -> None:
    params = load_simulation_params(FIXTURE)
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=10
    )

    assert dataset.newick.endswith(";")
    for leaf_name in dataset.alignment:
        assert leaf_name in dataset.newick
