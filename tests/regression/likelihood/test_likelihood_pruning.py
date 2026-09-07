"""Regression tests for ``phylo.likelihood.pruning``.

Four independent checks per issue #62, no two sharing an implementation:

- Brute-force agreement at ``n <= 6`` taxa, to machine precision
  (``test_pruning_matches_brute_force``) -- a genuinely different algorithm
  (direct marginalization, ``phylo.likelihood.brute_force``), not a second
  opinion from the same recursion.
- Rescaled and unrescaled paths agreeing on small problems where both run
  (``test_rescaled_and_unrescaled_agree_on_small_problems``), the check
  ``docs/tex/main.tex`` calls for after eq. (pruning).
- The pulley principle (``test_pulley_principle_is_invariant_to_root_position``):
  JC is reversible (pinned by
  ``tests/regression/test_jc_simulate.py``'s detailed-balance test), so
  sliding the root along the branch joining its two children -- splitting
  ``t`` into any ``t1 + t2`` -- must leave ``ln L`` unchanged.
- Scientific validity (``test_generating_topology_outscores_random_wrong_topologies``):
  on a simulated dataset, the generating topology scores above ``N`` random
  wrong topologies at sufficient sites.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from phylo.likelihood.brute_force import brute_force_log_likelihood
from phylo.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from phylo.likelihood.pruning import log_likelihood
from phylo.sim.params import load_simulation_params
from phylo.sim.simulate import simulate_alignment
from phylo.sim.tree import Node, preorder

from tests._fixtures import FIXTURES_DIR


def _small_tree_n4() -> Node:
    """4-taxon tree with a trifurcating root, mirroring the sim fixtures."""
    return Node(
        name="root",
        branch_length=None,
        children=(
            Node(name="A", branch_length=0.10),
            Node(name="B", branch_length=0.25),
            Node(
                name="ancestor_CD",
                branch_length=0.05,
                children=(
                    Node(name="C", branch_length=0.15),
                    Node(name="D", branch_length=0.40),
                ),
            ),
        ),
    )


def _small_tree_n6() -> Node:
    """6-taxon, fully binary tree -- exactly two children at the root, for
    the pulley-principle test's branch-length split."""
    return Node(
        name="root",
        branch_length=None,
        children=(
            Node(
                name="left",
                branch_length=0.08,
                children=(
                    Node(name="A", branch_length=0.10),
                    Node(name="B", branch_length=0.20),
                ),
            ),
            Node(
                name="right",
                branch_length=0.12,
                children=(
                    Node(
                        name="ancestor_CD",
                        branch_length=0.05,
                        children=(
                            Node(name="C", branch_length=0.15),
                            Node(name="D", branch_length=0.25),
                        ),
                    ),
                    Node(
                        name="ancestor_EF",
                        branch_length=0.05,
                        children=(
                            Node(name="E", branch_length=0.30),
                            Node(name="F", branch_length=0.10),
                        ),
                    ),
                ),
            ),
        ),
    )


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("tree_factory", "seed", "n_sites"),
    [
        (_small_tree_n4, 20260902, 20),
        (_small_tree_n6, 20260903, 15),
    ],
)
def test_pruning_matches_brute_force(
    tree_factory: Callable[[], Node], seed: int, n_sites: int
) -> None:
    tau = tree_factory()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(tau=tau, k=k, pi=pi, seed=seed, n_sites=n_sites)

    pruned = log_likelihood(tau, k, pi, dataset.alignment)
    brute = brute_force_log_likelihood(tau, k, pi, dataset.alignment)

    assert_allclose(pruned, brute, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.mathematical
def test_rescaled_and_unrescaled_agree_on_small_problems() -> None:
    tau = _small_tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(tau=tau, k=k, pi=pi, seed=20260904, n_sites=100)

    rescaled = log_likelihood(tau, k, pi, dataset.alignment, rescale=True)
    unrescaled = log_likelihood(tau, k, pi, dataset.alignment, rescale=False)

    assert_allclose(rescaled, unrescaled, rtol=1e-10)


def _relabel_leaves(node: Node, mapping: dict[str, str]) -> Node:
    """Rebuild ``node``'s subtree with every leaf name run through ``mapping``."""
    if node.is_leaf:
        return replace(node, name=mapping[node.name])
    return replace(
        node,
        children=tuple(_relabel_leaves(child, mapping) for child in node.children),
    )


@pytest.mark.mathematical
def test_pulley_principle_is_invariant_to_root_position() -> None:
    tau = _small_tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(tau=tau, k=k, pi=pi, seed=20260905, n_sites=200)

    left, right = tau.children
    assert left.branch_length is not None
    assert right.branch_length is not None
    total = left.branch_length + right.branch_length

    baseline = log_likelihood(tau, k, pi, dataset.alignment)

    for t1 in (0.01, total / 4, total / 2, total * 3 / 4, total - 0.01):
        slid = replace(
            tau,
            children=(
                replace(left, branch_length=t1),
                replace(right, branch_length=total - t1),
            ),
        )
        assert_allclose(
            log_likelihood(slid, k, pi, dataset.alignment),
            baseline,
            rtol=CROSS_DEVICE_RTOL_FLOAT64,
        )


@pytest.mark.simulated_truth
def test_generating_topology_outscores_random_wrong_topologies() -> None:
    params = load_simulation_params(FIXTURES_DIR / "simulation_params_8taxa.yaml")
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        seed=params.seed,
        n_sites=params.n_sites,
    )

    true_log_likelihood = log_likelihood(
        params.tau, params.k, params.pi, dataset.alignment
    )

    leaf_names = [node.name for node in preorder(params.tau) if node.is_leaf]
    rng = np.random.default_rng(20260906)

    n_wrong = 20
    wrong_log_likelihoods: list[float] = []
    while len(wrong_log_likelihoods) < n_wrong:
        permuted = rng.permutation(leaf_names)
        if np.array_equal(permuted, leaf_names):
            continue
        mapping = dict(zip(leaf_names, permuted, strict=True))
        wrong_tau = _relabel_leaves(params.tau, mapping)
        wrong_log_likelihoods.append(
            log_likelihood(wrong_tau, params.k, params.pi, dataset.alignment)
        )

    assert true_log_likelihood > max(wrong_log_likelihoods)
