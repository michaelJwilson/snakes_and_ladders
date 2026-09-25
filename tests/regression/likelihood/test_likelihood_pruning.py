"""Regression tests for ``sal.likelihood.pruning``.

Four independent checks per issue #62: brute force at ``n <= 6`` taxa to
machine precision, per route in ``ROUTES``
(``sal.likelihood.brute_force``); rescaled against unrescaled
(after ``eq:pruning``); the pulley principle, JC being reversible
(``test_jc_simulate.py``); and the generating topology outscoring ``N`` random
wrong ones at sufficient sites.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sal.likelihood.brute_force import brute_force_log_likelihood
from sal.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from sal.likelihood.pruning import log_likelihood
from sal.sim.simulate import simulate_alignment
from sal.sim.tree import Node, preorder

from tests._fixtures import SMALL_SITES, load_fixture, simulated_alignment
from tests.regression.likelihood.conftest import ROUTES, Route


def _small_tree_n6() -> Node:
    """6-taxon, fully binary tree -- exactly two children at the root, for
    the pulley-principle test's branch-length split; no fixture has one."""
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


def _fixture_tree() -> Node:
    """`tree_jc/ci`'s four-taxon tree, trifurcating at the root."""
    return load_fixture(SMALL_SITES).tau


#: Every route but the referee itself; each takes ``rescale``.
_JUDGED = {name: route for name, route in ROUTES.items() if name != "brute_force"}


@pytest.mark.oracle
@pytest.mark.parametrize("route", _JUDGED.values(), ids=list(_JUDGED))
@pytest.mark.parametrize(
    ("tree_factory", "seed", "n_sites"),
    [
        (_fixture_tree, 20260902, 20),
        (_small_tree_n6, 20260903, 15),
    ],
    ids=["n4", "n6"],
)
def test_pruning_matches_brute_force(
    route: Route, tree_factory: Callable[[], Node], seed: int, n_sites: int
) -> None:
    # Every route against direct marginalization, a different algorithm; one
    # body since issue #982 merged the Rust and Torch copies into it.
    tau = tree_factory()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(seed), n_sites=n_sites
    )

    pruned = float(route(tau, k, pi, dataset.alignment))
    brute = brute_force_log_likelihood(tau, k, pi, dataset.alignment)

    assert_allclose(pruned, brute, rtol=CROSS_DEVICE_RTOL_FLOAT64)


@pytest.mark.analytic
@pytest.mark.parametrize("route", _JUDGED.values(), ids=list(_JUDGED))
def test_rescaled_and_unrescaled_agree_on_small_problems(route: Route) -> None:
    tau = _small_tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260904), n_sites=100
    )

    rescaled = float(route(tau, k, pi, dataset.alignment, rescale=True))
    unrescaled = float(route(tau, k, pi, dataset.alignment, rescale=False))

    assert_allclose(rescaled, unrescaled, rtol=1e-10)


def _relabel_leaves(node: Node, mapping: dict[str, str]) -> Node:
    """Rebuild ``node``'s subtree with every leaf name run through ``mapping``."""
    if node.is_leaf:
        return replace(node, name=mapping[node.name])
    return replace(
        node,
        children=tuple(_relabel_leaves(child, mapping) for child in node.children),
    )


@pytest.mark.analytic
def test_pulley_principle_is_invariant_to_root_position() -> None:
    tau = _small_tree_n6()
    k = 4
    pi = np.full(k, 0.25)
    dataset = simulate_alignment(
        tau=tau, k=k, pi=pi, rng=np.random.default_rng(20260905), n_sites=200
    )

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


@pytest.mark.end2end
def test_generating_topology_outscores_random_wrong_topologies() -> None:
    params, alignment = simulated_alignment("tree_jc/release.yaml")

    true_log_likelihood = log_likelihood(params.tau, params.k, params.pi, alignment)

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
            log_likelihood(wrong_tau, params.k, params.pi, alignment)
        )

    assert true_log_likelihood > max(wrong_log_likelihoods)
