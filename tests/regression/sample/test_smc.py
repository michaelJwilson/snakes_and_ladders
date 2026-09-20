"""The coalescence SMC against the target it states, enumerated (issue #824).

At five taxa the 180 coalescence sequences are enumerated and each tree scored
by Felsenstein pruning, so the evidence the sampler estimates and the law its
particles follow are both held to an exact sum. The likelihood evaluator is the
package's own, pinned elsewhere against brute-force marginalization; nothing of
the sampler's incremental weighting, resampling or caching is shared with the
enumeration.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from snakes_and_ladders.likelihood.pruning import log_likelihood
from snakes_and_ladders.sample import smc
from snakes_and_ladders.sample.smc import (
    smc_topologies,
    split_support,
    topology_weights,
)
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.topology import leaf_bipartitions
from snakes_and_ladders.sim.tree import Node

BRANCH_LENGTH = 0.2
SEEDS = 20
PARTICLES = 256


def _params() -> SimulationParams:
    params = fixture("tree_search", "ci").params
    assert isinstance(params, SimulationParams)
    return params


def _alignment(n_sites: int | None = None) -> dict[str, np.ndarray]:
    params = _params()
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites if n_sites is None else n_sites,
    )
    return dict(dataset.alignment)


def _sequences(forest: list[Node]):  # type: ignore[no-untyped-def]
    """Every coalescence sequence's final tree, the proposal's support enumerated."""
    if len(forest) == 1:
        yield forest[0]
        return
    for i, j in itertools.combinations(range(len(forest)), 2):
        left, right = forest[i], forest[j]
        joined = Node(
            "x",
            BRANCH_LENGTH,
            (
                Node(left.name, BRANCH_LENGTH, left.children),
                Node(right.name, BRANCH_LENGTH, right.children),
            ),
        )
        rest = [tree for index, tree in enumerate(forest) if index not in (i, j)]
        yield from _sequences([*rest, joined])


def _enumerated(
    alignment: dict[str, np.ndarray],
) -> tuple[float, dict[frozenset[frozenset[str]], float]]:
    """The exact log evidence and the exact law over unrooted topologies."""
    params = _params()
    trees = list(_sequences([Node(name, BRANCH_LENGTH) for name in sorted(alignment)]))
    assert len(trees) == 180
    logs = np.array(
        [
            log_likelihood(
                Node(t.name, None, t.children),
                params.k,
                np.asarray(params.pi),
                alignment,
            )
            for t in trees
        ]
    )
    peak = logs.max()
    log_evidence = float(peak + math.log(np.mean(np.exp(logs - peak))))
    law: dict[frozenset[frozenset[str]], float] = {}
    for tree, value in zip(trees, logs, strict=True):
        key = leaf_bipartitions(tree)
        law[key] = law.get(key, 0.0) + math.exp(value - log_evidence) / len(trees)
    return log_evidence, law


@pytest.mark.oracle
def test_the_evidence_estimate_is_unbiased_for_the_enumerated_mean_likelihood() -> None:
    # Twenty independent runs; the mean of the estimates against the exact
    # value, within three standard errors of the mean, with the realized gap
    # stated beside the tolerance in the assertion message.
    alignment = _alignment()
    exact, _ = _enumerated(alignment)
    params = _params()
    estimates = np.array(
        [
            smc_topologies(
                alignment,
                params.k,
                np.random.default_rng(seed),
                PARTICLES,
                branch_length=BRANCH_LENGTH,
                pi=params.pi,
            ).log_evidence
            for seed in range(SEEDS)
        ]
    )
    error = float(np.std(estimates, ddof=1) / math.sqrt(SEEDS))
    gap = abs(float(np.mean(estimates)) - exact)
    assert gap <= 3.0 * error, f"gap {gap:.4f} against 3 x {error:.4f}"
    assert error < 0.2


@pytest.mark.oracle
def test_the_particles_follow_the_enumerated_law_over_topologies() -> None:
    # Forty sites, so the law spreads over several of the 15 unrooted
    # topologies rather than sitting on one; the weighted frequency pooled over
    # twenty runs against the exact law, within five standard errors of a
    # binomial at the pooled *effective* count --- each run's smallest
    # per-step effective sample size, since resampling leaves the final
    # particles correlated and the nominal count would overstate the evidence.
    alignment = _alignment(n_sites=40)
    _, law = _enumerated(alignment)
    assert max(law.values()) < 0.9, "the law must spread for the test to see it"
    params = _params()
    pooled: dict[frozenset[frozenset[str]], float] = {}
    effective = 0.0
    for seed in range(SEEDS):
        result = smc_topologies(
            alignment,
            params.k,
            np.random.default_rng(100 + seed),
            PARTICLES,
            branch_length=BRANCH_LENGTH,
            pi=params.pi,
        )
        effective += float(result.ess.min())
        for key, weight in topology_weights(result).items():
            pooled[key] = pooled.get(key, 0.0) + weight / SEEDS
    worst = max(abs(pooled.get(key, 0.0) - mass) for key, mass in law.items())
    tolerance = 5.0 * math.sqrt(0.25 / effective)
    assert worst <= tolerance, f"largest gap {worst:.4f} against {tolerance:.4f}"
    assert abs(sum(pooled.values()) - 1.0) < 1e-9
    support = split_support(result)
    assert all(0.0 <= weight <= 1.0 + 1e-9 for weight in support.values())


@pytest.mark.analytic
def test_every_cached_partial_is_what_a_recomputation_returns() -> None:
    # The cache is keyed by subtree structure; every final tree's score read
    # back from it must be the pruning call it stands for, bitwise.
    alignment = _alignment(n_sites=60)
    params = _params()
    partials = smc._Partials(alignment, params.k, np.asarray(params.pi))
    result = smc_topologies(
        alignment, params.k, np.random.default_rng(3), 32, branch_length=BRANCH_LENGTH
    )
    for tree in result.trees:
        rooted = Node(tree.name, BRANCH_LENGTH, tree.children)
        assert partials.score(rooted) == partials.recomputed(rooted)
        assert partials.score(rooted) == partials.recomputed(rooted)
    assert partials.evaluations <= len(result.trees)
    assert result.evaluations >= 5


@pytest.mark.analytic
def test_the_pair_index_enumerates_every_unordered_pair_once() -> None:
    for size in (2, 3, 5, 8):
        pairs = [smc._pair(size, index) for index in range(size * (size - 1) // 2)]
        assert sorted(pairs) == list(itertools.combinations(range(size), 2))
    with pytest.raises(ValueError, match="at least 3 taxa"):
        smc_topologies(
            {"A": np.zeros(3, dtype=int), "B": np.zeros(3, dtype=int)},
            4,
            np.random.default_rng(0),
            4,
            branch_length=0.1,
        )
    with pytest.raises(ValueError, match="n_particles"):
        smc_topologies(
            _alignment(n_sites=8), 4, np.random.default_rng(0), 0, branch_length=0.1
        )
