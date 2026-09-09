"""Site-pattern compression against the uncompressed log-likelihood (issue #408).

The claim is an identity, not an approximation: the weighted sum over
distinct columns is the sum over all columns, so the two numbers agree to
what floating-point reassociation allows and the weights agree with the
column counts exactly. Both halves are checked here -- the weights against a
count taken independently of :func:`compress`, and the value on every tree
fixture across all three backends.

The compression ratio is reported per fixture rather than asserted at a
value: it is a property of the data, it bounds the win, and pinning it would
pin the simulator's output rather than this module's.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
from snakes_and_ladders.likelihood import pruning, pruning_rust, pruning_torch
from snakes_and_ladders.likelihood.device import CROSS_DEVICE_RTOL_FLOAT64
from snakes_and_ladders.likelihood.patterns import SitePatterns, check_weights, compress
from snakes_and_ladders.likelihood.pruning_torch import branch_lengths_from_tree
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import preorder

from tests._fixtures import FIXTURES_DIR, load_fixture

#: Every tree fixture the registry declares, smallest first. `release` sits
#: at 200 000 sites and is the tier that shows the ratio saturating.
TREE_FIXTURES = (
    "tree_search/ci.yaml",
    "tree_jc/ci.yaml",
    "tree_search/stress.yaml",
    "tree_jc/stress.yaml",
    "tree_jc/release.yaml",
)

#: Sites simulated per fixture here. The fixture's own `n_sites` reaches
#: 200 000, which is a `release` cost for a claim that is an identity at any
#: length; 3 000 already exceeds the pattern count of every tree fixture, so
#: the compression is exercised in its saturated regime.
N_SITES = 3000


def _alignment(
    name: str, n_sites: int = N_SITES
) -> tuple[SimulationParams, dict[str, np.ndarray]]:
    params = load_fixture(name)
    dataset = simulate_alignment(
        params.tau, params.k, params.pi, np.random.default_rng(params.seed), n_sites
    )
    return params, dict(dataset.alignment)


@pytest.mark.mathematical
@pytest.mark.critical
def test_a_patterns_weight_is_its_column_count() -> None:
    # Against a Counter over the columns, which knows nothing of compress:
    # the weights are exact counts, they sum to the site count, and the
    # patterns are distinct.
    for name in TREE_FIXTURES:
        _, alignment = _alignment(name)
        patterns = compress(alignment)
        names = tuple(sorted(alignment))
        assert patterns.names == names
        columns = np.stack([alignment[leaf] for leaf in names])
        counted = Counter(map(tuple, columns.T.tolist()))
        found = {
            tuple(patterns.columns[:, index].tolist()): int(patterns.weights[index])
            for index in range(patterns.n_patterns)
        }
        assert found == dict(counted)
        assert int(patterns.weights.sum()) == patterns.n_sites == columns.shape[1]
        assert patterns.n_patterns == len(counted)


@pytest.mark.mathematical
@pytest.mark.critical
def test_the_compressed_log_likelihood_equals_the_uncompressed() -> None:
    # The identity of eq:site-independence, on every tree fixture and every
    # backend. The tolerance is the float64 cross-backend one: the two sums
    # differ only by the order they are accumulated in.
    ratios = {}
    for name in TREE_FIXTURES:
        params, alignment = _alignment(name)
        patterns = compress(alignment)
        ratios[name] = patterns.compression_ratio
        compressed = patterns.alignment
        pi = np.asarray(params.pi)

        full = pruning.log_likelihood(params.tau, params.k, pi, alignment)
        assert pruning.log_likelihood(
            params.tau, params.k, pi, compressed, weights=patterns.weights
        ) == pytest.approx(full, rel=CROSS_DEVICE_RTOL_FLOAT64)

        lengths = branch_lengths_from_tree(params.tau)
        assert float(
            pruning_torch.log_likelihood(
                params.tau,
                params.k,
                pi,
                compressed,
                lengths,
                weights=patterns.weights,
            )
        ) == pytest.approx(full, rel=CROSS_DEVICE_RTOL_FLOAT64)

        assert pruning_rust.log_likelihood(
            params.tau, params.k, pi, compressed, weights=patterns.weights
        ) == pytest.approx(full, rel=CROSS_DEVICE_RTOL_FLOAT64)

    assert min(ratios.values()) > 1.0


@pytest.mark.mathematical
def test_the_compression_ratio_at_each_fixtures_declared_size() -> None:
    # The ratio is a property of the data, so it is reported rather than
    # pinned; what is asserted is the bound it cannot exceed. A column is
    # one of `k ** n_taxa` values and there are `L` of them, so the pattern
    # count is at most the smaller, and the ratio is at least `L / that`.
    # This runs at the fixture's own `n_sites`, which the equality test
    # above does not: the identity holds at any length, the ratio does not.
    print("\ncompression at the declared size (L -> patterns):")
    for name in TREE_FIXTURES:
        params = load_fixture(name)
        dataset = simulate_alignment(
            params.tau,
            params.k,
            params.pi,
            np.random.default_rng(params.seed),
            params.n_sites,
        )
        patterns = compress(dict(dataset.alignment))
        ceiling = min(params.n_sites, params.k ** len(patterns.names))
        assert patterns.n_patterns <= ceiling
        assert patterns.compression_ratio >= params.n_sites / ceiling
        print(
            f"  {name}: {len(patterns.names)} taxa, "
            f"{patterns.n_sites} -> {patterns.n_patterns} columns, "
            f"{patterns.compression_ratio:.1f}x"
        )


@pytest.mark.mathematical
def test_weights_carry_through_the_gradient() -> None:
    # The weighted backend must differentiate the weighted sum, not the
    # unweighted one: the gradient through the compressed alignment is the
    # gradient through the full one, to float64 tolerance. Weights that
    # reached the graph as anything but constants would show up here.
    params, alignment = _alignment("tree_search/ci.yaml", 400)
    patterns = compress(alignment)
    pi = np.asarray(params.pi)

    def gradient(
        columns: dict[str, np.ndarray], weights: np.ndarray | None
    ) -> np.ndarray:
        lengths = branch_lengths_from_tree(params.tau).requires_grad_(True)
        value = pruning_torch.log_likelihood(
            params.tau, params.k, pi, columns, lengths, weights=weights
        )
        (found,) = torch.autograd.grad(value, lengths)
        return np.asarray(found.detach().numpy())

    assert np.allclose(
        gradient(patterns.alignment, patterns.weights),
        gradient(alignment, None),
        rtol=CROSS_DEVICE_RTOL_FLOAT64,
        atol=0.0,
    )


@pytest.mark.mathematical
def test_an_uncompressible_alignment_compresses_to_itself() -> None:
    # The identity at the other end: every column distinct means every
    # weight is one, the ratio is one, and the weighted call is the
    # unweighted call.
    params = load_fixture("tree_search/ci.yaml")
    names = sorted(node.name for node in preorder(params.tau) if node.is_leaf)
    # Distinct columns by construction: column s carries s in base k, which
    # needs k ** len(names) >= n_sites to stay injective.
    n_sites = 64
    digits = np.arange(n_sites)
    alignment = {
        name: ((digits // params.k**index) % params.k).astype(np.int64)
        for index, name in enumerate(names)
    }
    patterns = compress(alignment)
    assert patterns.n_patterns == n_sites
    assert patterns.compression_ratio == 1.0
    assert np.array_equal(patterns.weights, np.ones(n_sites, dtype=np.int64))
    pi = np.asarray(params.pi)
    assert pruning.log_likelihood(
        params.tau, params.k, pi, patterns.alignment, weights=patterns.weights
    ) == pytest.approx(
        pruning.log_likelihood(params.tau, params.k, pi, alignment),
        rel=CROSS_DEVICE_RTOL_FLOAT64,
    )


@pytest.mark.edge_case
def test_a_malformed_weight_or_alignment_is_refused() -> None:
    params, alignment = _alignment("tree_search/ci.yaml", 50)
    pi = np.asarray(params.pi)
    with pytest.raises(ValueError, match="one per column"):
        pruning.log_likelihood(params.tau, params.k, pi, alignment, weights=np.ones(3))
    with pytest.raises(ValueError, match="non-negative"):
        check_weights(-np.ones(4), 4)
    with pytest.raises(ValueError, match="no taxa"):
        compress({})
    ragged = dict(alignment)
    first = next(iter(ragged))
    ragged[first] = ragged[first][:-1]
    with pytest.raises(ValueError, match="ragged"):
        compress(ragged)


@pytest.mark.structural
def test_a_pattern_table_is_a_site_patterns() -> None:
    # The dataclass is the contract every backend reads; the alignment view
    # it exposes must be the rows of its own column block.
    _, alignment = _alignment("tree_jc/ci.yaml", 200)
    patterns = compress(alignment)
    assert isinstance(patterns, SitePatterns)
    view = patterns.alignment
    assert sorted(view) == list(patterns.names)
    for row, name in enumerate(patterns.names):
        assert np.array_equal(view[name], patterns.columns[row])
    assert FIXTURES_DIR.exists()
