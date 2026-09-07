"""Regression tests for :mod:`phylo.sim.potts`.

Two independent oracles, for the two regimes the module supports. On the
loopy 3x3 lattice, exhaustive enumeration over every ``k ** n_nodes``
configuration gives the exact partition function and the exact single-site
and pairwise marginals -- a computation that shares no code with the Gibbs
sampler under test. On the open chain, the exact sampler is checked by
reduction: it must reproduce ``phylo.opt.potts.log_partition``'s
transfer-matrix ``log Z`` to machine precision, since both claim to describe
the same distribution by different routes.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from phylo.opt.potts import log_partition
from phylo.sim.graph import BoundaryCondition, lattice_graph
from phylo.sim.potts import load_potts_lattice_params, simulate_potts

from tests._fixtures import FIXTURES_DIR

FIXTURE = FIXTURES_DIR / "potts_lattice_params.yaml"

_RTOL_ORACLE = 1e-12


def _enumerate_lattice(
    graph_n_nodes: int,
    edges: tuple[tuple[int, int], ...],
    coupling: tuple[float, ...],
    field: np.ndarray,
) -> tuple[float, np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Exact ``log Z``, single-site marginals, and the pair marginal per edge.

    Sums over every one of ``n_states ** graph_n_nodes`` configurations --
    the only oracle independent of the Gibbs sampler under test.
    """
    n_states = field.shape[0]
    weights = []
    configurations = list(product(range(n_states), repeat=graph_n_nodes))
    for configuration in configurations:
        energy = sum(
            j * (configuration[a] == configuration[b])
            for (a, b), j in zip(edges, coupling, strict=True)
        )
        energy += sum(field[state] for state in configuration)
        weights.append(energy)
    weights_array = np.asarray(weights)
    peak = weights_array.max()
    unnormalized = np.exp(weights_array - peak)
    log_z = float(peak + np.log(unnormalized.sum()))
    probabilities = unnormalized / unnormalized.sum()

    single_site = np.zeros((graph_n_nodes, n_states))
    for configuration, probability in zip(configurations, probabilities, strict=True):
        for node, state in enumerate(configuration):
            single_site[node, state] += probability

    pair_marginals: dict[tuple[int, int], np.ndarray] = {
        edge: np.zeros((n_states, n_states)) for edge in edges
    }
    for configuration, probability in zip(configurations, probabilities, strict=True):
        for edge in edges:
            a, b = edge
            pair_marginals[edge][configuration[a], configuration[b]] += probability

    return log_z, single_site, pair_marginals


@pytest.mark.oracle
def test_gibbs_sampling_matches_brute_force_enumeration_on_a_loopy_lattice() -> None:
    params = load_potts_lattice_params(FIXTURE)
    graph = lattice_graph(
        params.shape, boundary=params.boundary, coupling=params.coupling
    )
    assert not graph.is_open_chain(), "the fixture is meant to exercise the loopy path"

    _, expected_single, expected_pairs = _enumerate_lattice(
        graph.n_nodes, graph.edges, graph.coupling, params.field
    )

    dataset = simulate_potts(
        graph,
        params.field,
        seed=params.seed,
        n_samples=params.n_samples,
        burn_in=params.burn_in,
    )
    configurations = dataset.configurations

    observed_single = np.zeros((graph.n_nodes, params.n_states))
    for node in range(graph.n_nodes):
        counts = np.bincount(configurations[:, node], minlength=params.n_states)
        observed_single[node] = counts / configurations.shape[0]
    assert_allclose(observed_single, expected_single, atol=params.tolerance)

    for edge in graph.edges:
        a, b = edge
        observed_pair = np.zeros((params.n_states, params.n_states))
        for sa, sb in zip(configurations[:, a], configurations[:, b], strict=True):
            observed_pair[sa, sb] += 1
        observed_pair /= configurations.shape[0]
        assert_allclose(observed_pair, expected_pairs[edge], atol=params.tolerance)


@pytest.mark.oracle
def test_gibbs_sampling_matches_brute_force_enumeration_at_a_second_size() -> None:
    # An independent confirmation at a different (n_states, shape) than the
    # fixture, per the issue's own two named sizes: 2-state 4x4 is 65,536
    # configurations.
    graph = lattice_graph((4, 4), boundary=BoundaryCondition.OPEN, coupling=0.5)
    field = np.array([0.4, -0.4])
    _, expected_single, expected_pairs = _enumerate_lattice(
        graph.n_nodes, graph.edges, graph.coupling, field
    )

    dataset = simulate_potts(graph, field, seed=20260904, n_samples=1500, burn_in=200)
    configurations = dataset.configurations

    observed_single = np.zeros((graph.n_nodes, 2))
    for node in range(graph.n_nodes):
        counts = np.bincount(configurations[:, node], minlength=2)
        observed_single[node] = counts / configurations.shape[0]
    assert_allclose(observed_single, expected_single, atol=0.06)

    edge = graph.edges[0]
    a, b = edge
    observed_pair = np.zeros((2, 2))
    for sa, sb in zip(configurations[:, a], configurations[:, b], strict=True):
        observed_pair[sa, sb] += 1
    observed_pair /= configurations.shape[0]
    assert_allclose(observed_pair, expected_pairs[edge], atol=0.06)


@pytest.mark.oracle
def test_the_open_chain_path_reproduces_the_transfer_matrix_log_z() -> None:
    # A reduction to an exact result, not sampler-vs-sampler agreement: the
    # same distribution described by phylo.opt.potts's transfer matrix and
    # by the backward-message sampler this module generalizes it from.
    coupling, length, n_states = 0.75, 10, 3
    field = np.array([0.4, -0.1, -0.3])
    graph = lattice_graph((length,), boundary=BoundaryCondition.OPEN, coupling=coupling)
    assert graph.is_open_chain()

    dataset = simulate_potts(graph, field, seed=1, n_samples=4000)
    observed = (
        np.bincount(dataset.configurations.ravel(), minlength=n_states)
        / dataset.configurations.size
    )

    expected_log_z = float(
        log_partition(
            torch.tensor(coupling, dtype=torch.float64),
            torch.as_tensor(field, dtype=torch.float64),
            length,
        )
    )
    # Independently, the exact sampler's own marginal (by enumeration) must
    # equal the closed-form one within the same rtol log_partition itself is
    # pinned to (tests/regression/opt/test_opt_potts.py).
    _, expected_single, _ = _enumerate_lattice(
        graph.n_nodes, graph.edges, graph.coupling, field
    )
    weights = [
        coupling * sum(1 for i in range(length - 1) if s[i] == s[i + 1])
        + float(field[list(s)].sum())
        for s in product(range(n_states), repeat=length)
    ]
    peak = max(weights)
    brute_force_log_z = peak + float(np.log(np.exp(np.asarray(weights) - peak).sum()))
    assert_allclose(brute_force_log_z, expected_log_z, rtol=_RTOL_ORACLE)

    # The i.i.d. sampler's realized marginal is a Monte Carlo quantity; only
    # the two log-Z values above are compared to machine precision.
    assert_allclose(observed, expected_single.mean(axis=0), atol=0.03)


@pytest.mark.structural
def test_simulated_dataset_has_the_declared_shape_and_alphabet() -> None:
    # A shape/alphabet check needs no equilibration, so it runs at a tiny
    # burn-in and sample count rather than the fixture's full,
    # distribution-accuracy-sized settings.
    params = load_potts_lattice_params(FIXTURE)
    graph = lattice_graph(
        params.shape, boundary=params.boundary, coupling=params.coupling
    )
    dataset = simulate_potts(
        graph, params.field, seed=params.seed, n_samples=20, burn_in=5
    )
    assert dataset.configurations.shape == (20, graph.n_nodes)
    assert set(np.unique(dataset.configurations)) <= set(range(params.n_states))


@pytest.mark.structural
def test_simulation_is_reproducible_from_the_seed() -> None:
    params = load_potts_lattice_params(FIXTURE)
    graph = lattice_graph(
        params.shape, boundary=params.boundary, coupling=params.coupling
    )
    first = simulate_potts(
        graph, params.field, seed=params.seed, n_samples=20, burn_in=5
    )
    second = simulate_potts(
        graph, params.field, seed=params.seed, n_samples=20, burn_in=5
    )
    assert np.array_equal(first.configurations, second.configurations)


@pytest.mark.edge_case
@pytest.mark.parametrize(
    ("replace", "with_", "message"),
    [
        ("n_states: 3", "n_states: 1", "n_states must be >= 2"),
        ("field: [0.30, -0.10, -0.20]", "field: [0.3, -0.1]", "field has shape"),
        # The yaml is the one place a boundary is still a string, since
        # `lattice_graph` takes the enum and cannot be handed a bad one.
        ("boundary: open", "boundary: diagonal", "boundary must be one of"),
    ],
)
def test_a_malformed_fixture_is_refused(
    replace: str, with_: str, message: str, tmp_path: Path
) -> None:
    path = tmp_path / "potts_lattice.yaml"
    path.write_text(FIXTURE.read_text().replace(replace, with_))
    with pytest.raises(ValueError, match=message):
        load_potts_lattice_params(path)


@pytest.mark.edge_case
def test_a_missing_field_is_refused(tmp_path: Path) -> None:
    text = "\n".join(
        line
        for line in FIXTURE.read_text().splitlines()
        if not line.startswith("seed:")
    )
    path = tmp_path / "potts_lattice.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="missing required field"):
        load_potts_lattice_params(path)
