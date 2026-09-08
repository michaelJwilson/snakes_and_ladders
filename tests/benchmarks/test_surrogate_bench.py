"""What a surrogate costs against the evaluation it stands in for (issue #308).

One topology: a full fit, the plug-in bound, the parsimony bound, and a
learned prediction. One lattice: enumeration, the mean-field bound, the
spanning-tree bound. Correctness is pinned in
``tests/regression/likelihood/test_surrogate.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from snakes_and_ladders.learn.surrogate import MLPSurrogate, fit_surrogate
from snakes_and_ladders.likelihood.potts import enumerate_potts
from snakes_and_ladders.likelihood.surrogate import (
    ParsimonyUpperBound,
    PlugInLikelihood,
    mean_field_log_partition,
    spanning_tree_log_partition,
)
from snakes_and_ladders.search.infer import score_topology
from snakes_and_ladders.search.surrogate import (
    LearnedTreeSurrogate,
    fixed_length_target,
    tree_examples,
)
from snakes_and_ladders.search.topology import enumerate_topologies
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import load_fixture

FIVE_TAXA = "simulation_params_5taxa.yaml"
FIELD = np.array([0.3, -0.2, 0.1])


def _alignment() -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    params = load_fixture(FIVE_TAXA)
    dataset = simulate_alignment(
        params.tau, params.k, params.pi, np.random.default_rng(params.seed), 1000
    )
    return dict(dataset.alignment), params.k, np.asarray(params.pi)


def test_full_fit_benchmark(benchmark: BenchmarkFixture) -> None:
    alignment, k, _ = _alignment()
    topology = next(enumerate_topologies(sorted(alignment)))
    assert np.isfinite(benchmark(score_topology, topology, alignment, k))


@pytest.mark.parametrize("kind", ["plug_in", "parsimony"])
def test_analytic_tree_bound_benchmark(benchmark: BenchmarkFixture, kind: str) -> None:
    alignment, k, pi = _alignment()
    topology = next(enumerate_topologies(sorted(alignment)))
    surrogate = (
        PlugInLikelihood(k, pi) if kind == "plug_in" else ParsimonyUpperBound(k, pi)
    )
    assert torch.isfinite(benchmark(surrogate, topology, alignment))


def test_learned_tree_prediction_benchmark(benchmark: BenchmarkFixture) -> None:
    alignment, k, pi = _alignment()
    topologies = list(enumerate_topologies(sorted(alignment)))
    examples = tree_examples(
        [alignment], [topologies], k, pi, fixed_length_target(k, pi, 0.15)
    )
    fitted = fit_surrogate(
        MLPSurrogate(examples.features.shape[1]),
        examples,
        examples,
        generator=torch.Generator().manual_seed(0),
        max_epochs=20,
    )
    surrogate = LearnedTreeSurrogate(fitted, k, pi)
    assert torch.isfinite(benchmark(surrogate, topologies[0], alignment))


@pytest.mark.parametrize("kind", ["enumeration", "mean_field", "spanning_tree"])
def test_lattice_log_partition_benchmark(
    benchmark: BenchmarkFixture, kind: str
) -> None:
    graph = lattice_graph((3, 3), BoundaryCondition.OPEN, 0.7)
    if kind == "enumeration":
        assert np.isfinite(benchmark(enumerate_potts, graph, FIELD).log_partition)
        return
    bound = (
        mean_field_log_partition
        if kind == "mean_field"
        else spanning_tree_log_partition
    )
    assert torch.isfinite(benchmark(bound, graph, torch.as_tensor(FIELD)))
