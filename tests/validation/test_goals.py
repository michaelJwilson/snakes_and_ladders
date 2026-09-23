"""The runtimes the package has to meet, each set by an external framework (issue #972).

Each :class:`~tests.validation._goals.Goal` is a framework's runtime on a
declared fixture, measured on the 4-core reference host by the benchmark pair
its row names and written here as a number. The test times the package alone,
so no framework is installed to run it, and it fails until the package's
median meets the figure. It carries `goal` and runs in a non-blocking step of
CI's `validation` job.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.opt.hmm import baum_welch
from snakes_and_ladders.opt.mixture import expectation_maximization
from snakes_and_ladders.search.alpha_expansion import alpha_expansion
from snakes_and_ladders.search.ground_state import lattice_rung
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.hmm import HmmParams, simulate_sequences
from snakes_and_ladders.sim.potts import critical_coupling, site_field

from tests._fixtures import FIXTURES_DIR
from tests.validation._goals import Goal, assert_meets, median_seconds

pytestmark = pytest.mark.goal

#: PyMaxflow's graph build and cut on `lattice_rung(side, 2, seed=973)`, the
#: medians of five subprocess runs in `test_maxflow_pymaxflow_bench.py`.
PYMAXFLOW_CUT = {
    side: Goal(
        "pymaxflow",
        f"the Rust cut on lattice_rung({side}, 2, seed=973)",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 2.2, #973",
    )
    for side, seconds in ((142, 15.491e-3), (284, 68.667e-3))
}

#: gco's graph build and expansion to convergence, q = 10, on an open lattice
#: at `critical_coupling(10)` under a standard normal field drawn with seed
#: 974: the medians of three subprocess runs in
#: `test_alpha_expansion_gco_bench.py`.
GCO_EXPANSION = {
    side: Goal(
        "gco",
        f"alpha expansion on the Rust cut, {side}x{side}, q = 10",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.0, #974",
    )
    for side, seconds in ((71, 0.1402), (142, 0.6258))
}

#: hmmlearn's ten Baum--Welch iterations in log space on `hmm/ci.yaml`'s model
#: at sequences of 100, from the start `_hmm_start` draws: the medians of
#: three subprocess runs in `test_opt_external_em_bench.py`.
HMMLEARN_BAUM_WELCH = {
    n_sequences: Goal(
        "hmmlearn",
        f"ten Baum-Welch iterations at {100 * n_sequences:,} positions",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 0.7, #975",
    )
    for n_sequences, seconds in ((1_000, 1.91195), (10_000, 18.59566))
}

#: scikit-learn's ten EM iterations on a diagonal three-component mixture,
#: from the start below, on the draws `_mixture_draws` makes: the medians of
#: three subprocess runs in `test_opt_external_em_bench.py`.
SCIKIT_LEARN_EM = {
    n_samples: Goal(
        "scikit_learn",
        f"ten mixture EM iterations at {n_samples:,} draws",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.0, #975",
    )
    for n_samples, seconds in ((100_000, 0.30038), (1_000_000, 3.86556))
}


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(PYMAXFLOW_CUT))
def test_the_rust_cut_meets_pymaxflows_runtime(side: int) -> None:
    # The Rust kernel with its arrays prebuilt, which is its graph build and
    # cut in one call, against PyMaxflow's build and cut.
    rung = lattice_rung(side, 2, seed=973)
    field = np.ascontiguousarray(
        site_field(rung.field, rung.graph.n_nodes, n_states=2), dtype=np.float64
    ).reshape(-1)
    edges = rung.graph.edge_index.reshape(-1)
    coupling = rung.graph.edge_coupling
    ours = median_seconds(
        lambda: oxi_snakes_and_ladders.ising_ground_state(
            rung.graph.n_nodes, field, edges, coupling
        )
    )
    assert_meets(ours, PYMAXFLOW_CUT[side])


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(GCO_EXPANSION))
def test_the_expansion_meets_gcos_runtime(side: int) -> None:
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, critical_coupling(10))
    field = np.random.default_rng(974).normal(size=(graph.n_nodes, 10))
    ours = median_seconds(
        lambda: alpha_expansion(graph, field, 10, backend=Backend.RUST), repeats=3
    )
    assert_meets(ours, GCO_EXPANSION[side])


def _hmm_start() -> tuple[np.ndarray, ...]:
    """The benchmark's start: each row a perturbed uniform, seed 975."""
    rng = np.random.default_rng(975)
    shapes = ((3,), (3, 3), (3, 4))
    draws = [rng.random(shape) + 0.5 for shape in shapes]
    return tuple(draw / draw.sum(axis=-1, keepdims=True) for draw in draws)


@pytest.mark.experiment
@pytest.mark.parametrize("n_sequences", sorted(HMMLEARN_BAUM_WELCH))
def test_baum_welch_meets_hmmlearns_runtime(n_sequences: int) -> None:
    params = dataclasses.replace(
        load_params(FIXTURES_DIR / "hmm" / "ci.yaml", HmmParams),
        lengths=(100,) * n_sequences,
    )
    observations = simulate_sequences(params).observations
    initial, transition, emission = (
        torch.log(torch.as_tensor(p)) for p in _hmm_start()
    )
    ours = median_seconds(
        lambda: baum_welch(
            observations,
            initial,
            transition,
            emission,
            max_iterations=10,
            tolerance=-np.inf,
        ),
        repeats=3,
    )
    assert_meets(ours, HMMLEARN_BAUM_WELCH[n_sequences])


def _mixture_draws(n_samples: int) -> np.ndarray:
    """The benchmark's draws: components at -4, 0 and 5, deviations 1, 1.5, 1."""
    rng = np.random.default_rng(975)
    component = rng.choice(3, size=n_samples, p=[0.3, 0.3, 0.4])
    centre = np.array([-4.0, 0.0, 5.0])[component]
    spread = np.array([1.0, 1.5, 1.0])[component]
    return np.asarray(rng.normal(centre, spread))


@pytest.mark.experiment
@pytest.mark.parametrize("n_samples", sorted(SCIKIT_LEARN_EM))
def test_mixture_em_meets_scikit_learns_runtime(n_samples: int) -> None:
    observations = _mixture_draws(n_samples)
    start = GaussianEmission(
        np.array([-3.0, 0.5, 4.0]), np.array([1.2, 1.0, 1.3]), 1e-12
    )
    weights = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    ours = median_seconds(
        lambda: expectation_maximization(
            observations, weights, start, max_iterations=10, tolerance=-np.inf
        ),
        repeats=3,
    )
    assert_meets(ours, SCIKIT_LEARN_EM[n_samples])
