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
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.search.alpha_expansion import alpha_expansion
from snakes_and_ladders.search.ground_state import lattice_rung
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.hmm import HmmParams, simulate_sequences
from snakes_and_ladders.sim.potts import critical_coupling, site_field
from snakes_and_ladders.validation.gaussian import GaussianTarget, diagonal_precision
from snakes_and_ladders.validation.runner import package

from tests._fixtures import FIXTURES_DIR
from tests.validation._goals import (
    Goal,
    MemoryGoal,
    assert_fits,
    assert_meets,
    median_seconds,
)

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

#: PyMaxflow's peak added resident memory for the same build and cut, the
#: medians of three subprocess runs (#987).
PYMAXFLOW_CUT_MEMORY = {
    side: MemoryGoal(
        "pymaxflow",
        f"the Rust cut on lattice_rung({side}, 2, seed=973)",
        peak_bytes,
        "2026-09-23, 4-core reference host, #987",
    )
    for side, peak_bytes in ((142, 4_874_240), (284, 20_439_040))
}


#: JAX's per-point gradient under `jit` and its peak added memory over 100
#: points, the medians of three subprocess runs (#991): the diagonal Gaussian
#: at d = 10, 10^3 and 10^4, and the three-component mixture at n = 10^5.
#: Where torch's autograd was the faster or the lighter (the dense Gaussian at
#: d = 10^3) no goal is set.
JAX_GRADIENT = {
    case: Goal(
        "jax",
        f"hmc.gradient_at per point, {case}",
        seconds,
        "2026-09-23, 4-core reference host, #991",
    )
    for case, seconds in (
        ("diagonal d=10", 11.35e-6),
        ("diagonal d=1000", 28.62e-6),
        ("diagonal d=10000", 51.62e-6),
        ("mixture n=100000", 2843.98e-6),
    )
}
JAX_GRADIENT_MEMORY = {
    case: MemoryGoal(
        "jax",
        f"hmc.gradient_at over 100 points, {case}",
        peak_bytes,
        "2026-09-23, 4-core reference host, #991",
    )
    for case, peak_bytes in (
        ("diagonal d=10", 14_741_504),
        ("diagonal d=1000", 9_277_440),
        ("diagonal d=10000", 27_959_296),
        ("mixture n=100000", 29_954_048),
    )
}


def _gradient_inputs(case: str) -> dict[str, np.ndarray]:
    """The points and target of one JAX goal, as `test_gradient_jax_bench.py` draws them."""
    rng = np.random.default_rng(991)
    if case.startswith("diagonal"):
        dimension = int(case.split("=")[1])
        return {
            "precision": diagonal_precision(dimension),
            "points": rng.normal(size=(100, dimension)),
        }
    draw = np.random.default_rng(975)
    component = draw.choice(3, size=100_000, p=[0.3, 0.3, 0.4])
    observations = draw.normal(
        np.array([-4.0, 0.0, 5.0])[component], np.array([1.0, 1.5, 1.0])[component]
    )
    centre = np.array([0.0, 0.3, -3.0, 0.5, 4.0, 0.2, 0.0, 0.3])
    return {
        "observations": observations,
        "n_components": np.asarray(3),
        "points": centre + 0.05 * rng.normal(size=(100, 8)),
    }


def _cut_inputs(side: int) -> dict[str, np.ndarray]:
    """The Rust cut's arrays for `lattice_rung(side, 2, seed=973)`."""
    rung = lattice_rung(side, 2, seed=973)
    field = site_field(rung.field, rung.graph.n_nodes, n_states=2)
    return {
        "n_nodes": np.asarray(rung.graph.n_nodes),
        "field": np.ascontiguousarray(field, dtype=np.float64).reshape(-1),
        "edges": rung.graph.edge_index.reshape(-1),
        "coupling": rung.graph.edge_coupling,
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

#: gco's peak added resident memory for the same build and expansion, the
#: medians of three subprocess runs (#987).
GCO_EXPANSION_MEMORY = {
    side: MemoryGoal(
        "gco",
        f"alpha expansion on the Rust cut, {side}x{side}, q = 10",
        peak_bytes,
        "2026-09-23, 4-core reference host, #987",
    )
    for side, peak_bytes in ((71, 2_686_976), (142, 9_756_672))
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

#: hmmlearn's peak added resident memory for the same ten iterations, the
#: medians of three subprocess runs (#987). It works one sequence at a time.
HMMLEARN_BAUM_WELCH_MEMORY = {
    n_sequences: MemoryGoal(
        "hmmlearn",
        f"ten Baum-Welch iterations at {100 * n_sequences:,} positions",
        peak_bytes,
        "2026-09-23, 4-core reference host, #987",
    )
    for n_sequences, peak_bytes in ((1_000, 372_736), (10_000, 1_196_032))
}

#: scikit-learn's peak added resident memory for the same ten iterations at
#: 10^5 draws (#987). At 10^6 the package's is the lower, 163 MB against
#: 184 MB, and sets no goal.
SCIKIT_LEARN_EM_MEMORY = {
    100_000: MemoryGoal(
        "scikit_learn",
        "ten mixture EM iterations at 100,000 draws",
        18_132_992,
        "2026-09-23, 4-core reference host, #987",
    )
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


#: BlackJAX's compiled HMC chain, compilation excluded: 1,000 transitions of
#: ten leapfrog steps at unit mass on the diagonal Gaussian with precisions
#: from 1 to 4, step 0.9 / (2 d^(1/4)), the medians of three subprocess runs
#: in `test_hmc_blackjax_bench.py`.
BLACKJAX_HMC = {
    dimension: Goal(
        "blackjax",
        f"1,000 HMC transitions of ten leapfrog steps at d = {dimension:,}",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 0.8, #963",
    )
    for dimension, seconds in ((100, 0.0141), (1_000, 0.0868), (10_000, 0.5839))
}

#: BlackJAX's peak added resident memory for the same compiled chain, the
#: medians of three subprocess runs (#987). At d = 10^4 most of it is the
#: 1,000 x 10^4 draws both sides keep.
BLACKJAX_HMC_MEMORY = {
    dimension: MemoryGoal(
        "blackjax",
        f"1,000 HMC transitions of ten leapfrog steps at d = {dimension:,}",
        peak_bytes,
        "2026-09-23, 4-core reference host, #987",
    )
    for dimension, peak_bytes in (
        (100, 798_720),
        (1_000, 8_085_504),
        (10_000, 80_031_744),
    )
}


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(PYMAXFLOW_CUT_MEMORY))
def test_the_rust_cut_fits_pymaxflows_memory(side: int) -> None:
    # Read in a fresh interpreter by `scripts/package.py`, as PyMaxflow's was.
    inputs = _cut_inputs(side)
    peaks = [package("ising_cut", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), PYMAXFLOW_CUT_MEMORY[side])


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(GCO_EXPANSION_MEMORY))
def test_the_expansion_fits_gcos_memory(side: int) -> None:
    field = np.random.default_rng(974).normal(size=(side * side, 10))
    inputs = {
        "shape": np.asarray([side, side]),
        "coupling": np.asarray(critical_coupling(10)),
        "field": field,
    }
    peaks = [package("alpha_expansion", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), GCO_EXPANSION_MEMORY[side])


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


@pytest.mark.experiment
@pytest.mark.parametrize("dimension", sorted(BLACKJAX_HMC))
def test_hmc_meets_blackjaxs_runtime(dimension: int) -> None:
    target = GaussianTarget(diagonal_precision(dimension))
    step = 0.9 / (2.0 * dimension**0.25)
    ours = median_seconds(
        lambda: hmc.sample(
            target,
            torch.Generator().manual_seed(963),
            1_000,
            step_size=step,
            n_steps=10,
        ),
        repeats=3,
    )
    assert_meets(ours, BLACKJAX_HMC[dimension])


@pytest.mark.experiment
@pytest.mark.parametrize("n_sequences", sorted(HMMLEARN_BAUM_WELCH_MEMORY))
def test_baum_welch_fits_hmmlearns_memory(n_sequences: int) -> None:
    params = dataclasses.replace(
        load_params(FIXTURES_DIR / "hmm" / "ci.yaml", HmmParams),
        lengths=(100,) * n_sequences,
    )
    initial, transition, emission = _hmm_start()
    inputs = {
        "observations": simulate_sequences(params).observations,
        "initial": initial,
        "transition": transition,
        "emission": emission,
        "n_iter": np.asarray(10),
    }
    peaks = [package("baum_welch", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), HMMLEARN_BAUM_WELCH_MEMORY[n_sequences])


@pytest.mark.experiment
@pytest.mark.parametrize("n_samples", sorted(SCIKIT_LEARN_EM_MEMORY))
def test_mixture_em_fits_scikit_learns_memory(n_samples: int) -> None:
    inputs = {
        "observations": _mixture_draws(n_samples),
        "weights": np.array([0.3, 0.3, 0.4]),
        "mean": np.array([-3.0, 0.5, 4.0]),
        "scale": np.array([1.2, 1.0, 1.3]),
        "n_iter": np.asarray(10),
    }
    peaks = [package("mixture_em", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), SCIKIT_LEARN_EM_MEMORY[n_samples])


@pytest.mark.experiment
@pytest.mark.parametrize("dimension", sorted(BLACKJAX_HMC_MEMORY))
def test_hmc_fits_blackjaxs_memory(dimension: int) -> None:
    inputs = {
        "precision": diagonal_precision(dimension),
        "step_size": np.asarray(0.9 / (2.0 * dimension**0.25)),
        "n_steps": np.asarray(10),
        "n_draws": np.asarray(1_000),
        "seed": np.asarray(963),
    }
    peaks = [package("hmc_sample", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), BLACKJAX_HMC_MEMORY[dimension])


@pytest.mark.experiment
@pytest.mark.parametrize("case", sorted(JAX_GRADIENT))
def test_the_gradient_meets_jaxs_runtime(case: str) -> None:
    inputs = _gradient_inputs(case)
    runs = [package("gradient", inputs) for _ in range(3)]
    per_point = float(np.median([float(run.outputs["per_point"]) for run in runs]))
    assert_meets(per_point, JAX_GRADIENT[case])


@pytest.mark.experiment
@pytest.mark.parametrize("case", sorted(JAX_GRADIENT_MEMORY))
def test_the_gradient_fits_jaxs_memory(case: str) -> None:
    inputs = _gradient_inputs(case)
    peaks = [package("gradient", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), JAX_GRADIENT_MEMORY[case])
