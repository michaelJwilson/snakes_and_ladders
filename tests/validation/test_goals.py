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
from snakes_and_ladders import oxisal
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.ppo import generalized_advantages
from snakes_and_ladders.learn.reinforce import surrogate_loss
from snakes_and_ladders.learn.surrogate import (
    Examples,
    GraphSurrogate,
    SetSurrogate,
    _Batch,
)
from snakes_and_ladders.opt.hmm import GaussianHmmObjective, baum_welch
from snakes_and_ladders.opt.mixture import expectation_maximization
from snakes_and_ladders.sample import hmc, metropolis
from snakes_and_ladders.sample.potts_mcmc import _bond_probability
from snakes_and_ladders.search.alpha_expansion import alpha_expansion
from snakes_and_ladders.search.ground_state import lattice_rung
from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
from snakes_and_ladders.sim.hmm import HmmParams, simulate_sequences
from snakes_and_ladders.sim.potts import critical_coupling, site_field
from snakes_and_ladders.validation.gaussian import GaussianTarget, diagonal_precision
from snakes_and_ladders.validation.runner import package

from tests._fixtures import FIXTURES_DIR
from tests.regression.learn.conftest import potts_environment
from tests.validation._goals import (
    Goal,
    MemoryGoal,
    assert_fits,
    assert_meets,
    median_seconds,
)
from tests.validation._rl import (
    WEIGHTS,
    greedy_episodes,
    potts_decisions,
    ppo_loss_and_gradient,
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


#: rustworkx's graph build and `connected_components` on one Swendsen--Wang
#: bond mask at beta = 1, the medians of three subprocess runs (#976). Its
#: peak memory is above the union-find's at every size and sets no goal.
RUSTWORKX_COMPONENTS = {
    side: Goal(
        "rustworkx",
        f"the union-find labelling of one bond mask, {side}x{side}",
        seconds,
        "2026-09-23, 4-core reference host, #976",
    )
    for side, seconds in ((142, 11.176e-3), (284, 46.366e-3))
}


#: rustworkx's cost of one Swendsen--Wang sweep's clusters at q = 3 on an
#: open lattice at the critical coupling: building the `PyGraph` from that
#: sweep's bonds and `connected_components` on it, the medians of three
#: subprocess runs, and their peak added memory (#997). The package's figure
#: is the whole sweep on the compiled pass, bonds and recolouring included.
RUSTWORKX_SWEEP = {
    side: Goal(
        "rustworkx",
        f"one Swendsen-Wang sweep's clusters, {side}x{side}, q = 3",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.3, #997",
    )
    for side, seconds in ((142, 11.285e-3), (284, 42.542e-3))
}

RUSTWORKX_SWEEP_MEMORY = {
    side: MemoryGoal(
        "rustworkx",
        f"one Swendsen-Wang sweep's clusters, {side}x{side}, q = 3",
        peak_bytes,
        "2026-09-23, 4-core reference host, #997",
    )
    for side, peak_bytes in ((142, 6_078_464), (284, 18_046_976))
}


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(RUSTWORKX_SWEEP))
def test_the_sweep_meets_rustworkxs_runtime(side: int) -> None:
    inputs = {"side": np.asarray(side), "seed": np.asarray(976)}
    seconds = [package("swendsen_wang", inputs).seconds for _ in range(5)]
    assert_meets(float(np.median(seconds)), RUSTWORKX_SWEEP[side])


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(RUSTWORKX_SWEEP_MEMORY))
def test_the_sweep_fits_rustworkxs_memory(side: int) -> None:
    inputs = {"side": np.asarray(side), "seed": np.asarray(976)}
    peaks = [package("swendsen_wang", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), RUSTWORKX_SWEEP_MEMORY[side])


#: TorchRL's `GAE` at `gamma = 1`, `lmbda = 0.95`, one call per episode of
#: 100 decisions, `_gae_rollouts`' rewards and values with every other
#: episode terminated: the summed per-episode seconds, the median of three
#: subprocess runs (#997). TD(0)'s target is GAE at `lmbda = 0`, the same
#: recursion, and sets no goal of its own.
TORCHRL_GAE = {
    n_episodes: Goal(
        "torchrl",
        f"GAE over {n_episodes:,} episodes of 100",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.3, #997",
    )
    for n_episodes, seconds in ((100, 0.12086), (1_000, 1.24487))
}


def _gae_rollouts(n_episodes: int) -> list[tuple[list[float], list[float], bool]]:
    """Rewards, values and the end flag of each episode, seed 997."""
    rng = np.random.default_rng(997)
    return [
        (rng.normal(size=100).tolist(), rng.normal(size=101).tolist(), bool(i % 2))
        for i in range(n_episodes)
    ]


@pytest.mark.experiment
@pytest.mark.parametrize("n_episodes", sorted(TORCHRL_GAE))
def test_the_advantages_meet_torchrls_gae_runtime(n_episodes: int) -> None:
    rollouts = _gae_rollouts(n_episodes)

    def every() -> None:
        for rewards, values, terminated in rollouts:
            generalized_advantages(rewards, values, lam=0.95, terminated=terminated)

    assert_meets(median_seconds(every, repeats=3), TORCHRL_GAE[n_episodes])


#: PyG's twin of `SetSurrogate` (hidden 8): its encoder and decoder around
#: `global_add_pool`, one example of `side^2` tokens, the median of five
#: subprocess runs of the median of nine warm forwards, at one intra-op
#: thread as the suite runs (`tests/conftest.py`) (#997). At 142^2 the twin
#: is level with the package after two attempts --- pooling before the
#: encoder's last affine map (6.2 to 3.6 ms at 284^2 on four threads) and the
#: body run in place --- since both run the same encoder over 20,164 rows.
TORCH_GEOMETRIC_SET = {
    side: Goal(
        "torch_geometric",
        f"SetSurrogate forward on {side}x{side} tokens",
        seconds,
        "2026-09-23, 4-core reference host, one thread, load 0.8, #997",
    )
    for side, seconds in ((142, 2.714e-3), (284, 12.576e-3))
}


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(TORCH_GEOMETRIC_SET))
def test_the_set_surrogate_meets_pygs_runtime(side: int) -> None:
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, 1.0)
    rng = np.random.default_rng(977)
    examples = Examples(
        features=torch.as_tensor(rng.normal(size=(1, 3))),
        targets=torch.zeros(1, dtype=torch.float64),
        groups=np.zeros(1, dtype=np.int64),
        tokens=(torch.as_tensor(rng.normal(size=(graph.n_nodes, 4))),),
        adjacency=(np.asarray(graph.edge_index, dtype=np.int64),),
    )
    batch = _Batch(examples)
    torch.manual_seed(977)
    model = SetSurrogate(3, 4, hidden=8)
    with torch.no_grad():
        model(batch)  # warm-up
        seconds = median_seconds(lambda: model(batch))
    assert_meets(seconds, TORCH_GEOMETRIC_SET[side])


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
        lambda: oxisal.ising_ground_state(rung.graph.n_nodes, field, edges, coupling)
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


#: gco's alpha-beta `swap()` to convergence on the expansion goals' instance
#: (q = 10, `critical_coupling(10)`, field seed 974), graph build excluded,
#: and its peak added memory with the build: medians of three subprocess
#: runs (#997). The package's swap stood at 1.57x / 1.12x after two attempts
#: --- a proposal scored by its energy change (0.92 to 0.76 s at 142^2) and
#: each cut built on the moving sites alone (to 0.63 s) --- and at 1.27x /
#: 0.89x the memory.
GCO_SWAP = {
    side: Goal(
        "gco",
        f"alpha-beta swap on the Rust cut, {side}x{side}, q = 10",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.2, #997",
    )
    for side, seconds in ((71, 0.0964), (142, 0.5556))
}

GCO_SWAP_MEMORY = {
    side: MemoryGoal(
        "gco",
        f"alpha-beta swap on the Rust cut, {side}x{side}, q = 10",
        peak_bytes,
        "2026-09-23, 4-core reference host, #997",
    )
    for side, peak_bytes in ((71, 2_572_288), (142, 9_568_256))
}


def _swap_inputs(side: int) -> dict[str, np.ndarray]:
    """The expansion goals' instance, run as a swap."""
    return {
        "shape": np.asarray([side, side]),
        "coupling": np.asarray(critical_coupling(10)),
        "field": np.random.default_rng(974).normal(size=(side * side, 10)),
        "move": np.asarray("swap"),
    }


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(GCO_SWAP))
def test_the_swap_meets_gcos_runtime(side: int) -> None:
    seconds = [package("alpha_expansion", _swap_inputs(side)).seconds for _ in range(3)]
    assert_meets(float(np.median(seconds)), GCO_SWAP[side])


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(GCO_SWAP_MEMORY))
def test_the_swap_fits_gcos_memory(side: int) -> None:
    peaks = [
        package("alpha_expansion", _swap_inputs(side)).peak_bytes or 0 for _ in range(3)
    ]
    assert_fits(int(np.median(peaks)), GCO_SWAP_MEMORY[side])


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


#: hmmlearn's ten Baum--Welch iterations of a three-state Gaussian (diagonal)
#: and a Poisson HMM at sequences of 100, every prior and floor zero, from the
#: start `_FAMILY_START` gives, on `_family_observations`: the medians of
#: three subprocess runs (#997).
HMMLEARN_FAMILY_BAUM_WELCH = {
    (family, n_sequences): Goal(
        "hmmlearn",
        f"ten {family} Baum-Welch iterations at {100 * n_sequences:,} positions",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.3, #997",
    )
    for family, n_sequences, seconds in (
        ("gaussian", 1_000, 2.10545),
        ("gaussian", 10_000, 21.29643),
        ("poisson", 1_000, 3.68399),
        ("poisson", 10_000, 36.13588),
    )
}

#: hmmlearn's peak added resident memory for the same fits (#997).
HMMLEARN_FAMILY_BAUM_WELCH_MEMORY = {
    (family, n_sequences): MemoryGoal(
        "hmmlearn",
        f"ten {family} Baum-Welch iterations at {100 * n_sequences:,} positions",
        peak_bytes,
        "2026-09-23, 4-core reference host, #997",
    )
    for family, n_sequences, peak_bytes in (
        ("gaussian", 1_000, 684_032),
        ("gaussian", 10_000, 2_109_440),
        ("poisson", 1_000, 1_843_200),
        ("poisson", 10_000, 8_105_984),
    )
}

#: hmmlearn's Viterbi (`decode`, `algorithm="viterbi"`) at the start
#: parameters on the same sequences: the medians of three subprocess runs,
#: seconds and peak added resident bytes (#997).
HMMLEARN_VITERBI = {
    (family, n_sequences): Goal(
        "hmmlearn",
        f"{family} Viterbi at {100 * n_sequences:,} positions",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.3, #997",
    )
    for family, n_sequences, seconds in (
        ("gaussian", 1_000, 0.035513),
        ("gaussian", 10_000, 0.264086),
        ("poisson", 1_000, 0.151608),
        ("poisson", 10_000, 1.546611),
    )
}

HMMLEARN_VITERBI_MEMORY = {
    (family, n_sequences): MemoryGoal(
        "hmmlearn",
        f"{family} Viterbi at {100 * n_sequences:,} positions",
        peak_bytes,
        "2026-09-23, 4-core reference host, #997",
    )
    for family, n_sequences, peak_bytes in (
        ("gaussian", 1_000, 2_142_208),
        ("gaussian", 10_000, 19_349_504),
        ("poisson", 1_000, 2_457_600),
        ("poisson", 10_000, 19_689_472),
    )
}

#: hmmlearn's `score`, the summed log-likelihood at the start parameters on
#: the same sequences: medians of three subprocess runs (#997).
HMMLEARN_SCORE = {
    (family, n_sequences): Goal(
        "hmmlearn",
        f"{family} HMM log-likelihood at {100 * n_sequences:,} positions",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.2, #997",
    )
    for family, n_sequences, seconds in (
        ("gaussian", 1_000, 0.046337),
        ("gaussian", 10_000, 0.332986),
        ("poisson", 1_000, 0.156309),
        ("poisson", 10_000, 1.427731),
    )
}

HMMLEARN_SCORE_MEMORY = {
    (family, n_sequences): MemoryGoal(
        "hmmlearn",
        f"{family} HMM log-likelihood at {100 * n_sequences:,} positions",
        peak_bytes,
        "2026-09-23, 4-core reference host, #997",
    )
    for family, n_sequences, peak_bytes in (
        ("gaussian", 1_000, 442_368),
        ("gaussian", 10_000, 2_084_864),
        ("poisson", 1_000, 774_144),
        ("poisson", 10_000, 2_387_968),
    )
}

#: scikit-learn's `score_samples`, summed, at the mixture EM goal's start on
#: `_mixture_draws`: medians of three subprocess runs (#997).
SCIKIT_LEARN_SCORE = {
    n_samples: Goal(
        "scikit_learn",
        f"mixture log-likelihood at {n_samples:,} draws",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.2, #997",
    )
    for n_samples, seconds in ((100_000, 0.04040), (1_000_000, 0.63677))
}

SCIKIT_LEARN_SCORE_MEMORY = {
    n_samples: MemoryGoal(
        "scikit_learn",
        f"mixture log-likelihood at {n_samples:,} draws",
        peak_bytes,
        "2026-09-23, 4-core reference host, #997",
    )
    for n_samples, peak_bytes in ((100_000, 13_111_296), (1_000_000, 131_989_504))
}

#: The start both sides fit from: probabilities, and each family's parameters.
_FAMILY_START: dict[str, dict[str, np.ndarray]] = {
    "gaussian": {
        "mean": np.array([-1.5, 0.5, 2.5]),
        "variance": np.array([1.2, 1.0, 1.8]),
    },
    "poisson": {"rate": np.array([2.0, 4.0, 10.0])},
}
_FAMILY_INITIAL = np.array([0.4, 0.3, 0.3])
_FAMILY_TRANSITION = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])


def _family_observations(family: str, n_sequences: int) -> np.ndarray:
    """Sequences of 100 from a sticky three-state chain, seed 997."""
    rng = np.random.default_rng(997)
    transition = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    states = np.empty((n_sequences, 100), dtype=np.int64)
    states[:, 0] = rng.choice(3, size=n_sequences, p=[0.5, 0.3, 0.2])
    cumulative = transition.cumsum(axis=1)
    for t in range(1, 100):
        above = rng.random(n_sequences)[:, None] > cumulative[states[:, t - 1]]
        states[:, t] = above.sum(axis=1)
    if family == "gaussian":
        return np.asarray(
            rng.normal(
                np.array([-2.0, 0.0, 3.0])[states], np.array([1.0, 0.5, 1.5])[states]
            )
        )
    return np.asarray(rng.poisson(np.array([1.0, 5.0, 12.0])[states]))


def _family_inputs(family: str, n_sequences: int) -> dict[str, np.ndarray]:
    """The harness's inputs for one family fit."""
    return {
        "observations": _family_observations(family, n_sequences),
        "initial": _FAMILY_INITIAL,
        "transition": _FAMILY_TRANSITION,
        "family": np.asarray(family),
        **_FAMILY_START[family],
        "n_iter": np.asarray(10),
    }


@pytest.mark.experiment
@pytest.mark.parametrize(("family", "n_sequences"), sorted(HMMLEARN_FAMILY_BAUM_WELCH))
def test_family_baum_welch_meets_hmmlearns_runtime(
    family: str, n_sequences: int
) -> None:
    inputs = _family_inputs(family, n_sequences)
    seconds = [package("family_baum_welch", inputs).seconds for _ in range(3)]
    assert_meets(
        float(np.median(seconds)),
        HMMLEARN_FAMILY_BAUM_WELCH[(family, n_sequences)],
    )


@pytest.mark.experiment
@pytest.mark.parametrize(
    ("family", "n_sequences"), sorted(HMMLEARN_FAMILY_BAUM_WELCH_MEMORY)
)
def test_family_baum_welch_fits_hmmlearns_memory(family: str, n_sequences: int) -> None:
    inputs = _family_inputs(family, n_sequences)
    peaks = [package("family_baum_welch", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(
        int(np.median(peaks)),
        HMMLEARN_FAMILY_BAUM_WELCH_MEMORY[(family, n_sequences)],
    )


def _viterbi_inputs(family: str, n_sequences: int) -> dict[str, np.ndarray]:
    """The harness's inputs for one decode, at the fits' start."""
    inputs = _family_inputs(family, n_sequences)
    del inputs["n_iter"]
    return inputs


@pytest.mark.experiment
@pytest.mark.parametrize(("family", "n_sequences"), sorted(HMMLEARN_VITERBI))
def test_viterbi_meets_hmmlearns_runtime(family: str, n_sequences: int) -> None:
    inputs = _viterbi_inputs(family, n_sequences)
    seconds = [package("viterbi", inputs).seconds for _ in range(3)]
    assert_meets(float(np.median(seconds)), HMMLEARN_VITERBI[(family, n_sequences)])


@pytest.mark.experiment
@pytest.mark.parametrize(("family", "n_sequences"), sorted(HMMLEARN_VITERBI_MEMORY))
def test_viterbi_fits_hmmlearns_memory(family: str, n_sequences: int) -> None:
    inputs = _viterbi_inputs(family, n_sequences)
    peaks = [package("viterbi", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), HMMLEARN_VITERBI_MEMORY[(family, n_sequences)])


@pytest.mark.experiment
@pytest.mark.parametrize(("family", "n_sequences"), sorted(HMMLEARN_SCORE))
def test_hmm_log_likelihood_meets_hmmlearns_runtime(
    family: str, n_sequences: int
) -> None:
    inputs = {**_viterbi_inputs(family, n_sequences), "score": np.asarray(True)}
    seconds = [package("viterbi", inputs).seconds for _ in range(3)]
    assert_meets(float(np.median(seconds)), HMMLEARN_SCORE[(family, n_sequences)])


@pytest.mark.experiment
@pytest.mark.parametrize(("family", "n_sequences"), sorted(HMMLEARN_SCORE_MEMORY))
def test_hmm_log_likelihood_fits_hmmlearns_memory(
    family: str, n_sequences: int
) -> None:
    inputs = {**_viterbi_inputs(family, n_sequences), "score": np.asarray(True)}
    peaks = [package("viterbi", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), HMMLEARN_SCORE_MEMORY[(family, n_sequences)])


def _score_inputs(n_samples: int) -> dict[str, np.ndarray]:
    """The mixture EM goal's start and draws, for a log-likelihood alone."""
    return {
        "observations": _mixture_draws(n_samples),
        "weights": np.array([0.3, 0.3, 0.4]),
        "mean": np.array([-3.0, 0.5, 4.0]),
        "scale": np.array([1.2, 1.0, 1.3]),
    }


@pytest.mark.experiment
@pytest.mark.parametrize("n_samples", sorted(SCIKIT_LEARN_SCORE))
def test_mixture_log_likelihood_meets_scikit_learns_runtime(n_samples: int) -> None:
    inputs = _score_inputs(n_samples)
    seconds = [package("mixture_score", inputs).seconds for _ in range(3)]
    assert_meets(float(np.median(seconds)), SCIKIT_LEARN_SCORE[n_samples])


@pytest.mark.experiment
@pytest.mark.parametrize("n_samples", sorted(SCIKIT_LEARN_SCORE_MEMORY))
def test_mixture_log_likelihood_fits_scikit_learns_memory(n_samples: int) -> None:
    inputs = _score_inputs(n_samples)
    peaks = [package("mixture_score", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), SCIKIT_LEARN_SCORE_MEMORY[n_samples])


#: BlackJAX's MALA, `blackjax.mala` at `epsilon = h^2 / 2` for the package's
#: Langevin step `h = 0.9 / (2 d^(1/4))`, 1,000 transitions on
#: `diagonal_precision(d)`: the second call's seconds and peak, the chain
#: stored, and the first call's peak with no draw kept, compilation and the
#: buffers XLA keeps included as `BLACKJAX_HMC_CHAIN_FREE_MEMORY`'s are
#: (#997). Stored, both sides hold the same `n x d` doubles, so that goal is
#: bounded near 1.0x as `BLACKJAX_HMC_MEMORY`'s is.
BLACKJAX_MALA = {
    dimension: Goal(
        "blackjax",
        f"1,000 MALA transitions at d = {dimension:,}",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.4, #997",
    )
    for dimension, seconds in ((100, 0.01097), (1_000, 0.04474), (10_000, 0.35298))
}

BLACKJAX_MALA_MEMORY = {
    dimension: MemoryGoal(
        "blackjax",
        f"1,000 MALA transitions at d = {dimension:,}, the chain stored",
        peak_bytes,
        "2026-09-23, 4-core reference host, #997",
    )
    for dimension, peak_bytes in (
        (100, 815_104),
        (1_000, 8_065_024),
        (10_000, 80_494_592),
    )
}

BLACKJAX_MALA_CHAIN_FREE_MEMORY = {
    dimension: MemoryGoal(
        "blackjax",
        f"1,000 MALA transitions at d = {dimension:,}, no draws kept",
        peak_bytes,
        "2026-09-23, 4-core reference host, first call, compilation included, #997",
    )
    for dimension, peak_bytes in (
        (100, 31_760_384),
        (1_000, 28_065_792),
        (10_000, 32_493_568),
    )
}


def _mala_inputs(dimension: int, store_chain: bool) -> dict[str, np.ndarray]:
    """The harness's inputs for the MALA goals."""
    return {
        "precision": diagonal_precision(dimension),
        "step_size": np.asarray(0.9 / (2.0 * dimension**0.25)),
        "n_draws": np.asarray(1_000),
        "seed": np.asarray(963),
        "store_chain": np.asarray(store_chain),
    }


@pytest.mark.experiment
@pytest.mark.parametrize("dimension", sorted(BLACKJAX_MALA))
def test_mala_meets_blackjaxs_runtime(dimension: int) -> None:
    inputs = _mala_inputs(dimension, store_chain=True)
    seconds = [package("mala_sample", inputs).seconds for _ in range(3)]
    assert_meets(float(np.median(seconds)), BLACKJAX_MALA[dimension])


@pytest.mark.experiment
@pytest.mark.parametrize("dimension", sorted(BLACKJAX_MALA_MEMORY))
def test_mala_fits_blackjaxs_memory(dimension: int) -> None:
    inputs = _mala_inputs(dimension, store_chain=True)
    peaks = [package("mala_sample", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), BLACKJAX_MALA_MEMORY[dimension])


@pytest.mark.experiment
@pytest.mark.parametrize("dimension", sorted(BLACKJAX_MALA_CHAIN_FREE_MEMORY))
def test_mala_without_its_chain_fits_blackjaxs_memory(dimension: int) -> None:
    inputs = _mala_inputs(dimension, store_chain=False)
    peaks = [package("mala_sample", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), BLACKJAX_MALA_CHAIN_FREE_MEMORY[dimension])


#: JAX's `jit(value_and_grad)` of `GaussianHmmObjective`'s negative
#: log-likelihood (three states, sequences of 100) at 20 points about its
#: `initial()`, the per-point median, and the loop's peak added memory,
#: medians of three subprocess runs (#997).
JAX_HMM_GRADIENT = {
    n_sequences: Goal(
        "jax",
        f"Gaussian HMM gradient at {100 * n_sequences:,} positions",
        seconds,
        "2026-09-23, 4-core reference host at a 1-minute load of 1.0, #997",
    )
    for n_sequences, seconds in ((100, 2.58e-3), (1_000, 16.61e-3))
}

JAX_HMM_GRADIENT_MEMORY = {
    n_sequences: MemoryGoal(
        "jax",
        f"Gaussian HMM gradient at {100 * n_sequences:,} positions",
        peak_bytes,
        "2026-09-23, 4-core reference host, #997",
    )
    for n_sequences, peak_bytes in ((100, 49_197_056), (1_000, 60_833_792))
}


def _hmm_gradient_inputs(n_sequences: int) -> dict[str, np.ndarray]:
    """The gradient harness's inputs: the family fits' Gaussian sequences, 20 points."""
    observations = _family_observations("gaussian", n_sequences)
    start = GaussianHmmObjective(observations, 3).initial().numpy()
    rng = np.random.default_rng(997)
    return {
        "points": start + 0.1 * rng.normal(size=(20, start.size)),
        "observations": observations,
        "n_states": np.asarray(3),
    }


@pytest.mark.experiment
@pytest.mark.parametrize("n_sequences", sorted(JAX_HMM_GRADIENT))
def test_the_hmm_gradient_meets_jaxs_runtime(n_sequences: int) -> None:
    inputs = _hmm_gradient_inputs(n_sequences)
    runs = [package("gradient", inputs) for _ in range(3)]
    per_point = float(np.median([float(r.outputs["per_point"]) for r in runs]))
    assert_meets(per_point, JAX_HMM_GRADIENT[n_sequences])


@pytest.mark.experiment
@pytest.mark.parametrize("n_sequences", sorted(JAX_HMM_GRADIENT_MEMORY))
def test_the_hmm_gradient_fits_jaxs_memory(n_sequences: int) -> None:
    inputs = _hmm_gradient_inputs(n_sequences)
    peaks = [package("gradient", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), JAX_HMM_GRADIENT_MEMORY[n_sequences])


#: BlackJAX's peak added resident memory for the same compiled chain with no
#: draw kept: its scan carries the state and emits only the acceptance, the
#: medians of three subprocess runs (#997). Kept, the draws were 1.0x
#: `BLACKJAX_HMC_MEMORY`, so like is compared with like only with both off.
BLACKJAX_HMC_CHAIN_FREE_MEMORY = {
    dimension: MemoryGoal(
        "blackjax",
        f"1,000 HMC transitions of ten leapfrog steps at d = {dimension:,}, no draws kept",
        peak_bytes,
        "2026-09-23, 4-core reference host, first call, compilation included, #997",
    )
    # BlackJAX's first call: its compilation and the buffers XLA keeps
    # after it are charged, as every allocation of ours is charged per call.
    for dimension, peak_bytes in (
        (100, 30_777_344),
        (1_000, 30_580_736),
        (10_000, 31_281_152),
    )
}


@pytest.mark.experiment
@pytest.mark.parametrize("dimension", sorted(BLACKJAX_HMC_CHAIN_FREE_MEMORY))
def test_hmc_without_its_chain_fits_blackjaxs_memory(dimension: int) -> None:
    # Issue #997: the chain with its draws switched off on both sides
    # (`store_chain=False` here, no operators; a scan emitting only the
    # acceptance in BlackJAX's script).
    inputs = {
        "precision": diagonal_precision(dimension),
        "step_size": np.asarray(0.9 / (2.0 * dimension**0.25)),
        "n_steps": np.asarray(10),
        "n_draws": np.asarray(1_000),
        "seed": np.asarray(963),
        "store_chain": np.asarray(False),
        "observe": np.asarray(False),
    }
    peaks = [package("hmc_sample", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), BLACKJAX_HMC_CHAIN_FREE_MEMORY[dimension])


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
@pytest.mark.parametrize("side", sorted(RUSTWORKX_COMPONENTS))
def test_the_union_find_meets_rustworkxs_runtime(side: int) -> None:
    # Timed in a fresh interpreter by `scripts/package.py`, as the pair was.
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, critical_coupling(3))
    rng = np.random.default_rng(976)
    state = rng.integers(0, 3, graph.n_nodes)
    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    like = state[first] == state[second]
    active = like & (rng.random(len(graph.edges)) < _bond_probability(graph, 1.0))
    bonds = graph.edge_index[active]
    inputs = {
        "n_nodes": np.asarray(graph.n_nodes),
        "first": np.ascontiguousarray(bonds[:, 0]),
        "second": np.ascontiguousarray(bonds[:, 1]),
    }
    seconds = [package("cluster_labels", inputs).seconds for _ in range(3)]
    assert_meets(float(np.median(seconds)), RUSTWORKX_COMPONENTS[side])


#: TorchRL's warm call on 10⁵ Potts decisions, `WEIGHTS` in
#: `tests/validation/_rl.py`: `ClipPPOLoss` at clip 0.2 in episodes of 100,
#: and `ReinforceLoss` on greedy episodes of 10 at baseline 0.5, each with its
#: gradient; the median of ten warm calls over two interpreters (#977). The
#: package's REINFORCE recomputes every decision's features through the
#: environment, which TorchRL is handed.
TORCHRL_LOSS = {
    "clip_ppo": Goal(
        "torchrl",
        "ppo_loss and its gradient on 10^5 Potts decisions",
        14.66e-3,
        "2026-09-23, 4-core reference host, #977",
    ),
    "reinforce": Goal(
        "torchrl",
        "surrogate_loss and its gradient on 10^5 Potts decisions",
        17.97e-3,
        "2026-09-23, 4-core reference host, #977",
    ),
}

#: PyG's `GINConv` twin's forward pass on one open lattice with 4 random
#: token features per node, hidden 8, two layers, tied weights; the median of
#: ten warm calls over two interpreters (#977).
TORCH_GEOMETRIC_FORWARD = {
    side: Goal(
        "torch_geometric",
        f"GraphSurrogate's forward on the {side}x{side} open lattice",
        seconds,
        "2026-09-23, 4-core reference host, #977",
    )
    for side, seconds in ((142, 5.845e-3), (284, 18.87e-3))
}


@pytest.mark.experiment
def test_ppo_loss_meets_torchrls_runtime() -> None:
    features, taken = potts_decisions(100_000)
    rng = np.random.default_rng(9770)
    old = torch.as_tensor(rng.normal(scale=0.1, size=100_000)) - 1.5
    advantages = torch.as_tensor(rng.normal(size=100_000))
    ppo_loss_and_gradient(features, taken, old, advantages, 100, 0.2)  # warm-up
    seconds = median_seconds(
        lambda: ppo_loss_and_gradient(features, taken, old, advantages, 100, 0.2)
    )
    assert_meets(seconds, TORCHRL_LOSS["clip_ppo"])


@pytest.mark.experiment
def test_reinforce_loss_meets_torchrls_runtime() -> None:
    environment = potts_environment()
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor(WEIGHTS, dtype=torch.float64))
    episodes = greedy_episodes(environment, policy.weights, 10_000, 10, 977)

    def loss_and_gradient() -> None:
        value = surrogate_loss(environment, policy, episodes, 0.5)
        torch.autograd.grad(value, policy.weights)

    assert_meets(
        median_seconds(loss_and_gradient, repeats=3), TORCHRL_LOSS["reinforce"]
    )


@pytest.mark.experiment
@pytest.mark.parametrize("side", sorted(TORCH_GEOMETRIC_FORWARD))
def test_the_graph_surrogate_meets_pygs_runtime(side: int) -> None:
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, 1.0)
    rng = np.random.default_rng(977)
    examples = Examples(
        features=torch.as_tensor(rng.normal(size=(1, 3))),
        targets=torch.zeros(1, dtype=torch.float64),
        groups=np.zeros(1, dtype=np.int64),
        tokens=(torch.as_tensor(rng.normal(size=(graph.n_nodes, 4))),),
        adjacency=(np.asarray(graph.edge_index, dtype=np.int64),),
    )
    batch = _Batch(examples)
    torch.manual_seed(977)
    model = GraphSurrogate(3, 4, hidden=8, n_layers=2)
    with torch.no_grad():
        model(batch)  # warm-up
        seconds = median_seconds(lambda: model(batch))
    assert_meets(seconds, TORCH_GEOMETRIC_FORWARD[side])


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


#: The random-walk targets (#1006): name, dimension, the harness's target
#: inputs, the start and the step. The Gaussian is `diagonal_precision(d)`
#: from the origin at 1.2 / sqrt(d); Rosenbrock's function at a = 1, b = 100
#: from every coordinate at -1.2 at 0.06 / sqrt(d).
RWM_TARGETS = {
    **{
        f"gaussian-{d}": (
            {"target": np.asarray(0), "precision": diagonal_precision(d)},
            np.zeros(d),
            1.2 / np.sqrt(d),
        )
        for d in (100, 1_000, 10_000)
    },
    **{
        f"rosenbrock-{d}": (
            {"target": np.asarray(1), "constants": np.asarray([1.0, 100.0])},
            np.full(d, -1.2),
            0.06 / np.sqrt(d),
        )
        for d in (10, 100)
    },
}

_RWM_MEASURED = "2026-09-24, 4-core reference host at a 1-minute load of 0.6-1.9, #1006"

#: BlackJAX's `additive_step_random_walk` with a normal step at the same
#: scale, compiled, compilation excluded, the medians of three subprocess
#: runs in `test_rwm_blackjax_bench.py`: 1,000 transitions, and 2,000 for
#: the warm-up row, which the package spends as 1,000 warm-up proposals and
#: 1,000 draws (BlackJAX has no random-walk warm-up, so its figure is the
#: same number of proposals at a fixed step).
BLACKJAX_RWM = {
    name: Goal(
        "blackjax", f"1,000 random-walk transitions, {name}", seconds, _RWM_MEASURED
    )
    for name, seconds in (
        ("gaussian-100", 0.00977),
        ("gaussian-1000", 0.03877),
        ("gaussian-10000", 0.30202),
        ("rosenbrock-10", 0.00559),
        ("rosenbrock-100", 0.00964),
    )
}

BLACKJAX_RWM_WARMUP = {
    name: Goal(
        "blackjax",
        f"1,000 warm-up proposals and 1,000 random-walk transitions, {name}",
        seconds,
        _RWM_MEASURED,
    )
    for name, seconds in (
        ("gaussian-100", 0.01846),
        ("gaussian-1000", 0.08108),
        ("gaussian-10000", 0.57001),
        ("rosenbrock-10", 0.01147),
        ("rosenbrock-100", 0.01894),
    )
}

#: Stored, both sides hold the same `n x d` doubles, so this goal is bounded
#: near 1.0x as `BLACKJAX_HMC_MEMORY`'s is.
BLACKJAX_RWM_MEMORY = {
    name: MemoryGoal(
        "blackjax",
        f"1,000 random-walk transitions, {name}, the chain stored",
        peak_bytes,
        _RWM_MEASURED,
    )
    for name, peak_bytes in (
        ("gaussian-100", 823_296),
        ("gaussian-1000", 8_052_736),
        ("gaussian-10000", 80_261_120),
        ("rosenbrock-10", 90_112),
        ("rosenbrock-100", 815_104),
    )
}

#: No draw kept: BlackJAX's first call, compilation and the buffers XLA keeps
#: included, as `BLACKJAX_HMC_CHAIN_FREE_MEMORY`'s are (#997).
BLACKJAX_RWM_CHAIN_FREE_MEMORY = {
    name: MemoryGoal(
        "blackjax",
        f"1,000 random-walk transitions, {name}, no draws kept",
        peak_bytes,
        f"{_RWM_MEASURED}, first call, compilation included",
    )
    for name, peak_bytes in (
        ("gaussian-100", 33_366_016),
        ("gaussian-1000", 31_137_792),
        ("gaussian-10000", 27_283_456),
        ("rosenbrock-10", 22_724_608),
        ("rosenbrock-100", 29_630_464),
    )
}

#: Seconds per effective sample, the slowest coordinate's: 20,000 transitions
#: on `gaussian-100`, 0.186 s for an effective size of 15.3 (#1006). A chain
#: faster per step that mixes worse has not met this.
BLACKJAX_RWM_PER_EFFECTIVE_SAMPLE = Goal(
    "blackjax",
    "seconds per effective sample, 20,000 random-walk transitions, gaussian-100",
    0.18597 / 15.2825,
    _RWM_MEASURED,
)


def _rwm_inputs(name: str, *, warmup: int, store_chain: bool) -> dict[str, np.ndarray]:
    """The harness's inputs for the random-walk goals."""
    target, position, step = RWM_TARGETS[name]
    return {
        **target,
        "position": position,
        "step_size": np.asarray(step),
        "n_draws": np.asarray(1_000),
        "seed": np.asarray(1006),
        "warmup": np.asarray(warmup),
        "store_chain": np.asarray(store_chain),
    }


@pytest.mark.experiment
@pytest.mark.parametrize("name", sorted(BLACKJAX_RWM))
def test_random_walk_meets_blackjaxs_runtime(name: str) -> None:
    inputs = _rwm_inputs(name, warmup=0, store_chain=True)
    seconds = [package("random_walk_sample", inputs).seconds for _ in range(3)]
    assert_meets(float(np.median(seconds)), BLACKJAX_RWM[name])


@pytest.mark.experiment
@pytest.mark.parametrize("name", sorted(BLACKJAX_RWM_WARMUP))
def test_random_walk_with_its_warm_up_meets_blackjaxs_runtime(name: str) -> None:
    inputs = _rwm_inputs(name, warmup=1_000, store_chain=True)
    seconds = [package("random_walk_sample", inputs).seconds for _ in range(3)]
    assert_meets(float(np.median(seconds)), BLACKJAX_RWM_WARMUP[name])


@pytest.mark.experiment
@pytest.mark.parametrize("name", sorted(BLACKJAX_RWM_MEMORY))
def test_random_walk_fits_blackjaxs_memory(name: str) -> None:
    inputs = _rwm_inputs(name, warmup=0, store_chain=True)
    peaks = [package("random_walk_sample", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), BLACKJAX_RWM_MEMORY[name])


@pytest.mark.experiment
@pytest.mark.parametrize("name", sorted(BLACKJAX_RWM_CHAIN_FREE_MEMORY))
def test_random_walk_without_its_chain_fits_blackjaxs_memory(name: str) -> None:
    inputs = _rwm_inputs(name, warmup=0, store_chain=False)
    peaks = [package("random_walk_sample", inputs).peak_bytes or 0 for _ in range(3)]
    assert_fits(int(np.median(peaks)), BLACKJAX_RWM_CHAIN_FREE_MEMORY[name])


@pytest.mark.experiment
def test_random_walk_meets_blackjaxs_seconds_per_effective_sample() -> None:
    target = GaussianTarget(diagonal_precision(100))

    def chain() -> hmc.Chain:
        return metropolis.random_walk(
            target, torch.Generator().manual_seed(1006), 20_000, step_size=0.12
        )

    seconds = median_seconds(chain, repeats=3)
    size = float(hmc.effective_sample_size(chain().draws).min())
    assert_meets(seconds / size, BLACKJAX_RWM_PER_EFFECTIVE_SAMPLE)
