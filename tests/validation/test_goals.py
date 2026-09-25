"""The runtimes the package has to meet, each set by an external framework (issue #972).

Each :class:`~tests.validation._goals.Goal` is a framework's runtime on a
declared fixture, measured on the 4-core reference host by the benchmark pair
its row names and written here as a number. The test times the package alone,
so no framework is installed to run it, and it fails until the package's
median meets the figure. It carries `goal` and runs in a non-blocking step of
CI's `validation` job. Each goal is one row of `RUNTIME_GOALS` or
`MEMORY_GOALS`, its id the framework, the call and the key (issue #982).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch
from sal import oxisal
from sal.backend import Backend
from sal.emissions import GaussianEmission
from sal.fixtures import load_params
from sal.learn.policy import LinearPolicy
from sal.learn.ppo import generalized_advantages
from sal.learn.reinforce import surrogate_loss
from sal.learn.surrogate import (
    Examples,
    GraphSurrogate,
    SetSurrogate,
    _Batch,
)
from sal.opt.hmm import GaussianHmmObjective, baum_welch
from sal.opt.mixture import expectation_maximization
from sal.sample import hmc, metropolis
from sal.sample.potts_mcmc.sweeps import bond_probability
from sal.search.alpha_expansion import alpha_expansion
from sal.search.ground_state import lattice_rung
from sal.search.potts_starts import spatio_rung, tiling_rung
from sal.search.trws import trws
from sal.sim.fixtures import fixture
from sal.sim.graph import BoundaryCondition, lattice_graph
from sal.sim.hmm import HmmParams, simulate_sequences
from sal.sim.potts import critical_coupling, site_field
from sal.validation.gaussian import GaussianTarget, diagonal_precision

from tests._fixtures import FIXTURES_DIR
from tests.regression.learn.conftest import potts_environment
from tests.validation._goals import (
    Goal,
    MemoryGoal,
    assert_fits,
    assert_meets,
    median_package,
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


#: rustworkx: `PyGraph` from one Swendsen--Wang sweep's bonds plus
#: `connected_components`, q = 3, open, critical; medians of three and peak
#: memory (#997). The package's figure is the whole compiled sweep.
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


#: TorchRL's `GAE`, `gamma = 1`, `lmbda = 0.95`, per 100-decision episode, every
#: other terminated; summed seconds, median of three (#997). TD(0) is `lmbda = 0`.
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


#: PyG's `SetSurrogate` twin (hidden 8) around `global_add_pool`, `side^2`
#: tokens, median of five runs of nine warm forwards, one thread (#997). Level
#: at 142^2 after two attempts (pooling before the last affine map: 6.2 to
#: 3.6 ms at 284^2; in-place body): one encoder over 20,164 rows either way.
TORCH_GEOMETRIC_SET = {
    side: Goal(
        "torch_geometric",
        f"SetSurrogate forward on {side}x{side} tokens",
        seconds,
        "2026-09-23, 4-core reference host, one thread, load 0.8, #997",
    )
    for side, seconds in ((142, 2.714e-3), (284, 12.576e-3))
}


#: JAX's `jit` gradient and peak memory over 100 points, medians of three
#: (#991): diagonal Gaussian at d = 10, 10^3, 10^4; mixture at n = 10^5. No
#: goal where torch was faster or lighter (dense Gaussian at 10^3).
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


#: HiGHS's `linprog` on the explicit local-polytope LP at 5,041 sites and ten
#: states, the largest fixtures both it and `search.trws` solve: one
#: subprocess run each, the solve alone (#1063). TRW-S reaches the LP value on
#: the first and stops 0.0496 below it on the second; the goal is the time.
HIGHS_LOCAL_POLYTOPE = {
    name: Goal(
        "highs",
        f"the local-polytope bound on {name}/release by TRW-S",
        seconds,
        "2026-09-25, 4-core reference host, one run at a 1-minute load of 5.3, #1063",
    )
    for name, seconds in (("spatio_only", 343.86), ("spatio_tiling", 267.62))
}


def _trws_seconds(name: str) -> float:
    """`trws` at its defaults on one release fixture, after a warm-up that compiles it."""
    params = fixture(name, "release").params
    rung = (
        spatio_rung(params, "release")
        if name == "spatio_only"
        else tiling_rung(params, "release")
    )
    trws(rung.graph, rung.field, max_iterations=1)
    return median_seconds(lambda: trws(rung.graph, rung.field), repeats=3)


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


#: gco's `swap()` on the expansion instance (q = 10, `critical_coupling(10)`,
#: seed 974), build excluded, peak with build, medians of three (#997). Ours
#: stood at 1.57x / 1.12x after two attempts (energy-change scoring, 0.92 to
#: 0.76 s at 142^2; cuts on moving sites, 0.63 s), 1.27x / 0.89x memory.
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
    return {**_expansion_inputs(side), "move": np.asarray("swap")}


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


def _hmm_start() -> tuple[np.ndarray, ...]:
    """The benchmark's start: each row a perturbed uniform, seed 975."""
    rng = np.random.default_rng(975)
    shapes = ((3,), (3, 3), (3, 4))
    draws = [rng.random(shape) + 0.5 for shape in shapes]
    return tuple(draw / draw.sum(axis=-1, keepdims=True) for draw in draws)


def _mixture_draws(n_samples: int) -> np.ndarray:
    """The benchmark's draws: components at -4, 0 and 5, deviations 1, 1.5, 1."""
    rng = np.random.default_rng(975)
    component = rng.choice(3, size=n_samples, p=[0.3, 0.3, 0.4])
    centre = np.array([-4.0, 0.0, 5.0])[component]
    spread = np.array([1.0, 1.5, 1.0])[component]
    return np.asarray(rng.normal(centre, spread))


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


def _viterbi_inputs(family: str, n_sequences: int) -> dict[str, np.ndarray]:
    """The harness's inputs for one decode, at the fits' start."""
    inputs = _family_inputs(family, n_sequences)
    del inputs["n_iter"]
    return inputs


def _score_inputs(n_samples: int) -> dict[str, np.ndarray]:
    """The mixture EM goal's start and draws, for a log-likelihood alone."""
    return {
        "observations": _mixture_draws(n_samples),
        "weights": np.array([0.3, 0.3, 0.4]),
        "mean": np.array([-3.0, 0.5, 4.0]),
        "scale": np.array([1.2, 1.0, 1.3]),
    }


#: `blackjax.mala` at `epsilon = h^2 / 2`, `h = 0.9 / (2 d^(1/4))`, 1,000
#: transitions on `diagonal_precision(d)`: second call's seconds and peak
#: stored, first call's peak unstored, XLA buffers included (#997). Stored,
#: both hold `n x d` doubles: near 1.0x, as `BLACKJAX_HMC_MEMORY`.
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


#: TorchRL on 10⁵ Potts decisions (`_rl.py` `WEIGHTS`): `ClipPPOLoss` at 0.2,
#: episodes of 100; `ReinforceLoss`, greedy episodes of 10, baseline 0.5; with
#: gradients, median of ten warm calls over two interpreters (#977). Ours
#: recomputes features through the environment.
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

#: BlackJAX's `additive_step_random_walk`, same scale, compiled, medians of
#: three (`test_rwm_blackjax_bench.py`): 1,000 transitions; 2,000 against our
#: 1,000 warm-up + 1,000 draws (BlackJAX has no warm-up).
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


#: HMC on declared targets (#1008), ten leapfrog steps per transition: name,
#: the harness's target inputs, the start and the step. Rosenbrock at a = 1,
#: b = 100 from every coordinate at -1.2 at 0.01 / sqrt(d / 10); the Gaussian
#: is `diagonal_precision(d)` from the origin at 0.9 / (2 d^(1/4)).
HMC_TARGETS = {
    **{
        f"rosenbrock-{d}": (
            {"target": np.asarray(1), "constants": np.asarray([1.0, 100.0])},
            np.full(d, -1.2),
            0.01 / np.sqrt(d / 10),
        )
        for d in (10, 100)
    },
    **{
        f"gaussian-{d}": (
            {"target": np.asarray(0), "precision": diagonal_precision(d)},
            np.zeros(d),
            0.9 / (2.0 * d**0.25),
        )
        for d in (100, 1_000, 10_000)
    },
}

_HMC_MEASURED = "2026-09-24, 4-core reference host at a 1-minute load of 1.3-1.6, #1008"

#: BlackJAX's compiled HMC, compilation excluded, the medians of three
#: subprocess runs in `test_hmc_declared_blackjax_bench.py`: 1,000
#: transitions on Rosenbrock's function.
BLACKJAX_HMC_ROSENBROCK = {
    name: Goal("blackjax", f"1,000 HMC transitions, {name}", seconds, _HMC_MEASURED)
    for name, seconds in (("rosenbrock-10", 0.0077), ("rosenbrock-100", 0.01326))
}

#: `blackjax.window_adaptation` over 500 steps at a 0.65 target, then 1,000
#: transitions at what it settled on, one compiled call; the package runs
#: `Adaptation(500, 0.65, 0.0)` and the same 1,000.
BLACKJAX_HMC_WARMUP = {
    name: Goal(
        "blackjax",
        f"500 warm-up steps and 1,000 HMC transitions, {name}",
        seconds,
        _HMC_MEASURED,
    )
    for name, seconds in (
        ("rosenbrock-10", 0.0108),
        ("rosenbrock-100", 0.01982),
        ("gaussian-100", 0.01967),
        ("gaussian-1000", 0.09326),
        ("gaussian-10000", 0.83678),
    )
}

#: Stored, both sides hold the same `n x d` doubles, bounded near 1.0x as
#: `BLACKJAX_HMC_MEMORY`'s is; with no draw kept, BlackJAX's first call.
BLACKJAX_HMC_ROSENBROCK_MEMORY = {
    name: MemoryGoal(
        "blackjax",
        f"1,000 HMC transitions, {name}, the chain stored",
        peak,
        _HMC_MEASURED,
    )
    for name, peak in (("rosenbrock-10", 94_208), ("rosenbrock-100", 823_296))
}

BLACKJAX_HMC_ROSENBROCK_CHAIN_FREE_MEMORY = {
    name: MemoryGoal(
        "blackjax",
        f"1,000 HMC transitions, {name}, no draws kept",
        peak,
        f"{_HMC_MEASURED}, first call, compilation included",
    )
    for name, peak in (("rosenbrock-10", 30_060_544), ("rosenbrock-100", 34_598_912))
}


def _hmc_inputs(name: str, *, warmup: int, store_chain: bool) -> dict[str, np.ndarray]:
    """The harness's inputs for the declared-target HMC goals."""
    target, position, step = HMC_TARGETS[name]
    return {
        **target,
        "position": position,
        "step_size": np.asarray(step),
        "n_steps": np.asarray(10),
        "n_draws": np.asarray(1_000),
        "seed": np.asarray(1008),
        "warmup": np.asarray(warmup),
        "target_acceptance": np.asarray(0.65),
        "store_chain": np.asarray(store_chain),
    }


#: HMC on a mixture's NLL (#1008): 10^5 draws, weights (0.3, 0.3, 0.4), means
#: (-4, 0, 5), scales (1, 1.5, 1), seed 1008, 100 x 10 steps at 0.001; JAX
#: twin in `scripts/blackjax.py`, ours `src/energy.rs`.
def _mixture_hmc_inputs(*, warmup: int, store_chain: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1008)
    labels = rng.choice(3, size=100_000, p=[0.3, 0.3, 0.4])
    values = np.array([-4.0, 0.0, 5.0])[labels] + np.array([1.0, 1.5, 1.0])[
        labels
    ] * rng.normal(size=labels.size)
    return {
        "target": np.asarray(2),
        "values": values,
        "n_components": np.asarray(3),
        "position": np.array(
            [0.0, np.log(4 / 3), -4.0, 0.0, 5.0, 0.0, np.log(1.5), 0.0]
        ),
        "step_size": np.asarray(0.001),
        "n_steps": np.asarray(10),
        "n_draws": np.asarray(100),
        "seed": np.asarray(1008),
        "warmup": np.asarray(warmup),
        "target_acceptance": np.asarray(0.65),
        "store_chain": np.asarray(store_chain),
    }


_MIXTURE_MEASURED = (
    "2026-09-24, 4-core reference host at a 1-minute load of 1.6-2.1, #1008"
)

#: 100 transitions, and 100 warm-up steps before them; the gradient is the
#: cost on both sides, and after two attempts the package's is 0.56-0.58x
#: (#1008: the start gradient carried, 0.70x -> 0.57x; a vectorizable exp,
#: slower on the SSE2 baseline and reverted).
BLACKJAX_HMC_MIXTURE = {
    "plain": Goal(
        "blackjax", "100 HMC transitions, mixture of 10^5", 2.1415, _MIXTURE_MEASURED
    ),
    "warm-up": Goal(
        "blackjax",
        "100 warm-up steps and 100 HMC transitions, mixture of 10^5",
        4.1774,
        _MIXTURE_MEASURED,
    ),
}

#: Second-call peaks on both sides: BlackJAX's holds the (n, k) joint.
BLACKJAX_HMC_MIXTURE_MEMORY = MemoryGoal(
    "blackjax", "100 HMC transitions, mixture of 10^5", 11_218_944, _MIXTURE_MEASURED
)


def _hmm_hmc_inputs(*, warmup: int, store_chain: bool) -> dict[str, np.ndarray]:
    """A two-state Gaussian HMM, 100 sequences of 100 (#1008): stay 0.95, means -1 and 1, scale 0.5, seed 1008."""
    rng = np.random.default_rng(1008)
    states = np.zeros((100, 100), dtype=int)
    states[:, 0] = rng.integers(0, 2, 100)
    for t in range(1, 100):
        stay = rng.random(100) < 0.95
        states[:, t] = np.where(stay, states[:, t - 1], 1 - states[:, t - 1])
    values = np.array([-1.0, 1.0])[states] + 0.5 * rng.normal(size=states.shape)
    return {
        "target": np.asarray(3),
        "values": values,
        "n_states": np.asarray(2),
        "position": np.array(
            [
                0.0,
                np.log(0.05 / 0.95),
                np.log(0.95 / 0.05),
                -1.0,
                1.0,
                np.log(0.5),
                np.log(0.5),
            ]
        ),
        "step_size": np.asarray(0.01),
        "n_steps": np.asarray(10),
        "n_draws": np.asarray(300),
        "seed": np.asarray(1008),
        "warmup": np.asarray(warmup),
        "target_acceptance": np.asarray(0.65),
        "store_chain": np.asarray(store_chain),
    }


_HMM_MEASURED = "2026-09-24, 4-core reference host at a 1-minute load of 1.1-1.4, #1008"

#: 300 transitions of ten steps after 100 warm-up; BlackJAX's `vmap`ped log
#: forward recursion, second call; ours `src/energy.rs`, Fisher's identity.
BLACKJAX_HMC_HMM = {
    "plain": Goal(
        "blackjax", "300 HMC transitions, Gaussian HMM of 10^4", 3.085, _HMM_MEASURED
    ),
    "warm-up": Goal(
        "blackjax",
        "100 warm-up steps and 300 HMC transitions, Gaussian HMM of 10^4",
        4.1372,
        _HMM_MEASURED,
    ),
}

BLACKJAX_HMC_HMM_MEMORY = MemoryGoal(
    "blackjax", "300 HMC transitions, Gaussian HMM of 10^4", 1_187_840, _HMM_MEASURED
)


# --- The package's figures -------------------------------------------------
#
# Runtime in process or via `scripts/package.py`; memory always fresh.


def _package(
    call: str, inputs: Callable[..., dict[str, np.ndarray]], read: str = "seconds"
) -> Callable[[Any], float]:
    """The median figure ``read`` of the package's ``call`` on one key's inputs."""
    return lambda key: median_package(
        call, inputs(*key) if isinstance(key, tuple) else inputs(key), read
    )


def _cut_seconds(side: int) -> float:
    # The Rust kernel with its arrays prebuilt, which is its graph build and
    # cut in one call, against PyMaxflow's build and cut.
    rung = lattice_rung(side, 2, seed=973)
    field = np.ascontiguousarray(
        site_field(rung.field, rung.graph.n_nodes, n_states=2), dtype=np.float64
    ).reshape(-1)
    edges = rung.graph.edge_index.reshape(-1)
    coupling = rung.graph.edge_coupling
    return median_seconds(
        lambda: oxisal.ising_ground_state(rung.graph.n_nodes, field, edges, coupling)
    )


def _bond_inputs(side: int) -> dict[str, np.ndarray]:
    """One Swendsen--Wang bond mask at beta = 1, seed 976, as the pair drew it."""
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, critical_coupling(3))
    rng = np.random.default_rng(976)
    state = rng.integers(0, 3, graph.n_nodes)
    first, second = graph.edge_index[:, 0], graph.edge_index[:, 1]
    like = state[first] == state[second]
    active = like & (rng.random(len(graph.edges)) < bond_probability(graph, 1.0))
    bonds = graph.edge_index[active]
    return {
        "n_nodes": np.asarray(graph.n_nodes),
        "first": np.ascontiguousarray(bonds[:, 0]),
        "second": np.ascontiguousarray(bonds[:, 1]),
    }


def _sweep_inputs(side: int) -> dict[str, np.ndarray]:
    return {"side": np.asarray(side), "seed": np.asarray(976)}


def _gae_seconds(n_episodes: int) -> float:
    rollouts = _gae_rollouts(n_episodes)

    def every() -> None:
        for rewards, values, terminated in rollouts:
            generalized_advantages(rewards, values, lam=0.95, terminated=terminated)

    return median_seconds(every, repeats=3)


def _ppo_seconds(_: object) -> float:
    features, taken = potts_decisions(100_000)
    rng = np.random.default_rng(9770)
    old = torch.as_tensor(rng.normal(scale=0.1, size=100_000)) - 1.5
    advantages = torch.as_tensor(rng.normal(size=100_000))
    ppo_loss_and_gradient(features, taken, old, advantages, 100, 0.2)  # warm-up
    return median_seconds(
        lambda: ppo_loss_and_gradient(features, taken, old, advantages, 100, 0.2)
    )


def _reinforce_seconds(_: object) -> float:
    environment = potts_environment()
    policy = LinearPolicy(2)
    policy.set_weights(torch.tensor(WEIGHTS, dtype=torch.float64))
    episodes = greedy_episodes(environment, policy.weights, 10_000, 10, 977)

    def loss_and_gradient() -> None:
        value = surrogate_loss(environment, policy, episodes, 0.5)
        torch.autograd.grad(value, policy.weights)

    return median_seconds(loss_and_gradient, repeats=3)


def _forward_seconds(model: Callable[[], torch.nn.Module], side: int) -> float:
    """A warm forward on one open lattice's tokens, 4 features per node, seed 977."""
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
    forward = model()
    with torch.no_grad():
        forward(batch)  # warm-up
        return median_seconds(lambda: forward(batch))


def _expansion_seconds(side: int) -> float:
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, critical_coupling(10))
    field = np.random.default_rng(974).normal(size=(graph.n_nodes, 10))
    return median_seconds(
        lambda: alpha_expansion(graph, field, 10, backend=Backend.RUST), repeats=3
    )


def _expansion_inputs(side: int) -> dict[str, np.ndarray]:
    return {
        "shape": np.asarray([side, side]),
        "coupling": np.asarray(critical_coupling(10)),
        "field": np.random.default_rng(974).normal(size=(side * side, 10)),
    }


def _hmm_params(n_sequences: int) -> HmmParams:
    return dataclasses.replace(
        load_params(FIXTURES_DIR / "hmm" / "ci.yaml", HmmParams),
        lengths=(100,) * n_sequences,
    )


def _baum_welch_seconds(n_sequences: int) -> float:
    observations = simulate_sequences(_hmm_params(n_sequences)).observations
    initial, transition, emission = (
        torch.log(torch.as_tensor(p)) for p in _hmm_start()
    )
    return median_seconds(
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


def _baum_welch_inputs(n_sequences: int) -> dict[str, np.ndarray]:
    initial, transition, emission = _hmm_start()
    return {
        "observations": simulate_sequences(_hmm_params(n_sequences)).observations,
        "initial": initial,
        "transition": transition,
        "emission": emission,
        "n_iter": np.asarray(10),
    }


def _mixture_em_seconds(n_samples: int) -> float:
    observations = _mixture_draws(n_samples)
    start = GaussianEmission(
        np.array([-3.0, 0.5, 4.0]), np.array([1.2, 1.0, 1.3]), 1e-12
    )
    weights = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    return median_seconds(
        lambda: expectation_maximization(
            observations, weights, start, max_iterations=10, tolerance=-np.inf
        ),
        repeats=3,
    )


def _log_likelihood_inputs(family: str, n_sequences: int) -> dict[str, np.ndarray]:
    return {**_viterbi_inputs(family, n_sequences), "score": np.asarray(True)}


def _mixture_em_inputs(n_samples: int) -> dict[str, np.ndarray]:
    return {**_score_inputs(n_samples), "n_iter": np.asarray(10)}


def _hmc_seconds(dimension: int) -> float:
    target = GaussianTarget(diagonal_precision(dimension))
    step = 0.9 / (2.0 * dimension**0.25)
    return median_seconds(
        lambda: hmc.sample(
            target,
            torch.Generator().manual_seed(963),
            1_000,
            step_size=step,
            n_steps=10,
        ),
        repeats=3,
    )


def _hmc_chain_inputs(dimension: int, **switches: bool) -> dict[str, np.ndarray]:
    """The compiled chain's inputs; issue #997 switches its draws off on both sides."""
    return {
        "precision": diagonal_precision(dimension),
        "step_size": np.asarray(0.9 / (2.0 * dimension**0.25)),
        "n_steps": np.asarray(10),
        "n_draws": np.asarray(1_000),
        "seed": np.asarray(963),
        **{name: np.asarray(value) for name, value in switches.items()},
    }


def _seconds_per_effective_sample(_: object) -> float:
    target = GaussianTarget(diagonal_precision(100))

    def chain() -> hmc.Chain:
        return metropolis.random_walk(
            target, np.random.default_rng(1006), 20_000, step_size=0.12
        )

    seconds = median_seconds(chain, repeats=3)
    size = float(hmc.effective_sample_size(chain().draws).min())
    return seconds / size


def _goals(
    goals: Mapping[Any, Goal | MemoryGoal], measure: Callable[[Any], float]
) -> list[Any]:
    """One row per goal, its id the framework and the fixture the goal names."""
    return [
        pytest.param(
            goals[key],
            functools.partial(measure, key),
            id=f"{goals[key].framework}: {goals[key].what}",
        )
        for key in sorted(goals)
    ]


def _by_label(
    inputs: Callable[..., dict[str, np.ndarray]],
) -> Callable[[str], dict[str, np.ndarray]]:
    """A declared-target chain, with 100 warm-up steps under the `warm-up` label."""
    return lambda label: inputs(
        warmup=100 if label == "warm-up" else 0, store_chain=True
    )


class _Call(NamedTuple):
    """Goals read off one call of `scripts/package.py`, in a fresh interpreter."""

    runtime: Mapping[Any, Goal]
    memory: Mapping[Any, MemoryGoal]
    call: str
    inputs: Callable[..., dict[str, np.ndarray]]
    #: The output a runtime goal reads: the call's seconds or a scalar it reports.
    read: str = "seconds"


_partial = functools.partial

#: Every goal read in a fresh interpreter, runtime and memory from one row.
PACKAGE_GOALS = (
    _Call({}, PYMAXFLOW_CUT_MEMORY, "ising_cut", _cut_inputs),
    _Call(RUSTWORKX_COMPONENTS, {}, "cluster_labels", _bond_inputs),
    _Call(RUSTWORKX_SWEEP, RUSTWORKX_SWEEP_MEMORY, "swendsen_wang", _sweep_inputs),
    _Call(JAX_GRADIENT, JAX_GRADIENT_MEMORY, "gradient", _gradient_inputs, "per_point"),
    _Call(
        JAX_HMM_GRADIENT,
        JAX_HMM_GRADIENT_MEMORY,
        "gradient",
        _hmm_gradient_inputs,
        "per_point",
    ),
    _Call({}, GCO_EXPANSION_MEMORY, "alpha_expansion", _expansion_inputs),
    _Call(GCO_SWAP, GCO_SWAP_MEMORY, "alpha_expansion", _swap_inputs),
    _Call({}, HMMLEARN_BAUM_WELCH_MEMORY, "baum_welch", _baum_welch_inputs),
    _Call(
        HMMLEARN_FAMILY_BAUM_WELCH,
        HMMLEARN_FAMILY_BAUM_WELCH_MEMORY,
        "family_baum_welch",
        _family_inputs,
    ),
    _Call(HMMLEARN_VITERBI, HMMLEARN_VITERBI_MEMORY, "viterbi", _viterbi_inputs),
    _Call(HMMLEARN_SCORE, HMMLEARN_SCORE_MEMORY, "viterbi", _log_likelihood_inputs),
    _Call({}, SCIKIT_LEARN_EM_MEMORY, "mixture_em", _mixture_em_inputs),
    _Call(
        SCIKIT_LEARN_SCORE, SCIKIT_LEARN_SCORE_MEMORY, "mixture_score", _score_inputs
    ),
    _Call({}, BLACKJAX_HMC_MEMORY, "hmc_sample", _hmc_chain_inputs),
    _Call(
        {},
        BLACKJAX_HMC_CHAIN_FREE_MEMORY,
        "hmc_sample",
        _partial(_hmc_chain_inputs, store_chain=False, observe=False),
    ),
    _Call(
        BLACKJAX_MALA,
        BLACKJAX_MALA_MEMORY,
        "mala_sample",
        _partial(_mala_inputs, store_chain=True),
    ),
    _Call(
        {},
        BLACKJAX_MALA_CHAIN_FREE_MEMORY,
        "mala_sample",
        _partial(_mala_inputs, store_chain=False),
    ),
    _Call(
        BLACKJAX_RWM,
        BLACKJAX_RWM_MEMORY,
        "random_walk_sample",
        _partial(_rwm_inputs, warmup=0, store_chain=True),
    ),
    _Call(
        BLACKJAX_RWM_WARMUP,
        {},
        "random_walk_sample",
        _partial(_rwm_inputs, warmup=1_000, store_chain=True),
    ),
    _Call(
        {},
        BLACKJAX_RWM_CHAIN_FREE_MEMORY,
        "random_walk_sample",
        _partial(_rwm_inputs, warmup=0, store_chain=False),
    ),
    _Call(
        BLACKJAX_HMC_ROSENBROCK,
        BLACKJAX_HMC_ROSENBROCK_MEMORY,
        "hmc_declared",
        _partial(_hmc_inputs, warmup=0, store_chain=True),
    ),
    _Call(
        {},
        BLACKJAX_HMC_ROSENBROCK_CHAIN_FREE_MEMORY,
        "hmc_declared",
        _partial(_hmc_inputs, warmup=0, store_chain=False),
    ),
    _Call(
        BLACKJAX_HMC_WARMUP,
        {},
        "hmc_declared",
        _partial(_hmc_inputs, warmup=500, store_chain=True),
    ),
    _Call(
        BLACKJAX_HMC_MIXTURE,
        {"plain": BLACKJAX_HMC_MIXTURE_MEMORY},
        "hmc_declared",
        _by_label(_mixture_hmc_inputs),
    ),
    _Call(
        BLACKJAX_HMC_HMM,
        {"plain": BLACKJAX_HMC_HMM_MEMORY},
        "hmc_declared",
        _by_label(_hmm_hmc_inputs),
    ),
)

#: Every runtime goal and how the package's seconds are read: in process
#: where the benchmark pair timed the package in process, else as its row
#: in `PACKAGE_GOALS` reads it.
RUNTIME_GOALS = [
    *_goals(PYMAXFLOW_CUT, _cut_seconds),
    *_goals(TORCHRL_GAE, _gae_seconds),
    *_goals({"clip_ppo": TORCHRL_LOSS["clip_ppo"]}, _ppo_seconds),
    *_goals({"reinforce": TORCHRL_LOSS["reinforce"]}, _reinforce_seconds),
    *_goals(
        TORCH_GEOMETRIC_SET,
        _partial(_forward_seconds, lambda: SetSurrogate(3, 4, hidden=8)),
    ),
    *_goals(
        TORCH_GEOMETRIC_FORWARD,
        _partial(_forward_seconds, lambda: GraphSurrogate(3, 4, hidden=8, n_layers=2)),
    ),
    *_goals(GCO_EXPANSION, _expansion_seconds),
    *_goals(HIGHS_LOCAL_POLYTOPE, _trws_seconds),
    *_goals(HMMLEARN_BAUM_WELCH, _baum_welch_seconds),
    *_goals(SCIKIT_LEARN_EM, _mixture_em_seconds),
    *_goals(BLACKJAX_HMC, _hmc_seconds),
    *_goals(
        {"gaussian-100": BLACKJAX_RWM_PER_EFFECTIVE_SAMPLE},
        _seconds_per_effective_sample,
    ),
    *(
        row
        for c in PACKAGE_GOALS
        for row in _goals(c.runtime, _package(c.call, c.inputs, c.read))
    ),
]

#: Every memory goal: the peak added bytes of its row's call.
MEMORY_GOALS = [
    row
    for c in PACKAGE_GOALS
    for row in _goals(c.memory, _package(c.call, c.inputs, "peak_bytes"))
]


@pytest.mark.experiment
@pytest.mark.parametrize(("goal", "measure"), RUNTIME_GOALS)
def test_the_package_meets_each_runtime_goal(
    goal: Goal, measure: Callable[[], float]
) -> None:
    assert_meets(measure(), goal)


@pytest.mark.experiment
@pytest.mark.parametrize(("goal", "measure"), MEMORY_GOALS)
def test_the_package_fits_each_memory_goal(
    goal: MemoryGoal, measure: Callable[[], float]
) -> None:
    assert_fits(int(measure()), goal)
