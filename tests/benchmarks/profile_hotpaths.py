"""Self-time profiling over every module's representative workload.

Ranks each module's entry points by ``cProfile`` self time, at the enumerable
tier the suite pins and the mid-size tier `STATUS.md` reports (root
`CLAUDE.md`'s "Measurement" rule: rank on realistic sizes, not the smallest
that fits a CI budget). Written for issue #181, extended for #287 and again
for #341 -- the repository-wide runtime-optimization audit -- and left here
as the reusable first step of `DEV.md`'s "Profiling a Hot Path".

Not a ``test_*`` module: a self-time ranking is not a pass/fail scientific
assertion (root `CLAUDE.md`'s "No Coverage Theatre" rule), so it is not
pytest-collected, and prints its report instead. Run by hand, on fixed
hardware, per `DEV.md`'s "No CI Profiling" rule::

    python tests/benchmarks/profile_hotpaths.py --tier enumerable
    python tests/benchmarks/profile_hotpaths.py --tier mid --module likelihood

Each section prints the top five functions by self time with the fraction of
the run each carries; a loop under 10% of its run is recorded and not ported
(#341). The ``ROADMAP.md`` tier (1,000 taxa by 10,000 sites) is not a tier
here: `STATUS.md` records it as not measured until the memory footprint
helper of #232 exists.

``profile_harness`` is imported the same way
``tests/regression/test_select_tests.py`` imports ``select_tests``: insert
``infra/`` onto ``sys.path`` and import it by its bare module name, since
`infra/CLAUDE.md` keeps that directory a flat set of scripts with no
application reference, and this module is the one on the other side of that
boundary that may hold one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra"))

from profile_harness import format_table, self_time_table
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsEnvironment
from snakes_and_ladders.learn.ranking import fixed_length_target, tree_examples
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.learn.surrogate import MLPSurrogate, fit_surrogate
from snakes_and_ladders.likelihood import pruning, pruning_rust, pruning_torch
from snakes_and_ladders.likelihood.belief_propagation import belief_propagation
from snakes_and_ladders.likelihood.convolutional import bcjr
from snakes_and_ladders.likelihood.ldpc import DecodingAlgorithm, decode
from snakes_and_ladders.likelihood.message_passing import (
    MessageScheduleName,
    sum_product,
)
from snakes_and_ladders.likelihood.objective import BranchLengthObjective
from snakes_and_ladders.likelihood.parsimony import fitch_score
from snakes_and_ladders.likelihood.turbo import decode_turbo, noise_scale, split_streams
from snakes_and_ladders.numerics import sample_rows
from snakes_and_ladders.opt.budget import Budget, Outcome, compare
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.opt.hmm import forward_log_likelihood_from_density
from snakes_and_ladders.opt.potts import PottsObjective
from snakes_and_ladders.sample import hmc
from snakes_and_ladders.sample.gibbs import sample_factor_graph
from snakes_and_ladders.sample.potts_keyed import SwendsenWangMove
from snakes_and_ladders.sample.potts_mcmc import PottsMove, sample_potts
from snakes_and_ladders.search.alpha_expansion import alpha_expansion
from snakes_and_ladders.search.infer import infer
from snakes_and_ladders.search.maxflow import ising_ground_state
from snakes_and_ladders.sim import fixtures
from snakes_and_ladders.sim.convolutional import turbo_code
from snakes_and_ladders.sim.factor_graph import from_hmm, from_potts
from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from snakes_and_ladders.sim.ldpc import (
    BinarySymmetricChannel,
    all_zero_transmission,
    gallager_code,
)
from snakes_and_ladders.sim.potts import critical_coupling, energies
from snakes_and_ladders.sim.potts_chain import PottsParams, simulate_chains
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.topology import (
    MoveSet,
    Topology,
    enumerate_topologies,
    nni_neighbours,
    spr_neighbours,
)
from snakes_and_ladders.sim.tree import Node

FIELD = np.array([0.3, -0.7, 0.15])
Section = tuple[str, Callable[[], object], int]


def _caterpillar(n_taxa: int) -> Topology:
    """A caterpillar topology on ``n_taxa`` leaves, branch lengths unset."""
    leaves = [Node(name=f"t{i}", branch_length=None) for i in range(n_taxa)]
    tail: Node = leaves[-1]
    for leaf in reversed(leaves[2:-1]):
        tail = Node(name="i", branch_length=None, children=(leaf, tail))
    return Node(name="root", branch_length=None, children=(leaves[0], leaves[1], tail))


def _branched(n_taxa: int, seed: int) -> Node:
    """``_caterpillar`` with seeded, strictly positive branch lengths.

    Every node below the root gets a length and a unique name; the root
    itself keeps ``branch_length=None``, per ``Node``'s convention.
    """
    rng = np.random.default_rng(seed)
    counter = iter(range(10_000))

    def _assign(node: Node, *, is_root: bool) -> Node:
        children = tuple(_assign(child, is_root=False) for child in node.children)
        branch_length = None if is_root else float(rng.uniform(0.05, 0.3))
        name = node.name if node.is_leaf or is_root else f"i{next(counter)}"
        return Node(name=name, branch_length=branch_length, children=children)

    return _assign(_caterpillar(n_taxa), is_root=True)


def _alignment(n_taxa: int, n_sites: int) -> tuple[Node, dict[str, np.ndarray]]:
    tau = _branched(n_taxa, seed=0)
    dataset = simulate_alignment(
        tau, k=4, pi=np.full(4, 0.25), rng=np.random.default_rng(1), n_sites=n_sites
    )
    return tau, dict(dataset.alignment)


def _chain(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    log_initial = np.log(rng.dirichlet(np.ones(4)))
    log_transition = np.log(rng.dirichlet(np.ones(4), size=4))
    return log_initial, log_transition, rng.normal(size=(length, 4))


def _lattice(extent: int, coupling: float) -> PottsGraph:
    return lattice_graph((extent, extent), BoundaryCondition.OPEN, coupling)


# --- sim ------------------------------------------------------------------------


def sim_sections(mid: bool) -> list[Section]:
    n_taxa, n_sites, extent = (20, 2_000, 32) if mid else (6, 200, 8)
    tau = _branched(n_taxa, seed=0)
    pi = np.full(4, 0.25)
    graph = _lattice(extent, 0.3)
    transition = np.random.default_rng(0).dirichlet(np.ones(4), size=4)
    rows = np.random.default_rng(1).integers(0, 4, size=n_sites * n_taxa)

    def _simulate() -> None:
        simulate_alignment(
            tau, k=4, pi=pi, rng=np.random.default_rng(1), n_sites=n_sites
        )

    def _lattices() -> None:
        lattice_graph((extent, extent), BoundaryCondition.OPEN, 0.3)
        lattice_graph((extent, extent), BoundaryCondition.PERIODIC, 0.3)

    def _adapters() -> None:
        from_potts(graph, FIELD)
        from_hmm(*_chain(n_sites))

    def _sampler() -> None:
        sample_rows(np.random.default_rng(2), transition, rows)

    return [
        (f"sim.simulate_alignment @ {n_taxa} taxa x {n_sites} sites", _simulate, 5),
        (f"sim.lattice_graph @ {extent}x{extent}, open and periodic", _lattices, 5),
        (
            f"sim.factor_graph.from_potts {extent}x{extent} + from_hmm {n_sites}",
            _adapters,
            5,
        ),
        (f"sim Rust sample_rows @ {n_sites * n_taxa} draws", _sampler, 20),
    ]


# --- likelihood -----------------------------------------------------------------


def likelihood_sections(mid: bool) -> list[Section]:
    n_taxa, n_sites, extent, length = (20, 2_000, 8, 200) if mid else (4, 200, 3, 20)
    tau, alignment = _alignment(n_taxa, n_sites)
    pi = np.full(4, 0.25)
    lengths = pruning_torch.branch_lengths_from_tree(tau).requires_grad_(True)
    graph = _lattice(extent, 0.3)
    chain = _chain(length)
    hmm_graph = from_hmm(*chain)
    potts_graph = from_potts(graph, FIELD)
    tensors = (
        torch.as_tensor(chain[2])[None],
        torch.as_tensor(chain[0]),
        torch.as_tensor(chain[1]),
    )

    def _numpy() -> None:
        pruning.log_likelihood(tau, 4, pi, alignment)

    def _torch_with_gradient() -> None:
        value = pruning_torch.log_likelihood(tau, 4, pi, alignment, lengths)
        torch.autograd.grad(value, lengths)

    def _rust() -> None:
        pruning_rust.log_likelihood(tau, 4, pi, alignment)

    def _flooding() -> None:
        sum_product(potts_graph, schedule=MessageScheduleName.FLOODING)

    def _bp() -> None:
        belief_propagation(graph, FIELD)

    def _tree_schedule() -> None:
        sum_product(hmm_graph)

    def _forward() -> None:
        forward_log_likelihood_from_density(*tensors)

    def _fitch() -> None:
        fitch_score(tau, alignment, 4)

    return [
        (f"likelihood.pruning NumPy @ {n_taxa} taxa x {n_sites}", _numpy, 5),
        (
            f"likelihood.pruning_torch + backward @ {n_taxa} x {n_sites}",
            _torch_with_gradient,
            5,
        ),
        (f"likelihood.pruning_rust @ {n_taxa} x {n_sites}", _rust, 5),
        (f"likelihood.message_passing flooding @ {extent}x{extent}", _flooding, 1),
        (f"likelihood.belief_propagation @ {extent}x{extent}", _bp, 1),
        (
            f"likelihood.message_passing tree schedule @ chain {length}",
            _tree_schedule,
            5,
        ),
        (f"likelihood forward recursion @ chain {length}", _forward, 5),
        (f"likelihood.parsimony.fitch_score @ {n_taxa} x {n_sites}", _fitch, 20),
    ]


# --- opt ------------------------------------------------------------------------


def opt_sections(mid: bool) -> list[Section]:
    n_taxa, chain_length, draws = (20, 64, 1_000) if mid else (6, 4, 200)
    tau, alignment = _alignment(n_taxa, 500)
    tree_objective = BranchLengthObjective(tau, 4, np.full(4, 0.25), alignment)
    params = PottsParams(
        n_states=3,
        chain_length=chain_length,
        n_chains=200,
        coupling=0.6,
        field=np.array([0.2, -0.1, -0.1]),
        seed=0,
    )
    potts_objective = PottsObjective(simulate_chains(params), 3)

    def _tree_fit() -> None:
        fit(tree_objective, max_iterations=50)

    def _potts_fit() -> None:
        fit(potts_objective, max_iterations=50)

    def _hmc() -> None:
        hmc.sample(
            potts_objective,
            torch.Generator().manual_seed(0),
            draws,
            step_size=0.05,
            n_steps=5,
        )

    def _method(_instance: int, budget: Budget, rng: np.random.Generator) -> Outcome:
        return Outcome(float(rng.normal()), budget.size)

    def _budget() -> None:
        compare(
            {"a": _method, "b": _method},
            [0, 1],
            Budget("sweeps", 10),
            range(40),
            workers=1,
        )

    return [
        (f"opt.fit L-BFGS on the tree @ {n_taxa} taxa x 500 sites", _tree_fit, 1),
        (f"opt.fit L-BFGS on the Potts chain @ length {chain_length}", _potts_fit, 1),
        (f"sample.hmc.sample @ {draws} draws on the Potts chain", _hmc, 1),
        ("opt.budget.compare @ 2 methods x 2 instances x 40 seeds", _budget, 20),
    ]


# --- search ---------------------------------------------------------------------


def search_sections(mid: bool) -> list[Section]:
    n_taxa, extent, sweeps = (20, 32, 20) if mid else (8, 16, 100)
    _, alignment = _alignment(n_taxa, 1_000)
    topology = _caterpillar(n_taxa)
    graph = _lattice(extent, 0.6)
    ising_field = np.random.default_rng(extent).normal(size=(graph.n_nodes, 2))
    potts_field = np.random.default_rng(1).normal(size=(graph.n_nodes, 3))
    factor_graph = from_potts(graph, FIELD)
    configurations = np.random.default_rng(3).integers(0, 2, size=(64, graph.n_nodes))
    # The cluster pass's own instance: three states at the exact transition,
    # where the bond pass makes a cluster for every 1.7 sites, which is the
    # loop issue #754 ported.
    cluster_graph = lattice_graph(
        (extent, extent), BoundaryCondition.OPEN, critical_coupling(3)
    )
    cluster_move = SwendsenWangMove(cluster_graph, potts_field)
    cluster_state = np.ascontiguousarray(
        np.random.default_rng(4).integers(0, 3, size=cluster_graph.n_nodes),
        dtype=np.int64,
    )

    def _neighbourhoods() -> None:
        list(nni_neighbours(topology))
        list(spr_neighbours(topology))

    def _nni() -> None:
        infer(
            alignment,
            4,
            rng=np.random.default_rng(4),
            moves=MoveSet.NNI,
            max_evaluations=20,
        )

    def _spr() -> None:
        infer(
            alignment,
            4,
            rng=np.random.default_rng(4),
            moves=MoveSet.SPR,
            max_evaluations=20,
        )

    def _ground_state() -> None:
        ising_ground_state(graph, ising_field)

    def _energy() -> None:
        energies(graph, ising_field, configurations)

    def _expansion() -> None:
        alpha_expansion(graph, potts_field, 3)

    def _single_site() -> None:
        # The oracle sweep explicitly, though it is no longer the default:
        # this ranks *Python* self time to decide what to port, and the
        # default now spends the sweep in Rust (issue #599).
        sample_potts(
            graph,
            FIELD,
            PottsMove.SINGLE_SITE,
            np.random.default_rng(0),
            sweeps,
            backend=Backend.PYTHON,
        )

    def _wolff() -> None:
        sample_potts(graph, FIELD, PottsMove.WOLFF, np.random.default_rng(0), sweeps)

    def _swendsen_wang() -> None:
        # The oracle pass explicitly, as the single-site cell above: this
        # ranks *Python* self time, and the Rust cluster pass is opt-in
        # because it draws the same uniforms in another order (issue #754).
        cluster_move.propose(
            cluster_state,
            temperature=1.0,
            site=-1,
            label=-1,
            rng=np.random.default_rng(7),
        )

    def _gibbs() -> None:
        sample_factor_graph(factor_graph, np.random.default_rng(0), sweeps)

    return [
        (f"search NNI+SPR neighbourhoods @ n={n_taxa}", _neighbourhoods, 5),
        (f"search.infer NNI @ {n_taxa} taxa, 20 evaluations", _nni, 1),
        (f"search.infer SPR @ {n_taxa} taxa, 20 evaluations", _spr, 1),
        (f"search.maxflow.ising_ground_state @ {extent}x{extent}", _ground_state, 1),
        (f"sim.potts.energies @ {extent}x{extent}, 64 configurations", _energy, 20),
        (f"search.alpha_expansion @ {extent}x{extent}, 3 labels", _expansion, 1),
        (
            f"sample.potts_mcmc single-site @ {extent}x{extent}, {sweeps} sweeps",
            _single_site,
            1,
        ),
        (f"sample.potts_mcmc Wolff @ {extent}x{extent}, {sweeps} sweeps", _wolff, 1),
        (
            f"sample.potts_keyed SwendsenWangMove.propose @ {extent}x{extent}",
            _swendsen_wang,
            5,
        ),
        (
            f"sample.gibbs.sample_factor_graph @ {extent}x{extent}, {sweeps} sweeps",
            _gibbs,
            1,
        ),
    ]


# --- codes ---------------------------------------------------------------------


def codes_sections(mid: bool) -> list[Section]:
    """The decoders, at the two block lengths the fixtures declare.

    Added for issue #754: the codes were the one family with no workload
    here, so nothing ranked them. Both instances are read from the fixture
    store rather than restated (issue #622). Which file each tier reads is a
    statement about cost, and the fixture tiers are not: `ldpc/stress.yaml`
    is 96 bits because dense encoding is cubic and refuses past 512, and
    `ldpc/release.yaml` is 996 because that is where the thresholds turn. So
    the enumerable tier here reads `stress` and the mid tier reads
    `release`, which orders the two by the work they do.
    """
    ldpc = fixtures.fixture("ldpc", "release" if mid else "stress").params
    turbo = fixtures.fixture("turbo", "release" if mid else "stress").params
    code = gallager_code(
        ldpc.n_bits,
        ldpc.column_weight,
        ldpc.row_weight,
        np.random.default_rng(ldpc.seed),
    )
    llr = all_zero_transmission(
        code,
        BinarySymmetricChannel(ldpc.flip_probability),
        np.random.default_rng(0),
    )
    trellis_code = turbo_code(
        turbo.feedback,
        turbo.feedforward,
        turbo.memory,
        turbo.message_length,
        np.random.default_rng(turbo.seed),
    )
    sigma = noise_scale(1.0, trellis_code.rate)
    received = 1.0 + sigma * np.random.default_rng(9).standard_normal(
        trellis_code.block_length
    )
    channel_llr = np.asarray(2.0 * received / sigma**2)
    streams = split_streams(trellis_code, channel_llr)

    def _sum_product() -> None:
        decode(
            code,
            llr,
            algorithm=DecodingAlgorithm.SUM_PRODUCT,
            max_iterations=50,
            early_stop=False,
        )

    def _min_sum() -> None:
        decode(
            code,
            llr,
            algorithm=DecodingAlgorithm.MIN_SUM,
            max_iterations=50,
            early_stop=False,
        )

    def _bcjr() -> None:
        bcjr(trellis_code.trellis, streams.systematic, streams.parity_first)

    def _turbo() -> None:
        decode_turbo(trellis_code, channel_llr, iterations=turbo.iterations)

    return [
        (
            f"codes.ldpc sum-product @ {ldpc.n_bits} bits, 50 iterations",
            _sum_product,
            1,
        ),
        (f"codes.ldpc min-sum @ {ldpc.n_bits} bits, 50 iterations", _min_sum, 1),
        (f"codes.convolutional.bcjr @ K={turbo.message_length}", _bcjr, 5),
        (
            f"codes.turbo.decode_turbo @ K={turbo.message_length}, "
            f"{turbo.iterations} iterations",
            _turbo,
            1,
        ),
    ]


# --- learn ----------------------------------------------------------------------


def learn_sections(mid: bool) -> list[Section]:
    chain_length, iterations = (8, 60) if mid else (4, 10)
    environment = PottsEnvironment(
        0.75, np.array([0.4, -0.1, -0.3]), chain_length=chain_length
    )
    _, alignment = _alignment(5, 1_000)
    topologies = list(enumerate_topologies(sorted(alignment)))
    pi = np.full(4, 0.25)

    def _reinforce() -> None:
        reinforce(
            environment,
            LinearPolicy(2),
            np.random.default_rng(0),
            iterations=iterations,
            batch=32,
            max_steps=6,
        )

    def _surrogate() -> None:
        examples = tree_examples(
            [alignment], [topologies], 4, pi, fixed_length_target(4, pi, 0.15)
        )
        fit_surrogate(
            MLPSurrogate(examples.features.shape[1]),
            examples,
            examples,
            generator=torch.Generator().manual_seed(0),
            max_epochs=20,
        )

    # PPO and PUCT are not built (ROADMAP.md, Milestone 2.1), so nothing is
    # profiled for them; STATUS.md records the gap.
    return [
        (
            f"learn.reinforce @ {iterations} x 32 episodes, chain {chain_length}",
            _reinforce,
            1,
        ),
        ("learn.fit_surrogate on the 5-taxon neighbourhoods, 20 epochs", _surrogate, 1),
    ]


MODULES: dict[str, Callable[[bool], list[Section]]] = {
    "sim": sim_sections,
    "likelihood": likelihood_sections,
    "opt": opt_sections,
    "search": search_sections,
    "codes": codes_sections,
    "learn": learn_sections,
}


def main(argv: list[str] | None = None) -> int:
    """Print the self-time ranking for each module at one tier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["enumerable", "mid"], default="enumerable")
    parser.add_argument("--module", choices=sorted(MODULES), action="append")
    parser.add_argument("--top", type=int, default=5)
    arguments = parser.parse_args(argv)
    mid = arguments.tier == "mid"
    for name in arguments.module or sorted(MODULES):
        for title, fn, repeats in MODULES[name](mid):
            rows, total = self_time_table(fn, repeats=repeats, top_n=arguments.top)
            print(f"=== [{arguments.tier}] {title} (x{repeats}) ===")
            print(format_table(rows, total))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
