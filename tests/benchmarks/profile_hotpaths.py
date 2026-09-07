"""Self-time profiling over ``snakes_and_ladders.sim``, ``snakes_and_ladders.search`` and ``snakes_and_ladders.learn``.

Ranks each module's hot entry points by ``cProfile`` self time, at a
CI-sized fixture and one larger, non-CI size (root `CLAUDE.md`'s
"Measurement" rule: rank on realistic sizes, not the smallest that fits a
CI budget). Written for issue #181 -- "identify opportunities to build out
the rust backend" -- and left here as a reusable tool for the next audit
rather than a one-off.

Not a ``test_*`` module: a self-time ranking is not a pass/fail scientific
assertion (root `CLAUDE.md`'s "No Coverage Theatre" rule), so it is not
pytest-collected, and prints its report instead. Run by hand, on fixed
hardware, per `DEV.md`'s "No CI Profiling" rule::

    python tests/benchmarks/profile_hotpaths.py

``profile_harness.self_time_ranking`` is imported the same way
``tests/regression/test_select_tests.py`` imports ``select_tests``: insert
``infra/`` onto ``sys.path`` and import it by its bare module name, since
`infra/CLAUDE.md` keeps that directory a flat set of scripts with no
application reference, and this module is the one on the other side of that
boundary that may hold one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra"))

from profile_harness import self_time_ranking
from snakes_and_ladders.learn.policy import LinearPolicy
from snakes_and_ladders.learn.potts import PottsLandscape
from snakes_and_ladders.learn.reinforce import reinforce
from snakes_and_ladders.search.infer import MoveSet, infer
from snakes_and_ladders.search.topology import Topology, nni_neighbours, spr_neighbours
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node


def _caterpillar(n_taxa: int) -> Topology:
    """A caterpillar topology on ``n_taxa`` leaves, branch lengths unset."""
    leaves = [Node(name=f"t{i}", branch_length=None) for i in range(n_taxa)]
    tail: Node = leaves[-1]
    for leaf in reversed(leaves[2:-1]):
        tail = Node(name="i", branch_length=None, children=(leaf, tail))
    return Node(name="root", branch_length=None, children=(leaves[0], leaves[1], tail))


def _branched(n_taxa: int, seed: int) -> Node:
    """``_caterpillar`` with seeded, strictly positive branch lengths.

    Every node below the root gets a length; the root itself keeps
    ``branch_length=None``, per ``Node``'s convention.
    """
    rng = np.random.default_rng(seed)

    def _assign(node: Node, *, is_root: bool) -> Node:
        children = tuple(_assign(child, is_root=False) for child in node.children)
        branch_length = None if is_root else float(rng.uniform(0.05, 0.3))
        return Node(name=node.name, branch_length=branch_length, children=children)

    return _assign(_caterpillar(n_taxa), is_root=True)


def profile_sim(n_sites: int) -> str:
    """Self time inside ``simulate_alignment`` at ``n_sites``, 8 taxa, k=4."""
    tau = _branched(8, seed=0)
    pi = np.full(4, 0.25)

    def _run() -> None:
        simulate_alignment(tau, k=4, pi=pi, seed=1, n_sites=n_sites)

    return self_time_ranking(_run, repeats=5)


def profile_search_neighbourhoods(n_taxa: int) -> str:
    """Self time generating the NNI and SPR neighbourhoods at ``n_taxa``.

    Capped at `search/CLAUDE.md`'s topological-test size (``n <= 10`` in
    CI); the caller may pass a larger, non-CI size for the audit's second
    problem size.
    """
    topology = _caterpillar(n_taxa)

    def _run() -> None:
        list(nni_neighbours(topology))
        list(spr_neighbours(topology))

    return self_time_ranking(_run, repeats=20)


def profile_search_hill_climb(n_sites: int) -> str:
    """Self time inside one full NNI hill-climb, the realistic search unit.

    `search/CLAUDE.md` measures neighbourhood generation against the
    candidate fit it feeds, not in isolation -- one fit dwarfs an entire
    NNI neighbourhood. This profiles the whole loop so that ranking holds.
    """
    tau = _branched(6, seed=2)
    pi = np.full(4, 0.25)
    dataset = simulate_alignment(tau, k=4, pi=pi, seed=3, n_sites=n_sites)

    def _run() -> None:
        infer(dataset.alignment, k=4, seed=4, moves=MoveSet.NNI, max_evaluations=20)

    return self_time_ranking(_run, repeats=1)


def profile_learn_reinforce(iterations: int) -> str:
    """Self time inside ``reinforce`` over ``iterations`` gradient updates."""
    landscape = PottsLandscape(0.75, np.array([0.4, -0.1, -0.3]), chain_length=4)

    def _run() -> None:
        reinforce(
            landscape,
            LinearPolicy(2),
            np.random.default_rng(0),
            iterations=iterations,
            batch=32,
            max_steps=6,
        )

    return self_time_ranking(_run, repeats=1)


def main() -> None:
    """Print the self-time ranking for each module, at two problem sizes."""
    sections = [
        ("sim.simulate_alignment @ 200_000 sites (CI-sized)", profile_sim(200_000)),
        ("sim.simulate_alignment @ 2_000_000 sites (large)", profile_sim(2_000_000)),
        (
            "search NNI+SPR neighbourhoods @ n=10 (CI cap)",
            profile_search_neighbourhoods(10),
        ),
        (
            "search NNI+SPR neighbourhoods @ n=30 (large)",
            profile_search_neighbourhoods(30),
        ),
        (
            "search hill_climb @ 20 evaluations, 2000 sites",
            profile_search_hill_climb(2000),
        ),
        ("learn.reinforce @ 5 gradient updates", profile_learn_reinforce(5)),
        ("learn.reinforce @ 50 gradient updates", profile_learn_reinforce(50)),
    ]
    for title, report in sections:
        print(f"=== {title} ===")
        print(report)


if __name__ == "__main__":
    main()
