"""QA figure: the tree the search found, beside the best one it rejected.

Milestone 5 asks for the inferred and generating trees compared. On a fixture
where the search recovers the truth those two drawings are identical, which
is the right result and an uninformative picture. The informative comparison
is against the runner-up: the highest-scoring topology the search ruled out,
with the split that separates them marked.

Renders what `snakes_and_ladders.search` and `snakes_and_ladders.likelihood` computed; it reimplements
no move set, no optimizer and no recursion (`qa/CLAUDE.md`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.likelihood.objective import BranchLengthObjective
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.qa.figure import QAFigure, latex_integer
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.qa.sim_tree import render_sim_tree
from snakes_and_ladders.qa.style import ONE_COLUMN_WIDE, letter_style
from snakes_and_ladders.search.infer import MoveSet, infer
from snakes_and_ladders.search.topology import (
    Topology,
    enumerate_topologies,
    leaf_bipartitions,
)
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node, preorder


def _fitted(
    topology: Topology, alignment: dict[str, np.ndarray], k: int, pi: np.ndarray
) -> tuple[Node, float]:
    objective = BranchLengthObjective(topology, k, pi, alignment)
    result = fit(objective)
    return objective.fitted_tree(result.theta), -result.value


def found_and_runner_up(
    params: SimulationParams,
) -> tuple[Node, float, Node, float, frozenset[str], bool]:
    """Fit the search's tree and the best topology it rejected.

    Parameters
    ----------
    params : SimulationParams
        The generating truth.

    Returns
    -------
    tuple[Node, float, Node, float, frozenset[str], bool]
        The found tree and its score, the runner-up and its score, one leaf
        set separating them, and whether the found tree is the generating
        one.
    """
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    alignment = dict(dataset.alignment)

    result = infer(alignment, params.k, rng=np.random.default_rng(0), moves=MoveSet.NNI)
    found_key = leaf_bipartitions(result.topology)

    ranked = sorted(
        (
            (
                _fitted(topology, alignment, params.k, params.pi)[1],
                leaf_bipartitions(topology),
                topology,
            )
            for topology in enumerate_topologies(sorted(alignment))
            if leaf_bipartitions(topology) != found_key
        ),
        key=lambda entry: entry[0],
        reverse=True,
    )
    runner_up_topology = ranked[0][2]

    found_tree, found_score = _fitted(result.topology, alignment, params.k, params.pi)
    other_tree, other_score = _fitted(
        runner_up_topology, alignment, params.k, params.pi
    )
    # A split present in one and absent from the other: what the likelihood
    # difference is actually about.
    difference = sorted(found_key - ranked[0][1], key=len)[0]
    return (
        found_tree,
        found_score,
        other_tree,
        other_score,
        difference,
        found_key == leaf_bipartitions(params.tau),
    )


def build_figure(
    found: Node,
    found_score: float,
    other: Node,
    other_score: float,
    difference: frozenset[str],
    recovered_truth: bool,
    params: SimulationParams,
) -> tuple[Figure, str]:
    """Assemble the side-by-side comparison and its caption.

    Parameters
    ----------
    found, other : Node
        The found tree and the best rejected topology, with fitted lengths.
    found_score, other_score : float
        Their maximized log-likelihoods.
    difference : frozenset[str]
        A leaf set forming a split present in one and absent from the other.
    recovered_truth : bool
        Whether the found tree is the generating one.
    params : SimulationParams
        The generating truth, for the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption text.
    """
    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)
        for ax, tree, title in (
            (axes[0], found, "(a) found"),
            (axes[1], other, "(b) best rejected"),
        ):
            render_sim_tree(tree, ax)
            ax.set_title(title, loc="left")
        fig.tight_layout()

    grouped = " ".join(sorted(difference))
    # A wrong grouping is typically only supportable by collapsing the edge
    # that would create it, so the shortest internal branch is diagnostic.
    shortest = min(
        node.branch_length
        for node in preorder(other)
        if not node.is_leaf and node.branch_length is not None
    )
    caption = (
        f"The topology hill climbing selected, beside the highest-scoring one "
        f"it rejected, each with its own branch lengths fitted. Fixture: "
        f"{len(params.pi)}-state Jukes-Cantor over "
        f"{latex_integer(params.n_sites)} sites, seed {params.seed}. "
        f"(a) scores {found_score:.1f} and "
        f"{'is' if recovered_truth else 'is not'} the generating topology; "
        f"(b) scores {other_score:.1f}, a difference of "
        f"{found_score - other_score:.1f} log units. They differ by one "
        f"split: {grouped} are a group in one and not in the other. On this "
        f"fixture the search recovers the generating tree, so a drawing of "
        f"found against true would show the same tree twice; the runner-up is "
        f"what the likelihood actually had to discriminate against. Note its "
        f"shortest internal branch, fitted at {shortest:.3f}: a wrong grouping "
        f"is generally only supportable by collapsing the edge that would "
        f"create it."
    )
    return fig, caption


def main(argv: list[str] | None = None) -> QAFigure:
    """Render the figure from the command line.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; ``None`` reads ``sys.argv``.

    Returns
    -------
    QAFigure
        Paths written, and the caption.
    """
    return figure_main(
        stem="search_topologies",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=lambda params: build_figure(*found_and_runner_up(params), params),
        argv=argv,
    )


if __name__ == "__main__":
    main()
