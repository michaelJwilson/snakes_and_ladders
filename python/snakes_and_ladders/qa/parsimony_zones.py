"""QA figure: parsimony where it is wrong by theorem, and where it is right.

Two panels, one claim each. Panel (a) is the Felsenstein zone against the
Farris zone: four taxa, two long branches, and the Fitch score of the tree
that groups the long branches minus the score of the generating tree, per
site, as the alignment grows. Where the long branches are not a cherry the
gap is negative and does not close --- parsimony prefers the wrong tree and
more data entrenches it --- and where they are a cherry it is positive:
statistical inconsistency and consistency, drawn side by side so that a
broken implementation, which would fail both, can be told from the theorem.
Panel (b) is large parsimony on the eight-taxon fixture: every unrooted
topology scored and ranked, the generating tree marked, and the endpoint of
a nearest-neighbour-interchange climb from a random start, against the
exhaustive enumeration that referees it below eight taxa.

Renders what `snakes_and_ladders.likelihood.parsimony` and
`snakes_and_ladders.search.infer` computed; it reimplements no score and no
search (`qa/CLAUDE.md`). The textbook cites it as ``fig:parsimony-zones``. The zone trees are the standard proportions the
regression suite scores the same criteria against.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from snakes_and_ladders.likelihood.parsimony import fitch_score
from snakes_and_ladders.qa.figure import QAFigure, latex_integer
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN_WIDE,
    letter_style,
    series_style,
)
from snakes_and_ladders.search.infer import MoveSet, parsimony_search
from snakes_and_ladders.search.topology import enumerate_topologies, leaf_bipartitions
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment
from snakes_and_ladders.sim.tree import Node

#: The zone proportions: a long branch on which convergent change is common,
#: a short one, and an internal branch too short to carry much signal.
ZONE_LONG, ZONE_SHORT, ZONE_INTERNAL = 0.75, 0.02, 0.02

#: Sites per zone alignment, and independent alignments per site count.
SITE_COUNTS: tuple[int, ...] = (100, 200, 500, 1000, 2000, 5000)
REPLICATES = 8

#: The eight-taxon fixture carries 200,000 sites; the ranking panel scores
#: 10,395 topologies, so it draws this many from the same tree and seed.
RANKED_SITES = 2000
SEARCH_BUDGET = 400

_ZONE_STATES = 4


def zone_tree(first: float, second: float, third: float, fourth: float) -> Node:
    """``((A,B),(C,D))`` with the four pendant branch lengths given."""
    return Node(
        "root",
        None,
        (
            Node("i1", ZONE_INTERNAL, (Node("A", first), Node("B", second))),
            Node("i2", ZONE_INTERNAL, (Node("C", third), Node("D", fourth))),
        ),
    )


#: Felsenstein: the long branches A and C are not a cherry, so grouping them
#: is the mistake convergent change invites. Farris: A and B are a cherry.
FELSENSTEIN_ZONE = zone_tree(ZONE_LONG, ZONE_SHORT, ZONE_LONG, ZONE_SHORT)
FARRIS_ZONE = zone_tree(ZONE_LONG, ZONE_LONG, ZONE_SHORT, ZONE_SHORT)

#: The tree that groups the long branches of the Felsenstein zone, ``AC|BD``.
LONG_BRANCH_GROUPING = Node(
    "root",
    None,
    (
        Node("i1", ZONE_INTERNAL, (Node("A", ZONE_LONG), Node("C", ZONE_LONG))),
        Node("i2", ZONE_INTERNAL, (Node("B", ZONE_SHORT), Node("D", ZONE_SHORT))),
    ),
)


def zone_gaps(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-site Fitch gap, long-branch grouping minus generating tree, per zone.

    Parameters
    ----------
    rng : np.random.Generator
        Spawns one generator per zone, in a fixed order; passed in rather
        than seeded here (``sim/CLAUDE.md``).

    Returns
    -------
    dict[str, np.ndarray]
        ``"Felsenstein"`` and ``"Farris"`` to an array of shape
        ``(len(SITE_COUNTS), REPLICATES)``. Negative means parsimony prefers
        the wrong tree.
    """
    generators = rng.spawn(2)
    uniform = np.full(_ZONE_STATES, 1.0 / _ZONE_STATES)
    gaps: dict[str, np.ndarray] = {}
    for name, tau, zone_rng in (
        ("Felsenstein", FELSENSTEIN_ZONE, generators[0]),
        ("Farris", FARRIS_ZONE, generators[1]),
    ):
        table = np.zeros((len(SITE_COUNTS), REPLICATES))
        for row, n_sites in enumerate(SITE_COUNTS):
            for replicate in range(REPLICATES):
                dataset = simulate_alignment(
                    tau=tau, k=_ZONE_STATES, pi=uniform, rng=zone_rng, n_sites=n_sites
                )
                wrong = fitch_score(
                    LONG_BRANCH_GROUPING, dataset.alignment, _ZONE_STATES
                )
                right = fitch_score(tau, dataset.alignment, _ZONE_STATES)
                table[row, replicate] = (wrong - right) / n_sites
        gaps[name] = table
    return gaps


@dataclass(frozen=True)
class Ranking:
    """Every topology's Fitch score on the fixture, and where the search landed.

    Parameters
    ----------
    scores : np.ndarray
        Sorted ascending, one per unrooted topology.
    truth : int
        The generating tree's score.
    found : int
        The score the climb stopped at.
    evaluations : int
        Candidates the climb scored.
    found_truth : bool
        Whether the climb's topology is the generating one.
    """

    scores: np.ndarray
    truth: int
    found: int
    evaluations: int
    found_truth: bool


def ranking(params: SimulationParams) -> Ranking:
    """Score every topology on the fixture and climb from a random start.

    Returns
    -------
    Ranking
    """
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=RANKED_SITES,
    )
    alignment = dict(dataset.alignment)
    scores = np.array(
        sorted(
            fitch_score(topology, alignment, params.k)
            for topology in enumerate_topologies(sorted(alignment))
        )
    )
    result = parsimony_search(
        alignment,
        params.k,
        moves=MoveSet.NNI,
        rng=np.random.default_rng(params.seed),
        max_evaluations=SEARCH_BUDGET,
    )
    return Ranking(
        scores=scores,
        truth=fitch_score(params.tau, alignment, params.k),
        found=int(result.score),
        evaluations=result.evaluations,
        found_truth=leaf_bipartitions(result.topology) == leaf_bipartitions(params.tau),
    )


def build_figure(
    gaps: dict[str, np.ndarray], ranked: Ranking, params: SimulationParams
) -> tuple[Figure, str]:
    """Assemble the two panels and the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
    """
    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)
        for index, (name, table) in enumerate(gaps.items()):
            style = series_style(index)
            for column in range(table.shape[1]):
                axes[0].plot(
                    SITE_COUNTS,
                    table[:, column],
                    marker=style["marker"],
                    linestyle="none",
                    color=style["color"],
                    markersize=2.5,
                    alpha=0.5,
                )
            axes[0].plot(
                SITE_COUNTS,
                table.mean(axis=1),
                linestyle=style["linestyle"],
                color=style["color"],
                label=f"{name} zone",
            )
        axes[0].axhline(0.0, color=INK_MUTED, linewidth=0.6)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("sites")
        axes[0].set_ylabel("Fitch gap per site, long-branch tree minus truth")
        axes[0].set_title("(a) the two zones", loc="left")
        axes[0].legend(loc="center right", frameon=False, fontsize="small")

        style = series_style(2)
        axes[1].plot(
            np.arange(1, ranked.scores.size + 1),
            ranked.scores,
            color=style["color"],
            linewidth=1.0,
        )
        axes[1].axhline(
            ranked.truth,
            color=INK_MUTED,
            linestyle=":",
            linewidth=0.8,
            label="generating tree",
        )
        axes[1].axhline(
            ranked.found,
            color=series_style(1)["color"],
            linestyle="--",
            linewidth=0.8,
            label="NNI climb",
        )
        axes[1].set_xscale("log")
        axes[1].set_xlabel("topologies, ranked")
        axes[1].set_ylabel("Fitch score")
        axes[1].set_title("(b) large parsimony, 8 taxa", loc="left")
        axes[1].legend(loc="lower right", frameon=False, fontsize="small")
        fig.tight_layout()

    felsenstein = gaps["Felsenstein"].mean(axis=1)
    farris = gaps["Farris"].mean(axis=1)
    caption = (
        f"Parsimony in the two zones, and large parsimony on the eight-taxon "
        f"fixture. (a) Four taxa, pendant branches {ZONE_LONG} and {ZONE_SHORT} "
        f"expected substitutions per site with internal branches "
        f"{ZONE_INTERNAL}, {_ZONE_STATES}-state Jukes-Cantor, seed "
        f"{params.seed}, {REPLICATES} independent alignments at each of "
        f"{len(SITE_COUNTS)} site counts from {SITE_COUNTS[0]} to "
        f"{SITE_COUNTS[-1]}. Each marker is the Fitch score of the tree "
        f"grouping the two long branches minus the generating tree's, per "
        f"site; the line is the mean. In the Felsenstein zone the long "
        f"branches are not a cherry and the mean gap is {felsenstein[0]:.3f} "
        f"at {SITE_COUNTS[0]} sites and {felsenstein[-1]:.3f} at "
        f"{SITE_COUNTS[-1]}: negative, so the criterion prefers the wrong "
        f"tree, and not closing, which is statistical inconsistency. In the "
        f"Farris zone the same branches are a cherry and the gap is "
        f"{farris[0]:.3f} and {farris[-1]:.3f}. (b) Every one of the "
        f"{latex_integer(ranked.scores.size)} unrooted topologies on the "
        f"8-taxon fixture, scored at {latex_integer(RANKED_SITES)} sites "
        f"(seed {params.seed}) and ranked; the generating tree scores "
        f"{ranked.truth} and is the minimum. A nearest-neighbour-interchange "
        f"climb from a random start reaches {ranked.found} after scoring "
        f"{ranked.evaluations} candidates of a {SEARCH_BUDGET} budget"
        f"{', the generating tree' if ranked.found_truth else ''}; the "
        f"enumeration is the oracle that says so."
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
        stem="parsimony_zones",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=lambda params: build_figure(
            zone_gaps(np.random.default_rng(params.seed)), ranking(params), params
        ),
        argv=argv,
    )


if __name__ == "__main__":
    main()
