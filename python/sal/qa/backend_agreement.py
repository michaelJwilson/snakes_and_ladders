"""QA figure: the backends against the oracle, and against each other.

``ROADMAP.md``'s numerics requirement is that cross-backend agreement is
checked against a declared tolerance, never bitwise. This figure is that
check, made visible: every backend's log-likelihood against brute-force
marginalization over all ``k ** m`` ancestral assignments, which is the one
computation here that owes nothing to the pruning recursion it validates.

Brute force is what bounds the problem sizes. It costs ``k ** m`` for ``m``
internal nodes, so the fixtures are small by necessity, and that is the point:
the recursion is pinned where an independent answer exists, and the larger
sizes inherit the guarantee from the code being the same.

Renders what `sal.likelihood` computed; it reimplements no backend
(`qa/CLAUDE.md`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sal.backend import Backend
from sal.likelihood import pruning
from sal.likelihood.brute_force import brute_force_log_likelihood
from sal.qa.figure import QAFigure
from sal.qa.runner import FIXTURE_PARAMS, figure_main
from sal.qa.style import (
    INK_MUTED,
    ONE_COLUMN,
    letter_style,
    series_style,
)
from sal.sim.params import SimulationParams
from sal.sim.simulator import simulate_tree

# Site counts the comparison is run at. Chosen to span two orders of
# magnitude, because the quantity being compared is a sum over sites: an
# absolute discrepancy grows with the sum while a relative one does not, and
# the figure exists partly to show which of the two is stable.
SITE_COUNTS = (100, 300, 1000, 3000)

#: Every `Backend` member `pruning.log_likelihood` runs, reached through its
#: `backend=` (issue #1059).
BACKENDS = (Backend.PYTHON, Backend.TORCH, Backend.RUST)

#: The legend's name for each: the oracle is the NumPy recursion.
LABELS = {Backend.PYTHON: "numpy", Backend.TORCH: "torch", Backend.RUST: "rust"}


def agreement(params: SimulationParams) -> dict[Backend, list[tuple[int, float]]]:
    """Relative deviation of each backend from brute force, per site count.

    Returns
    -------
    dict[Backend, list[tuple[int, float]]]
        Per backend, ``(n_sites, |backend - oracle| / |oracle|)`` pairs.
    """
    measured: dict[Backend, list[tuple[int, float]]] = {
        backend: [] for backend in BACKENDS
    }
    for n_sites in SITE_COUNTS:
        dataset = simulate_tree(
            params, np.random.default_rng(params.seed), n_sites=n_sites
        )
        alignment = dict(dataset.alignment)
        oracle = brute_force_log_likelihood(params.tau, params.k, params.pi, alignment)
        for backend in BACKENDS:
            value = pruning.log_likelihood(
                params.tau, params.k, params.pi, alignment, backend=backend
            )
            measured[backend].append((n_sites, abs(value - oracle) / abs(oracle)))
    return measured


def build_figure(
    measured: dict[Backend, list[tuple[int, float]]], params: SimulationParams
) -> tuple[Figure, str]:
    """Assemble the figure and its caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption text.
    """
    worst = max(value for points in measured.values() for _, value in points)
    with letter_style():
        fig, axis = plt.subplots(figsize=ONE_COLUMN)
        for index, backend in enumerate(BACKENDS):
            style = series_style(index)
            points = measured[backend]
            axis.plot(
                [x for x, _ in points],
                [max(y, 1e-18) for _, y in points],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                markersize=4,
                linewidth=1.0,
                label=LABELS[backend],
            )
        axis.axhline(
            np.finfo(np.float64).eps,
            color=INK_MUTED,
            linestyle=":",
            linewidth=0.8,
            zorder=1,
        )
        axis.annotate(
            "float64 epsilon",
            xy=(0.03, np.finfo(np.float64).eps),
            xycoords=("axes fraction", "data"),
            va="bottom",
            color=INK_MUTED,
            fontsize="small",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("sites")
        axis.set_ylabel("relative deviation from brute force")
        axis.legend(loc="upper right", frameon=False, fontsize="small")
        fig.tight_layout()

    caption = (
        f"Every pruning backend against the oracle that owes it nothing. "
        f"Brute-force marginalization sums over all ancestral assignments, so "
        f"it shares no code with the recursion it checks, and its cost is what "
        f"keeps the fixture small. Fixture: {len(params.pi)}-state "
        f"Jukes-Cantor on the {sum(1 for _ in params.tau.children)}-child-root "
        f"tree, seed {params.seed}. The deviation is relative because the "
        f"log-likelihood is a sum over sites: an absolute bound fixed at one "
        f"site count does not transfer to another, which is why the "
        f"x-axis spans a factor of {max(SITE_COUNTS) // min(SITE_COUNTS)}. "
        f"Worst deviation across every backend and size: {worst:.2e}, against "
        f"a float64 epsilon of {np.finfo(np.float64).eps:.2e}."
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
        stem="backend_agreement",
        description=__doc__,
        params=[FIXTURE_PARAMS],
        build=lambda params: build_figure(agreement(params), params),
        argv=argv,
    )


if __name__ == "__main__":
    main()
