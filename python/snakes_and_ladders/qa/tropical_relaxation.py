"""QA figure: the tropical Grassmannian relaxation, and the surface it relaxes.

Two panels, one claim each. Panel (a) is the corner agreement against the
softmin temperature: at the metric of the generating tree, the measured
``|F_tau - D|`` and the bound ``sum_Q 2 exp(-g_Q / tau) R_Q`` that certifies
it, over four decades of ``tau``, with the temperature the bound is inverted
for marked. It shows the bound holding where the softmin is loose, and shows
that below the derived temperature the agreement stops improving because what
remains is float64 rounding of a sum and not the relaxation. Panel (b) is the
quartet surface against the fitted likelihood, every unrooted topology of the
five-taxon fixture scored on both: the two are different surfaces, and what
licenses relaxing one in place of the other is that they agree at the argmax.

Renders what ``snakes_and_ladders.search.tropical`` and
``snakes_and_ladders.search.infer`` computed; it reimplements no score
(``qa/CLAUDE.md``). The paper cites it as ``fig:tropical-relaxation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from snakes_and_ladders.likelihood.distance import tree_distances
from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.qa.style import INK_MUTED, ONE_COLUMN_WIDE, letter_style
from snakes_and_ladders.search.infer import score_topology
from snakes_and_ladders.search.topology import enumerate_topologies, leaf_bipartitions
from snakes_and_ladders.search.tropical import (
    condensed,
    corner_bound,
    discrete_score,
    pairing_positions,
    quartet_table,
    relaxed_score,
    temperature_for,
)
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment

#: Temperatures panel (a) sweeps, geometrically, and the agreement panel (a)
#: inverts the bound for.
TEMPERATURES = np.geomspace(1.0, 1e-3, 25)
TOLERANCE = 1e-11


@dataclass(frozen=True)
class Sweep:
    """Panel (a): the measured corner gap and its bound, per temperature.

    Parameters
    ----------
    measured, bound : np.ndarray
        ``|F_tau - D|`` and the certificate, one per :data:`TEMPERATURES`.
    derived : float
        The temperature :func:`temperature_for` returns at
        :data:`TOLERANCE`.
    discrete : float
        ``D`` at the generating tree, the magnitude the residual is relative
        to.
    """

    measured: np.ndarray
    bound: np.ndarray
    derived: float
    discrete: float


@dataclass(frozen=True)
class Surfaces:
    """Panel (b): every topology scored on both surfaces.

    Parameters
    ----------
    quartet, likelihood : np.ndarray
        One entry per unrooted topology, in enumeration order.
    argmax_agrees : bool
        Whether the two surfaces have the same maximizer.
    """

    quartet: np.ndarray
    likelihood: np.ndarray
    argmax_agrees: bool


def measure(params: SimulationParams) -> tuple[Sweep, Surfaces]:
    """Fit the quartet table once and take both panels from it.

    Returns
    -------
    tuple[Sweep, Surfaces]
    """
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=params.pi,
        rng=np.random.default_rng(params.seed),
        n_sites=params.n_sites,
    )
    alignment = dict(dataset.alignment)
    table = quartet_table(alignment, params.k)
    positions = pairing_positions(table.n_taxa)

    _, square = tree_distances(params.tau)
    metric = condensed(square)
    discrete = discrete_score(table, params.tau)
    measured = np.array(
        [
            abs(
                float(
                    relaxed_score(
                        table, positions, torch.from_numpy(metric), temperature
                    )
                )
                - discrete
            )
            for temperature in TEMPERATURES
        ]
    )
    bound = np.array(
        [
            corner_bound(table, positions, metric, temperature)
            for temperature in TEMPERATURES
        ]
    )
    sweep = Sweep(
        measured=measured,
        bound=bound,
        derived=temperature_for(table, positions, metric, TOLERANCE),
        discrete=discrete,
    )

    topologies = list(enumerate_topologies(sorted(alignment)))
    quartet = np.array([discrete_score(table, topology) for topology in topologies])
    likelihood = np.array(
        [score_topology(topology, alignment, params.k) for topology in topologies]
    )
    keys = [leaf_bipartitions(topology) for topology in topologies]
    surfaces = Surfaces(
        quartet=quartet,
        likelihood=likelihood,
        argmax_agrees=keys[int(quartet.argmax())] == keys[int(likelihood.argmax())],
    )
    return sweep, surfaces


def build_figure(
    sweep: Sweep, surfaces: Surfaces, params: SimulationParams
) -> tuple[Figure, str]:
    """Assemble the two panels and the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
    """
    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)
        axes[0].loglog(TEMPERATURES, sweep.bound, label="bound", color=INK_MUTED)
        axes[0].loglog(
            TEMPERATURES,
            np.maximum(sweep.measured, 1e-18),
            marker="o",
            markersize=2.5,
            linestyle="none",
            label="measured",
        )
        axes[0].axvline(sweep.derived, linestyle=":", color=INK_MUTED)
        axes[0].axhline(TOLERANCE, linestyle="--", color=INK_MUTED)
        axes[0].set_xlabel(r"softmin temperature $\tau$")
        axes[0].set_ylabel(r"$|F_\tau(d) - D|$")
        axes[0].set_title("(a) corner agreement", loc="left")
        axes[0].legend(loc="lower right", frameon=False, fontsize="small")

        axes[1].plot(
            surfaces.likelihood,
            surfaces.quartet,
            marker="o",
            markersize=3,
            linestyle="none",
        )
        best = int(surfaces.likelihood.argmax())
        axes[1].plot(
            surfaces.likelihood[best],
            surfaces.quartet[best],
            marker="*",
            markersize=11,
            linestyle="none",
            label="likelihood maximum",
        )
        axes[1].set_xlabel("fitted log-likelihood")
        axes[1].set_ylabel("quartet score $D$")
        axes[1].set_title(
            f"(b) two surfaces, {len(surfaces.quartet)} topologies", loc="left"
        )
        axes[1].legend(loc="lower right", frameon=False, fontsize="small")
        fig.tight_layout()

    relative = sweep.measured.min() / abs(sweep.discrete)
    caption = (
        f"The tropical Grassmannian relaxation on the {len(params.pi)}-state "
        f"Jukes-Cantor fixture of {len(surfaces.quartet)} unrooted topologies "
        f"(seed {params.seed}, {params.n_sites} sites). (a) At the generating "
        f"tree's metric, the measured gap between the relaxed objective "
        f"$F_\\tau$ and the discrete quartet score $D$ against the softmin "
        f"temperature, with the certificate $\\sum_Q 2 e^{{-g_Q/\\tau}} R_Q$ "
        f"over it. The dotted line is the temperature the certificate is "
        f"inverted for at the dashed tolerance, $\\tau = "
        f"{sweep.derived:.3g}$; below it the measured gap stops falling at "
        f"{sweep.measured.min():.2g}, which is {relative:.1g} of $|D| = "
        f"{abs(sweep.discrete):.0f}$ and is float64 rounding of a sum rather "
        f"than the relaxation. (b) Every topology scored on both surfaces: "
        f"$D$ is a quartet decomposition and not the tree's log-likelihood, "
        f"and the two "
        f"{'share their maximizer' if surfaces.argmax_agrees else 'do not share their maximizer'}, "
        f"which is what licenses relaxing the one in place of the other."
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

    def build(params: SimulationParams) -> tuple[Figure, str]:
        sweep, surfaces = measure(params)
        return build_figure(sweep, surfaces, params)

    return figure_main(
        stem="tropical_relaxation",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=build,
        argv=argv,
    )


if __name__ == "__main__":
    main()
