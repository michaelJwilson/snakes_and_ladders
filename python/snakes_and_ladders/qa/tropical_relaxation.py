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

Renders what ``snakes_and_ladders.sandbox.tropical`` and
``snakes_and_ladders.search.infer`` computed; it reimplements no score
(``qa/CLAUDE.md``). The paper cites it as ``fig:tropical-relaxation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.qa.style import INK_MUTED, ONE_COLUMN_WIDE, letter_style
from snakes_and_ladders.sandbox.tropical import (
    corner,
    corner_bound,
    discrete_score,
    pairing_positions,
    quartet_table,
    relaxed_score,
    temperature_for,
)
from snakes_and_ladders.search.infer import score_topology
from snakes_and_ladders.search.topology import enumerate_topologies, leaf_bipartitions
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
        The largest ``|F_tau - D|`` over the corners of every enumerated
        topology, and the largest certificate over the same, one per
        :data:`TEMPERATURES`. Taken over every corner rather than at the
        generating tree's: a relaxation that is an extension only near the
        truth optimizes a different problem everywhere else, and at the
        generating tree alone the gap reaches exactly zero and says nothing.
    derived : float
        The largest temperature :func:`temperature_for` certifies
        :data:`TOLERANCE` at over the same corners.
    at_derived : float
        The largest measured gap there.
    discrete : float
        ``D`` at the generating tree, the magnitude the residual is relative
        to.
    """

    measured: np.ndarray
    bound: np.ndarray
    derived: float
    at_derived: float
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

    topologies = list(enumerate_topologies(sorted(alignment)))
    corners = [corner(topology) for topology in topologies]
    discretes = [discrete_score(table, topology) for topology in topologies]

    def gap(metric: np.ndarray, discrete: float, temperature: float) -> float:
        relaxed = float(
            relaxed_score(table, positions, torch.from_numpy(metric), temperature)
        )
        return abs(relaxed - discrete)

    measured = np.array(
        [
            max(
                gap(metric, discrete, temperature)
                for metric, discrete in zip(corners, discretes, strict=True)
            )
            for temperature in TEMPERATURES
        ]
    )
    bound = np.array(
        [
            max(
                corner_bound(table, positions, metric, temperature)
                for metric in corners
            )
            for temperature in TEMPERATURES
        ]
    )
    derived = max(
        temperature_for(table, positions, metric, TOLERANCE) for metric in corners
    )
    sweep = Sweep(
        measured=measured,
        bound=bound,
        derived=derived,
        at_derived=max(
            gap(metric, discrete, derived)
            for metric, discrete in zip(corners, discretes, strict=True)
        ),
        discrete=discrete_score(table, params.tau),
    )

    quartet = np.array(discretes)
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

    relative = sweep.at_derived / abs(sweep.discrete)
    agreement = (
        "share their maximizer"
        if surfaces.argmax_agrees
        else "do not share their maximizer"
    )
    caption = (
        f"The tropical Grassmannian relaxation on the {len(params.pi)}-state "
        f"Jukes-Cantor fixture of {len(surfaces.quartet)} unrooted topologies "
        f"(seed {params.seed}, {params.n_sites} sites). (a) At the tree metric "
        f"of each of them, the largest gap between the relaxed "
        f"objective and the discrete quartet score D against the softmin "
        f"temperature, with the certificate (the sum over quartets of twice "
        f"the score range times the exponential of minus the quartet gap "
        f"over the temperature) drawn over it. The dotted line is the "
        f"temperature the certificate is inverted for at the dashed "
        f"tolerance, {sweep.derived:.3g}, where the measured gap is "
        f"{sweep.at_derived:.2g} -- {relative:.1g} of the score magnitude "
        f"{abs(sweep.discrete):.0f}, which is float64 rounding of a sum over "
        f"quartets rather than the relaxation. (b) Every topology scored on both "
        f"surfaces: D is a quartet decomposition and not the tree's "
        f"log-likelihood, and the two {agreement}, which is what licenses "
        f"relaxing the one in place of the other."
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
