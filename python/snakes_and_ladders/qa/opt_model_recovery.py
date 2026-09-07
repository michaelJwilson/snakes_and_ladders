"""QA figure: recovering a substitution model, and not inventing one.

Milestone 4 asks for the rate matrix and root distribution fitted, not only
branch lengths. Jukes-Cantor has no free rate parameters, so this needs the
general time-reversible model (:mod:`snakes_and_ladders.sim.gtr`), and the figure makes
the two claims that matter for it.

Panel (a): data simulated under an asymmetric GTR truth recovers that truth,
with intervals from the observed information. Panel (b): data simulated under
Jukes-Cantor recovers a Jukes-Cantor-like model -- equal exchangeabilities
and a uniform stationary distribution -- rather than inventing structure the
data does not contain. A model flexible enough to fit the first is only
useful if it also declines to overfit the second.

Renders what `snakes_and_ladders.likelihood` and `snakes_and_ladders.opt` computed; it reimplements no
model and no optimizer (`qa/CLAUDE.md`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from snakes_and_ladders.likelihood.objective import SubstitutionModelObjective
from snakes_and_ladders.opt.fit import constrained_standard_errors, covers, fit
from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import SIMULATION_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN_WIDE,
    letter_style,
    series_style,
)
from snakes_and_ladders.sim.gtr import gtr_rate_matrix
from snakes_and_ladders.sim.params import SimulationParams
from snakes_and_ladders.sim.simulate import simulate_alignment

# The generating truth, matching tests/regression/test_gtr.py: no two
# exchangeabilities equal and no two frequencies equal, so a fit that
# collapsed either would be visible rather than flattering.
TRUE_EXCHANGEABILITIES = np.array([1.6, 0.4, 0.9, 0.7, 2.1, 1.0])
TRUE_PI = np.array([0.35, 0.15, 0.30, 0.20])

SITES = 20000

_Z_95 = 1.959963984540054


def fit_model(
    params: SimulationParams, rate_matrix: np.ndarray | None, pi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate under a model, fit the general one, and return the estimates.

    Parameters
    ----------
    params : SimulationParams
        Topology, alphabet size and seed.
    rate_matrix : np.ndarray | None
        Rate matrix to simulate under; ``None`` for Jukes-Cantor.
    pi : np.ndarray
        Stationary distribution to simulate under.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Fitted values and standard errors, over the free exchangeabilities
        followed by the stationary frequencies. Coverage is decided where the
        truth is known, in :func:`build_figure`.
    """
    dataset = simulate_alignment(
        tau=params.tau,
        k=params.k,
        pi=pi,
        rng=np.random.default_rng(params.seed),
        n_sites=SITES,
        rate_matrix=rate_matrix,
    )
    objective = SubstitutionModelObjective(
        params.tau, params.k, dict(dataset.alignment)
    )
    result = fit(objective)
    estimate = objective.constrain(result.theta)
    error = constrained_standard_errors(objective, result.theta)

    fitted = torch.cat([estimate["exchangeabilities"][:-1], estimate["pi"]])
    spread = torch.cat([error["exchangeabilities"][:-1], error["pi"]])
    return fitted.numpy(), spread.numpy()


def truth_vector(exchangeabilities: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """The free exchangeabilities, rescaled so the last is 1, then ``pi``."""
    scaled = exchangeabilities / exchangeabilities[-1]
    return np.concatenate([scaled[:-1], pi])


def _draw(
    ax: Axes,
    truth: np.ndarray,
    fitted: np.ndarray,
    spread: np.ndarray,
    index: int,
) -> np.ndarray:
    hits = covers(
        torch.as_tensor(fitted), torch.as_tensor(spread), torch.as_tensor(truth)
    ).numpy()
    style = series_style(index)
    span = np.array([0.0, max(truth.max(), fitted.max()) * 1.15])
    ax.plot(span, span, color=INK_MUTED, linestyle=":", linewidth=0.8, zorder=1)
    for covered, face in ((hits, style["color"]), (~hits, "white")):
        if not covered.any():
            continue
        ax.errorbar(
            truth[covered],
            fitted[covered],
            yerr=_Z_95 * spread[covered],
            linestyle="none",
            marker=style["marker"],
            markersize=4,
            markerfacecolor=face,
            markeredgecolor=style["color"],
            color=style["color"],
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=2,
            zorder=2,
        )
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    ax.set_xlabel("generating value")
    return hits


def build_figure(
    params: SimulationParams,
    general: tuple[np.ndarray, np.ndarray],
    jukes_cantor: tuple[np.ndarray, np.ndarray],
) -> tuple[Figure, str]:
    """Assemble the two-panel figure and its caption.

    Parameters
    ----------
    params : SimulationParams
        The topology and seed used, for the caption.
    general, jukes_cantor : tuple[np.ndarray, np.ndarray]
        Fitted values and standard errors from each simulation.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption text.
    """
    gtr_truth = truth_vector(TRUE_EXCHANGEABILITIES, TRUE_PI)
    jc_truth = truth_vector(
        np.ones(TRUE_EXCHANGEABILITIES.size), np.full(params.k, 1.0 / params.k)
    )

    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)
        gtr_hits = _draw(axes[0], gtr_truth, *general, index=0)
        jc_hits = _draw(axes[1], jc_truth, *jukes_cantor, index=1)
        axes[0].set_ylabel("fitted value")
        axes[0].set_title("(a) generated under GTR", loc="left")
        axes[1].set_title("(b) generated under JC", loc="left")
        fig.tight_layout()

    caption = (
        f"Fitting the general time-reversible model: five free "
        f"exchangeabilities (the sixth is pinned at 1) and four stationary "
        f"frequencies, alongside the branch lengths, by the same "
        f"model-agnostic optimizer used for the Potts and hidden Markov "
        f"instances. Bars are 95 percent Wald intervals from the observed "
        f"information; the dotted line is y = x; markers are open where an "
        f"interval misses. Topology: the {len(params.tau.children)}-child-root "
        f"fixture, seed {params.seed}, {SITES} sites. (a) Data simulated under "
        f"an asymmetric truth, {int(gtr_hits.sum())} of {gtr_hits.size} "
        f"intervals covering. (b) Data simulated under Jukes-Cantor, where "
        f"every exchangeability is 1 and every frequency 1/4: "
        f"{int(jc_hits.sum())} of {jc_hits.size} covering. The second panel "
        f"is the half that is easy to forget. A model flexible enough to fit "
        f"panel (a) is only useful if it also declines to invent structure "
        f"that is not there, and Jukes-Cantor is exactly the point of the "
        f"parameter space where the general model could most easily do so. "
        f"Coverage on a single dataset is a draw and not a rate: the nominal "
        f"0.95 is checked over independent replicates by the release-gated "
        f"regression suite."
    )
    return fig, caption


def _fit_and_build(params: SimulationParams) -> tuple[Figure, str]:
    """Fit both models the figure compares, then render them.

    Returns
    -------
    tuple[Figure, str]
        The figure and its caption.
    """
    general = fit_model(
        params, gtr_rate_matrix(TRUE_EXCHANGEABILITIES, TRUE_PI), TRUE_PI
    )
    jukes_cantor = fit_model(params, None, np.full(params.k, 1.0 / params.k))
    return build_figure(params, general, jukes_cantor)


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
        stem="opt_model_recovery",
        description=__doc__,
        params=[SIMULATION_PARAMS],
        build=_fit_and_build,
        argv=argv,
    )


if __name__ == "__main__":
    main()
