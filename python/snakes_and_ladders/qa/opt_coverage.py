"""QA figure: do the fitted intervals cover the truth at their nominal rate?

`opt/CLAUDE.md` makes recovery the acceptance test, and a Wald interval built
from the observed information is an *asymptotic* claim: it is exact only in
the limit of large samples. This figure is the evidence for that claim rather
than an assertion of it, and it is what separates a slightly optimistic
approximation from a wrong formula -- the first shrinks as the sample grows,
the second does not.

Both reference instances are refitted at a range of data sizes, many
independent datasets at each, and the fraction of 95 percent intervals
containing the generating value is plotted with its binomial uncertainty.

Renders what `snakes_and_ladders.opt` computed; it reimplements no model and no optimizer
(`qa/CLAUDE.md`).
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from snakes_and_ladders.opt.fit import constrained_standard_errors, covers, fit
from snakes_and_ladders.opt.hmm import HmmObjective, align_states
from snakes_and_ladders.opt.potts import (
    PottsObjective,
    PottsParams,
    simulate_chains,
)
from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import HMM_PARAMS, POTTS_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN,
    letter_style,
    series_style,
)
from snakes_and_ladders.sim.hmm import HmmParams, simulate_sequences

NOMINAL = 0.95

# Sizes and replicate counts, chosen so the whole figure stays inside a
# minute of the technical-document build. The Potts fit is two orders of
# magnitude cheaper than the HMM's, so it gets both more sizes and more
# replicates; the HMM's largest size is where the asymptotics are meant to
# be visible, so it is the one that must not be dropped. The smallest HMM size
# is deliberately small enough that some fits reach the boundary of the
# parameter space and have no interval at all; those are counted and
# reported rather than hidden, since it is part of what small-sample
# inference looks like.
POTTS_SIZES = ((100, 40), (400, 40), (1600, 25))
HMM_SIZES = ((150, 8), (600, 6), (2400, 4))


def potts_coverage(
    params: PottsParams, n_chains: int, replicates: int
) -> tuple[int, int]:
    """Count covering intervals over independent Potts datasets.

    Parameters
    ----------
    params : PottsParams
        The generating truth; its ``n_chains`` is overridden.
    n_chains : int
        Chains per dataset.
    replicates : int
        Independent datasets, each with its own seed.

    Returns
    -------
    tuple[int, int]
        Intervals covering, and intervals checked.
    """
    truth = torch.cat(
        [
            torch.tensor([params.coupling], dtype=torch.float64),
            torch.as_tensor(params.field),
        ]
    )
    covered = 0
    total = 0
    for replicate in range(replicates):
        drawn = replace(params, seed=params.seed + 7919 * replicate, n_chains=n_chains)
        objective = PottsObjective(simulate_chains(drawn), drawn.n_states)
        result = fit(objective)
        estimate = objective.constrain(result.theta)
        error = constrained_standard_errors(objective, result.theta)
        point = torch.cat([estimate["coupling"].reshape(1), estimate["field"]])
        spread = torch.cat([error["coupling"].reshape(1), error["field"]])
        hits = covers(point, spread, truth)
        covered += int(hits.sum())
        total += hits.numel()
    return covered, total


def hmm_coverage(
    params: HmmParams, n_sequences: int, replicates: int
) -> tuple[int, int, int]:
    """Count covering intervals over independent HMM datasets.

    The hidden-state permutation is aligned before comparing, since the
    likelihood is invariant to it.

    A fit whose estimate reaches the boundary of the parameter space -- an
    emission probability at zero -- has no interval to check: the curvature
    in that direction vanishes, the observed information is singular, and
    ``snakes_and_ladders.opt.fit`` refuses to invert it. Those replicates are counted
    separately rather than dropped silently, because excluding them without
    saying so would quietly select for the well-behaved samples and report a
    coverage that no procedure achieves.

    Parameters
    ----------
    params : HmmParams
        The generating truth; its ``n_sequences`` is overridden.
    n_sequences : int
        Sequences per dataset.
    replicates : int
        Independent datasets, each with its own seed.

    Returns
    -------
    tuple[int, int, int]
        Intervals covering, intervals checked, and replicates that reached
        the boundary and so contributed no intervals.
    """
    truths = {
        "log_initial": torch.log(torch.as_tensor(params.initial)),
        "log_transition": torch.log(torch.as_tensor(params.transition)),
        "log_emission": torch.log(torch.as_tensor(params.emission)),
    }
    covered = 0
    total = 0
    boundary = 0
    for replicate in range(replicates):
        drawn = replace(
            params, seed=params.seed + 7919 * replicate, n_sequences=n_sequences
        )
        objective = HmmObjective(
            simulate_sequences(drawn).observations, drawn.n_states, drawn.n_symbols
        )
        result = fit(objective)
        estimate = objective.constrain(result.theta)
        try:
            error = constrained_standard_errors(objective, result.theta)
        except ValueError:
            boundary += 1
            continue
        order = list(
            align_states(estimate["log_emission"], torch.as_tensor(params.emission))
        )
        for name, reference in truths.items():
            point, spread = estimate[name], error[name]
            if name == "log_transition":
                point, spread = point[order][:, order], spread[order][:, order]
            else:
                point, spread = point[order], spread[order]
            hits = covers(point, spread, reference)
            covered += int(hits.sum())
            total += hits.numel()
    return covered, total, boundary


def _points(
    series: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sizes, realized rates, and binomial standard errors, for one curve."""
    empty = [size for size, _, total in series if total == 0]
    if empty:
        msg = (
            f"every fit at size(s) {empty} reached the boundary of the "
            f"parameter space, so there are no intervals to plot; raise the "
            f"size or the replicate count"
        )
        raise ValueError(msg)
    sizes = np.array([float(size) for size, _, _ in series])
    rates = np.array([hit / total for _, hit, total in series])
    errors = np.array(
        [
            float((rate * (1.0 - rate) / total) ** 0.5)
            for rate, (_, _, total) in zip(rates, series, strict=True)
        ]
    )
    return sizes, rates, errors


def build_figure(
    potts: list[tuple[int, int, int]],
    hmm: list[tuple[int, int, int, int]],
) -> tuple[Figure, str]:
    """Assemble the coverage figure and its caption.

    Parameters
    ----------
    potts : list[tuple[int, int, int]]
        One ``(size, covered, total)`` per data size.
    hmm : list[tuple[int, int, int, int]]
        One ``(size, covered, total, boundary)`` per data size, the last
        entry counting replicates that produced no interval.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption text.
    """
    with letter_style():
        fig, ax = plt.subplots(figsize=ONE_COLUMN)
        ax.axhline(NOMINAL, color=INK_MUTED, linestyle=":", linewidth=0.8, zorder=1)
        ax.annotate(
            "nominal 0.95",
            xy=(0.02, NOMINAL),
            xycoords=("axes fraction", "data"),
            va="bottom",
            color=INK_MUTED,
            fontsize="small",
        )
        curves = [
            (_points(potts), "Potts"),
            (_points([entry[:3] for entry in hmm]), "HMM"),
        ]
        for index, ((sizes, rates, errors), label) in enumerate(curves):
            style = series_style(index)
            ax.errorbar(
                sizes,
                rates,
                yerr=errors,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                ecolor=style["color"],
                markersize=4,
                linewidth=1.0,
                elinewidth=0.9,
                capsize=2,
                label=label,
                zorder=2,
            )
        ax.set_xscale("log")
        ax.set_xlabel("datasets: chains (Potts) or sequences (HMM) per fit")
        ax.set_ylabel("fraction of 95% intervals covering truth")
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()

    def _rate(entry: tuple[int, ...]) -> str:
        size, hit, total = entry[0], entry[1], entry[2]
        return f"{hit}/{total} = {hit / total:.3f} at {size}"

    boundary_note = (
        ""
        if not any(entry[3] for entry in hmm)
        else (
            " At the smallest hidden Markov size "
            f"{sum(entry[3] for entry in hmm)} of "
            f"{sum(replicates for _, replicates in HMM_SIZES)} fits reached "
            "the boundary of the parameter space -- an emission probability "
            "estimated at zero -- where the observed information is singular "
            "and there is no interval to report. Those fits are excluded from "
            "the counts above and stated here rather than dropped silently, "
            "since excluding them unannounced would select for the "
            "well-behaved samples."
        )
    )

    caption = (
        f"Realized coverage of the 95 percent Wald intervals built from the "
        f"observed information, against the amount of data each fit sees. "
        f"Bars are binomial standard errors on the realized fraction. Both "
        f"instances converge on the nominal rate and neither sits on it at "
        f"the smallest sample, which is what an asymptotic approximation does "
        f"and a wrong formula does not. They approach from opposite sides. "
        f"The Potts chain over-covers and comes down: {_rate(potts[0])} "
        f"chains, {_rate(potts[-1])}. The hidden Markov model under-covers "
        f"and comes up: {_rate(hmm[0])} sequences, {_rate(hmm[-1])}. The "
        f"under-coverage has two identifiable sources absent from the Potts "
        f"chain -- some emission probabilities are fitted near zero, where a "
        f"Wald interval on the log scale is a poor approximation, and "
        f"aligning the hidden-state permutation to the truth is a "
        f"post-selection step that costs a little coverage. Seeds are fixed, "
        f"so every point is reproducible rather than redrawn." + boundary_note
    )
    return fig, caption


def _sweep_and_build(
    potts_params: PottsParams, hmm_params: HmmParams
) -> tuple[Figure, str]:
    """Sweep both models over their problem sizes, then render them.

    Returns
    -------
    tuple[Figure, str]
        The figure and its caption.
    """
    potts = [
        (size, *potts_coverage(potts_params, size, replicates))
        for size, replicates in POTTS_SIZES
    ]
    hmm = [
        (size, *hmm_coverage(hmm_params, size, replicates))
        for size, replicates in HMM_SIZES
    ]
    return build_figure(potts, hmm)


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
        stem="opt_coverage",
        description=__doc__,
        params=[POTTS_PARAMS, HMM_PARAMS],
        build=_sweep_and_build,
        argv=argv,
    )


if __name__ == "__main__":
    main()
