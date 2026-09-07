"""QA figure: parameter recovery for both reference instances of ``Objective``.

`opt/CLAUDE.md` makes recovery the acceptance test for a fit, and issue #63
claims one optimizer serves models that share nothing but a factorized
structure. Both claims are visible in one figure: the same `fit` is run on a
Potts chain and on an HMM, and each fitted parameter is plotted against the
value that generated the data, with the interval the observed information
implies.

Panel (a) is the Potts chain -- a coupling and a gauge-fixed field, in
natural units. Panel (b) is the HMM, in probabilities, after aligning the
hidden-state permutation the likelihood is invariant to. Points on the
diagonal are recovered; the bars say by how much the data constrains each.

Renders what `snakes_and_ladders.opt` computed; it reimplements no model and no optimizer
(`qa/CLAUDE.md`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
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
    ONE_COLUMN_WIDE,
    letter_style,
    series_style,
)
from snakes_and_ladders.sim.hmm import HmmParams, simulate_sequences

# Two-sided normal quantile for the 95% bars drawn here, matching
# snakes_and_ladders.opt.fit.covers.
_Z_95 = 1.959963984540054


def potts_recovery(
    params: PottsParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the Potts fixture and return truth, estimate, error and coverage.

    Parameters
    ----------
    params : PottsParams
        The generating truth.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        True values, fitted values, standard errors, and a boolean array of
        whether each 95% interval covers its truth. Ordered as the coupling
        followed by the field entries.
    """
    objective = PottsObjective(simulate_chains(params), params.n_states)
    result = fit(objective)
    estimate = objective.constrain(result.theta)
    error = constrained_standard_errors(objective, result.theta)

    truth = torch.cat(
        [
            torch.tensor([params.coupling], dtype=torch.float64),
            torch.as_tensor(params.field),
        ]
    )
    fitted = torch.cat([estimate["coupling"].reshape(1), estimate["field"]])
    spread = torch.cat([error["coupling"].reshape(1), error["field"]])
    return (
        truth.numpy(),
        fitted.numpy(),
        spread.numpy(),
        covers(fitted, spread, truth).numpy(),
    )


def hmm_recovery(
    params: HmmParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the HMM fixture and return truth, estimate, error and coverage.

    Reported in probabilities rather than the log-probabilities the objective
    works in, and after aligning the hidden-state permutation: the likelihood
    is invariant to relabelling the states, so an unaligned comparison would
    show a correct fit as a failure.

    Parameters
    ----------
    params : HmmParams
        The generating truth.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        True values, fitted values, standard errors, and a boolean array of
        whether each 95% interval covers its truth, over the initial,
        transition and emission entries in that order.
    """
    objective = HmmObjective(
        simulate_sequences(params).observations, params.n_states, params.n_symbols
    )
    result = fit(objective)
    estimate = objective.constrain(result.theta)
    error = constrained_standard_errors(objective, result.theta)
    order = list(
        align_states(estimate["log_emission"], torch.as_tensor(params.emission))
    )

    truths, fitted, spreads, hits = [], [], [], []
    for name, true_value in (
        ("log_initial", params.initial),
        ("log_transition", params.transition),
        ("log_emission", params.emission),
    ):
        point, deviation = estimate[name], error[name]
        if name == "log_transition":
            point, deviation = point[order][:, order], deviation[order][:, order]
        else:
            point, deviation = point[order], deviation[order]
        reference = torch.log(torch.as_tensor(true_value))
        hits.append(covers(point, deviation, reference).reshape(-1).numpy())
        # The delta method again, from log-probability to probability:
        # d exp(x) = exp(x) dx, so the error scales by the estimate itself.
        probability = torch.exp(point)
        truths.append(np.asarray(true_value).reshape(-1))
        fitted.append(probability.reshape(-1).numpy())
        spreads.append((probability * deviation).reshape(-1).numpy())

    return (
        np.concatenate(truths),
        np.concatenate(fitted),
        np.concatenate(spreads),
        np.concatenate(hits),
    )


def _draw(
    ax: Axes,
    truth: np.ndarray,
    fitted: np.ndarray,
    spread: np.ndarray,
    hits: np.ndarray,
    index: int,
) -> None:
    """Draw one panel, distinguishing intervals that miss their truth.

    Covered and missed points differ by marker fill rather than by colour, so
    the figure's own claim about coverage survives greyscale printing
    (root ``CLAUDE.md``'s figure contract).
    """
    style = series_style(index)
    span = np.array([min(truth.min(), fitted.min()), max(truth.max(), fitted.max())])
    pad = 0.08 * (span[1] - span[0])
    span = span + np.array([-pad, pad])
    ax.plot(span, span, color=INK_MUTED, linestyle=":", linewidth=0.8, zorder=1)
    for covered, face, label in (
        (hits, style["color"], "interval covers truth"),
        (~hits, "white", "interval misses"),
    ):
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
            label=label,
            zorder=2,
        )
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    ax.set_xlabel("generating value")
    ax.set_ylabel("fitted value")


def build_figure(
    potts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    hmm: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    potts_params: PottsParams,
    hmm_params: HmmParams,
) -> tuple[Figure, str]:
    """Assemble the two-panel recovery figure and its caption.

    Parameters
    ----------
    potts, hmm : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Output of :func:`potts_recovery` and :func:`hmm_recovery`.
    potts_params, hmm_params : PottsParams | HmmParams
        The generating truths, for the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption text.
    """
    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)
        _draw(axes[0], *potts, index=0)
        _draw(axes[1], *hmm, index=1)
        axes[0].set_title("(a) Potts chain", loc="left")
        axes[1].set_title("(b) hidden Markov model", loc="left")
        axes[1].set_ylabel("")
        axes[1].legend(loc="upper left", frameon=False, fontsize="small")
        fig.tight_layout()

    potts_rate = float(potts[3].mean())
    hmm_rate = float(hmm[3].mean())
    caption = (
        f"Parameter recovery for the two reference instances of the "
        f"optimization interface, fitted by the same model-agnostic "
        f"optimizer. Bars are 95 percent Wald intervals from the observed "
        f"information, propagated to the reported parameters by the delta "
        f"method; the dotted line is y = x. (a) A {potts_params.n_states}-state "
        f"Potts chain, {potts_params.n_chains} chains of length "
        f"{potts_params.chain_length}, seed {potts_params.seed}: coupling and "
        f"gauge-fixed field, {int(potts[3].sum())} of {potts[3].size} intervals "
        f"covering ({potts_rate:.2f}). (b) A {hmm_params.n_states}-state, "
        f"{hmm_params.n_symbols}-symbol hidden Markov model, "
        f"{hmm_params.n_sequences} sequences of length "
        f"{hmm_params.sequence_length}, seed {hmm_params.seed}: initial, "
        f"transition and emission probabilities after aligning the "
        f"hidden-state permutation the likelihood is invariant to, "
        f"{int(hmm[3].sum())} of {hmm[3].size} intervals covering "
        f"({hmm_rate:.2f}). Coverage on a single dataset is a draw, not a "
        f"rate: the nominal 0.95 is checked over independent replicates by "
        f"the fitting regression suite, and against sample size in the "
        f"companion coverage figure."
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
        stem="opt_recovery",
        description=__doc__,
        params=[POTTS_PARAMS, HMM_PARAMS],
        build=lambda potts_params, hmm_params: build_figure(
            potts_recovery(potts_params),
            hmm_recovery(hmm_params),
            potts_params,
            hmm_params,
        ),
        argv=argv,
    )


if __name__ == "__main__":
    main()
