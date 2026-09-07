"""QA figure: branch-length recovery, and the root pair that cannot be recovered.

Milestone 4 asks for branch lengths fitted by autodiff and validated against
known truth. Panel (a) is that: every estimable branch length of two
fixtures, plotted against the value that generated the alignment, with the
interval the observed information implies.

Panel (b) is why "estimable" is doing work in that sentence. Under a
reversible model the likelihood does not depend on where the root sits along
the branch it subdivides, so on a rooted binary tree the two branches below
the root are confounded -- only their sum can be estimated. The panel moves
mass between them at fixed sum and shows the log-likelihood does not move,
against a control doing the same to two non-root siblings, where it moves a
great deal. Fitting the pair separately leaves the observed information
singular and every interval undefined, which is the reason the objective
merges them.

Renders what `snakes_and_ladders.likelihood` and `snakes_and_ladders.opt` computed; it reimplements no
recursion and no optimizer (`qa/CLAUDE.md`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from snakes_and_ladders.likelihood import pruning_torch
from snakes_and_ladders.likelihood.objective import BranchLengthObjective
from snakes_and_ladders.opt.fit import constrained_standard_errors, covers, fit
from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import ParamsArgument, figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN_WIDE,
    letter_style,
    series_style,
)
from snakes_and_ladders.sim.params import SimulationParams, load_simulation_params
from snakes_and_ladders.sim.simulate import simulate_alignment

# Enough sites for a well-determined estimate without a slow build; the
# fixtures' own 2e5 sites are for the Monte Carlo validation tests.
SITES = 5000

# Splits of a branch pair's total length, for panel (b).
SPLITS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

_Z_95 = 1.959963984540054


def recovery(
    params: SimulationParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one fixture's branch lengths and return truth, estimate, error, coverage.

    Parameters
    ----------
    params : SimulationParams
        The generating truth.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        True values, fitted values, standard errors, and whether each 95%
        interval covers, one entry per *estimable* parameter.
    """
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=SITES
    )
    objective = BranchLengthObjective(
        params.tau, params.k, params.pi, dict(dataset.alignment)
    )
    result = fit(objective)
    estimate = objective.constrain(result.theta)["branch_lengths"]
    error = constrained_standard_errors(objective, result.theta)["branch_lengths"]
    truth = torch.exp(objective.theta_from_truth(params.tau))
    return (
        truth.numpy(),
        estimate.numpy(),
        error.numpy(),
        covers(estimate, error, truth).numpy(),
    )


def split_profile(
    params: SimulationParams, use_root_pair: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Log-likelihood as mass moves between two sibling branches at fixed sum.

    Parameters
    ----------
    params : SimulationParams
        The generating truth; its tree must have a two-child root.
    use_root_pair : bool
        Whether to move mass between the two branches below the root (the
        confounded pair) or between two non-root siblings (the control).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The splits, and the log-likelihood change from the even split.
    """
    dataset = simulate_alignment(
        tau=params.tau, k=params.k, pi=params.pi, seed=params.seed, n_sites=SITES
    )
    alignment = dict(dataset.alignment)
    order = pruning_torch.branch_order(params.tau)
    lengths = pruning_torch.branch_lengths_from_tree(params.tau)

    pair = params.tau.children if use_root_pair else params.tau.children[0].children
    first, second = order.index(pair[0].name), order.index(pair[1].name)
    total = float(lengths[first] + lengths[second])

    scores = []
    for fraction in SPLITS:
        candidate = lengths.clone()
        candidate[first] = total * fraction
        candidate[second] = total * (1.0 - fraction)
        scores.append(
            float(
                pruning_torch.log_likelihood(
                    params.tau, params.k, params.pi, alignment, candidate
                )
            )
        )
    values = np.asarray(scores)
    return np.asarray(SPLITS), values - values[len(SPLITS) // 2]


def build_figure(
    unrooted: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rooted: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    root_profile: tuple[np.ndarray, np.ndarray],
    sibling_profile: tuple[np.ndarray, np.ndarray],
    unrooted_params: SimulationParams,
    rooted_params: SimulationParams,
) -> tuple[Figure, str]:
    """Assemble the two-panel figure and its caption.

    Parameters
    ----------
    unrooted, rooted : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Output of :func:`recovery` for each fixture.
    root_profile, sibling_profile : tuple[np.ndarray, np.ndarray]
        Output of :func:`split_profile`, for the root pair and the control.
    unrooted_params, rooted_params : SimulationParams
        The generating truths, for the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
        The figure and its caption text.
    """
    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)

        span = np.array([0.0, 0.0])
        for index, (data, label) in enumerate(
            ((unrooted, "4 taxa, unrooted"), (rooted, "8 taxa, rooted"))
        ):
            truth, estimate, error, hits = data
            style = series_style(index)
            span = np.array([0.0, max(span[1], truth.max(), estimate.max()) * 1.1])
            for covered, face in ((hits, style["color"]), (~hits, "white")):
                if not covered.any():
                    continue
                axes[0].errorbar(
                    truth[covered],
                    estimate[covered],
                    yerr=_Z_95 * error[covered],
                    linestyle="none",
                    marker=style["marker"],
                    markersize=4,
                    markerfacecolor=face,
                    markeredgecolor=style["color"],
                    color=style["color"],
                    ecolor=style["color"],
                    elinewidth=0.9,
                    capsize=2,
                    label=label if covered is hits else None,
                    zorder=2,
                )
        axes[0].plot(
            span, span, color=INK_MUTED, linestyle=":", linewidth=0.8, zorder=1
        )
        axes[0].set_xlim(*span)
        axes[0].set_ylim(*span)
        axes[0].set_xlabel("generating branch length")
        axes[0].set_ylabel("fitted branch length")
        axes[0].set_title("(a) recovery", loc="left")
        axes[0].legend(loc="upper left", frameon=False, fontsize="small")

        for index, (profile, label) in enumerate(
            (
                (root_profile, "the two root branches"),
                (sibling_profile, "two non-root siblings"),
            )
        ):
            splits, delta = profile
            style = series_style(index + 2)
            axes[1].plot(
                splits,
                delta,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                markersize=4,
                linewidth=1.0,
                label=label,
            )
        axes[1].set_xlabel("share of the pair's total length")
        axes[1].set_ylabel("change in log-likelihood")
        axes[1].set_title("(b) what is estimable", loc="left")
        axes[1].legend(loc="lower center", frameon=False, fontsize="small")
        fig.tight_layout()

    flat = float(np.abs(root_profile[1]).max())
    curved = float(np.abs(sibling_profile[1]).max())
    # On this fixture the change is often exactly zero in float64. "at most
    # 0.0e+00 -- floating-point noise" would say two contradictory things,
    # so an exact zero is reported as one.
    flat_phrase = (
        "not at all: the log-likelihood is bit-identical across a 9 to 1 range"
        if flat == 0.0
        else (f"by at most {flat:.1e} across a 9 to 1 range -- floating-point noise")
    )
    caption = (
        f"Branch-length fitting by autodiff, using the same model-agnostic "
        f"optimizer as the Potts and hidden Markov instances. (a) Every "
        f"estimable branch length against the value that generated the "
        f"alignment, with 95 percent Wald intervals from the observed "
        f"information; markers are open where an interval misses. Fixtures: "
        f"{len(unrooted_params.tau.children)}-child root, seed "
        f"{unrooted_params.seed}, and {len(rooted_params.tau.children)}-child "
        f"root, seed {rooted_params.seed}, both scored at {SITES} sites. "
        f"(b) Why some branch lengths are not estimable at all. Moving mass "
        f"between the two branches below a rooted binary tree's root, at "
        f"fixed sum, changes the log-likelihood {flat_phrase}. "
        f"The same move between two "
        f"non-root siblings changes it by up to {curved:.1f}. This is the "
        f"pulley principle: under a reversible model the likelihood does not "
        f"depend on where the root sits along the branch it subdivides. Only "
        f"the pair's sum is estimable, so the objective fits it as one "
        f"parameter; fitting both would leave the observed information "
        f"singular and every interval in panel (a) undefined."
    )
    return fig, caption


# Two alignments through the same loader, so the flags are declared here
# rather than shared: the figure's claim is about the rooted and unrooted
# fixtures specifically.
UNROOTED_PARAMS = ParamsArgument("unrooted-params", load_simulation_params)
ROOTED_PARAMS = ParamsArgument("rooted-params", load_simulation_params)


def _build_from_params(
    unrooted_params: SimulationParams, rooted_params: SimulationParams
) -> tuple[Figure, str]:
    """Recover both fixtures and profile the rooted split, then render them.

    Returns
    -------
    tuple[Figure, str]
        The figure and its caption.
    """
    return build_figure(
        recovery(unrooted_params),
        recovery(rooted_params),
        split_profile(rooted_params, use_root_pair=True),
        split_profile(rooted_params, use_root_pair=False),
        unrooted_params,
        rooted_params,
    )


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
        stem="opt_branch_recovery",
        description=__doc__,
        params=[UNROOTED_PARAMS, ROOTED_PARAMS],
        build=_build_from_params,
        argv=argv,
    )


if __name__ == "__main__":
    main()
