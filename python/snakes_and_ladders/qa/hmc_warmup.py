"""QA figure: what a Hamiltonian Monte Carlo warm-up buys, and what it costs.

The warm-up of ``snakes_and_ladders.opt.hmc.Adaptation`` discards every draw
it takes, so the question it raises is whether the draws that follow repay
them. This figure answers it on the one target whose answer is known in closed
form: a correlated Gaussian, whose mean and covariance are declared rather
than estimated.

Two chains run from the same start, at the same starting step size, over the
same number of recorded draws. One adapts its step size and mass diagonal over
a warm-up and then runs at fixed values; the other runs at the starting step
and unit mass throughout. Both are kernels with the right stationary
distribution --- the comparison is of cost per effective draw and of nothing
else, which is why the caption reports both chains' agreement with the closed
form alongside their convergence.

Renders what `snakes_and_ladders.opt.hmc` sampled; it reimplements no sampler
(`qa/CLAUDE.md`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from snakes_and_ladders.opt.hmc import (
    Adaptation,
    HmcChain,
    effective_sample_size,
    sample,
)
from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN_WIDE,
    letter_style,
    series_style,
)

#: The target's mean, declared. The figure's horizontal reference lines are
#: these numbers and not an estimate of them.
MEAN: tuple[float, ...] = (20.0, -2.0)

#: The target's covariance. Its two eigenvalues differ by a factor of roughly
#: 500, which is the whole point: at unit mass one step size serves both
#: directions, and no value serves either well. A step above the stability
#: limit of the narrow direction diverges; one below it crosses the wide
#: direction by a random walk. The mass diagonal a warm-up learns rescales
#: the coordinates and removes the choice.
COVARIANCE: tuple[tuple[float, ...], ...] = ((100.0, 2.5), (2.5, 0.25))

#: The coordinate the running mean is drawn for: the wide one, which is the
#: one a step small enough to be stable on the narrow direction crosses
#: slowly. The start sits two of its standard deviations from the mean, so a
#: chain that does not travel cannot hide it.
TRACKED = 0

#: Recorded draws per chain, equal for both so the panels share an x-axis.
N_SAMPLES = 1200

#: Proposals the adapted chain spends before recording. Charged to it in the
#: cost panel, and discarded.
WARMUP = 400

#: The step both chains start from, and the one the fixed chain keeps. Below
#: the narrow direction's stability limit, so the fixed chain is a correct
#: sampler and not a stuck one --- the comparison is of mixing, which is the
#: situation adaptation exists for, and not of a divergence.
STEP_SIZE = 0.6

#: Leapfrog steps per proposal, the same for both chains, so a proposal costs
#: the same in each and the gradient counts compare directly.
N_STEPS = 12

#: The stream both chains are drawn from. One seed, stated in the caption;
#: `sim/CLAUDE.md` requires the generator be constructed at the call site.
SEED = 20260914


@dataclass(frozen=True)
class CorrelatedGaussian:
    """``-log N(mean, covariance)`` up to a constant, over an unconstrained vector.

    The analytic target: its mean and covariance are the declared constants
    above, so the figure compares a chain against truth rather than against a
    second chain. Pinned to ``tests._objective_checks.AnalyticGaussian``, the
    same target the sampler's own regression tests use, by
    ``tests/regression/qa/test_qa_hmc_warmup.py``.

    Parameters
    ----------
    mean : torch.Tensor
        The target's mean, 1-D.
    covariance : torch.Tensor
        The target's covariance, symmetric positive definite.
    """

    mean: torch.Tensor
    covariance: torch.Tensor

    @property
    def precision(self) -> torch.Tensor:
        """The inverse covariance, which is also this objective's Hessian."""
        inverse: torch.Tensor = torch.linalg.inv(self.covariance)
        return inverse

    def initial(self) -> torch.Tensor:
        """A start away from the mode, identical for both chains."""
        return torch.zeros_like(self.mean)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """The named parameters; the coordinates are already unconstrained."""
        return {"x": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector behind ``named``."""
        return named["x"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """The negative log density at ``theta``, up to its constant."""
        deviation = theta - self.mean
        quadratic: torch.Tensor = 0.5 * deviation @ self.precision @ deviation
        return quadratic


def target() -> CorrelatedGaussian:
    """The declared target.

    Returns
    -------
    CorrelatedGaussian
        The correlated Gaussian of ``MEAN`` and ``COVARIANCE``, in float64.
    """
    return CorrelatedGaussian(
        mean=torch.tensor(MEAN, dtype=torch.float64),
        covariance=torch.tensor(COVARIANCE, dtype=torch.float64),
    )


def chains() -> tuple[HmcChain, HmcChain]:
    """Draw the adapted chain and the fixed-parameter chain.

    Both start at ``objective.initial()`` and record ``N_SAMPLES`` draws at
    ``N_STEPS`` leapfrog steps, so a proposal costs the same in each. Each
    takes its own generator seeded identically, so the difference between the
    two is adaptation and not the random stream.

    Returns
    -------
    tuple[HmcChain, HmcChain]
        The adapted chain, then the fixed-parameter chain.
    """
    objective = target()
    adapted = sample(
        objective,
        torch.Generator().manual_seed(SEED),
        N_SAMPLES,
        step_size=STEP_SIZE,
        n_steps=N_STEPS,
        adaptation=Adaptation(warmup=WARMUP, target_acceptance=0.65, step_jitter=0.2),
    )
    fixed = sample(
        objective,
        torch.Generator().manual_seed(SEED),
        N_SAMPLES,
        step_size=STEP_SIZE,
        n_steps=N_STEPS,
    )
    return adapted, fixed


def running_mean(chain: HmcChain, coordinate: int) -> np.ndarray:
    """The running mean of one coordinate, draw by draw.

    Parameters
    ----------
    chain : HmcChain
        A drawn chain.
    coordinate : int
        Which coordinate to average.

    Returns
    -------
    np.ndarray
        ``mean(theta[:t, coordinate])`` for every ``t``, 1-D of length
        ``n_samples``.
    """
    draws: np.ndarray = chain.theta[:, coordinate].detach().numpy()
    counts = np.arange(1, draws.size + 1)
    trace: np.ndarray = np.cumsum(draws) / counts
    return trace


#: How close to the exact mean counts as arrived, in exact standard
#: deviations. A tenth is well inside any interval a chain of this length
#: could report, and well outside the Monte Carlo noise of the last draws.
SETTLED = 0.1


def settling_draw(trace: np.ndarray, exact_mean: float, exact_sd: float) -> int | None:
    """The first draw after which the running mean stays within ``SETTLED``.

    "Stays": the last excursion outside the band decides it, not the first
    entry into it, so a chain that wanders back out is not credited with the
    draw it first touched.

    Parameters
    ----------
    trace : np.ndarray
        The running mean, draw by draw.
    exact_mean : float
        The declared mean of the tracked coordinate.
    exact_sd : float
        Its declared standard deviation, the unit the tolerance is in.

    Returns
    -------
    int | None
        A 1-based draw index, or ``None`` if the run ends outside the band.
    """
    outside = np.flatnonzero(np.abs(trace - exact_mean) / exact_sd > SETTLED)
    if outside.size == 0:
        return 1
    if outside[-1] + 1 >= trace.size:
        return None
    return int(outside[-1]) + 2


def _ess_per_gradient(chain: HmcChain) -> float:
    """Effective draws per gradient evaluation, the smallest over coordinates.

    The warm-up's gradients are inside ``force_evaluations``, so this charges
    the adapted chain for them rather than comparing at equal recorded draws.
    """
    sizes: np.ndarray = effective_sample_size(chain.theta).detach().numpy()
    return float(sizes.min()) / chain.force_evaluations


def build_figure(adapted: HmcChain, fixed: HmcChain) -> tuple[Figure, str]:
    """Draw the two panels and write the caption.

    Parameters
    ----------
    adapted : HmcChain
        The chain that warmed up.
    fixed : HmcChain
        The chain at the starting step and unit mass.

    Returns
    -------
    tuple[Figure, str]
        The figure, and its caption.
    """
    objective = target()
    exact_mean = float(objective.mean[TRACKED])
    exact_sd = float(torch.sqrt(objective.covariance[TRACKED, TRACKED]))

    series = {"with warm-up": adapted, "no warm-up": fixed}
    means = {name: running_mean(chain, TRACKED) for name, chain in series.items()}
    draws = np.arange(1, N_SAMPLES + 1)

    # Stacked vertically rather than side by side: the review of the textbook
    # asked for it, and the two panels share the draw axis.
    with letter_style():
        fig, (upper, lower) = plt.subplots(2, 1, figsize=ONE_COLUMN_WIDE, sharex=True)
        for index, (name, trace) in enumerate(means.items()):
            style = series_style(index)
            upper.plot(
                draws,
                trace,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.1,
                label=name,
            )
            lower.plot(
                draws,
                np.abs(trace - exact_mean) / exact_sd,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.1,
                label=name,
            )
        # The reference goes in the legend rather than an annotation: both
        # traces sit on it for most of the run, so there is nowhere beside it
        # to put a label that does not land on a curve.
        upper.axhline(
            exact_mean,
            color=INK_MUTED,
            linestyle=":",
            linewidth=0.8,
            zorder=1,
            label="exact mean",
        )
        upper.set_ylabel(f"running mean of $x_{TRACKED}$")
        upper.legend(loc="lower right", frameon=False, fontsize="small")
        lower.set_yscale("log")
        lower.set_xlabel("recorded draws")
        lower.set_ylabel("error, in exact s.d.")
        fig.tight_layout()

    final = {
        name: abs(trace[-1] - exact_mean) / exact_sd for name, trace in means.items()
    }
    arrival = {
        name: (
            f"draw {draw}"
            if (draw := settling_draw(trace, exact_mean, exact_sd)) is not None
            else "not within the run"
        )
        for name, trace in means.items()
    }
    adapted_state = adapted.adapted
    assert adapted_state is not None, "the adapted chain was drawn with an Adaptation"
    mass = adapted_state.mass_diagonal.detach().numpy()
    eigenvalues = np.linalg.eigvalsh(np.asarray(COVARIANCE))

    caption = (
        "What a warm-up buys, on the target whose answer is closed form. Two "
        f"chains of {N_SAMPLES} draws at {N_STEPS} leapfrog steps run from the "
        f"same start on a 2-dimensional Gaussian of mean {MEAN} and covariance "
        f"{COVARIANCE}, whose eigenvalues differ by a factor of "
        f"{eigenvalues.max() / eigenvalues.min():.0f}. One adapts its step size "
        f"and mass diagonal over {WARMUP} discarded proposals and then runs at "
        f"fixed values; the other keeps the starting step of {STEP_SIZE} and "
        "unit mass throughout. Upper: the running mean of the wider "
        "coordinate, the one a step small enough to be stable on the narrower "
        "direction crosses by a random walk, against its exact value of "
        f"{exact_mean:g}; the start sits two of its standard deviations away. "
        "Lower: the same error in units of that coordinate's exact standard "
        f"deviation of {exact_sd:g}. The running mean reaches the final "
        f"{SETTLED:g} standard deviations and stays there at "
        f"{arrival['with warm-up']} with the warm-up and "
        f"{arrival['no warm-up']} without; at the last draw the error is "
        f"{final['with warm-up']:.3f} and {final['no warm-up']:.3f}. "
        "The warm-up "
        f"settled on a step of {adapted_state.step_size:.3f} and a mass "
        f"diagonal of ({mass[0]:.3f}, {mass[1]:.3f}), accepting "
        f"{adapted_state.warmup_acceptance:.2f} over the warm-up against a "
        f"target of 0.65. Both chains are correct: the acceptance is "
        f"{adapted.acceptance_rate:.2f} adapted and "
        f"{fixed.acceptance_rate:.2f} fixed, and each is a kernel with the "
        "right stationary distribution, so what separates them is cost per "
        "effective draw --- "
        f"{_ess_per_gradient(adapted):.2e} effective draws per gradient with "
        f"the warm-up, its own gradients charged to it, against "
        f"{_ess_per_gradient(fixed):.2e} without. "
        f"Seed {SEED}."
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
        stem="hmc_warmup",
        description=__doc__,
        params=[],
        build=lambda: build_figure(*chains()),
        argv=argv,
    )


if __name__ == "__main__":
    main()
