"""QA figure: the Gaussian mixture, its fit, and what seeding buys.

Panel (a) is the five-component fixture the equal-budget comparison of
tempering against restarts was run on: the observations, the generating
density, and the density expectation--maximization reaches from one
k-means++ start. Panel (b) is the seeding guarantee measured: the k-means
cost of a k-means++ seeding and of a uniform seeding, each divided by the
exact optimal cost from the one-dimensional dynamic programme, over
independent seedings of the same data, against the published bound on the
expectation of the first.

Renders what `snakes_and_ladders.sim.mixture` and
`snakes_and_ladders.opt.mixture` computed; it reimplements no density, no
seeding and no M step (`qa/CLAUDE.md`). The textbook cites it as ``fig:mixture-seeding``.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from snakes_and_ladders.emissions import GaussianEmission
from snakes_and_ladders.opt.mixture import (
    GaussianMixtureObjective,
    KMeansPlusPlus,
    MixtureFit,
    clustering_cost,
    expectation_maximization,
    kmeans_plus_plus,
    optimal_clustering_cost,
    seeding_guarantee,
    uniform_seeds,
)
from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import figure_main
from snakes_and_ladders.qa.style import (
    INK_MUTED,
    ONE_COLUMN_WIDE,
    blend_with_white,
    letter_style,
    series_style,
)
from snakes_and_ladders.sim.mixture import MixtureParams, simulate_mixture

#: The five-component fixture of the tempering-against-restarts comparison:
#: adjacent means 1.5 standard deviations apart, unequal weights.
MEANS: tuple[float, ...] = (-3.0, -1.5, 0.0, 1.5, 3.0)
SCALES: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
WEIGHTS: tuple[float, ...] = (0.30, 0.10, 0.25, 0.15, 0.20)
N_SAMPLES = 500
SEED = 20260908

#: Independent seedings of the same observations, per strategy.
SEEDINGS = 200

#: The variance floor the fitting suite declares for this fixture.
VARIANCE_FLOOR = 1e-12

#: The EM cap. Components 1.5 standard deviations apart converge linearly,
#: and the equal-budget comparison measured raw EM still moving at this
#: count; the caption says whether the cap was reached.
EM_ITERATIONS = 500


def fixture() -> MixtureParams:
    """The generating truth, declared once.

    Returns
    -------
    MixtureParams
    """
    return MixtureParams(
        weights=np.array(WEIGHTS),
        components=GaussianEmission(MEANS, SCALES, VARIANCE_FLOOR),
        n_samples=N_SAMPLES,
        seed=SEED,
        tolerance=0.05,
    )


@dataclass(frozen=True)
class SeedingRatios:
    """Seeding cost over the optimal cost, per strategy.

    Parameters
    ----------
    kmeans_plus_plus : np.ndarray
        One ratio per k-means++ seeding, at least 1 each.
    uniform : np.ndarray
        One ratio per uniform seeding.
    optimal : float
        The exact optimal k-means cost the ratios divide by.
    """

    kmeans_plus_plus: np.ndarray
    uniform: np.ndarray
    optimal: float


def seeding_ratios(observations: np.ndarray, seed: int) -> SeedingRatios:
    """Seed the observations `SEEDINGS` times each way and cost every seeding.

    Returns
    -------
    SeedingRatios
    """
    n_centres = len(MEANS)
    optimal = optimal_clustering_cost(observations, n_centres)
    rng = np.random.default_rng(seed)
    plus_plus = np.array(
        [
            clustering_cost(
                observations, kmeans_plus_plus(observations, n_centres, rng)
            )
            / optimal
            for _ in range(SEEDINGS)
        ]
    )
    uniform = np.array(
        [
            clustering_cost(observations, uniform_seeds(observations, n_centres, rng))
            / optimal
            for _ in range(SEEDINGS)
        ]
    )
    return SeedingRatios(kmeans_plus_plus=plus_plus, uniform=uniform, optimal=optimal)


def fitted(observations: np.ndarray, seed: int) -> MixtureFit:
    """Expectation--maximization from one k-means++ start.

    Returns
    -------
    MixtureFit
    """
    objective = GaussianMixtureObjective(observations, len(MEANS))
    start = KMeansPlusPlus(1, np.random.default_rng(seed)).starts(objective)[0]
    named = objective.constrain(start)
    return expectation_maximization(
        observations,
        torch.exp(named["log_weight"]).detach(),
        objective.components(start),
        max_iterations=EM_ITERATIONS,
    )


def density(
    grid: np.ndarray, weights: np.ndarray, components: GaussianEmission
) -> np.ndarray:
    """The mixture density on ``grid``, from the family's own log-density.

    Returns
    -------
    np.ndarray
        Shape ``grid.shape``.
    """
    log_weight = torch.log(torch.as_tensor(weights, dtype=torch.float64))
    values = torch.as_tensor(grid, dtype=torch.float64)
    return (
        torch.logsumexp(log_weight + components.log_density(values), dim=-1)
        .exp()
        .numpy()
    )


def build_figure(
    observations: np.ndarray, fit: MixtureFit, ratios: SeedingRatios
) -> tuple[Figure, str]:
    """Assemble the two panels and the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
    """
    truth = fixture()
    grid = np.linspace(observations.min() - 1.0, observations.max() + 1.0, 400)
    generating = density(grid, truth.weights, truth.components)
    reached = density(grid, fit.weights.numpy(), fit.components)
    bound = seeding_guarantee(len(MEANS))
    with letter_style():
        fig, axes = plt.subplots(1, 2, figsize=ONE_COLUMN_WIDE)
        axes[0].hist(
            observations,
            bins=30,
            density=True,
            color=blend_with_white(series_style(0)["color"], 0.35),
            edgecolor="none",
            label="observations",
        )
        for index, (values, label) in enumerate(
            ((generating, "generating density"), (reached, "EM from k-means++"))
        ):
            style = series_style(index + 1)
            axes[0].plot(
                grid,
                values,
                linestyle=style["linestyle"],
                color=style["color"],
                label=label,
            )
        axes[0].set_xlabel("observation")
        axes[0].set_ylabel("density")
        axes[0].set_title("(a) the fixture and its fit", loc="left")
        axes[0].legend(loc="upper right", frameon=False, fontsize="small")

        edges = np.linspace(1.0, max(ratios.uniform.max(), bound) + 1.0, 30)
        for index, (values, label) in enumerate(
            ((ratios.kmeans_plus_plus, "k-means++"), (ratios.uniform, "uniform"))
        ):
            style = series_style(index)
            axes[1].hist(
                values,
                bins=edges,
                histtype="step",
                color=style["color"],
                linestyle=style["linestyle"],
                label=label,
            )
        axes[1].axvline(bound, color=INK_MUTED, linestyle=":", linewidth=0.8)
        axes[1].annotate(
            "8 (ln k + 2)",
            xy=(bound, 0.95),
            xycoords=("data", "axes fraction"),
            ha="right",
            color=INK_MUTED,
            fontsize="small",
        )
        axes[1].set_xlabel("seeding cost / optimal cost")
        axes[1].set_ylabel("seedings")
        axes[1].set_title("(b) the seeding guarantee", loc="left")
        axes[1].legend(loc="center right", frameon=False, fontsize="small")
        fig.tight_layout()

    stopped = (
        f"at the {EM_ITERATIONS}-iteration cap, still moving"
        if fit.iterations >= EM_ITERATIONS
        else f"in {fit.iterations} iterations"
    )
    caption = (
        f"A Gaussian mixture of {len(MEANS)} components with means "
        f"{', '.join(f'{mean:g}' for mean in MEANS)}, unit scales and weights "
        f"{', '.join(f'{weight:g}' for weight in WEIGHTS)}: {N_SAMPLES} "
        f"observations at seed {SEED}. (a) The observations, the generating "
        f"density, and the density expectation-maximization reaches from one "
        f"k-means++ start {stopped}, at log-likelihood "
        f"{fit.log_likelihood:.1f}. (b) The k-means cost of a seeding divided "
        f"by the exact optimal cost from the one-dimensional dynamic "
        f"programme, over {SEEDINGS} independent seedings of the same "
        f"observations each way: k-means++ averages "
        f"{ratios.kmeans_plus_plus.mean():.2f} (worst "
        f"{ratios.kmeans_plus_plus.max():.2f}) and uniform seeding "
        f"{ratios.uniform.mean():.2f} (worst {ratios.uniform.max():.2f}), "
        f"against the published bound of {bound:.2f} on the k-means++ "
        f"expectation. The bound is a statement about the mean, and the mean "
        f"is what is held to it."
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

    def build() -> tuple[Figure, str]:
        observations = simulate_mixture(fixture()).observations
        return build_figure(
            observations, fitted(observations, SEED), seeding_ratios(observations, SEED)
        )

    return figure_main(
        stem="mixture_seeding",
        description=__doc__,
        params=(),
        build=build,
        argv=argv,
    )


if __name__ == "__main__":
    main()
