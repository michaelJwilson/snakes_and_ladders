"""QA figure: the three test functions, and where a multi-start fit lands.

One panel per function: its contours over the standard domain, the
minimizer known in closed form, and the endpoint of every L-BFGS fit from a
set of random restarts. Rosenbrock's one minimum is reached from every
start; Himmelblau's four equal minima are each reached from the basin the
start fell in, so the endpoints say which basin and not "the" optimum; and
on Rastrigin every fit stops at the local minimum of the cell it began in,
each reporting convergence, which is the measurement that bounds what a
single fit of a multimodal surface may claim.

Renders what `snakes_and_ladders.opt.testfunctions` and
`snakes_and_ladders.opt.fit` computed; it reimplements no objective and no
optimizer (`qa/CLAUDE.md`). The textbook cites it as ``fig:optimizer-landscapes``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from snakes_and_ladders.opt.fit import MultiStartResult, fit_from
from snakes_and_ladders.opt.initialize import RandomRestart
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.testfunctions import (
    HIMMELBLAU_MINIMA,
    Himmelblau,
    Rastrigin,
    Rosenbrock,
)
from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import figure_main
from snakes_and_ladders.qa.style import (
    INK,
    ONE_COLUMN_SHORT,
    letter_style,
    series_style,
)

SEED = 20260908

#: Restarts per function and the scale of the Gaussian displacement around
#: each function's own start, in its coordinates.
RESTARTS: dict[str, int] = {"Rosenbrock": 4, "Rastrigin": 16, "Himmelblau": 8}
SCALES: dict[str, float] = {"Rosenbrock": 1.0, "Rastrigin": 2.0, "Himmelblau": 3.0}

#: The domain each function is drawn over: ``(x_min, x_max, y_min, y_max)``.
DOMAINS: dict[str, tuple[float, float, float, float]] = {
    "Rosenbrock": (-2.0, 2.0, -1.0, 3.0),
    "Rastrigin": (-5.12, 5.12, -5.12, 5.12),
    "Himmelblau": (-5.0, 5.0, -5.0, 5.0),
}

#: Grid points per axis for the contours.
GRID = 81

#: A fit within this distance of a closed-form minimizer counts as at it.
AT_MINIMUM = 1e-4


def objectives() -> dict[str, Objective]:
    """The three functions, two-dimensional.

    Returns
    -------
    dict[str, Objective]
    """
    return {
        "Rosenbrock": Rosenbrock(dimension=2),
        # Restarts are drawn around the objective's own start, so Rastrigin's
        # is put at the origin: the question is which cell a start falls in.
        "Rastrigin": Rastrigin(dimension=2, start=0.0),
        "Himmelblau": Himmelblau(),
    }


def known_minimizers(name: str) -> np.ndarray:
    """The closed-form minimizers of the named function, shape ``(m, 2)``.

    Returns
    -------
    np.ndarray
    """
    if name == "Himmelblau":
        return np.array(HIMMELBLAU_MINIMA)
    objective = objectives()[name]
    assert isinstance(objective, Rosenbrock | Rastrigin)
    return objective.minimizer().numpy()[None, :]


def surface(
    objective: Objective, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The objective evaluated on the function's drawing grid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``xs``, ``ys`` of length `GRID`, and ``values`` of shape
        ``(GRID, GRID)`` indexed ``[y, x]``.
    """
    x_min, x_max, y_min, y_max = DOMAINS[name]
    xs = np.linspace(x_min, x_max, GRID)
    ys = np.linspace(y_min, y_max, GRID)
    values = np.array(
        [
            [float(objective(torch.tensor([x, y], dtype=torch.float64))) for x in xs]
            for y in ys
        ]
    )
    return xs, ys, values


def endpoints(seed: int = SEED) -> dict[str, MultiStartResult]:
    """Fit every function from its random restarts.

    Returns
    -------
    dict[str, MultiStartResult]
    """
    generators = np.random.default_rng(seed).spawn(len(RESTARTS))
    return {
        name: fit_from(
            objective,
            RandomRestart(RESTARTS[name], SCALES[name], rng, include_nominal=False),
            workers=1,
        )
        for (name, objective), rng in zip(objectives().items(), generators, strict=True)
    }


def reached(name: str, result: MultiStartResult) -> int:
    """How many of the fits ended within `AT_MINIMUM` of a global minimizer.

    Returns
    -------
    int
    """
    minimizers = known_minimizers(name)
    return sum(
        bool(
            np.min(np.linalg.norm(minimizers - fit.theta.numpy(), axis=1)) < AT_MINIMUM
        )
        for fit in result.all_fits
    )


def build_figure(results: dict[str, MultiStartResult]) -> tuple[Figure, str]:
    """Assemble the three panels and the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
    """
    hits = {name: reached(name, result) for name, result in results.items()}
    with letter_style():
        fig, axes = plt.subplots(1, 3, figsize=ONE_COLUMN_SHORT)
        for axis, (label, (name, objective)) in zip(
            axes, zip("abc", objectives().items(), strict=True), strict=True
        ):
            xs, ys, values = surface(objective, name)
            axis.contour(
                xs,
                ys,
                np.log1p(values),
                levels=12,
                colors=INK,
                linewidths=0.4,
                alpha=0.6,
            )
            axis.grid(False)
            minimizers = known_minimizers(name)
            truth_style = series_style(1)
            axis.plot(
                minimizers[:, 0],
                minimizers[:, 1],
                marker=truth_style["marker"],
                linestyle="none",
                color=truth_style["color"],
                markersize=6,
                label="closed-form minimizer",
            )
            ends = np.array([fit.theta.numpy() for fit in results[name].all_fits])
            fit_style = series_style(0)
            axis.plot(
                ends[:, 0],
                ends[:, 1],
                marker=fit_style["marker"],
                linestyle="none",
                color=fit_style["color"],
                markersize=3,
                label="fit endpoint",
            )
            axis.set_title(f"({label}) {name}", loc="left")
            axis.set_xlabel("x")
            axis.set_aspect("equal")
        axes[0].set_ylabel("y")
        axes[2].legend(loc="lower center", frameon=False, fontsize="x-small")
        fig.tight_layout()

    caption = (
        f"The three continuous test functions in two dimensions, with the "
        f"minimizers known in closed form and the endpoints of L-BFGS fits "
        f"from Gaussian random restarts around each function's own start, "
        f"the origin for Rastrigin (seed {SEED}). (a) Rosenbrock, {RESTARTS['Rosenbrock']} restarts at "
        f"scale {SCALES['Rosenbrock']:g}: {hits['Rosenbrock']} of "
        f"{RESTARTS['Rosenbrock']} reach the minimizer (1, 1). (b) Rastrigin, "
        f"{RESTARTS['Rastrigin']} restarts at scale {SCALES['Rastrigin']:g}: "
        f"{hits['Rastrigin']} of {RESTARTS['Rastrigin']} reach the global "
        f"minimum at the origin, every fit stopping at the local minimum of "
        f"the cell it started in, so a single fit of this surface reports "
        f"convergence and not optimality. (c) Himmelblau, "
        f"{RESTARTS['Himmelblau']} restarts at scale {SCALES['Himmelblau']:g}: "
        f"{hits['Himmelblau']} of {RESTARTS['Himmelblau']} reach one of the "
        f"four equal minima, which one depending on the basin the start fell "
        f"in. Contours are drawn in log(1 + value)."
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
        stem="optimizer_landscapes",
        description=__doc__,
        params=(),
        build=lambda: build_figure(endpoints()),
        argv=argv,
    )


if __name__ == "__main__":
    main()
