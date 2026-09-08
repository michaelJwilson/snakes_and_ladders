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
    TestFunctionParams,
    TestFunctionSuite,
)
from snakes_and_ladders.qa.figure import QAFigure
from snakes_and_ladders.qa.runner import TEST_FUNCTION_PARAMS, figure_main
from snakes_and_ladders.qa.style import (
    INK,
    ONE_COLUMN_SHORT,
    letter_style,
    series_style,
)


def objectives(suite: TestFunctionSuite) -> dict[str, Objective]:
    """The declared functions, in the fixture's order.

    Returns
    -------
    dict[str, Objective]
    """
    return {function.name: function.objective() for function in suite.functions}


def known_minimizers(function: TestFunctionParams) -> np.ndarray:
    """The published minimizers of one function, shape ``(m, dimension)``.

    Returns
    -------
    np.ndarray
    """
    return np.array(function.minimizers)


def surface(
    objective: Objective, function: TestFunctionParams, grid: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The objective evaluated on the function's declared drawing grid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``xs``, ``ys`` of length ``grid``, and ``values`` of shape
        ``(grid, grid)`` indexed ``[y, x]``.
    """
    x_min, x_max, y_min, y_max = function.domain
    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)
    values = np.array(
        [
            [float(objective(torch.tensor([x, y], dtype=torch.float64))) for x in xs]
            for y in ys
        ]
    )
    return xs, ys, values


def endpoints(
    suite: TestFunctionSuite, rng: np.random.Generator
) -> dict[str, MultiStartResult]:
    """Fit every declared function from its declared random restarts.

    Parameters
    ----------
    suite : TestFunctionSuite
        The functions and their restart counts and scales.
    rng : np.random.Generator
        Spawns one generator per function, in a fixed order; passed in
        rather than seeded here (``sim/CLAUDE.md``).

    Returns
    -------
    dict[str, MultiStartResult]
    """
    generators = rng.spawn(len(suite.functions))
    return {
        function.name: fit_from(
            function.objective(),
            RandomRestart(
                function.restarts, function.scale, rng, include_nominal=False
            ),
            workers=1,
        )
        for function, rng in zip(suite.functions, generators, strict=True)
    }


def reached(
    function: TestFunctionParams, result: MultiStartResult, at_minimum: float
) -> int:
    """How many of the fits ended within ``at_minimum`` of a global minimizer.

    Returns
    -------
    int
    """
    minimizers = known_minimizers(function)
    return sum(
        bool(
            np.min(np.linalg.norm(minimizers - fit.theta.numpy(), axis=1)) < at_minimum
        )
        for fit in result.all_fits
    )


def build_figure(
    suite: TestFunctionSuite, results: dict[str, MultiStartResult]
) -> tuple[Figure, str]:
    """Assemble the three panels and the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, str]
    """
    declared = suite.named()
    hits = {
        name: reached(declared[name], result, suite.at_minimum)
        for name, result in results.items()
    }
    restarts = {name: function.restarts for name, function in declared.items()}
    scales = {name: function.scale for name, function in declared.items()}
    with letter_style():
        fig, axes = plt.subplots(1, 3, figsize=ONE_COLUMN_SHORT)
        for axis, (label, (name, objective)) in zip(
            axes, zip("abc", objectives(suite).items(), strict=True), strict=True
        ):
            xs, ys, values = surface(objective, declared[name], suite.grid)
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
            minimizers = known_minimizers(declared[name])
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
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=2, frameon=False, fontsize="small"
        )
        fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))

    caption = (
        f"The three continuous test functions in two dimensions, with the "
        f"minimizers known in closed form and the endpoints of L-BFGS fits "
        f"from Gaussian random restarts around each function's own start, "
        f"the origin for Rastrigin (seed {suite.seed}). (a) Rosenbrock, {restarts['Rosenbrock']} restarts at "
        f"scale {scales['Rosenbrock']:g}: {hits['Rosenbrock']} of "
        f"{restarts['Rosenbrock']} reach the minimizer (1, 1). (b) Rastrigin, "
        f"{restarts['Rastrigin']} restarts at scale {scales['Rastrigin']:g}: "
        f"{hits['Rastrigin']} of {restarts['Rastrigin']} reach the global "
        f"minimum at the origin, every fit stopping at the local minimum of "
        f"the cell it started in, so a single fit of this surface reports "
        f"convergence and not optimality. (c) Himmelblau, "
        f"{restarts['Himmelblau']} restarts at scale {scales['Himmelblau']:g}: "
        f"{hits['Himmelblau']} of {restarts['Himmelblau']} reach one of the "
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

    def build(suite: TestFunctionSuite) -> tuple[Figure, str]:
        return build_figure(suite, endpoints(suite, np.random.default_rng(suite.seed)))

    return figure_main(
        stem="optimizer_landscapes",
        description=__doc__,
        params=(TEST_FUNCTION_PARAMS,),
        build=build,
        argv=argv,
    )


if __name__ == "__main__":
    main()
