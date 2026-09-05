"""The optimizer against minimizers known in closed form, not against a model.

Every other test of `fit` measures a statistical property of a likelihood
surface, so an optimizer that stops early and a parameter that is weakly
identified produce the same symptom and nothing separates them. These
functions have analytic minimizers and no model behind them, so a failure
here is the optimizer's.

Three properties, three functions, and the third is the one that changes what
may be claimed elsewhere: on a multimodal surface `converged` means the
first-order condition holds, not that the global minimum was found, and the
measured rate below says how far apart those are.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.testfunctions import (
    HIMMELBLAU_MINIMA,
    Himmelblau,
    Rastrigin,
    Rosenbrock,
)

# The published Himmelblau minima are quoted to six decimals, so no test
# against them can be tighter than that. Rosenbrock's and Rastrigin's are
# exact, and are held to `likelihood/CLAUDE.md`'s float64 bound instead.
PUBLISHED_PRECISION = 1e-5
EXACT = 1e-11


@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_rosenbrock_reaches_its_analytic_minimizer(dimension: int) -> None:
    # The valley is curved and nearly flat along its floor, so a line search
    # that terminates on the wrong condition lands short of `(1, ..., 1)`
    # while still reporting a small gradient.
    objective = Rosenbrock(dimension=dimension)

    result = fit(objective)

    assert result.converged
    assert float(torch.linalg.vector_norm(result.theta - objective.minimizer())) < EXACT
    assert float(result.value) == pytest.approx(0.0, abs=1e-20)


def test_rastrigin_reaches_its_analytic_minimizer_from_inside_the_central_cell() -> (
    None
):
    # Started inside the central well, the ripple is not in the way and the
    # global minimum must be found exactly. This separates "cannot optimize
    # this function" from "cannot find the global basin", which is the
    # distinction the measurement below rests on.
    objective = Rastrigin(dimension=2, start=0.3)

    result = fit(objective)

    assert float(torch.linalg.vector_norm(result.theta - objective.minimizer())) < EXACT


@pytest.mark.parametrize(
    "objective",
    [Rosenbrock(dimension=4), Rastrigin(dimension=3), Himmelblau(start=(0.7, -1.3))],
)
def test_the_autodiff_gradient_matches_the_closed_form(objective: object) -> None:
    # The closed forms are written out in `testfunctions`, not differentiated
    # from the implementation: an error shared between a value and its
    # derivative is exactly what differentiating the implementation hides.
    point = objective.initial().clone().requires_grad_(True)  # type: ignore[attr-defined]
    objective(point).backward()  # type: ignore[operator]

    realized = point.grad
    expected = objective.gradient(objective.initial())  # type: ignore[attr-defined]

    assert realized is not None
    np.testing.assert_allclose(realized.numpy(), expected.numpy(), rtol=EXACT)


@pytest.mark.parametrize(
    ("start", "expected_index"),
    [((1.0, 1.0), 0), ((-3.0, 2.0), 1), ((-3.0, -3.0), 2), ((3.0, -2.0), 3)],
)
def test_himmelblau_converges_to_the_basin_it_started_in(
    start: tuple[float, float], expected_index: int
) -> None:
    result = fit(Himmelblau(start=start))

    index, distance = Himmelblau.nearest_minimum(result.theta)

    assert index == expected_index
    assert distance < PUBLISHED_PRECISION
    assert float(result.value) == pytest.approx(0.0, abs=1e-20)


def test_all_four_himmelblau_minima_are_reachable() -> None:
    # The property a single-minimum function cannot test. All four have value
    # 0, so no ordering distinguishes them and "the" optimum is not a
    # well-formed question; a method that returns the same point from every
    # start is reporting its own initialization.
    starts = [(1.0, 1.0), (-3.0, 2.0), (-3.0, -3.0), (3.0, -2.0)]

    found = {
        Himmelblau.nearest_minimum(fit(Himmelblau(start=s)).theta)[0] for s in starts
    }

    assert found == set(range(len(HIMMELBLAU_MINIMA)))


def test_a_converged_fit_on_rastrigin_is_not_a_global_minimum() -> None:
    # Measured, not asserted as a success: over 40 starts drawn uniformly from
    # the standard domain, a single L-BFGS fit reached the global minimum 0
    # times in 200 at this spread, and 4% when the draw is restricted to
    # +/- 2. What is asserted is the consequence -- that `converged` reports
    # the first-order condition and nothing about global optimality, so any
    # claim built on a single fit of a multimodal surface has to say so.
    rng = np.random.default_rng(20260904)
    objective = Rastrigin(dimension=2)

    converged_but_not_global = 0
    for _ in range(40):
        start = torch.tensor(rng.uniform(-5.12, 5.12, size=2), dtype=torch.float64)
        result = fit(objective, theta0=start)
        distance = float(torch.linalg.vector_norm(result.theta - objective.minimizer()))
        if result.converged and distance > 1e-4:
            converged_but_not_global += 1

    # Overwhelmingly the common outcome, so a loose bound still fails loudly
    # if the surface or the optimizer ever changes character.
    assert converged_but_not_global >= 30


def test_a_one_dimensional_rosenbrock_is_refused() -> None:
    # The function is a sum over adjacent pairs, so one coordinate has no
    # terms at all and the "minimum" would be every point.
    with pytest.raises(ValueError, match="at least two coordinates"):
        Rosenbrock(dimension=1)


@pytest.mark.parametrize("objective", [Rosenbrock(), Rastrigin(), Himmelblau()])
def test_the_value_at_the_stated_minimizer_is_zero(objective: object) -> None:
    # All three are constructed to have value 0 at their minima, which is a
    # property of the functions rather than of any optimizer -- so this fails
    # if a constant or a sign in the implementation is wrong, independently of
    # whether `fit` can find it.
    if isinstance(objective, Himmelblau):
        points = [
            torch.tensor(minimum, dtype=torch.float64) for minimum in HIMMELBLAU_MINIMA
        ]
    else:
        points = [objective.minimizer()]  # type: ignore[attr-defined]

    for point in points:
        assert float(objective(point)) == pytest.approx(0.0, abs=1e-10)  # type: ignore[operator]


@pytest.mark.parametrize(
    "objective",
    [Rosenbrock(dimension=3), Rastrigin(dimension=4), Himmelblau()],
    ids=["Rosenbrock", "Rastrigin", "Himmelblau"],
)
def test_the_test_functions_invert_their_own_constraint_map(
    objective: Objective,
) -> None:
    # These carry no parameter transformation, so the inverse is a repacking
    # rather than arithmetic -- and Himmelblau's is the one that can go wrong,
    # because it splits `theta` into two named scalars and an inverse that
    # stacked them in the other order would still typecheck.
    theta = torch.linspace(-1.5, 2.5, objective.initial().shape[0], dtype=torch.float64)

    recovered = objective.theta_from(objective.constrain(theta))

    assert torch.equal(recovered, theta)
