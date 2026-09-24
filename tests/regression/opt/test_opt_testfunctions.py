"""The optimizer against minimizers known in closed form, not against a model.

With no model behind them a failure here is the optimizer's, not weak
identification. On a multimodal surface `converged` means the first-order
condition, not the global minimum, and the measured rate says how far apart
those are. Built inline at the dimensions and starts each property needs;
the fixture declares the 2-D instances the figure uses.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.fixtures import load_params
from snakes_and_ladders.opt.fit import fit
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.opt.testfunctions import (
    HIMMELBLAU_MINIMA,
    Himmelblau,
    Rastrigin,
    Rosenbrock,
    TestFunctionSuite,
)
from snakes_and_ladders.sim.fixtures import path_of

from tests._rows import every_row, every_value

# The published Himmelblau minima are quoted to six decimals, so no test
# against them can be tighter than that. Rosenbrock's and Rastrigin's are
# exact, and are held to `likelihood/CLAUDE.md`'s float64 bound instead.
PUBLISHED_PRECISION = 1e-5
EXACT = 1e-11

#: The minimizers written out from the sources `testfunctions` names, so the
#: fixture is refereed by the published values and not by the module constant
#: it was copied from: Rosenbrock's ``(a, ..., a)`` at ``a = 1``, Rastrigin's
#: origin, and Himmelblau (1972)'s four, quoted to six decimals.
CLOSED_FORM_MINIMIZERS = {
    "Rosenbrock": ((1.0, 1.0),),
    "Rastrigin": ((0.0, 0.0),),
    "Himmelblau": (
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ),
}


@pytest.mark.oracle
def test_rosenbrock_reaches_its_analytic_minimizer() -> None:
    # The valley is curved and nearly flat along its floor, so a line search
    # that terminates on the wrong condition lands short of `(1, ..., 1)`
    # while still reporting a small gradient.
    def check(dimension: int) -> None:
        objective = Rosenbrock(dimension=dimension)

        result = fit(objective)

        assert result.converged
        assert (
            float(torch.linalg.vector_norm(result.theta - objective.minimizer()))
            < EXACT
        )
        assert float(result.value) == pytest.approx(0.0, abs=1e-20)

    every_value([2, 3, 5], check)


@pytest.mark.oracle
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


@pytest.mark.analytic
@pytest.mark.oracle
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


@pytest.mark.analytic
def test_himmelblau_converges_to_the_basin_it_started_in() -> None:
    def check(start: tuple[float, float], expected_index: int) -> None:
        result = fit(Himmelblau(start=start))

        index, distance = Himmelblau.nearest_minimum(result.theta)

        assert index == expected_index
        assert distance < PUBLISHED_PRECISION
        assert float(result.value) == pytest.approx(0.0, abs=1e-20)

    every_row(
        [((1.0, 1.0), 0), ((-3.0, 2.0), 1), ((-3.0, -3.0), 2), ((3.0, -2.0), 3)], check
    )


@pytest.mark.oracle
def test_all_four_himmelblau_minima_are_reachable() -> None:
    # The property a single-minimum function cannot test. All four have value
    # 0, so no ordering distinguishes them and "the" optimum is not a
    # well-formed question; a method that returns the same point from every
    # start is reporting its own initialization.
    starts = [(1.0, 1.0), (-3.0, 2.0), (-3.0, -3.0), (3.0, -2.0)]

    found = {
        Himmelblau.nearest_minimum(fit(Himmelblau(start=s)).theta).index for s in starts
    }

    assert found == set(range(len(HIMMELBLAU_MINIMA)))


@pytest.mark.smoke
def test_a_converged_fit_on_rastrigin_is_not_a_global_minimum() -> None:
    # Measured: 0 of 200 fits reached the global minimum at this spread, 4%
    # at +/- 2; asserted: `converged` says nothing about global optimality.
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


@pytest.mark.smoke
def test_a_one_dimensional_rosenbrock_is_refused() -> None:
    # The function is a sum over adjacent pairs, so one coordinate has no
    # terms at all and the "minimum" would be every point.
    with pytest.raises(ValueError, match="at least two coordinates"):
        Rosenbrock(dimension=1)


@pytest.mark.oracle
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


@pytest.mark.oracle
@pytest.mark.critical
def test_the_fixtures_declared_instances_are_what_they_declare() -> None:
    """Every minimizer the fixture declares, against the closed forms, and a
    fit from each declared start against the minimizer it declares.

    Referees: published values, the value and gradient at the point, and a
    positive Hessian (``sec:testfunctions``). Rosenbrock and Rastrigin land
    bitwise (0.0, 0.0); Himmelblau's six-decimal points give value <= 1.10e-11
    (1e-09) and gradient <= 4.19e-05 (1e-03). Smallest eigenvalue 0.40
    (Rosenbrock), Rastrigin 396.78. Each declared start converges to its
    minimizer at distance 0.0 (``at_minimum`` 1e-04).
    """
    suite = load_params(path_of("test_functions", "ci"), TestFunctionSuite)
    declared = suite.named()

    assert set(declared) == set(CLOSED_FORM_MINIMIZERS)
    assert suite.at_minimum > 0.0
    print("\nvalue, gradient norm and smallest Hessian eigenvalue per minimizer:")
    for name, expected in CLOSED_FORM_MINIMIZERS.items():
        params = declared[name]
        objective = params.objective()
        assert params.minimizers == expected

        for minimizer in params.minimizers:
            point = torch.tensor(minimizer, dtype=torch.float64)
            gradient = objective.gradient(point)
            hessian = torch.autograd.functional.hessian(  # type: ignore[no-untyped-call]
                objective.__call__, point
            )
            curvature = float(torch.linalg.eigvalsh(hessian).min())
            print(
                f"  {name} {minimizer}: {float(objective(point)):.2e} "
                f"{float(torch.linalg.vector_norm(gradient)):.2e} {curvature:.2f}"
            )

            assert float(objective(point)) == pytest.approx(0.0, abs=1e-9)
            assert float(torch.linalg.vector_norm(gradient)) < 1e-3
            assert curvature > 0.0

        result = fit(objective)
        distance = min(
            float(
                torch.linalg.vector_norm(
                    result.theta - torch.tensor(minimizer, dtype=torch.float64)
                )
            )
            for minimizer in params.minimizers
        )
        print(f"  {name} from {params.start}: {distance:.2e} to a declared minimizer")

        assert result.converged
        assert distance < suite.at_minimum


@pytest.mark.parametrize(
    "objective",
    [Rosenbrock(dimension=3), Rastrigin(dimension=4), Himmelblau()],
    ids=["Rosenbrock", "Rastrigin", "Himmelblau"],
)
@pytest.mark.smoke
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
