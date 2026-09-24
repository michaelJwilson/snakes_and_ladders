"""Standard continuous test functions, as :class:`~snakes_and_ladders.opt.objective.Objective`.

Every other test of :func:`snakes_and_ladders.opt.fit.fit` measures a
*statistical* property of a likelihood surface, and there the two failure
modes are confounded: a fit that lands away from the truth reads as a weakly
identified parameter, and an optimizer that stops early produces the same
symptom.

These three functions separate them: their minimizers are known in closed form
(``sec:testfunctions``). Each targets a different failure:

* :class:`Rosenbrock` --- a narrow curved valley, where a wrong line search
  shows up as slowness rather than as a wrong answer.
* :class:`Rastrigin` --- a global bowl under roughly ``10 ** n`` local
  minima, which is the honest test of what one fit from one start can claim.
* :class:`Himmelblau` --- four *equal* global minima, which catches a method
  that reports "the" optimum without saying which basin it found.

**These are not likelihoods**, so the Hessian at the optimum is not an
observed information matrix and
:func:`snakes_and_ladders.opt.fit.constrained_standard_errors` must not be
called on them (issue #122 covers the general case). :meth:`constrain`
therefore returns the point itself, under the name each function's minimizer
is stated in.

Sources: Rosenbrock (1960); Rastrigin (1974); Himmelblau (1972). Nocedal &
Wright use the first as the standing example for quasi-Newton methods.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import torch

from snakes_and_ladders.opt.objective import Objective


@dataclass(frozen=True)
class Rosenbrock(Objective):
    """``sum_i b (x_{i+1} - x_i^2)^2 + (a - x_i)^2``, minimized at ``(a, ..., a)`` (``eq:rosenbrock``).

    The valley is curved and its floor is nearly flat, so the gradient points
    across it and steepest descent zig-zags. The difficulty is entirely
    conditioning --- the function is smooth, unimodal in the relevant region,
    and has an analytic minimizer --- which is what makes it the standing
    quasi-Newton example.

    Parameters
    ----------
    dimension : int
        Number of coordinates, ``>= 2``.
    a, b : float
        The standard constants. ``b = 100`` is what makes the valley narrow.
    start : float
        Every coordinate of the starting point.
    """

    dimension: int = 2
    a: float = 1.0
    b: float = 100.0
    start: float = -1.2

    def __post_init__(self) -> None:
        if self.dimension < 2:
            msg = f"Rosenbrock needs at least two coordinates, got {self.dimension}"
            raise ValueError(msg)

    def initial(self) -> torch.Tensor:
        return torch.full((self.dimension,), self.start, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return named["x"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        head, tail = theta[:-1], theta[1:]
        return (self.b * (tail - head**2) ** 2 + (self.a - head) ** 2).sum()

    @property
    def rosenbrock_constants(self) -> tuple[float, float]:
        """``(a, b)``: declares this a compiled chain's family (issue #1006)."""
        return (self.a, self.b)

    def minimizer(self) -> torch.Tensor:
        """The analytic minimizer, ``(a, ..., a)``, where the value is 0."""
        return torch.full((self.dimension,), self.a, dtype=torch.float64)

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        """The closed-form gradient, for checking autodiff against.

        Written out rather than differentiated, so it is an independent
        reference: an error shared between the value and its derivative is
        exactly what differentiating the implementation would hide.
        """
        grad = torch.zeros_like(theta)
        head, tail = theta[:-1], theta[1:]
        residual = tail - head**2
        grad[:-1] += -4.0 * self.b * residual * head - 2.0 * (self.a - head)
        grad[1:] += 2.0 * self.b * residual
        return grad


@dataclass(frozen=True)
class Rastrigin(Objective):
    """``10 n + sum_i (x_i^2 - 10 cos(2 pi x_i))``, minimized at the origin (``eq:rastrigin``).

    A quadratic bowl with a cosine ripple: roughly ``10 ** n`` local minima,
    one per lattice cell, each satisfying the first-order condition. A single
    fit from a single start lands in whichever cell it began in, which is why
    the test built on this reports a *success rate*.
    """

    dimension: int = 2
    amplitude: float = 10.0
    start: float = 4.4

    def initial(self) -> torch.Tensor:
        return torch.full((self.dimension,), self.start, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return named["x"]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        ripple = theta**2 - self.amplitude * torch.cos(2.0 * math.pi * theta)
        return self.amplitude * theta.shape[0] + ripple.sum()

    def minimizer(self) -> torch.Tensor:
        """The analytic global minimizer, the origin, where the value is 0."""
        return torch.zeros(self.dimension, dtype=torch.float64)

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        """The closed-form gradient."""
        return 2.0 * theta + 2.0 * math.pi * self.amplitude * torch.sin(
            2.0 * math.pi * theta
        )


# Himmelblau's four global minima, each with value 0. The first is exact; the
# rest are the standard published values, quoted to the precision they are
# usually given to, which is what the tolerance below is set from.
HIMMELBLAU_MINIMA = (
    (3.0, 2.0),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126),
)


@dataclass(frozen=True)
class NearestMinimum:
    """Which minimizer a point landed on, and how far off it is.

    Parameters
    ----------
    index : int
        Index into :data:`HIMMELBLAU_MINIMA`. Which of the four a start
        reaches is the fact the basin tests read, so it is named rather
        than discarded by every caller that wants only the distance.
    distance : float
        The Euclidean distance to that minimizer. Zero exactly at one.
    """

    index: int
    distance: float

    def __iter__(self) -> Iterator[Any]:
        """``(index, distance)``: the order callers unpack.

        ``Any`` and not a union: an unpacking gives both names the element
        type, so a union would mistype each of them.
        """
        yield from (self.index, self.distance)


@dataclass(frozen=True)
class Himmelblau(Objective):
    """``(x^2 + y - 11)^2 + (x + y^2 - 7)^2``, with four equal global minima (``eq:himmelblau``).

    Two dimensions only. The four minima all have value 0, so "the" optimum
    is not a well-formed question, and a method that always returns the same
    one is reading its own initialization.
    """

    start: tuple[float, float] = (0.0, 0.0)

    def initial(self) -> torch.Tensor:
        return torch.tensor(self.start, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta[0], "y": theta[1]}

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([named["x"], named["y"]])

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        first, second = theta[0], theta[1]
        return (first**2 + second - 11.0) ** 2 + (first + second**2 - 7.0) ** 2

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        """The closed-form gradient."""
        first, second = theta[0], theta[1]
        outer, inner = first**2 + second - 11.0, first + second**2 - 7.0
        return torch.stack(
            [4.0 * first * outer + 2.0 * inner, 2.0 * outer + 4.0 * second * inner]
        )

    @staticmethod
    def nearest_minimum(point: torch.Tensor) -> NearestMinimum:
        """Which of the four minima ``point`` landed on, and how far off it is.

        Returns
        -------
        NearestMinimum
            Index into :data:`HIMMELBLAU_MINIMA`, and the Euclidean distance.
        """
        distances = [
            float(torch.linalg.vector_norm(point - torch.tensor(minimum)))
            for minimum in HIMMELBLAU_MINIMA
        ]
        best = min(range(len(distances)), key=distances.__getitem__)
        return NearestMinimum(best, distances[best])


@dataclass(frozen=True)
class TestFunctionMetrics:
    """What a point on a test function means: its value, and how far it is from a minimizer (issue #778).

    A :class:`snakes_and_ladders.track.Metrics` over ``theta``, satisfied
    structurally. These are the only three objectives in the package whose
    minimizer is known in closed form, which is what makes the second metric
    a fact rather than an estimate.

    * ``value`` is the objective itself, ``objective(theta)``.
    * ``distance_to_minimizer`` is the Euclidean distance
      :meth:`Himmelblau.nearest_minimum` returns --- to the nearest of the
      four minima for Himmelblau, and to the single minimizer
      :meth:`Rosenbrock.minimizer` or :meth:`Rastrigin.minimizer` returns
      otherwise. Zero exactly at a minimizer, which is what
      ``TestFunctionSuite.at_minimum`` judges a fit against.

    Parameters
    ----------
    objective : Rosenbrock | Rastrigin | Himmelblau
        The function the point is on.
    """

    objective: Rosenbrock | Rastrigin | Himmelblau

    names: tuple[str, ...] = ("value", "distance_to_minimizer")

    def __call__(self, state: torch.Tensor) -> dict[str, float]:
        """The value at ``state`` and its distance to the nearest known minimizer."""
        with torch.no_grad():
            if isinstance(self.objective, Himmelblau):
                distance = Himmelblau.nearest_minimum(state).distance
            else:
                distance = float(
                    torch.linalg.vector_norm(state - self.objective.minimizer())
                )
            return {
                "value": float(self.objective(state)),
                "distance_to_minimizer": distance,
            }


_REQUIRED_FIELDS = frozenset({"seed", "grid", "at_minimum", "functions"})
_FUNCTION_FIELDS = frozenset(
    {"name", "dimension", "start", "restarts", "scale", "domain", "minimizers"}
)

#: The three functions this module defines, by the name a fixture states.
_FUNCTIONS = ("Rosenbrock", "Rastrigin", "Himmelblau")


@dataclass(frozen=True)
class TestFunctionParams:
    """One test function as a declared instance: where it starts and where it ends.

    Parameters
    ----------
    name : str
        One of :data:`_FUNCTIONS`.
    dimension : int
        Coordinates. Himmelblau is two-dimensional by definition and the
        loader refuses any other value for it.
    start : float
        Every coordinate of the nominal start.
    restarts : int
        Random restarts a multi-start fit of this function draws.
    scale : float
        Standard deviation of the displacement each restart is drawn at, in
        the function's own coordinates.
    domain : tuple[float, float, float, float]
        ``(x_min, x_max, y_min, y_max)``, the region a surface of it is drawn
        over.
    minimizers : tuple[tuple[float, ...], ...]
        The published global minimizers, the oracle: quoted from the sources
        the module docstring names, never computed here.
    """

    name: str
    dimension: int
    start: float
    restarts: int
    scale: float
    domain: tuple[float, float, float, float]
    minimizers: tuple[tuple[float, ...], ...]

    def objective(self) -> Rosenbrock | Rastrigin | Himmelblau:
        """The function itself, at this instance's dimension and start.

        Returns
        -------
        Rosenbrock | Rastrigin | Himmelblau

        Raises
        ------
        ValueError
            If the name is not one of the three.
        """
        if self.name == "Rosenbrock":
            return Rosenbrock(dimension=self.dimension, start=self.start)
        if self.name == "Rastrigin":
            return Rastrigin(dimension=self.dimension, start=self.start)
        if self.name == "Himmelblau":
            return Himmelblau(start=(self.start, self.start))
        msg = f"unknown test function {self.name!r}, expected one of {_FUNCTIONS}"
        raise ValueError(msg)


@dataclass(frozen=True)
class TestFunctionSuite:
    """The functions a fixture declares, and the knobs a study of them shares.

    Parameters
    ----------
    functions : tuple[TestFunctionParams, ...]
        In the file's order, which is the order a figure lays them out in.
    seed : int
        Seed a study of this suite builds its generator from.
    grid : int
        Points per axis when a surface is drawn.
    at_minimum : float
        Distance within which a fit counts as having reached a minimizer.
    """

    functions: tuple[TestFunctionParams, ...]
    seed: int
    grid: int
    at_minimum: float

    def named(self) -> dict[str, TestFunctionParams]:
        """The functions by name, in the file's order.

        Returns
        -------
        dict[str, TestFunctionParams]
        """
        return {function.name: function for function in self.functions}

    #: The fields :func:`snakes_and_ladders.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the truth from a continuous-test-function fixture's declared mapping.

        ``declared`` is the mapping
        :func:`snakes_and_ladders.fixtures.load_params` read from ``path``
        with :attr:`required_fields` present; ``path`` names the file in
        every error.

        The parsed suite.

        Raises
        ------
        ValueError
            If a required field is missing, a function names one this module does
            not define, or a declared minimizer does not have the function's
            dimension.
        """
        functions: list[TestFunctionParams] = []
        for entry in declared["functions"]:
            missing = _FUNCTION_FIELDS - set(entry)
            if missing:
                msg = f"{path}: a function lacks {sorted(missing)}"
                raise ValueError(msg)
            name = str(entry["name"])
            if name not in _FUNCTIONS:
                msg = f"{path}: unknown test function {name!r}, expected one of {_FUNCTIONS}"
                raise ValueError(msg)
            dimension = int(entry["dimension"])
            minimizers = tuple(
                tuple(float(value) for value in minimizer)
                for minimizer in entry["minimizers"]
            )
            if any(len(minimizer) != dimension for minimizer in minimizers):
                msg = f"{path}: {name} declares a minimizer that is not {dimension}-dimensional"
                raise ValueError(msg)
            domain = tuple(float(edge) for edge in entry["domain"])
            if len(domain) != 4:
                msg = f"{path}: {name} declares a domain of {len(domain)} edges, expected 4"
                raise ValueError(msg)
            functions.append(
                TestFunctionParams(
                    name=name,
                    dimension=dimension,
                    start=float(entry["start"]),
                    restarts=int(entry["restarts"]),
                    scale=float(entry["scale"]),
                    domain=(domain[0], domain[1], domain[2], domain[3]),
                    minimizers=minimizers,
                )
            )

        return cls(
            functions=tuple(functions),
            seed=int(declared["seed"]),
            grid=int(declared["grid"]),
            at_minimum=float(declared["at_minimum"]),
        )
