"""Standard continuous test functions, as :class:`~snakes_and_ladders.opt.objective.Objective`.

Every other test of :func:`snakes_and_ladders.opt.fit.fit` measures a *statistical*
property of a likelihood surface --- the first-order condition, interval
coverage at the nominal rate, agreement with Baum-Welch, distance to the
generating truth. None of them checks that L-BFGS and the convergence
criterion find a minimum that is known independently of the model being
fitted, and the two failure modes are confounded: a fit that lands away from
the truth reads as a weakly identified parameter, and an optimizer that stops
early produces the same symptom.

These three functions separate them, because their minimizers are known in
closed form and have nothing to do with phylogenetics. Each targets a
different failure:

* :class:`Rosenbrock` --- a narrow curved valley, where a wrong line search
  shows up as slowness rather than as a wrong answer.
* :class:`Rastrigin` --- a global bowl under roughly ``10 ** n`` local
  minima, which is the honest test of what one fit from one start can claim.
* :class:`Himmelblau` --- four *equal* global minima, which catches a method
  that reports "the" optimum without saying which basin it found.

**These are not likelihoods**, so the Hessian at the optimum is not an
observed information matrix and :func:`snakes_and_ladders.opt.fit.constrained_standard_errors`
must not be called on them: an interval built from it would be a number with
no meaning attached (issue #122 covers the general case).
:meth:`constrain` therefore returns the point itself, under the name each
function's minimizer is stated in, rather than pretending to a parameter
transformation there is none of.

Sources: Rosenbrock (1960); Rastrigin (1974); Himmelblau (1972). Nocedal &
Wright use the first as the standing example for quasi-Newton methods.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Rosenbrock:
    """``sum_i b (x_{i+1} - x_i^2)^2 + (a - x_i)^2``, minimized at ``(a, ..., a)``.

    The valley is curved and its floor is nearly flat, so the gradient points
    across it rather than along it and steepest descent zig-zags. What makes
    it the standing quasi-Newton example is that the difficulty is entirely
    conditioning: the function is smooth, unimodal in the relevant region, and
    has an analytic minimizer, so a method that struggles here is struggling
    for a reason that has nothing to do with the surface being hard to search.

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

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        head, tail = theta[:-1], theta[1:]
        return (self.b * (tail - head**2) ** 2 + (self.a - head) ** 2).sum()

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
class Rastrigin:
    """``10 n + sum_i (x_i^2 - 10 cos(2 pi x_i))``, minimized at the origin.

    A quadratic bowl with a cosine ripple, so the global structure points at
    the answer and the local structure does not: there are roughly ``10 ** n``
    local minima, one per lattice cell, and every one of them satisfies the
    first-order condition. A single fit from a single start is expected to
    land in whichever cell it began in, which is why the test built on this
    reports a *success rate* rather than asserting success.
    """

    dimension: int = 2
    amplitude: float = 10.0
    start: float = 4.4

    def initial(self) -> torch.Tensor:
        return torch.full((self.dimension,), self.start, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta}

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
class Himmelblau:
    """``(x^2 + y - 11)^2 + (x + y^2 - 7)^2``, with four equal global minima.

    Two dimensions only; the function is defined that way. The four minima all
    have value 0, so no ordering distinguishes them and "the" optimum is not a
    well-formed question. A method that always returns the same one regardless
    of where it started is reading its own initialization, and nothing else in
    this repository's suite would notice.
    """

    start: tuple[float, float] = (0.0, 0.0)

    def initial(self) -> torch.Tensor:
        return torch.tensor(self.start, dtype=torch.float64)

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"x": theta[0], "y": theta[1]}

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
    def nearest_minimum(point: torch.Tensor) -> tuple[int, float]:
        """Which of the four minima ``point`` landed on, and how far off it is.

        Returns
        -------
        tuple[int, float]
            Index into :data:`HIMMELBLAU_MINIMA`, and the Euclidean distance.
        """
        distances = [
            float(torch.linalg.vector_norm(point - torch.tensor(minimum)))
            for minimum in HIMMELBLAU_MINIMA
        ]
        best = min(range(len(distances)), key=distances.__getitem__)
        return best, distances[best]
