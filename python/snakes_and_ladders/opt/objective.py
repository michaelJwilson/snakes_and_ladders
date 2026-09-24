"""The optimization interface, and nothing that knows what is being optimized.

Root ``CLAUDE.md`` requires the fitting machinery to serve HMMs, the Potts
model and phylogenetic trees alike (issue #63), which holds only if the
interface names none of them. ``snakes_and_ladders.opt`` therefore imports
nothing from ``snakes_and_ladders.sim``, ``snakes_and_ladders.likelihood`` or
``snakes_and_ladders.search`` -- asserted by a test, not left to review.

Three pieces are enough:

* an unconstrained parameter vector, so a gradient step is always legal;
* a differentiable scalar to minimize;
* a way to read the constrained parameters back out, because the
  unconstrained vector is an implementation detail and recovery is stated
  against the parameters a person named.

**Discrete moves are deliberately outside this interface.** A discrete move
changes the structure -- topology, chain length, state count -- and so changes
what ``theta`` means and how long it is. It constructs a *new* ``Objective``
rather than stepping inside a fit, and the loop that proposes such moves owns
that construction. An optimizer owning the outer loop would have to know what
a move is.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Objective(Protocol):
    """A differentiable scalar objective over an unconstrained parameter vector.

    Implementations are free to be stateful (holding data, sizes, fixed
    structure); only ``theta`` is optimized.
    """

    @abstractmethod
    def initial(self) -> torch.Tensor:
        """A starting point in unconstrained coordinates.

        Returns
        -------
        torch.Tensor
            1-D tensor of unconstrained parameters. Its length defines the
            dimension of the problem.
        """
        ...  # pragma: no cover

    @abstractmethod
    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """The named, feasible parameters ``theta`` encodes.

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters, as returned by :meth:`initial`.

        Returns
        -------
        Mapping[str, torch.Tensor]
            Constrained parameters under the names the model uses. This is
            what a recovery test compares against truth; ``theta`` itself is
            not meaningful to compare.
        """
        ...  # pragma: no cover

    @abstractmethod
    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The unconstrained vector whose :meth:`constrain` is ``named``.

        The inverse of :meth:`constrain`, required rather than optional.
        Without it an expectation-maximization run, which works in the model's
        own parameters and never builds a ``theta``, could carry no interval
        while the gradient fit's could (issue #268). The observed information
        is a property of *this objective at a point*, not of the route that
        reached it, so any route may ask for it. A constraint map that cannot
        be inverted is a problem worth failing on.

        Parameters
        ----------
        named : Mapping[str, torch.Tensor]
            Constrained parameters, under the keys :meth:`constrain` returns.

        Returns
        -------
        torch.Tensor
            ``theta`` such that ``constrain(theta)`` returns ``named``.
        """
        ...  # pragma: no cover

    @abstractmethod
    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """The value to **minimize**, differentiable with respect to ``theta``.

        Minimization is the convention throughout ``snakes_and_ladders.opt``, so a
        likelihood-based objective returns a *negative* log-likelihood.

        Parameters
        ----------
        theta : torch.Tensor
            Unconstrained parameters.

        Returns
        -------
        torch.Tensor
            Scalar tensor.
        """
        ...  # pragma: no cover


@runtime_checkable
class DeclaredGradient(Protocol):
    """An objective that states its own gradient.

    Autograd records and replays a graph per call, 57 µs at d = 10 where the
    closed form of a Gaussian takes 1.7 µs (issue #991); an objective whose
    gradient has a closed form states it here (issue #986). It must equal
    autograd's through ``__call__``, which each implementation's test pins.
    """

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        """``dU/dtheta`` at ``theta``, detached."""
        ...  # pragma: no cover


@runtime_checkable
class DeclaredValueAndGradient(Protocol):
    """An objective that states its value and gradient together (issue #1000).

    Optional beside :class:`Objective`: a consumer that reads a gradient ---
    the L-BFGS closure, the convergence test, a Hamiltonian kick --- takes it
    from :func:`value_and_gradient`. An HMM objective declares one so its
    gradient can come from a compiled backend rather than the graph.
    """

    def value_and_gradient(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(U(theta), dU/dtheta)``, both detached."""
        ...  # pragma: no cover


def declares_gradient(objective: object) -> bool:
    """Whether ``objective`` is a :class:`DeclaredGradient` with a *method* ``gradient``.

    A runtime-checkable protocol checks only that the attribute exists, and
    the phylogenetic objective's ``gradient`` is a property naming its route
    (`likelihood.objective`), not a function of ``theta``.
    """
    return isinstance(objective, DeclaredGradient) and callable(
        getattr(type(objective), "gradient", None)
    )


def value_and_gradient(
    objective: Objective, theta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(U(theta), dU/dtheta)`` detached.

    The objective's own where it declares them together, its declared
    gradient beside a graph-free value where it declares that alone, and
    autograd through ``__call__`` otherwise.
    """
    if isinstance(objective, DeclaredValueAndGradient):
        return objective.value_and_gradient(theta)
    if declares_gradient(objective):
        assert isinstance(objective, DeclaredGradient)
        with torch.no_grad():
            value = objective(theta.detach())
        return value, objective.gradient(theta.detach())
    point = theta.detach().clone().requires_grad_(True)
    value = objective(point)
    (gradient,) = torch.autograd.grad(value, point)
    return value.detach(), gradient
