"""The optimization interface, and nothing that knows what is being optimized.

Root ``CLAUDE.md`` requires the fitting machinery to serve HMMs, the Potts
model and phylogenetic trees alike (issue #63). That only holds if the
interface is written without reference to any of them, so this module
mentions no model, and ``snakes_and_ladders.opt`` imports nothing from ``snakes_and_ladders.sim``,
``snakes_and_ladders.likelihood`` or ``snakes_and_ladders.search`` -- asserted by a test, not left to
review.

Three pieces are enough:

* an unconstrained parameter vector, so a gradient step is always legal;
* a differentiable scalar to minimize;
* a way to read the constrained parameters back out, because the
  unconstrained vector is an implementation detail and recovery is stated
  against the parameters a person named.

**Discrete moves are deliberately outside this interface.** A discrete move
changes the structure -- a different topology, chain length or state count --
and therefore changes what ``theta`` means and how long it is. It cannot be
a step inside a fit over a fixed-length vector; it constructs a *new*
``Objective``. The loop that proposes such moves owns that construction and
calls ``fit`` per candidate. An optimizer that tried to own the outer loop
would have to know what a move is, which is exactly the model knowledge this
interface exists to exclude.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Objective(Protocol):
    """A differentiable scalar objective over an unconstrained parameter vector.

    Implementations are free to be stateful (holding data, sizes, fixed
    structure); only ``theta`` is optimized.
    """

    def initial(self) -> torch.Tensor:
        """A starting point in unconstrained coordinates.

        Returns
        -------
        torch.Tensor
            1-D tensor of unconstrained parameters. Its length defines the
            dimension of the problem.
        """
        ...  # pragma: no cover

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
