"""The optimization interface, and nothing that knows what is being optimized.

Root ``CLAUDE.md`` requires the fitting machinery to serve HMMs, the Potts
model and phylogenetic trees alike (issue #63), which holds only if the
interface names none of them. ``sal.opt`` therefore imports
nothing from ``sal.sim``, ``sal.likelihood`` or
``sal.search`` -- asserted by a test, not left to review.

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
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, runtime_checkable

import numpy as np
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

        Minimization is the convention throughout ``sal.opt``, so a
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


@runtime_checkable
class DeclaredEnergy(Protocol):
    """An objective that states its value on an array, without the tensor type (issue #1011).

    Optional beside :class:`Objective`: a consumer that reads a value and
    never a derivative --- a random-walk proposal, an annealer --- takes it
    from :func:`energy_of`, which falls back to ``__call__`` where this is
    not declared, so no objective breaks. It must equal ``__call__`` at the
    same point, which each implementation's test pins: bitwise where the
    arithmetic is the same, and at its stated tolerance where a reduction
    is ordered differently.
    """

    def energy(self, x: np.ndarray) -> float:
        """``U(x)`` at a ``float64`` array ``x``."""
        ...  # pragma: no cover


#: :func:`energy_of`'s answers, by type: whether it declares ``energy``.
_ENERGY: dict[type, bool] = {}


def energy_of(objective: Objective, x: np.ndarray) -> float:
    """``U(x)`` as a ``float``: the objective's declared :meth:`~DeclaredEnergy.energy`, or ``__call__`` without a graph.

    The seam a value-only consumer evaluates through (issue #1011): it
    passes and receives arrays whether or not the objective has a NumPy
    route, and the fallback is ``float(objective(torch.as_tensor(x)))``
    under ``no_grad``, what such a consumer computed before the seam. Read
    from the class once per type, as :func:`_routes` is.
    """
    kind = type(objective)
    declared = _ENERGY.get(kind)
    if declared is None:
        declared = _ENERGY[kind] = _declares(kind, "energy")
    if declared:
        return objective.energy(x)  # type: ignore[attr-defined, no-any-return]
    with torch.no_grad():
        return float(objective(torch.as_tensor(x)))


def declares_gradient(objective: object) -> bool:
    """Whether ``objective`` is a :class:`DeclaredGradient` with a *method* ``gradient``.

    A runtime-checkable protocol checks only that the attribute exists, and
    the phylogenetic objective's ``gradient`` is a property naming its route
    (`likelihood.objective`), not a function of ``theta``. Read from the
    class, so the answer is one per type (:func:`_routes`).
    """
    return _routes(type(objective))[0]


def _declares(kind: type, name: str) -> bool:
    """Whether ``kind`` has a *method* ``name``: a class attribute that is callable."""
    return callable(getattr(kind, name, None))


#: :func:`_routes`' answers, by type.
_ROUTES: dict[type, tuple[bool, bool]] = {}


def _routes(kind: type) -> tuple[bool, bool]:
    """``(declares gradient, declares value and gradient)`` for a type, resolved once (issue #1008).

    What the runtime protocols answer, read from the class: an
    ``isinstance`` against a runtime-checkable protocol walks its members on
    every call, 16% of a Rosenbrock chain when it was asked per leapfrog step.
    """
    routes = _ROUTES.get(kind)
    if routes is None:
        routes = _declares(kind, "gradient"), _declares(kind, "value_and_gradient")
        _ROUTES[kind] = routes
    return routes


def value_and_gradient(
    objective: Objective, theta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(U(theta), dU/dtheta)`` detached.

    The objective's own where it declares them together, its declared
    gradient beside a graph-free value where it declares that alone, and
    autograd through ``__call__`` otherwise.
    """
    gradient, together = _routes(type(objective))
    if together:
        return objective.value_and_gradient(theta)  # type: ignore[attr-defined, no-any-return]
    if gradient:
        with torch.no_grad():
            value = objective(theta.detach())
        return value, objective.gradient(theta.detach())  # type: ignore[attr-defined]
    return autograd_value_and_gradient(objective, theta)


def autograd_value_and_gradient(
    objective: Callable[[torch.Tensor], torch.Tensor], theta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(U(theta), dU/dtheta)`` detached, by autograd through ``objective`` (issue #1010).

    The oracle every declared gradient is pinned to, and the route an
    objective takes where it declares none; written out at each site that
    fell back to it before it had one home.
    """
    point = theta.detach().clone().requires_grad_(True)
    value = objective(point)
    (derivative,) = torch.autograd.grad(value, point)
    return value.detach(), derivative


@runtime_checkable
class DeclaredBlocks(Protocol):
    """An objective that names the blocks of its ``theta`` (issue #1168).

    Optional beside :class:`Objective`: each name is a key :meth:`~Objective.constrain`
    returns, and its slice the coordinates of ``theta`` that parameter is
    read from. The slices are disjoint and cover ``theta``, which each
    implementation's test pins. :func:`coordinates` turns names into the
    indices :class:`Restricted` varies.
    """

    @property
    def blocks(self) -> Mapping[str, slice]:
        """Each named parameter's coordinates in ``theta``, in ``theta``'s order."""
        ...  # pragma: no cover


def coordinates(objective: object, names: Iterable[str]) -> torch.Tensor:
    """The indices into ``theta`` of the blocks ``names``, ascending (issue #1168).

    Parameters
    ----------
    objective : object
        A :class:`DeclaredBlocks`.
    names : Iterable[str]
        Keys of its :attr:`~DeclaredBlocks.blocks`; a bare string is one name.

    Returns
    -------
    torch.Tensor
        1-D ``torch.long``, ascending and without repeats: the ``varied``
        argument of :class:`Restricted`.

    Raises
    ------
    TypeError
        If ``objective`` declares no blocks.
    ValueError
        If a name is not one of its blocks.
    """
    if not isinstance(objective, DeclaredBlocks):
        msg = f"{type(objective).__name__} declares no blocks"
        raise TypeError(msg)
    blocks = objective.blocks
    # A bare string is one name, not an iterable of its characters.
    wanted = [names] if isinstance(names, str) else list(names)
    unknown = [name for name in wanted if name not in blocks]
    if unknown:
        msg = f"unknown block(s) {unknown}; the blocks are {list(blocks)}"
        raise ValueError(msg)
    indices = sorted(
        {i for name in wanted for i in range(blocks[name].start, blocks[name].stop)}
    )
    return torch.tensor(indices, dtype=torch.long)


class Restricted(Objective):
    """``objective`` on the coordinates ``varied``, every other one held at ``at`` (issue #1168).

    The protocol is satisfied on the reduced vector of length
    ``len(varied)``: a coordinate ``theta`` is scattered into a copy of
    ``at`` before the inner objective sees it, and a gradient is gathered
    back onto ``varied``. :meth:`initial` is ``at[varied]``, the point the
    restriction is taken at, not the inner objective's own start: a caller
    that holds parameters fixed holds them at a point it chose, and a start
    elsewhere on the varied coordinates is passed to the consumer as any
    start is. Promoted from ``search.infer``'s partial re-optimization
    (issue #408).

    ``value_and_gradient``, ``gradient`` and ``energy`` are taken through the
    inner objective's own routes (:func:`value_and_gradient`,
    :func:`energy_of`), so a compiled gradient stays compiled inside the
    restriction, as :class:`~sal.sample.chain._Scaled` keeps it. A
    declaration a consumer compiles a full-dimensional energy from ---
    ``jax_energy``, ``gaussian_hmm_declaration`` --- is not forwarded: it
    would describe ``objective``, not this restriction of it.

    Parameters
    ----------
    objective : Objective
        The objective restricted.
    at : torch.Tensor
        A full ``theta`` of ``objective``: the values every coordinate outside
        ``varied`` is held at.
    varied : torch.Tensor
        1-D integer indices into ``at``, without repeats: the coordinates left
        variable. Named ``varied`` rather than ``free``, which means an
        unconstrained coordinate (:mod:`sal.opt.constrain`).
        :func:`coordinates` builds it from block names.

    Raises
    ------
    ValueError
        If ``at`` is not 1-D, or ``varied`` is not 1-D integer indices into
        ``at`` without repeats.
    """

    def __init__(
        self, objective: Objective, at: torch.Tensor, varied: torch.Tensor
    ) -> None:
        if at.dim() != 1:
            msg = f"at is a full theta, 1-D; got shape {tuple(at.shape)}"
            raise ValueError(msg)
        if varied.dim() != 1 or varied.dtype not in (torch.int32, torch.int64):
            msg = f"varied is 1-D integer indices; got {varied.dtype} of shape {tuple(varied.shape)}"
            raise ValueError(msg)
        n = at.shape[0]
        if varied.numel() and (int(varied.min()) < 0 or int(varied.max()) >= n):
            msg = f"varied indexes outside a theta of length {n}"
            raise ValueError(msg)
        if torch.unique(varied).numel() != varied.numel():
            msg = "varied repeats an index"
            raise ValueError(msg)
        self.objective = objective
        self._at = at.detach()
        self._varied = varied.to(torch.long)
        self._at_array = self._at.cpu().numpy()
        self._varied_array = self._varied.cpu().numpy()

    @property
    def at(self) -> torch.Tensor:
        """The full ``theta`` the held coordinates are read from."""
        return self._at

    @property
    def varied(self) -> torch.Tensor:
        """The indices into ``at`` left variable."""
        return self._varied

    def embed(self, theta: torch.Tensor) -> torch.Tensor:
        """The full ``theta`` of ``objective``: ``at`` with ``varied`` set to ``theta``."""
        return self._at.index_copy(0, self._varied, theta)

    def initial(self) -> torch.Tensor:
        """``at[varied]``: the restriction starts where it was taken."""
        return self._at[self._varied]

    def constrain(self, theta: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """``objective``'s named parameters at the embedded point."""
        return self.objective.constrain(self.embed(theta))

    def theta_from(self, named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The varied coordinates of ``objective``'s inverse of ``named``."""
        return self.objective.theta_from(named)[self._varied]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return self.objective(self.embed(theta))

    def value_and_gradient(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``objective``'s value and gradient at the embedded point, the gradient gathered onto ``varied``."""
        value, gradient = value_and_gradient(self.objective, self.embed(theta.detach()))
        return value, gradient[self._varied]

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        """``objective``'s gradient at the embedded point, gathered: its declared gradient alone where it has one."""
        full = self.embed(theta.detach())
        if declares_gradient(self.objective):
            gradient: torch.Tensor = self.objective.gradient(full)  # type: ignore[attr-defined]
        else:
            gradient = value_and_gradient(self.objective, full)[1]
        return gradient[self._varied]

    def energy(self, x: np.ndarray) -> float:
        """``objective``'s :func:`energy_of` at ``at`` with ``varied`` set to ``x``."""
        full = self._at_array.copy()
        full[self._varied_array] = x
        return energy_of(self.objective, full)
