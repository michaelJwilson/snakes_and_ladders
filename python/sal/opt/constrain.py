"""Constraint maps: unconstrained reals in, feasible parameters out.

``opt/CLAUDE.md`` requires constraints by construction rather than by
projection, so this module holds the maps every instance shares: an optimizer
never sees a constrained quantity, because the feasible set is the image of
the map.

Nothing here knows what the parameters mean --- this is the vocabulary the
phylogenetic, Potts and HMM objectives are all written in. A family names
the set each of its parameters lives in as a
:class:`~sal.emissions.base.Domain`, and :func:`constrained` and
:func:`free_from` apply the map that value names (issue #1164).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from sal.emissions.base import Domain


def log_simplex(free: torch.Tensor) -> torch.Tensor:
    """Map ``n - 1`` unconstrained reals to ``n`` log-probabilities.

    The first logit is pinned to zero rather than optimized. A plain softmax
    over ``n`` logits is invariant to adding a constant to all of them, so one
    direction of the parameter space would be flat -- harmless for a gradient
    step, fatal for a Hessian-based interval, whose observed information would
    be singular. Pinning makes the map a bijection onto the simplex.

    Parameters
    ----------
    free : torch.Tensor
        Unconstrained parameters, shape ``(..., n - 1)``. The last axis is
        the one normalized; leading axes are batched, so a stack of
        transition-matrix rows maps in one call.

    Returns
    -------
    torch.Tensor
        Log-probabilities of shape ``(..., n)``, whose ``exp`` sums to 1
        along the last axis.
    """
    zeros = torch.zeros(*free.shape[:-1], 1, dtype=free.dtype, device=free.device)
    return torch.log_softmax(torch.cat([zeros, free], dim=-1), dim=-1)


def free_from_log_simplex(log_probs: torch.Tensor) -> torch.Tensor:
    """Invert :func:`log_simplex`.

    Needed to place a known truth, or a chosen starting point, in the
    unconstrained coordinates the optimizer works in.

    Parameters
    ----------
    log_probs : torch.Tensor
        Log-probabilities, shape ``(..., n)``, normalized along the last
        axis.

    Returns
    -------
    torch.Tensor
        Unconstrained parameters of shape ``(..., n - 1)`` satisfying
        ``log_simplex(free) == log_probs``.
    """
    return log_probs[..., 1:] - log_probs[..., :1]


def positive(free: torch.Tensor) -> torch.Tensor:
    """Map unconstrained reals to strictly positive values.

    An exponential rather than a softplus: it is exactly invertible in
    floating point, where ``log(expm1(x))`` loses precision for small ``x``,
    and a branch length near zero is exactly the case that has to round-trip.

    Parameters
    ----------
    free : torch.Tensor
        Unconstrained parameters, any shape.

    Returns
    -------
    torch.Tensor
        Positive values, same shape.
    """
    return torch.exp(free)


def free_from_positive(values: torch.Tensor) -> torch.Tensor:
    """Invert :func:`positive`.

    Parameters
    ----------
    values : torch.Tensor
        Strictly positive values, any shape.

    Returns
    -------
    torch.Tensor
        Unconstrained parameters satisfying ``positive(free) == values``.
    """
    return torch.log(values)


def probability(free: torch.Tensor) -> torch.Tensor:
    """Map unconstrained reals to strictly interior probabilities.

    A logistic rather than a clamp: a probability has two boundaries, so
    positivity is not enough, and a projection lands iterates exactly on a
    boundary, where the inverse is infinite.

    Parameters
    ----------
    free : torch.Tensor
        Unconstrained parameters, any shape.

    Returns
    -------
    torch.Tensor
        Values in ``(0, 1)``, same shape.
    """
    return torch.sigmoid(free)


def free_from_probability(values: torch.Tensor) -> torch.Tensor:
    """Invert :func:`probability`.

    ``log(p) - log1p(-p)`` rather than ``log(p / (1 - p))``: the difference is
    what ``1 - p`` costs as ``p`` approaches one.

    Parameters
    ----------
    values : torch.Tensor
        Probabilities in ``(0, 1)``, any shape.

    Returns
    -------
    torch.Tensor
        Unconstrained parameters satisfying ``probability(free) == values``.
    """
    return torch.log(values) - torch.log1p(-values)


def real(free: torch.Tensor) -> torch.Tensor:
    """Map unconstrained reals to the reals: the identity, so ``REAL`` names a map like every other domain.

    Parameters
    ----------
    free : torch.Tensor
        Unconstrained parameters, any shape.

    Returns
    -------
    torch.Tensor
        ``free`` itself.
    """
    return free


def free_from_real(values: torch.Tensor) -> torch.Tensor:
    """Invert :func:`real`.

    Parameters
    ----------
    values : torch.Tensor
        Real values, any shape.

    Returns
    -------
    torch.Tensor
        ``values`` itself.
    """
    return values


type _Map = Callable[[torch.Tensor], torch.Tensor]

#: Each domain's map from the free coordinates and its inverse.
_MAPS: dict[Domain, tuple[_Map, _Map]] = {
    Domain.REAL: (real, free_from_real),
    Domain.POSITIVE: (positive, free_from_positive),
    Domain.PROBABILITY: (probability, free_from_probability),
    Domain.LOG_SIMPLEX: (log_simplex, free_from_log_simplex),
}


def constrained(domain: Domain, free: torch.Tensor) -> torch.Tensor:
    """Map ``free`` onto ``domain`` by the map it names.

    Parameters
    ----------
    domain : Domain
        The set the result lives in.
    free : torch.Tensor
        Unconstrained parameters, shaped as :func:`free_shape` states.

    Returns
    -------
    torch.Tensor
    """
    return _MAPS[domain][0](free)


def free_from(domain: Domain, values: torch.Tensor) -> torch.Tensor:
    """Invert :func:`constrained`.

    Parameters
    ----------
    domain : Domain
        The set ``values`` lives in.
    values : torch.Tensor
        Values in ``domain``.

    Returns
    -------
    torch.Tensor
        Unconstrained parameters satisfying ``constrained(domain, free) == values``
        to rounding.
    """
    return _MAPS[domain][1](values)


def free_shape(domain: Domain, shape: tuple[int, ...]) -> tuple[int, ...]:
    """The shape of the free coordinates of a value of ``shape`` in ``domain``.

    ``shape`` for every domain but :attr:`~Domain.LOG_SIMPLEX`, which pins
    one logit per normalized row and so has one entry fewer on the last axis.

    Returns
    -------
    tuple[int, ...]
    """
    if domain is Domain.LOG_SIMPLEX:
        return (*shape[:-1], shape[-1] - 1)
    return shape


@dataclass(frozen=True)
class DomainBlock:
    """Where one named parameter sits in ``theta``, and the map onto it.

    Shared by the objectives that read a family's declared domains ---
    :class:`~sal.opt.emission_mixture.EmissionMixtureObjective` and
    :class:`~sal.opt.hmm.EmissionHmmObjective` --- so the layout is stated
    once (issues #1164, #1169).

    Parameters
    ----------
    domain : Domain
        The family's declared domain for the parameter.
    shape : tuple[int, ...]
        The parameter's shape.
    free : tuple[int, ...]
        The shape of its free coordinates (:func:`free_shape`).
    offset : int
        Its first entry in ``theta``.
    """

    domain: Domain
    shape: tuple[int, ...]
    free: tuple[int, ...]
    offset: int

    @property
    def stop(self) -> int:
        """One past its last entry in ``theta``."""
        return self.offset + math.prod(self.free)

    def read(self, theta: torch.Tensor) -> torch.Tensor:
        """The constrained parameter ``theta`` encodes in this block."""
        free = theta[self.offset : self.stop]
        if len(self.free) != 1:
            free = free.reshape(self.free)
        return constrained(self.domain, free)

    def free_of(self, value: torch.Tensor) -> torch.Tensor:
        """The block's free coordinates of ``value``, flat."""
        value = torch.as_tensor(value, dtype=torch.float64)
        return free_from(self.domain, value.reshape(self.shape)).reshape(-1)


def domain_blocks(
    named: Mapping[str, torch.Tensor], domains: Mapping[str, Domain], offset: int
) -> dict[str, DomainBlock]:
    """One :class:`DomainBlock` per parameter of ``named``, laid end to end from ``offset``.

    Parameters
    ----------
    named : Mapping[str, torch.Tensor]
        A family's :meth:`~sal.emissions.EmissionFamily.named_parameters`.
    domains : Mapping[str, Domain]
        Its :meth:`~sal.emissions.EmissionFamily.parameter_domains`.
    offset : int
        The first block's first entry in ``theta``.

    Raises
    ------
    ValueError
        If a parameter lies outside its domain, where its free coordinates
        are not finite.
    """
    blocks: dict[str, DomainBlock] = {}
    for name, value in named.items():
        domain = domains[name]
        if not bool(torch.isfinite(free_from(domain, value)).all()):
            msg = f"parameter {name!r} lies outside its {domain} domain"
            raise ValueError(msg)
        shape = tuple(value.shape)
        blocks[name] = DomainBlock(domain, shape, free_shape(domain, shape), offset)
        offset = blocks[name].stop
    return blocks
