"""Constraint maps: unconstrained reals in, feasible parameters out.

``opt/CLAUDE.md`` requires constraints by construction rather than by
projection, so this module holds the maps every instance shares. An
optimizer never sees a constrained quantity and is never asked to stay
inside a feasible set, because the set is the image of the map.

Nothing here knows what the parameters mean. That is the point: this is the
vocabulary the phylogenetic, Potts and HMM objectives are all written in.
"""

from __future__ import annotations

import torch


def log_simplex(free: torch.Tensor) -> torch.Tensor:
    """Map ``n - 1`` unconstrained reals to ``n`` log-probabilities.

    The first logit is pinned to zero rather than optimized. A plain softmax
    over ``n`` logits is invariant to adding a constant to all of them, so it
    would leave one direction of the parameter space flat -- harmless for a
    gradient step, fatal for a Hessian-based interval (the observed
    information would be singular). Pinning makes the map a bijection onto
    the simplex, so a fitted parameter has an identifiable value.

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
