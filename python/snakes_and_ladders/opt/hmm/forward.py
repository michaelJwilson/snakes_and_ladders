"""The forward recursion, and the alignment of fitted states to true ones: the kernels the objectives and the EM drivers share.

:func:`forward_log_likelihood_from_density` is the scaled forward pass on
per-position emission densities; :func:`align_by_key` and its two wrappers
resolve the label switching the package docstring states. Imports no other
submodule of :mod:`snakes_and_ladders.opt.hmm`.
"""

from __future__ import annotations

from itertools import permutations

import torch

from snakes_and_ladders.emissions import (
    CategoricalEmission,
    EmissionFamily,
)
from snakes_and_ladders.numerics import constant_chain_kernel


def forward_log_likelihood(
    observations: torch.Tensor,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    log_emission: torch.Tensor,
) -> torch.Tensor:
    """Total log-likelihood of ``observations`` by the forward recursion.

    Parameters
    ----------
    observations : torch.Tensor
        Integer symbols, shape ``(n_sequences, sequence_length)``.
    log_initial : torch.Tensor
        Log initial distribution, shape ``(m,)``.
    log_transition : torch.Tensor
        Log transition matrix, shape ``(m, m)``.
    log_emission : torch.Tensor
        Log emission matrix, shape ``(m, o)``.

    Returns
    -------
    torch.Tensor
        Scalar: the summed log-likelihood over sequences, differentiable
        with respect to every parameter.
    """
    return forward_log_likelihood_from_density(
        CategoricalEmission.from_log(log_emission).log_density(observations),
        log_initial,
        log_transition,
    )


def align_states(
    log_emission: torch.Tensor, reference: torch.Tensor
) -> tuple[int, ...]:
    """Permutation of fitted hidden states best matching ``reference``.

    The likelihood is invariant to relabelling the hidden states, so a
    recovery test has to choose a permutation. Emissions are the
    discriminating signal -- two states with the same emission distribution
    are the same state -- so it is the one minimizing total absolute emission
    difference, found by enumeration (``m!`` is small, and a greedy match can
    be wrong).

    Parameters
    ----------
    log_emission : torch.Tensor
        Fitted log emission matrix, shape ``(m, o)``.
    reference : torch.Tensor
        Emission matrix to align to, shape ``(m, o)``, as probabilities.

    Returns
    -------
    tuple[int, ...]
        ``order`` such that ``exp(log_emission)[list(order)]`` lines up with
        ``reference``.
    """
    return align_by_key(torch.exp(log_emission), reference)


def align_families(
    fitted: EmissionFamily, reference: EmissionFamily
) -> tuple[int, ...]:
    """Permutation of ``fitted``'s states best matching ``reference``'s.

    The same enumeration as :func:`align_states`, over whatever signature each
    family says distinguishes its states --- symbol probabilities for a
    categorical emission, means for a Gaussian one. The signature is the
    family's to define: a Gaussian fit has no emission matrix to align by.

    Parameters
    ----------
    fitted : EmissionFamily
        The fitted family.
    reference : EmissionFamily
        The family to align to.

    Returns
    -------
    tuple[int, ...]
        ``order`` such that state ``order[i]`` of ``fitted`` lines up with
        state ``i`` of ``reference``.
    """
    return align_by_key(fitted.alignment_key(), reference.alignment_key())


def align_by_key(fitted: torch.Tensor, reference: torch.Tensor) -> tuple[int, ...]:
    """The permutation minimizing total absolute distance between two key sets.

    Found by enumeration: ``m!`` is small, and a greedy match can be wrong.

    Parameters
    ----------
    fitted, reference : torch.Tensor
        Per-state signatures, shape ``(m, d)``.

    Returns
    -------
    tuple[int, ...]
        ``order`` such that ``fitted[list(order)]`` lines up with
        ``reference``.
    """
    best: tuple[int, ...] = ()
    best_cost = float("inf")
    for order in permutations(range(fitted.shape[0])):
        cost = float((fitted[list(order)] - reference).abs().sum())
        if cost < best_cost:
            best, best_cost = order, cost
    return best


def forward_log_likelihood_from_density(
    log_density: torch.Tensor,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
) -> torch.Tensor:
    """Total log-likelihood from per-site emission scores already computed.

    The recursion that does not know what a state emits, so one forward
    algorithm serves a family over an alphabet and one over the reals: both
    arrive here as the same block of numbers.

    Parameters
    ----------
    log_density : torch.Tensor
        Emission scores, shape ``(n_sequences, length, n_states)``. A
        log-probability for a discrete family, a log-density otherwise.
    log_initial : torch.Tensor
        Log initial distribution, shape ``(m,)``.
    log_transition : torch.Tensor
        Log transition matrix. Either ``(m, m)``, one kernel for the whole
        chain, or ``(length - 1, m, m)``, one per step, so a kernel that is a
        function of position can be written down (issue #653). The step axis
        is shared across the batch: the kernel varies along the sequence, not
        between sequences.

    Returns
    -------
    torch.Tensor
        Scalar: the summed log-likelihood over sequences, differentiable with
        respect to every parameter. **Not** bounded above by zero where the
        emission family is continuous, since it then sums densities.

    Raises
    ------
    ValueError
        If ``log_transition`` is neither shape.

    Notes
    -----
    The constant form is expanded to the step axis with
    :meth:`torch.Tensor.expand`, a stride-zero view rather than a copy, and the
    choice between the two is hoisted out of the recursion, so the constant
    case indexes nothing and sees the same numbers in the same order --- the
    result is the one this function returned before the second shape was
    admitted, bitwise. The ``length``-fold memory the varying form costs is
    paid only by a caller who asks for it. The same reasoning and the
    measurement behind the hoist are in
    :func:`snakes_and_ladders.likelihood.forward_backward.step_kernels`, whose
    shape check this shares
    (:func:`snakes_and_ladders.numerics.constant_chain_kernel`, issue #857):
    ``opt`` may not import ``likelihood``, so the dispatch sits at the root
    rather than in either recursion.
    """
    length, n_states = log_density.shape[1], log_density.shape[2]
    constant: torch.Tensor | None = None
    if constant_chain_kernel(tuple(log_transition.shape), length, n_states):
        kernels = log_transition.expand(max(length - 1, 0), n_states, n_states)
        constant = log_transition
    else:
        kernels = log_transition
    alpha = log_initial.unsqueeze(0) + log_density[:, 0]
    for t in range(1, length):
        step = kernels[t - 1] if constant is None else constant
        alpha = (
            torch.logsumexp(alpha.unsqueeze(2) + step.unsqueeze(0), dim=1)
            + log_density[:, t]
        )
    return torch.logsumexp(alpha, dim=1).sum()
