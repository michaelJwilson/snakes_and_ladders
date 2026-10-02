"""The forward recursion, and the alignment of fitted states to true ones: the kernels the objectives and the EM drivers share.

:func:`forward_messages` is the scaled forward pass on per-position
emission densities over a padded batch, and
:func:`forward_log_likelihood_from_density` its differentiable total; :func:`align_by_key` and its two wrappers
resolve the label switching the package docstring states. Imports no other
submodule of :mod:`sal.opt.hmm`.
"""

from __future__ import annotations

from itertools import permutations

import torch

from sal.emissions import (
    CategoricalEmission,
    EmissionFamily,
)
from sal.numerics import constant_chain_kernel


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
    The recursion is :func:`forward_messages`'s, which the Baum--Welch Python
    E step shares (issue #1162). The choice between the two forms is hoisted
    out of the recursion, so the constant case indexes nothing and sees the
    same numbers in the same order --- the result is the one this function
    returned before the second shape was admitted, bitwise. The
    ``length``-fold memory the varying form costs is paid only by a caller who
    asks for it. The same reasoning and the measurement behind the hoist are in
    :func:`sal.likelihood.forward_backward.step_kernels`, whose
    shape check this shares
    (:func:`sal.numerics.constant_chain_kernel`, issue #857):
    ``opt`` may not import ``likelihood``, so the dispatch sits at the root
    rather than in either recursion.
    """
    length, n_states = log_density.shape[1], log_density.shape[2]
    # The shape check refuses a third form and reads `(m, m)` first, so a chain
    # of `m + 1` positions carrying one matrix is one kernel; the recursion then
    # reads the form off the kernel's rank.
    constant_chain_kernel(tuple(log_transition.shape), length, n_states)
    log_evidence, _ = forward_messages(log_density, log_initial, log_transition)
    return log_evidence.sum()


def forward_messages(
    log_density: torch.Tensor,
    log_initial: torch.Tensor,
    kernels: torch.Tensor,
    *,
    final: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The scaled forward recursion over a padded batch: each chain's evidence, and the messages.

    The one torch forward pass in :mod:`sal.opt.hmm` (issue #1162):
    :func:`forward_log_likelihood_from_density` differentiates through it and
    the Baum--Welch Python E step reads its messages for the backward pass and
    the pair counts. Shapes are not checked here; each caller refuses the
    kernels it does not admit, in its own words.

    Parameters
    ----------
    log_density : torch.Tensor
        Emission scores, shape ``(n, length, m)``, padded to the longest
        chain. A padded position must already score zero (log one), so it adds
        nothing wherever it is reached: the caller masks it, because its own
        backward pass reads the same masked block.
    log_initial : torch.Tensor
        Log initial distribution, shape ``(m,)``.
    kernels : torch.Tensor
        Log transition kernels, read off their rank: ``(m, m)``, one for every
        step and chain; ``(length - 1, m, m)``, one per step, shared by the
        chains; or ``(n, length - 1, m, m)``, each chain's own per step.
    final : torch.Tensor | None
        Each chain's last live position, shape ``(n,)``, integer. Given, the
        message table is kept and each chain's evidence is gathered at its own
        last position, since a message beyond it is not a probability of
        anything. ``None``, every chain runs the full ``length`` and only the
        running column is kept --- the form a gradient is taken through, which
        then writes no table.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        Each chain's log evidence, shape ``(n,)``, and the forward messages
        ``alpha``, shape ``(n, length, m)``, where ``final`` is given, else
        ``None``.

    Notes
    -----
    The kernel's form is read once, outside the recursion, so the constant
    case indexes nothing per step and sees the same numbers in the same order
    as the recursion :func:`forward_log_likelihood_from_density` carried before
    the per-step form was admitted (issue #653) --- the result is unchanged
    bitwise. Each step is the same elementwise sum and the same ``logsumexp``
    whichever caller asks, and keeping the table copies each column after it is
    computed, so the evidence on equal lengths is the same bitwise with or
    without ``final`` (``tests/regression/opt/test_opt_hmm_forward_messages.py``).
    """
    n, length = log_density.shape[0], log_density.shape[1]
    alpha = log_initial.unsqueeze(0) + log_density[:, 0]
    table: torch.Tensor | None = None
    if final is not None:
        table = torch.empty((n, length, alpha.shape[1]), dtype=alpha.dtype)
        table[:, 0] = alpha
    # Hoisted: the one-kernel form broadcasts the same `(1, m, m)` view at
    # every step.
    constant = kernels.unsqueeze(0) if kernels.ndim == 2 else None
    for t in range(1, length):
        if constant is not None:
            step = constant
        elif kernels.ndim == 3:
            step = kernels[t - 1].unsqueeze(0)
        else:
            step = kernels[:, t - 1]
        alpha = torch.logsumexp(alpha.unsqueeze(2) + step, dim=1) + log_density[:, t]
        if table is not None:
            table[:, t] = alpha
    if table is None or final is None:
        return torch.logsumexp(alpha, dim=1), None
    return torch.logsumexp(table[torch.arange(n), final], dim=1), table
