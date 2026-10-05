"""The forward recursion, and the alignment of fitted states to true ones: the kernels the objectives and the EM drivers share.

:func:`forward_messages` is the scaled forward pass on per-position
emission densities over a padded batch, and
:func:`forward_log_likelihood_from_density` its differentiable total;
:func:`forward_log_likelihood_ragged` is the same total over segments of
unequal length (issue #1167); :func:`align_by_key` and its two wrappers
resolve the label switching the package docstring states.
:class:`Posteriors` is the ragged E step's result and :class:`SwitchKind` the
form of its switched step: both are defined here, where ``opt`` can name
them, and :mod:`sal.likelihood.ragged` re-exports them, since ``opt`` may not
import ``likelihood`` (issues #1166, #1186). Imports no other
submodule of :mod:`sal.opt.hmm`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import permutations

import numpy as np
import torch

from sal.emissions import (
    CategoricalEmission,
    EmissionFamily,
)
from sal.numerics import constant_chain_kernel


@dataclass(frozen=True)
class Posteriors:
    """What one ragged forward--backward pass returns, in the log domain.

    Parameters
    ----------
    log_posterior : np.ndarray
        ``(total, n_states)``, the log marginal at every position.
    log_counts : np.ndarray
        ``(n_states, n_states)``, the log transition counts summed over
        segments. The pair spanning a boundary is in none of them.
    log_evidence : np.ndarray
        One log evidence per segment.
    """

    log_posterior: np.ndarray
    log_counts: np.ndarray
    log_evidence: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        """``(log_posterior, log_counts, log_evidence)``: the order callers unpack."""
        yield from (self.log_posterior, self.log_counts, self.log_evidence)


class SwitchKind(StrEnum):
    """How a per-position switch probability ``s_t`` enters the step into ``t``.

    ``STAY_OR_MOVE`` is ``(1 - s_t) I + s_t A`` (issue #1082). The other two
    are a slow chain ``A`` over ``K`` states coupled to a fast binary layer,
    ``2 K`` states with ``(i, a)`` at ``2 i + a`` as ``np.kron`` lays them
    out (issue #1133):

    - ``KRONECKER``: ``A ⊗ S_t``, ``S_t = [[1 - s_t, s_t], [s_t, 1 - s_t]]``.
      No ``A'`` makes it ``(1 - s) I + s A'`` unless ``A = I``, since both of
      ``(1 - s)(A ⊗ I) + s (A ⊗ J)`` carry ``A``.
    - ``KRONECKER_DIAGONAL``: ``S_t`` on ``A``'s diagonal blocks alone, the
      layer switching only where the slow chain stays; an off-diagonal move
      lands on either layer with probability one half.

    The order is state-major (``2 i + a``), as ``np.kron(A, S)``;
    :func:`sal.likelihood.ragged.kronecker_order` maps a layer-major
    (``a K + i``) array onto it.

    Defined here, where ``opt`` can name it, for
    :func:`forward_log_likelihood_ragged`; :mod:`sal.likelihood.ragged`
    re-exports it, as it does :class:`Posteriors` (issue #1186).
    """

    STAY_OR_MOVE = "stay_or_move"
    KRONECKER = "kronecker"
    KRONECKER_DIAGONAL = "kronecker_diagonal"


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
    whichever caller asks, and keeping the table stacks the columns after they
    are computed, so the evidence on equal lengths is the same bitwise with or
    without ``final`` (``tests/regression/opt/test_opt_hmm_forward_messages.py``).
    """
    n, length = log_density.shape[0], log_density.shape[1]
    alpha = log_initial.unsqueeze(0) + log_density[:, 0]
    # The columns are collected and stacked once rather than written into a
    # preallocated table: a slice assignment is one autograd node per step
    # whose backward copies the whole table, quadratic in `length` (#1167).
    columns: list[torch.Tensor] | None = [alpha] if final is not None else None
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
        if columns is not None:
            columns.append(alpha)
    if columns is None or final is None:
        return torch.logsumexp(alpha, dim=1), None
    table = torch.stack(columns, dim=1)
    return torch.logsumexp(table[torch.arange(n), final], dim=1), table


def forward_log_likelihood_ragged(
    log_density: torch.Tensor,
    lengths: Sequence[int],
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    *,
    switch: torch.Tensor | None = None,
    switch_kind: SwitchKind = SwitchKind.STAY_OR_MOVE,
) -> torch.Tensor:
    """Total log-likelihood over segments of unequal length, differentiable.

    :func:`forward_log_likelihood_from_density` for a batch whose chains do
    not share a length (issue #1167). Each segment restarts at
    ``log_initial``, and no transition spans a boundary.

    Parameters
    ----------
    log_density : torch.Tensor
        Emission scores, shape ``(total, m)``: the segments end to end, in
        :attr:`sal.ragged.Ragged.values` layout. A tensor rather than a
        :class:`~sal.ragged.Ragged`, whose ``values`` is typed NumPy: a
        tensor-holding ``Ragged`` would be a second reading of that type.
    lengths : Sequence[int]
        One length per segment, summing to ``total``, as
        :func:`sal.oxisal.ragged_posteriors` takes them. Each is at least one:
        the recursion admits a segment of one position, whose evidence is
        ``logsumexp(log_initial + log_density[t])``.
        :class:`~sal.ragged.Ragged` refuses fewer than
        :data:`sal.ragged.MINIMUM_LENGTH`, so a caller building from one
        never passes it.
    log_initial : torch.Tensor
        Log initial distribution, shape ``(m,)``.
    log_transition : torch.Tensor
        Log transition matrix, shape ``(m, m)``, one kernel for every step of
        every segment; under either Kronecker kind, the slow chain's
        ``(m / 2, m / 2)``.
    switch : torch.Tensor | None
        One probability ``s`` in ``[0, 1]`` per position, shape ``(total,)``,
        or ``None`` for ``log_transition`` at every step; as
        :func:`sal.likelihood.ragged.posteriors` takes it (issue #1186). The
        step into position ``t`` is built from ``s[t]`` and the one
        transition, and a segment's first entry is never read, so its
        gradient there is zero.
    switch_kind : SwitchKind
        How ``switch`` enters; see :class:`SwitchKind`. Under either
        Kronecker kind ``log_density`` and ``log_initial`` are over the
        ``m = 2 K`` states, and ``switch`` is required.

    Returns
    -------
    torch.Tensor
        Scalar: the summed log evidence over segments, differentiable with
        respect to every tensor argument. Its gradient with respect to
        ``log_density`` is the posterior marginal at every position (the
        Fisher identity), in the same ``(total, m)`` layout.

    Raises
    ------
    ValueError
        If ``lengths`` is empty, holds a length below one or does not sum to
        ``total``, or a tensor's shape disagrees with ``m``; or, as
        :func:`sal.likelihood.ragged.posteriors` refuses them, a Kronecker
        kind is given no ``switch`` or an odd ``m``, or ``switch`` is not one
        probability per position.

    Notes
    -----
    The recursion is :func:`forward_messages`'s. The segments are scattered
    into an ``(n, longest, m)`` block that scores log one at every padded
    position, as the Baum--Welch E step masks it, and each segment's evidence
    is gathered at its own last position through ``final``. Where every
    length is equal the block is a reshape and ``final`` is not passed, so no
    table is written and the result is
    :func:`forward_log_likelihood_from_density`'s, bitwise.

    With ``switch`` the recursion is ``_switched_evidence``'s, on the
    same padded block and the same ``final``: each step is taken in its
    factors (``_switched_step``) rather than as a materialized
    ``(m, m)`` matrix per position, and ``switch=None`` under
    ``STAY_OR_MOVE`` runs the unswitched path above, unchanged.
    """
    if log_density.ndim != 2:
        msg = f"log_density {tuple(log_density.shape)} must be (total, m)"
        raise ValueError(msg)
    total, n_states = log_density.shape[0], log_density.shape[1]
    sizes = [int(length) for length in lengths]
    if not sizes or min(sizes) < 1 or sum(sizes) != total:
        msg = (
            f"lengths {sizes} must be at least one segment, each of at least "
            f"one position, summing to the {total} rows of log_density"
        )
        raise ValueError(msg)
    switch_kind = SwitchKind(switch_kind)
    kronecker = switch_kind is not SwitchKind.STAY_OR_MOVE
    if kronecker and (switch is None or n_states % 2):
        msg = (
            f"a {switch_kind} switch takes 2 K states and a switch per "
            f"position; got {n_states} states and switch={switch is not None}"
        )
        raise ValueError(msg)
    width = n_states // 2 if kronecker else n_states
    if tuple(log_transition.shape) != (width, width):
        msg = f"log_transition {tuple(log_transition.shape)} must be ({width}, {width})"
        raise ValueError(msg)
    if tuple(log_initial.shape) != (n_states,):
        msg = f"log_initial {tuple(log_initial.shape)} must be ({n_states},)"
        raise ValueError(msg)
    if switch is not None and (
        tuple(switch.shape) != (total,)
        or not bool(((switch >= 0.0) & (switch <= 1.0)).all())
    ):
        msg = (
            f"switch {tuple(switch.shape)} must be ({total},), one probability "
            "in [0, 1] per position"
        )
        raise ValueError(msg)
    n, longest = len(sizes), max(sizes)
    equal = min(sizes) == longest
    if switch is None and equal:
        block = log_density.reshape(n, longest, n_states)
        evidence, _ = forward_messages(block, log_initial, log_transition)
        return evidence.sum()
    counts = torch.as_tensor(sizes, dtype=torch.long)
    if equal:
        final = None
        block = log_density.reshape(n, longest, n_states)
        steps = None if switch is None else switch.reshape(n, longest)
    else:
        final = counts - 1
        segment = torch.repeat_interleave(torch.arange(n), counts)
        starts = torch.cumsum(counts, dim=0) - counts
        position = torch.arange(total) - starts[segment]
        # Log one at every padded position, so it adds nothing wherever the
        # recursion reaches it; the scatter is out of place, so the gradient
        # flows back to the live rows alone.
        block = log_density.new_zeros((n, longest, n_states)).index_put(
            (segment, position), log_density
        )
        # A padded step reads a switch of zero; it lies past `final` and is
        # never gathered.
        steps = (
            None
            if switch is None
            else switch.new_zeros((n, longest)).index_put((segment, position), switch)
        )
    if steps is None:
        evidence, _ = forward_messages(block, log_initial, log_transition, final=final)
        return evidence.sum()
    return _switched_evidence(
        block, log_initial, log_transition, steps, switch_kind, final
    ).sum()


def _switched_step(
    probability: torch.Tensor,
    step: torch.Tensor,
    moved: torch.Tensor,
    switch_kind: SwitchKind,
) -> torch.Tensor:
    """One switched step in its factors: ``probability`` times the step's matrix.

    ``probability`` is ``(n, m)``, ``step`` the ``(n, 1)`` switch of the step
    and ``moved`` the transition in probability space. The matrix
    :func:`sal.likelihood.ragged.step_transitions` materializes is never
    built: ``STAY_OR_MOVE`` is ``(1 - s) p + s p A``, ``m^2 + 2 m`` terms;
    ``KRONECKER`` applies ``S_t`` to the layer axis, ``4 K`` terms, then
    ``A`` to the slow axis, ``2 K^2``; ``KRONECKER_DIAGONAL`` takes ``A``'s
    off-diagonal part on the layer sums, ``K^2``, and ``S_t`` on the
    diagonal, ``4 K``. Against ``4 K^2`` per row for the explicit ``2 K``
    matrix.
    """
    if switch_kind is SwitchKind.STAY_OR_MOVE:
        return (1.0 - step) * probability + step * (probability @ moved)
    n = probability.shape[0]
    # `(n, K, 2)`: state `(i, a)` at `2 i + a`, as `np.kron(A, S)` lays it.
    pair = probability.reshape(n, -1, 2)
    layer = step.unsqueeze(2)
    # `sum_a p[i, a] S[a, b]`: stay on the layer with `1 - s`, flip with `s`.
    layered = (1.0 - layer) * pair + layer * pair.flip(2)
    if switch_kind is SwitchKind.KRONECKER:
        # `sum_i layered[i, b] A[i, j]`, the slow axis last for the matmul.
        return (layered.transpose(1, 2) @ moved).transpose(1, 2).reshape(n, -1)
    kept = torch.diagonal(moved)
    # A move off the diagonal lands on either layer with probability 1/2.
    spread = 0.5 * (pair.sum(2) @ (moved - torch.diag(kept)))
    return (spread.unsqueeze(2) + kept.unsqueeze(1) * layered).reshape(n, -1)


def _switched_evidence(
    log_density: torch.Tensor,
    log_initial: torch.Tensor,
    log_transition: torch.Tensor,
    switch: torch.Tensor,
    switch_kind: SwitchKind,
    final: torch.Tensor | None,
) -> torch.Tensor:
    """Each chain's log evidence under the switched step, over a padded batch.

    :func:`forward_messages` with the step :func:`_switched_step` takes:
    ``log_density`` is ``(n, length, m)``, padded positions scoring zero,
    ``switch`` ``(n, length)`` and ``final`` as there. Each step rescales the
    message by its row maximum, multiplies in probability space and returns
    to logs; the maximum is detached, since the result does not depend on it,
    so the gradient is the one through the product alone.
    """
    n = log_density.shape[0]
    moved = torch.exp(log_transition)
    # Split once: a slice per step is one autograd node whose backward writes
    # a zero block the size of the whole input, quadratic in `length`; the
    # backward of `unbind` stacks the per-step gradients once.
    scores = log_density.unbind(dim=1)
    steps = switch.unsqueeze(2).unbind(dim=1)
    alpha = log_initial.unsqueeze(0) + scores[0]
    columns: list[torch.Tensor] | None = [alpha] if final is not None else None
    for score, step in zip(scores[1:], steps[1:], strict=True):
        shift = alpha.max(dim=1, keepdim=True).values.detach()
        product = _switched_step(torch.exp(alpha - shift), step, moved, switch_kind)
        alpha = shift + torch.log(product) + score
        if columns is not None:
            columns.append(alpha)
    if columns is None or final is None:
        return torch.logsumexp(alpha, dim=1)
    table = torch.stack(columns, dim=1)
    return torch.logsumexp(table[torch.arange(n), final], dim=1)
