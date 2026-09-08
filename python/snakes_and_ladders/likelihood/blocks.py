"""The block-frequency bound: exact where blocks repeat, bounded where they do not (issue #408).

Site-pattern compression (:mod:`snakes_and_ladders.likelihood.patterns`) is
exact and saturates: once every distinct column has appeared, a longer
alignment costs nothing more. Blocks of ``N > 1`` consecutive sites do not
saturate --- there are ``k ** (N n)`` of them and they repeat rarely --- so
exact block compression buys nothing. What it buys is a **bound**.

Partition the alignment into blocks of ``N`` consecutive sites and count how
often each distinct block occurs. Blocks occurring at least ``min_count``
times are evaluated exactly, through the pattern compression of their
columns. The rare tail is not evaluated at all: every one of its sites
contributes a per-site log-likelihood lying between two numbers that depend
on the tree and the branch lengths alone, so the tail contributes an
interval of width ``(number of bounded sites) x (upper - lower)``. The
result contains the exact log-likelihood by construction, at a cost the
cutoff sets.

**The two numbers.** :func:`site_log_likelihood_extremes` brackets
``log Pr(x)`` over *every* column ``x``, in one pass over the tree and
independently of the alignment's length. The recursion is the pruning
recursion with the leaf messages replaced by their extremes: a leaf sends
its parent ``P(s, x)`` for its observed state ``x``, so over all ``x`` that
message lies in ``[min_j P(s, j), max_j P(s, j)]``, and the products and
non-negative sums above it are monotone, so bounds propagate. It is a
bound and not the exact extreme --- the maximum over each child is taken
independently, which no single column need attain --- and that is what makes
it sound and loose rather than tight and wrong.

**What the interval brackets.** ``block_frequency_interval`` brackets the
log-likelihood **at the branch lengths it is given**. :class:`BlockFrequencyBound`
presents that as a :class:`~snakes_and_ladders.bound.Surrogate` at
least-squares lengths, as :class:`~snakes_and_ladders.likelihood.surrogate.PlugInLikelihood`
does. Only the lower end is then a bound on the *maximized* log-likelihood
--- a value at feasible lengths is at most the maximum --- so only the lower
end may be certified against a fit. The upper end bounds the evaluation, not
the fit, and the class says so where it is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.bound import Bound
from snakes_and_ladders.likelihood.patterns import compress_columns
from snakes_and_ladders.likelihood.pruning_torch import (
    branch_order,
    log_likelihood,
    transition_probabilities,
)
from snakes_and_ladders.likelihood.surrogate import jc_distances, least_squares_lengths
from snakes_and_ladders.search.topology import Topology
from snakes_and_ladders.sim.tree import Node


@dataclass(frozen=True)
class Interval:
    """What the bound returns: two ends, and what it cost to get them.

    Parameters
    ----------
    lower : torch.Tensor
        0-dimensional; at most the exact log-likelihood at the same branch
        lengths.
    upper : torch.Tensor
        0-dimensional; at least it.
    exact : torch.Tensor
        0-dimensional; the frequent blocks' own log-likelihood, which both
        ends are built from. It is a log-likelihood over a subset of the
        sites and claims nothing about the whole, which is what makes it a
        ranking signal rather than a bound (``likelihood/CLAUDE.md``, "A
        cheap value that claims nothing ranks").
    exact_sites : int
        Sites inside a block frequent enough to evaluate.
    bounded_sites : int
        Sites in the rare tail, which the interval's width is proportional
        to.
    exact_blocks : int
        Distinct blocks evaluated, of ``n_blocks`` in the partition.
    n_blocks : int
        Blocks the alignment partitions into, distinct or not.
    evaluated_columns : int
        Columns the exact evaluation actually ran over, after the frequent
        blocks' columns were pattern-compressed. This is the cost, and what
        it is compared against is the alignment's length.
    """

    lower: torch.Tensor
    upper: torch.Tensor
    exact: torch.Tensor
    exact_sites: int
    bounded_sites: int
    exact_blocks: int
    n_blocks: int
    evaluated_columns: int

    @property
    def extrapolated(self) -> torch.Tensor:
        """The exact half scaled to the whole alignment: a point estimate, not a bound.

        The frequent blocks are the same sites for every topology, so this
        is one subsample scored the same way each time, and it orders
        topologies where the interval's ends -- dominated by a tail term
        that varies with the tree and measures nothing about fit -- do not.

        Raises
        ------
        ValueError
            If no block was frequent enough to evaluate. The exact half is
            then zero for every topology, which is above every
            log-likelihood and orders nothing; refusing beats returning it
            (``likelihood/CLAUDE.md``, "Refuse rather than return an
            unconverged number"). Lower the cutoff or the block size.
        """
        if self.exact_sites == 0:
            msg = (
                f"no block of the {self.n_blocks} reached the cutoff, so the "
                "exact half is empty and cannot be scaled to the alignment; "
                "lower min_count or block_size"
            )
            raise ValueError(msg)
        return self.exact * (self.exact_sites + self.bounded_sites) / self.exact_sites

    @property
    def width(self) -> float:
        """``upper - lower``: zero where nothing was bounded."""
        return float(self.upper - self.lower)

    def contains(self, value: float, *, tolerance: float = 0.0) -> bool:
        """Whether ``value`` lies inside, allowing ``tolerance`` relative slack.

        The slack admits the floating-point noise of two summation orders,
        and nothing more: a bound that needs a wider one is not a bound.
        """
        slack = tolerance * abs(value)
        return float(self.lower) <= value + slack and float(self.upper) >= value - slack


def site_log_likelihood_extremes(
    tau: Node,
    k: int,
    pi: np.ndarray | torch.Tensor,
    branch_lengths: torch.Tensor,
    *,
    rate_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounds on ``log Pr(x)`` holding for every column ``x``, in one pass over the tree.

    The pruning recursion in the log domain with the leaf messages replaced
    by their extremes over the leaf's state, which is what makes the result
    independent of the data. Sound rather than tight: each child's extreme
    is taken independently of the others, so no column need attain either
    end.

    Parameters
    ----------
    tau : Node
        Root of the topology. Its own ``branch_length`` fields are ignored.
    k : int
        Number of states.
    pi : np.ndarray | torch.Tensor
        Root state distribution, shape ``(k,)``.
    branch_lengths : torch.Tensor
        Shape ``(len(branch_order(tau)),)``, as
        :func:`~snakes_and_ladders.likelihood.pruning_torch.log_likelihood`
        takes it.
    rate_matrix : torch.Tensor | None
        A general rate matrix, or ``None`` for the closed-form Jukes-Cantor
        transition probabilities.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(lower, upper)``, both 0-dimensional. ``lower`` is ``-inf`` where
        a branch of length zero makes some column impossible, which is
        correct rather than a failure: such a column has log-likelihood
        ``-inf``.

    Raises
    ------
    ValueError
        If ``pi`` or ``branch_lengths`` has the wrong shape.
    """
    dtype = branch_lengths.dtype
    device = branch_lengths.device
    pi_t = torch.as_tensor(pi, dtype=dtype, device=device)
    if pi_t.shape != (k,):
        msg = f"pi has shape {tuple(pi_t.shape)}, expected ({k},)"
        raise ValueError(msg)
    order = branch_order(tau)
    if branch_lengths.shape != (len(order),):
        msg = (
            f"branch_lengths has shape {tuple(branch_lengths.shape)}, "
            f"expected ({len(order)},) to match branch_order(tau)"
        )
        raise ValueError(msg)
    index = {name: position for position, name in enumerate(order)}
    transitions = transition_probabilities(branch_lengths, k, rate_matrix)

    def message(child: Node) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounds on ``log sum_j P(s, j) L_child(j)``, shape ``(k,)`` each."""
        log_transition = torch.log(transitions[index[child.name]])
        if child.is_leaf:
            # The message is the column P(., x) for the observed state x, so
            # over every x it lies between the row's smallest and largest
            # entry. log is monotone, so the extremes commute with it.
            return log_transition.amin(dim=1), log_transition.amax(dim=1)
        lower, upper = _partial(child)
        return (
            torch.logsumexp(log_transition + lower[None, :], dim=1),
            torch.logsumexp(log_transition + upper[None, :], dim=1),
        )

    def _partial(node: Node) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounds on ``log L_node(s)``, shape ``(k,)`` each."""
        lower = torch.zeros(k, dtype=dtype, device=device)
        upper = torch.zeros(k, dtype=dtype, device=device)
        for child in node.children:
            child_lower, child_upper = message(child)
            lower = lower + child_lower
            upper = upper + child_upper
        return lower, upper

    root_lower, root_upper = _partial(tau)
    log_pi = torch.log(pi_t)
    return (
        torch.logsumexp(log_pi + root_lower, dim=0),
        torch.logsumexp(log_pi + root_upper, dim=0),
    )


def _partition(columns: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Blocks of ``block_size`` consecutive columns, flattened one block per row.

    Returns the full blocks and the short remainder separately: a remainder
    is a block of a different length, so it is never equal to a full one and
    counting it among them would be wrong.
    """
    n_leaves, n_sites = columns.shape
    n_full = n_sites // block_size
    full = (
        columns[:, : n_full * block_size]
        .reshape(n_leaves, n_full, block_size)
        .transpose(1, 0, 2)
        .reshape(n_full, n_leaves * block_size)
    )
    return np.ascontiguousarray(full), columns[:, n_full * block_size :]


def block_frequency_interval(
    tau: Node,
    k: int,
    pi: np.ndarray | torch.Tensor,
    alignment: Mapping[str, np.ndarray],
    branch_lengths: torch.Tensor,
    *,
    block_size: int,
    min_count: int,
    rate_matrix: torch.Tensor | None = None,
) -> Interval:
    """An interval containing the log-likelihood, exact on frequent blocks.

    Parameters
    ----------
    tau : Node
        Root of the topology. Its own ``branch_length`` fields are ignored.
    k : int
        Number of states.
    pi : np.ndarray | torch.Tensor
        Root state distribution, shape ``(k,)``.
    alignment : Mapping[str, np.ndarray]
        Leaf name to its observed states, each of shape ``(n_sites,)``.
    branch_lengths : torch.Tensor
        Shape ``(len(branch_order(tau)),)``. The interval brackets the
        log-likelihood at *these* lengths, not the maximized one.
    block_size : int
        Sites per block. ``1`` makes every block a column, and the frequent
        set is then the site-pattern table itself.
    min_count : int
        Occurrences a distinct block needs before it is evaluated exactly.
        ``1`` evaluates everything and returns an interval of zero width.
    rate_matrix : torch.Tensor | None
        A general rate matrix, or ``None`` for closed-form Jukes-Cantor.

    Returns
    -------
    Interval
        Containing the exact log-likelihood at ``branch_lengths``.

    Raises
    ------
    ValueError
        If ``block_size`` or ``min_count`` is below one, or the alignment is
        empty or ragged.
    """
    if block_size < 1:
        msg = f"block_size must be at least 1, got {block_size}"
        raise ValueError(msg)
    if min_count < 1:
        msg = f"min_count must be at least 1, got {min_count}"
        raise ValueError(msg)
    if not alignment:
        msg = "an alignment with no taxa has no blocks to partition"
        raise ValueError(msg)

    names = tuple(sorted(alignment))
    rows = [np.asarray(alignment[name], dtype=np.int64) for name in names]
    shapes = {row.shape for row in rows}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        msg = f"the alignment is ragged: rows have shapes {sorted(shapes)}"
        raise ValueError(msg)
    columns = np.ascontiguousarray(np.stack(rows))
    n_leaves, n_sites = columns.shape

    full, remainder = _partition(columns, block_size)
    unique, counts = (
        np.unique(full, axis=0, return_counts=True)
        if full.shape[0]
        else (full, np.zeros(0, dtype=np.int64))
    )
    frequent = counts >= min_count
    n_blocks = int(full.shape[0]) + int(remainder.shape[1] > 0)
    exact_blocks = int(frequent.sum())

    # The frequent blocks' columns, each carrying its block's count. A
    # column repeated across two frequent blocks is one column with the
    # counts added, which is the pattern compression of the exact half.
    pieces: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    if exact_blocks:
        chosen = unique[frequent]
        pieces.append(
            chosen.reshape(chosen.shape[0], n_leaves, block_size)
            .transpose(1, 0, 2)
            .reshape(n_leaves, chosen.shape[0] * block_size)
        )
        weights.append(np.repeat(counts[frequent], block_size))
    # The remainder is one block of a shorter length, so it occurs once.
    if remainder.shape[1] and min_count <= 1:
        exact_blocks += 1
        pieces.append(remainder)
        weights.append(np.ones(remainder.shape[1], dtype=np.int64))

    if pieces:
        patterns = compress_columns(
            names,
            np.ascontiguousarray(np.concatenate(pieces, axis=1)),
            np.concatenate(weights),
        )
        exact = log_likelihood(
            tau,
            k,
            pi,
            patterns.alignment,
            branch_lengths,
            weights=patterns.weights,
            rate_matrix=rate_matrix,
        )
        exact_sites = patterns.n_sites
        evaluated_columns = patterns.n_patterns
    else:
        exact = torch.zeros(
            (), dtype=branch_lengths.dtype, device=branch_lengths.device
        )
        exact_sites = 0
        evaluated_columns = 0

    bounded_sites = n_sites - exact_sites
    lower, upper = site_log_likelihood_extremes(
        tau, k, pi, branch_lengths, rate_matrix=rate_matrix
    )
    return Interval(
        lower=exact + bounded_sites * lower,
        upper=exact + bounded_sites * upper,
        exact=exact,
        exact_sites=exact_sites,
        bounded_sites=bounded_sites,
        exact_blocks=exact_blocks,
        n_blocks=n_blocks,
        evaluated_columns=evaluated_columns,
    )


class BlockFrequencyBound:
    """The block-frequency interval as a :class:`~snakes_and_ladders.bound.Surrogate`.

    Branch lengths come from the least-squares fit to the Jukes-Cantor
    pairwise distances, as
    :class:`~snakes_and_ladders.likelihood.surrogate.PlugInLikelihood`'s do,
    so the surrogate is a function of the topology and the alignment alone
    and plugs into ``search.infer``'s lazy ranking without a fit.

    **Which end claims what.** Both ends bracket the log-likelihood at those
    lengths. Only the lower end also bounds the *maximized* log-likelihood,
    because a value at feasible lengths is at most the maximum; it is the
    end that may be certified against a fit, and the default. The upper end
    is declared ``Bound.UPPER`` against the evaluation at those lengths,
    never against a fit.

    **The cutoff is a count, so it scales with the alignment.** How much of
    the alignment stays exact is what a cutoff means, and that is
    ``min_count`` against the length. On the five-taxon fixture a cutoff of
    4 at 1200 sites and one of 8 at 2000 retain the same fraction and rank
    the same way; a cutoff of 8 at 1200 sites loses the optimum from half
    the starts. A caller choosing one for a new length scales it rather than
    carrying the number across.

    **Neither end ranks, and the measurement says so.** The tail term is
    ``(bounded sites) x (per-site extreme)``, the extreme varies with the
    tree, and at any cutoff above one it is larger than the differences
    between neighbouring topologies -- so an ordering by either end is an
    ordering by the tail's looseness. Over the 15 five-taxon topologies at
    2000 sites and cutoff 8, the lower end's best is topology 2 against the
    fitted best 13; the upper end's happens to coincide, which is a fact
    about that instance and not a property of the end. The ``POINT`` claim
    exists for this: it drops the tail entirely and orders by the frequent
    blocks alone, the same sites for every candidate. It refuses where the
    cutoff leaves no frequent block, since it would otherwise rank every
    topology by the same zero.

    Parameters
    ----------
    k : int
        Number of states.
    pi : np.ndarray
        Root state distribution, shape ``(k,)``.
    block_size : int
        Sites per block.
    min_count : int
        Occurrences a block needs before it is evaluated exactly.
    claim : Bound
        What :meth:`__call__` returns and :attr:`kind` reports: an end of
        the interval, or ``POINT`` for
        :attr:`Interval.extrapolated` -- the frequent blocks' own
        log-likelihood scaled to the alignment's length, which claims
        nothing and is what ranks.
    """

    def __init__(
        self,
        k: int,
        pi: np.ndarray,
        *,
        block_size: int,
        min_count: int,
        claim: Bound = Bound.LOWER,
    ) -> None:
        if claim not in (Bound.LOWER, Bound.UPPER, Bound.POINT):
            msg = f"claim must be a Bound, got {claim}"
            raise ValueError(msg)
        self.k = k
        self.pi = np.asarray(pi, dtype=float)
        self.block_size = block_size
        self.min_count = min_count
        self._claim = claim

    @property
    def kind(self) -> Bound:
        """Which end this surrogate reports."""
        return self._claim

    def lengths(
        self, topology: Topology, alignment: Mapping[str, np.ndarray]
    ) -> torch.Tensor:
        """The feasible branch lengths the interval is evaluated at."""
        return least_squares_lengths(topology, jc_distances(alignment, self.k))

    def interval(
        self, topology: Topology, alignment: Mapping[str, np.ndarray]
    ) -> Interval:
        """The whole interval, and what it cost, at :meth:`lengths`."""
        return block_frequency_interval(
            topology,
            self.k,
            self.pi,
            alignment,
            self.lengths(topology, alignment),
            block_size=self.block_size,
            min_count=self.min_count,
        )

    def __call__(self, structure: object, data: object) -> torch.Tensor:
        if not isinstance(structure, Node) or not isinstance(data, Mapping):
            msg = "a tree surrogate takes a topology and an alignment"
            raise TypeError(msg)
        interval = self.interval(structure, data)
        if self._claim is Bound.LOWER:
            return interval.lower
        if self._claim is Bound.UPPER:
            return interval.upper
        return interval.extrapolated


__all__ = [
    "BlockFrequencyBound",
    "Interval",
    "block_frequency_interval",
    "site_log_likelihood_extremes",
]
