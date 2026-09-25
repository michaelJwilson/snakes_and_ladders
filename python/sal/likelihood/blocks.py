"""The block-frequency bound: exact where blocks repeat, bounded where they do not (issue #408).

Site-pattern compression (:mod:`sal.likelihood.patterns`) is
exact and saturates. Blocks of ``N > 1`` consecutive sites do not --- there
are ``k ** (N n)`` of them and they repeat rarely --- so exact block
compression buys nothing. What it buys is a **bound**.

Partition the alignment into blocks of ``N`` consecutive sites and count how
often each distinct block occurs. Blocks occurring at least ``min_count``
times are evaluated exactly, through the pattern compression of their
columns. The rare tail is not evaluated: each of its sites contributes a
per-site log-likelihood between two numbers depending on the tree and the
branch lengths alone, so the tail contributes an interval of width
``(bounded sites) x (upper - lower)``. The result contains the exact
log-likelihood by construction, at a cost the cutoff sets.

**The two numbers.** :func:`site_log_likelihood_extremes` brackets
``log Pr(x)`` over *every* column ``x``, in one pass over the tree and
independently of the alignment's length. It is the pruning recursion with the
leaf messages replaced by their extremes: a leaf's message lies in
``[min_j P(s, j), max_j P(s, j)]``, and the products and non-negative sums
above it are monotone. Each child's maximum is taken independently, which no
single column need attain, so it is sound and loose rather than tight and
wrong.

**What the interval brackets.** ``block_frequency_interval`` brackets the
log-likelihood **at the branch lengths it is given**.
:class:`BlockFrequencyBound` presents that as a
:class:`~sal.bound.Surrogate` at least-squares lengths, as
:class:`~sal.likelihood.surrogate.PlugInLikelihood` does. Only
the lower end then bounds the *maximized* log-likelihood --- a value at
feasible lengths is at most the maximum --- so only it may be certified
against a fit.

**What it costs, which is more than the evaluation it replaces.** Measured,
one thread: the interval is 4.0x the uncompressed evaluation at five taxa by
2,000 sites and 5.8x at four taxa by 20,000, and the cutoff moves it by 17%
and 2.5% because the block partition's sort dominates. The saving is in
*fits* --- a branch-length fit is 254 ms against a 3.65 ms interval --- never
in forward passes; ``tests/benchmarks/test_likelihood_blocks_bench.py``
carries the table and `STATUS.md` the conclusion. For a cheaper evaluation
the right tool is :mod:`sal.likelihood.patterns`.

**Arrays in, floats out.** No derivative is taken through the interval, so
it takes array-likes and returns ``float`` (issue #1011). Two quantities come
from :mod:`~sal.likelihood.torch.pruning`, whose tensors the
fits differentiate: the exact half's evaluation and the transition matrices
``P(t)`` the extremes are read from. Each crosses there once per call, and
``torch`` is imported at that call rather than with the module.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from sal.bound import Bound, Surrogate
from sal.likelihood.patterns import compress_columns
from sal.likelihood.pruning_common import (
    check_branch_lengths_shape,
    check_pi_shape,
)
from sal.likelihood.surrogate import jc_distances, least_squares_lengths
from sal.numerics import logsumexp
from sal.sim.topology import Topology
from sal.sim.tree import Node

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Interval:
    """What the bound returns: two ends, and what it cost to get them.

    Parameters
    ----------
    lower : float
        At most the exact log-likelihood at the same branch lengths.
    upper : float
        At least it.
    exact : float
        The frequent blocks' own log-likelihood, which both
        ends are built from. Over a subset of the sites, so it claims nothing
        about the whole and is a ranking signal rather than a bound
        (``likelihood/CLAUDE.md``, "A cheap value that claims nothing
        ranks").
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

    lower: float
    upper: float
    exact: float
    exact_sites: int
    bounded_sites: int
    exact_blocks: int
    n_blocks: int
    evaluated_columns: int

    @property
    def extrapolated(self) -> float:
        """The exact half scaled to the whole alignment: a point estimate, not a bound.

        The frequent blocks are the same sites for every topology, so this is
        one subsample scored the same way each time. It orders topologies
        where the interval's ends -- dominated by a tail term varying with the
        tree -- do not.

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
        return self.upper - self.lower

    def contains(self, value: float, *, tolerance: float = 0.0) -> bool:
        """Whether ``value`` lies inside, allowing ``tolerance`` relative slack.

        The slack admits the floating-point noise of two summation orders,
        and nothing more: a bound that needs a wider one is not a bound.
        """
        slack = tolerance * abs(value)
        return self.lower <= value + slack and self.upper >= value - slack


@dataclass(frozen=True)
class Extremes:
    """The two ends :func:`site_log_likelihood_extremes` returns.

    Parameters
    ----------
    lower : float
        At most every column's ``log Pr(x)``. ``-inf`` where a branch of
        length zero makes some column impossible.
    upper : float
        At least every column's.
    """

    lower: float
    upper: float

    def __iter__(self) -> Iterator[float]:
        """``(lower, upper)``: the order callers unpack."""
        yield from (self.lower, self.upper)


def _log_total(terms: np.ndarray, axis: int) -> np.ndarray:
    """``log sum exp`` along ``axis``, ``-inf`` where every term is ``-inf``.

    :func:`~sal.numerics.logsumexp` shifts by the maximum, and
    a maximum of ``-inf`` makes that shift ``nan``. A child message of
    ``-inf`` in every state is what a zero-length leaf branch gives the lower
    end, and the sum it stands for is zero, so its logarithm is ``-inf``.
    """
    with np.errstate(invalid="ignore"):
        total = logsumexp(terms, axis=axis)
    return np.where(np.isneginf(terms.max(axis=axis)), -np.inf, total)


def site_log_likelihood_extremes(
    tau: Node,
    k: int,
    pi: npt.ArrayLike,
    branch_lengths: npt.ArrayLike,
    *,
    rate_matrix: npt.ArrayLike | None = None,
) -> Extremes:
    """Bounds on ``log Pr(x)`` holding for every column ``x``, in one pass over the tree.

    The pruning recursion in the log domain with the leaf messages replaced
    by their extremes over the leaf's state, which makes the result
    independent of the data. Sound rather than tight: each child's extreme is
    taken independently, so no column need attain either end.

    Parameters
    ----------
    tau : Node
        Root of the topology. Its own ``branch_length`` fields are ignored.
    k : int
        Number of states.
    pi : npt.ArrayLike
        Root state distribution, shape ``(k,)``.
    branch_lengths : npt.ArrayLike
        Shape ``(len(branch_order(tau)),)``, in the order
        :func:`~sal.likelihood.torch.pruning.log_likelihood`
        takes it; read as ``float64``.
    rate_matrix : npt.ArrayLike | None
        A general rate matrix, or ``None`` for the closed-form Jukes-Cantor
        transition probabilities.

    Returns
    -------
    Extremes
        The two ends. ``lower`` is ``-inf`` where a branch of length zero
        makes some column impossible, which is correct rather than a
        failure: such a column has log-likelihood ``-inf``.

    Raises
    ------
    ValueError
        If ``pi`` or ``branch_lengths`` has the wrong shape.
    """
    import torch

    from sal.likelihood.torch.pruning import (
        branch_order,
        transition_probabilities,
    )

    lengths = np.asarray(branch_lengths, dtype=np.float64)
    pi_values = np.asarray(pi, dtype=np.float64)
    check_pi_shape(pi_values.shape, k)
    order = branch_order(tau)
    check_branch_lengths_shape(lengths.shape, len(order))
    index = {name: position for position, name in enumerate(order)}
    # `P(t)` is the model's definition and lives with the tensors the fits
    # differentiate; it crosses here once, and everything after is NumPy.
    transitions = transition_probabilities(
        torch.from_numpy(lengths), k, _rate_tensor(rate_matrix)
    ).numpy()
    with np.errstate(divide="ignore"):
        log_transitions = np.log(transitions)
        log_pi = np.log(pi_values)

    def message(child: Node) -> tuple[np.ndarray, np.ndarray]:
        """Bounds on ``log sum_j P(s, j) L_child(j)``, shape ``(k,)`` each."""
        log_transition = log_transitions[index[child.name]]
        if child.is_leaf:
            # The message is the column P(., x) for the observed state x, so
            # over every x it lies between the row's smallest and largest
            # entry. log is monotone, so the extremes commute with it.
            return log_transition.min(axis=1), log_transition.max(axis=1)
        lower, upper = _partial(child)
        return (
            _log_total(log_transition + lower[None, :], axis=1),
            _log_total(log_transition + upper[None, :], axis=1),
        )

    def _partial(node: Node) -> tuple[np.ndarray, np.ndarray]:
        """Bounds on ``log L_node(s)``, shape ``(k,)`` each."""
        lower = np.zeros(k)
        upper = np.zeros(k)
        for child in node.children:
            child_lower, child_upper = message(child)
            lower = lower + child_lower
            upper = upper + child_upper
        return lower, upper

    root_lower, root_upper = _partial(tau)
    return Extremes(
        float(_log_total(log_pi + root_lower, axis=0)),
        float(_log_total(log_pi + root_upper, axis=0)),
    )


def _rate_tensor(rate_matrix: npt.ArrayLike | None) -> torch.Tensor | None:
    """``rate_matrix`` as the ``float64`` tensor ``likelihood.torch.pruning`` takes, or ``None``."""
    if rate_matrix is None:
        return None
    import torch

    return torch.from_numpy(np.array(rate_matrix, dtype=np.float64))


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
    pi: npt.ArrayLike,
    alignment: Mapping[str, np.ndarray],
    branch_lengths: npt.ArrayLike,
    *,
    block_size: int,
    min_count: int,
    rate_matrix: npt.ArrayLike | None = None,
) -> Interval:
    """An interval containing the log-likelihood, exact on frequent blocks.

    Parameters
    ----------
    tau : Node
        Root of the topology. Its own ``branch_length`` fields are ignored.
    k : int
        Number of states.
    pi : npt.ArrayLike
        Root state distribution, shape ``(k,)``.
    alignment : Mapping[str, np.ndarray]
        Leaf name to its observed states, each of shape ``(n_sites,)``.
    branch_lengths : npt.ArrayLike
        Shape ``(len(branch_order(tau)),)``, read as ``float64``. The
        interval brackets the log-likelihood at *these* lengths, not the
        maximized one.
    block_size : int
        Sites per block. ``1`` makes every block a column, and the frequent
        set is then the site-pattern table itself.
    min_count : int
        Occurrences a distinct block needs before it is evaluated exactly.
        ``1`` evaluates everything and returns an interval of zero width.
    rate_matrix : npt.ArrayLike | None
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

    lengths = np.asarray(branch_lengths, dtype=np.float64)
    if pieces:
        patterns = compress_columns(
            names,
            np.ascontiguousarray(np.concatenate(pieces, axis=1)),
            np.concatenate(weights),
        )
        # The exact half is one evaluation of the tensor recursion the fits
        # differentiate; the lengths cross into it here, once.
        import torch

        from sal.likelihood.torch.pruning import log_likelihood

        exact = float(
            log_likelihood(
                tau,
                k,
                np.asarray(pi, dtype=np.float64),
                patterns.alignment,
                torch.from_numpy(lengths),
                weights=patterns.weights,
                rate_matrix=_rate_tensor(rate_matrix),
            )
        )
        exact_sites = patterns.n_sites
        evaluated_columns = patterns.n_patterns
    else:
        exact = 0.0
        exact_sites = 0
        evaluated_columns = 0

    bounded_sites = n_sites - exact_sites
    extremes = site_log_likelihood_extremes(
        tau, k, pi, lengths, rate_matrix=rate_matrix
    )
    return Interval(
        lower=exact + bounded_sites * extremes.lower,
        upper=exact + bounded_sites * extremes.upper,
        exact=exact,
        exact_sites=exact_sites,
        bounded_sites=bounded_sites,
        exact_blocks=exact_blocks,
        n_blocks=n_blocks,
        evaluated_columns=evaluated_columns,
    )


class BlockFrequencyBound(Surrogate):
    """The block-frequency interval as a :class:`~sal.bound.Surrogate`.

    Branch lengths come from the least-squares fit to the Jukes-Cantor
    pairwise distances, as
    :class:`~sal.likelihood.surrogate.PlugInLikelihood`'s do,
    so the surrogate is a function of the topology and the alignment alone and
    plugs into ``search.infer``'s lazy ranking without a fit.

    **Which end claims what.** Both ends bracket the log-likelihood at those
    lengths. Only the lower end also bounds the *maximized* log-likelihood, so
    only it may be certified against a fit, and it is the default. The upper
    end is declared ``Bound.UPPER`` against the evaluation, never against a
    fit.

    **The cutoff is a count, so it scales with the alignment.** On the
    five-taxon fixture a cutoff of 4 at 1200 sites and one of 8 at 2000 retain
    the same fraction and rank the same way; a cutoff of 8 at 1200 sites loses
    the optimum from half the starts. A caller choosing one for a new length
    scales it rather than carrying the number across.

    **Neither end ranks, and the measurement says so.** The tail term is
    ``(bounded sites) x (per-site extreme)``, the extreme varies with the
    tree, and at any cutoff above one it exceeds the differences between
    neighbouring topologies. Over the 15 five-taxon topologies at 2000 sites
    and cutoff 8, the lower end's best is topology 2 against the fitted best
    13; the upper end's coincidence is a fact about that instance. The
    ``POINT`` claim drops the tail and orders by the frequent blocks alone,
    the same sites for every candidate, and refuses where the cutoff leaves no
    frequent block rather than ranking every topology by the same zero.

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
    ) -> np.ndarray:
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

    def __call__(self, structure: object, data: object) -> float:
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
    "Extremes",
    "Interval",
    "block_frequency_interval",
    "site_log_likelihood_extremes",
]
