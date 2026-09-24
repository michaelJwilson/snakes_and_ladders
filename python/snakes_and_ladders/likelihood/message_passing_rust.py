"""Rust tree schedule (``snakes_and_ladders.oxisal.tree_message_passing``),
pinned against :func:`snakes_and_ladders.likelihood.message_passing.sum_product`
on its Python route, the NumPy oracle (``likelihood/CLAUDE.md``, "The reference
implementation is the oracle and it stays"), which is itself pinned against
:mod:`snakes_and_ladders.likelihood.message_passing_reference`.

**Why this one is a port.** Root ``CLAUDE.md`` reserves the Rust backend for
CPU-bound hot paths, and the tree schedule's cost is the dispatch rather than
the arithmetic: on a chain each level carries one message, so a chain of 200 is
798 steps of a handful of NumPy calls over a single four-wide row, with the
plan that groups them rebuilt per call. Issue #754's stress profile put
``tree_passes`` at 29.8% of the run and ``_logsumexp_last`` at 10.6%;
``docs/experiments/020`` carries the ranking and ``027`` the measurement that
admitted the port.

**The boundary is crossed once and the marshalling is not stored.**
:class:`~snakes_and_ladders.likelihood.schedule.Layout` holds its incidences as
lists of lists, which root ``CLAUDE.md`` keeps on the Python side of the memory
rule and issue #586 measured; they cross as offsets and a flat array, built per
call. That is the recompute side of the recompute-or-store decision, made
rather than defaulted: the marshalling is **0.266 ms** against **18.8 ms** of
Python sends at the chain of 200, and a store on the layout would buy nothing,
since the layout is itself built per call.

**Bitwise, at every cardinality this repository's fixtures declare.** The
kernel sums a variable's incoming rows onto zeros in factor order, shifts a
log-sum-exp by the row maximum and sums left to right --- which is NumPy's
pairwise reduction below eight terms --- adds a factor's incoming rows onto its
table in axis order, and reduces the axes it does not send along in row-major
order, which is the layout ``transpose(...).reshape(n, c, -1)`` hands
``logsumexp``. So agreement with the oracle is **bitwise** on the chain, the
Potts tree, the coupled fixture and random trees of degree up to four; above
eight terms in a reduction NumPy's pairwise sum reassociates and this does not,
which is a floating sum's order and nothing else, bounded by
:data:`~snakes_and_ladders.likelihood.device.CROSS_DEVICE_RTOL_FLOAT64`.
``tests/regression/likelihood/test_message_passing_rust.py`` asserts the
equality and the bound separately rather than the looser one everywhere.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from snakes_and_ladders import oxisal
from snakes_and_ladders.likelihood.schedule import Layout


def _offsets(lists: list[list[int]]) -> np.ndarray:
    """``[0, len(lists[0]), ...]``: where each row starts in the flat array."""
    out = np.zeros(len(lists) + 1, dtype=np.int64)
    np.cumsum([len(row) for row in lists], out=out[1:])
    return out


@dataclass(frozen=True)
class TreeMessages:
    """Both directions of the tree schedule, as edge rows.

    Parameters
    ----------
    to_variable : np.ndarray
        ``(n_edges, width)``, factor to variable.
    to_factor : np.ndarray
        ``(n_edges, width)``, variable to factor.
    """

    to_variable: np.ndarray
    to_factor: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        """``(to_variable, to_factor)``: the order callers unpack."""
        yield from (self.to_variable, self.to_factor)


def tree_messages(layout: Layout, *, maximum: bool) -> TreeMessages:
    """The tree schedule's two passes, computed in Rust.

    Leaves to root and then root to leaves, rooted at
    :attr:`~snakes_and_ladders.likelihood.schedule.Layout.root` as the Python
    route and the reference are, so the messages compare edge for edge. The
    kernel derives the rooted walk from the layout rather than taking a plan:
    it is ``O(nodes)`` against the arithmetic of the passes, and deriving it
    checks the tree they rest on.

    Parameters
    ----------
    layout : Layout
    maximum : bool
        Max-product rather than sum-product: a factor reduces the axes it does
        not send along by ``max`` rather than by ``logsumexp``.

    Returns
    -------
    TreeMessages
        Both ``(n_edges, width)``, as
        :func:`snakes_and_ladders.likelihood.message_passing._run` carries
        them. The arrays are the extension's own, reshaped here and not
        copied, so a pin reads what the kernel wrote. A row's columns past
        its variable's cardinality are zero and never read.

    Raises
    ------
    ValueError
        If the graph has a cycle or is disconnected --- the tree schedule is
        exact only on a tree --- or if the layout's arrays disagree with each
        other.
    """
    n_edges = layout.n_edges
    to_variable, to_factor = oxisal.tree_message_passing(
        np.asarray(layout.cardinality, dtype=np.int64),
        _offsets(layout.variable_edges),
        np.fromiter(
            itertools.chain.from_iterable(layout.variable_edges),
            dtype=np.int64,
            count=n_edges,
        ),
        _offsets(layout.factor_edges),
        np.fromiter(
            itertools.chain.from_iterable(layout.factor_edges),
            dtype=np.int64,
            count=n_edges,
        ),
        np.asarray(layout.edge_variable, dtype=np.int64),
        # The tables cross stacked and flat, in factor order; `reshape(-1)` on
        # a C-contiguous table is a view, so a factor of any degree crosses
        # without a copy of its own.
        _table_offsets(layout),
        np.concatenate(
            [
                np.ascontiguousarray(factor.log_table, dtype=np.float64).reshape(-1)
                for factor in layout.graph.factors
            ]
        ),
        layout.width,
        maximum,
    )
    shape = (n_edges, layout.width)
    return TreeMessages(
        np.asarray(to_variable).reshape(shape), np.asarray(to_factor).reshape(shape)
    )


def _table_offsets(layout: Layout) -> np.ndarray:
    """Where each factor's table starts in the concatenated tables."""
    out = np.zeros(len(layout.graph.factors) + 1, dtype=np.int64)
    np.cumsum([factor.log_table.size for factor in layout.graph.factors], out=out[1:])
    return out
