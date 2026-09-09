"""`GraphSurrogate` against the PyTorch Geometric front that lost to it (issues #322, #390).

The front is :mod:`snakes_and_ladders.sandbox.pyg_surrogate`, kept there
because issue #390 declined it: the ``index_add`` it would replace is 7.0% of
a fit, under #341's 10% bar, and PyG is 1.14x slower at the size these tests
run. These tests are what stop that decline from becoming an assertion about
a version nobody is holding.

The graph model's layer is ``x + MLP([x, sum_j x_j])``; ``GINConv`` with
``eps = 0`` is ``MLP(x + sum_j x_j)``. The two coincide exactly when the first
linear map treats the node and its neighbour sum alike -- both halves of its
weight equal -- and
:func:`~snakes_and_ladders.sandbox.pyg_surrogate.tie_weights` puts a model
there. On such weights PyG's message passing reproduces ours to the bound
below, which pins the neighbour sum, the residual, the pooling and the
decoder. On general weights they are different architectures, and the third
test says so rather than approximating one by the other. ``GCNConv`` is not a
candidate at all: it normalizes by degree and adds self-loops, neither of
which our layer does.

**The pinned bound is 1.33e-15**, the largest entry-wise gap issue #390
recorded, on 60 examples over 480 nodes and 420 edges. Reproduced here at that
same size: 0 to 8.88e-16 over model seeds 0 to 9, and 0 to 4.44e-16 on the
twelve-example fixture. Absolute rather than relative because the quantity is
one decoder output of order 1e-2, not a sum over sites.

Skips without the ``frameworks`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.surrogate import Examples, GraphSurrogate, _Batch
from snakes_and_ladders.search.surrogate import fixed_length_target, tree_examples
from snakes_and_ladders.search.topology import enumerate_topologies
from snakes_and_ladders.sim.simulate import simulate_alignment

from tests._fixtures import load_fixture

pytest.importorskip("torch_geometric")
from snakes_and_ladders.sandbox.pyg_surrogate import (
    PyGGraphSurrogate,
    tie_weights,
)

FIVE_TAXA = "tree_search/ci.yaml"
HIDDEN = 8

#: The largest entry-wise gap issue #390 recorded between the two forwards on
#: tied weights. This module's docstring says what is reproduced against it.
TIED_GAP = 1.33e-15


def _tree_examples(n_alignments: int = 2, n_topologies: int = 6) -> Examples:
    """Five-taxon neighbourhoods as surrogate examples.

    The default is twelve examples. ``(4, 15)`` is the 60 examples, 480 nodes
    and 420 edges issue #390 timed and measured the gap on.
    """
    params = load_fixture(FIVE_TAXA)
    alignments = [
        dict(
            simulate_alignment(
                params.tau, params.k, params.pi, np.random.default_rng(seed), 60
            ).alignment
        )
        for seed in range(1000, 1000 + n_alignments)
    ]
    topologies = [
        list(enumerate_topologies(sorted(a)))[:n_topologies] for a in alignments
    ]
    return tree_examples(
        alignments,
        topologies,
        params.k,
        np.asarray(params.pi),
        fixed_length_target(params.k, np.asarray(params.pi), 0.1),
    )


def _ours(examples: Examples, seed: int) -> GraphSurrogate:
    torch.manual_seed(seed)
    return GraphSurrogate(
        examples.features.shape[1],
        examples.tokens[0].shape[1],
        hidden=HIDDEN,
        n_layers=2,
    )


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("seed", [0, 1])
def test_pyg_s_gin_reproduces_the_graph_surrogate_on_tied_weights(seed: int) -> None:
    examples = _tree_examples()
    ours = _ours(examples, seed)
    tie_weights(ours)
    front = PyGGraphSurrogate(ours)
    batch = _Batch(examples)
    with torch.no_grad():
        mine, theirs = ours(batch), front(batch)
    assert mine.shape == (len(examples),)
    assert float(mine.std()) > 1e-6  # not a constant, or the pin is empty
    torch.testing.assert_close(theirs, mine, rtol=0.0, atol=TIED_GAP)


@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(10))
def test_the_gap_at_the_size_390_measured_stays_inside_the_recorded_bound(
    seed: int,
) -> None:
    # The declining experiment's own size, asserted rather than assumed: 60
    # examples, 480 nodes, 420 edges. The bound is what #390 recorded, and a
    # PyG release that widened the gap fails here rather than quietly
    # rewriting the finding.
    examples = _tree_examples(n_alignments=4, n_topologies=15)
    batch = _Batch(examples)
    assert (len(examples), batch.tokens.shape[0], batch.edges.shape[0]) == (
        60,
        480,
        420,
    )
    ours = _ours(examples, seed)
    tie_weights(ours)
    with torch.no_grad():
        gap = float((ours(batch) - PyGGraphSurrogate(ours)(batch)).abs().max())

    assert gap <= TIED_GAP


@pytest.mark.structural
def test_on_general_weights_gin_is_a_different_architecture() -> None:
    # What differs: GIN sums the node and its neighbours *before* its
    # network, so one weight serves both; ours concatenates them, so two
    # do. The front built from the node half is then not our model, and the
    # gap is the neighbour half's contribution rather than round-off.
    examples = _tree_examples()
    ours = _ours(examples, 2)
    front = PyGGraphSurrogate(ours)
    batch = _Batch(examples)
    with torch.no_grad():
        gap = float((ours(batch) - front(batch)).abs().max())
    assert gap > 1e-6
    tie_weights(ours)
    with torch.no_grad():
        closed = float((ours(batch) - PyGGraphSurrogate(ours)(batch)).abs().max())
    assert closed <= TIED_GAP


@pytest.mark.structural
def test_the_front_shares_the_decoder_rather_than_copying_it() -> None:
    # A copied decoder would put a second set of parameters into a comparison
    # that is supposed to be at fixed ones, and the two forwards would then
    # agree by construction of the copy rather than by the architectures
    # coinciding. Identity, not equality of values.
    examples = _tree_examples()
    ours = _ours(examples, 3)

    assert PyGGraphSurrogate(ours).decode is ours.decode
