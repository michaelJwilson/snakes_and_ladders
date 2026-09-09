"""PyTorch Geometric fronting :class:`~snakes_and_ladders.learn.surrogate.GraphSurrogate`'s forward (issues #322, #390).

The front that was measured and declined. The graph surrogate's layer is
``x + MLP([x, sum_j x_j])``, one ``index_add`` for the neighbour sum and one
for the pooling; ``torch_geometric.nn.GINConv`` at ``eps = 0`` is
``MLP(x + sum_j x_j)`` and ``global_add_pool`` is the pooling, so the same
architecture is expressible in PyG's vocabulary whenever the first linear map
reads the node and its neighbour sum alike. :func:`tie_weights` puts a
``GraphSurrogate`` on those weights and :class:`PyGGraphSurrogate` copies them
across, which is what lets the two be compared at fixed parameters rather
than through a fit.

**Declined on two numbers**, both in `STATUS.md` and
`docs/experiments/008-frameworks-on-three-hot-paths.md`. The term PyG would
front is 7.0% of a 120-example fit by `cProfile` self time --- the two
``index_add`` calls --- which is under issue #341's 10% bar, so the port pays
for nothing whichever side is faster. It is also slower where the tests run:
0.972 ms against 0.849 ms per forward at 60 examples, 480 nodes and 420
edges, 1.14x, and 1.15x faster at four times that size. One thread on a
shared four-core Linux x86-64 host under the exclusive lock, best of 20.

`tests/regression/learn/test_learn_surrogate_pyg.py` pins the two forwards
against each other on tied weights and states the gap it measures, so a later
PyG that moves either number is a failing test rather than a claim about a
version nobody is holding.

On general weights the two are different architectures and this module does
not pretend otherwise: GIN sums the node and its neighbours before its
network, so one weight matrix serves both, where ``GraphSurrogate``
concatenates them and two do. The tie is the subspace where they coincide,
named here and asserted in the test rather than approximated.

Imports ``torch_geometric`` at module scope: it is the ``frameworks`` extra,
and a caller without it gets an ``ImportError`` here rather than a silent
fallback to the model this exists to referee.
"""

from __future__ import annotations

import torch
from torch_geometric.nn import GINConv, global_add_pool

from snakes_and_ladders.learn.surrogate import GraphSurrogate, _Batch


def tie_weights(model: GraphSurrogate) -> None:
    """Make each layer's first map read the node and its neighbour sum alike.

    ``GraphSurrogate``'s layer applies its MLP to ``[x, sum_j x_j]``, so its
    first map has two halves; GIN applies its MLP to ``x + sum_j x_j``, so it
    has one. Setting the two halves equal is the subspace where the two
    architectures compute the same function, and it is where the comparison
    at fixed parameters is made.

    Mutates ``model`` in place, under ``no_grad``.

    Parameters
    ----------
    model : GraphSurrogate
        The model to place on the tied subspace.
    """
    hidden = model.embed.out_features
    with torch.no_grad():
        for layer in model.layers:
            assert isinstance(layer, torch.nn.Sequential)
            first = layer[0]
            assert isinstance(first, torch.nn.Linear)
            first.weight[:, hidden:] = first.weight[:, :hidden]


class PyGGraphSurrogate(torch.nn.Module):
    """``GraphSurrogate``'s forward built from ``GINConv`` and ``global_add_pool``.

    Layer for layer what
    :class:`~snakes_and_ladders.learn.surrogate.GraphSurrogate` is, in PyG's
    vocabulary: the token embedding and the decoder are shared objects rather
    than copies --- the decoder is not graph code and copying it would put a
    second set of parameters in the comparison --- and each message-passing
    layer becomes one ``GINConv`` at ``eps = 0`` carrying the node half of
    that layer's first map. The neighbour half is what GIN's own sum supplies,
    which is why the two agree only on :func:`tie_weights`'s subspace.

    Takes and returns what ``GraphSurrogate`` takes and returns, so the two
    can be pinned against each other and timed against each other with no
    adapter in between.

    Parameters
    ----------
    model : GraphSurrogate
        The model whose parameters this reads. Read once, at construction:
        weights tied after this is built are not seen.
    """

    def __init__(self, model: GraphSurrogate) -> None:
        super().__init__()
        hidden = model.embed.out_features
        self.embed = torch.nn.Linear(model.embed.in_features, hidden).to(torch.float64)
        self.embed.load_state_dict(model.embed.state_dict())
        convs = []
        for layer in model.layers:
            assert isinstance(layer, torch.nn.Sequential)
            first, last = layer[0], layer[2]
            assert isinstance(first, torch.nn.Linear)
            assert isinstance(last, torch.nn.Linear)
            first_map = torch.nn.Linear(hidden, hidden).to(torch.float64)
            last_map = torch.nn.Linear(hidden, hidden).to(torch.float64)
            conv = GINConv(
                torch.nn.Sequential(first_map, torch.nn.SiLU(), last_map), eps=0.0
            )
            # After construction, and not before: `GINConv.__init__`
            # re-initializes the network it is handed, so weights copied into
            # it earlier are lost.
            with torch.no_grad():
                first_map.weight.copy_(first.weight[:, :hidden])
                first_map.bias.copy_(first.bias)
                last_map.weight.copy_(last.weight)
                last_map.bias.copy_(last.bias)
            convs.append(conv)
        self.convs = torch.nn.ModuleList(convs)
        self.decode = model.decode

    def forward(self, batch: _Batch) -> torch.Tensor:
        """One prediction per example, from the batch's tokens, edges and features.

        Parameters
        ----------
        batch : _Batch
            The flattened tokens, the edge list into them, and the example
            each token row belongs to, as
            :class:`~snakes_and_ladders.learn.surrogate.GraphSurrogate` reads
            them.

        Returns
        -------
        torch.Tensor
            Shape ``(batch.n,)``.
        """
        state = torch.nn.functional.silu(self.embed(batch.tokens))
        # PyG takes both directions explicitly, as `(2, n_arcs)`; the batch
        # carries one row per undirected edge, which is the same convention
        # `GraphSurrogate` reverses inline.
        edge_index = torch.cat([batch.edges, batch.edges.flip(1)]).T.contiguous()
        for conv in self.convs:
            state = state + conv(state, edge_index)
        pooled = global_add_pool(state, batch.owner, size=batch.n)
        out: torch.Tensor = self.decode(torch.cat([pooled, batch.features], dim=1))[
            :, 0
        ]
        return out


__all__ = ["PyGGraphSurrogate", "tie_weights"]
