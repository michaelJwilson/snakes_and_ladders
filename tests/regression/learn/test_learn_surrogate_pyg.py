"""`GraphSurrogate` against PyTorch Geometric's `GINConv`, weights copied across, on the tree tokens (issue #322).

The graph model's layer is ``x + MLP([x, sum_j x_j])``; ``GINConv`` with
``eps = 0`` is ``MLP(x + sum_j x_j)``. The two coincide exactly when the
first linear map treats the node and its neighbour sum alike -- both halves
of its weight equal -- and on such weights PyG's message passing reproduces
ours to ``1e-12`` on twelve five-taxon trees, which pins the neighbour sum,
the residual, the pooling and the decoder. On general weights they are
different architectures, and the second test says so rather than
approximating one by the other. ``GCNConv`` is not a candidate at all: it
normalizes by degree and adds self-loops, neither of which our layer does.

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

torch_geometric = pytest.importorskip("torch_geometric")
from torch_geometric.nn import GINConv, global_add_pool  # noqa: E402

FIVE_TAXA = "simulation_params_5taxa.yaml"
HIDDEN = 8


def _tree_examples() -> Examples:
    params = load_fixture(FIVE_TAXA)
    alignments = [
        dict(
            simulate_alignment(
                params.tau, params.k, params.pi, np.random.default_rng(seed), 60
            ).alignment
        )
        for seed in (1000, 1001)
    ]
    topologies = [list(enumerate_topologies(sorted(a)))[:6] for a in alignments]
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


def _tie(model: GraphSurrogate) -> None:
    """Make each layer's first map read the node and its neighbour sum alike."""
    with torch.no_grad():
        for layer in model.layers:
            assert isinstance(layer, torch.nn.Sequential)
            first = layer[0]
            assert isinstance(first, torch.nn.Linear)
            first.weight[:, HIDDEN:] = first.weight[:, :HIDDEN]


class _Twin(torch.nn.Module):
    """The same architecture in PyG's vocabulary, weights copied from ``ours``.

    Each ``GINConv``'s network is our layer's MLP with the first map's
    node half; the neighbour half is what GIN's own sum supplies.
    """

    def __init__(self, ours: GraphSurrogate) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(ours.embed.in_features, HIDDEN).to(torch.float64)
        self.embed.load_state_dict(ours.embed.state_dict())
        convs = []
        for layer in ours.layers:
            assert isinstance(layer, torch.nn.Sequential)
            first, last = layer[0], layer[2]
            assert isinstance(first, torch.nn.Linear)
            assert isinstance(last, torch.nn.Linear)
            first_map = torch.nn.Linear(HIDDEN, HIDDEN).to(torch.float64)
            last_map = torch.nn.Linear(HIDDEN, HIDDEN).to(torch.float64)
            conv = GINConv(
                torch.nn.Sequential(first_map, torch.nn.SiLU(), last_map), eps=0.0
            )
            # After construction: `GINConv.__init__` re-initializes the
            # network it is handed, so weights copied before it are lost.
            with torch.no_grad():
                first_map.weight.copy_(first.weight[:, :HIDDEN])
                first_map.bias.copy_(first.bias)
                last_map.weight.copy_(last.weight)
                last_map.bias.copy_(last.bias)
            convs.append(conv)
        self.convs = torch.nn.ModuleList(convs)
        self.decode = ours.decode

    def forward(self, batch: _Batch) -> torch.Tensor:
        state = torch.nn.functional.silu(self.embed(batch.tokens))
        edge_index = torch.cat([batch.edges, batch.edges.flip(1)]).T.contiguous()
        for conv in self.convs:
            state = state + conv(state, edge_index)
        pooled = global_add_pool(state, batch.owner, size=batch.n)
        out: torch.Tensor = self.decode(torch.cat([pooled, batch.features], dim=1))[
            :, 0
        ]
        return out


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [0, 1])
def test_pyg_s_gin_reproduces_the_graph_surrogate_on_tied_weights(seed: int) -> None:
    examples = _tree_examples()
    ours = _ours(examples, seed)
    _tie(ours)
    twin = _Twin(ours)
    batch = _Batch(examples)
    with torch.no_grad():
        mine, theirs = ours(batch), twin(batch)
    assert mine.shape == (len(examples),)
    assert float(mine.std()) > 1e-6  # not a constant, or the pin is empty
    torch.testing.assert_close(theirs, mine, rtol=0.0, atol=1e-12)


@pytest.mark.structural
def test_on_general_weights_gin_is_a_different_architecture() -> None:
    # What differs: GIN sums the node and its neighbours *before* its
    # network, so one weight serves both; ours concatenates them, so two
    # do. The twin built from the node half is then not our model, and the
    # gap is the neighbour half's contribution rather than round-off.
    examples = _tree_examples()
    ours = _ours(examples, 2)
    twin = _Twin(ours)
    batch = _Batch(examples)
    with torch.no_grad():
        gap = float((ours(batch) - twin(batch)).abs().max())
    assert gap > 1e-6
    _tie(ours)
    with torch.no_grad():
        closed = float((ours(batch) - _Twin(ours)(batch)).abs().max())
    assert closed < 1e-12
