"""`GraphSurrogate` against PyTorch Geometric's `GINConv`, weights copied across, in a subprocess (issues #322, #977).

On tied first-layer weights the ``GINConv`` twin reproduces ours to
``1e-12`` on twelve five-taxon trees at two seeds (neighbour sum, residual,
pooling, decoder); on general weights the gap exceeds ``1e-6`` and tying
closes it below ``1e-12``. ``GCNConv`` normalizes by degree and adds
self-loops, which our layer does not.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from snakes_and_ladders.learn.ranking import fixed_length_target, tree_examples
from snakes_and_ladders.learn.surrogate import (
    Examples,
    GraphSurrogate,
    SetSurrogate,
    _Batch,
)
from snakes_and_ladders.sim.simulator import simulate_tree
from snakes_and_ladders.sim.topology import enumerate_topologies
from snakes_and_ladders.validation import torch_geometric

from tests._fixtures import load_fixture
from tests._frameworks import requires

pytestmark = [
    pytest.mark.validation,
    requires("torch_geometric"),
]

FIVE_TAXA = "tree_search/ci.yaml"
HIDDEN = 8
N_LAYERS = 2


def _tree_examples() -> Examples:
    params = load_fixture(FIVE_TAXA)
    alignments = [
        dict(simulate_tree(params, np.random.default_rng(seed), n_sites=60).alignment)
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
        n_layers=N_LAYERS,
    )


def _tied(model: GraphSurrogate) -> GraphSurrogate:
    """``model`` with each layer's first map reading the node and its neighbour sum alike."""
    with torch.no_grad():
        for layer in model.layers:
            assert isinstance(layer, torch.nn.Sequential)
            first = layer[0]
            assert isinstance(first, torch.nn.Linear)
            first.weight[:, HIDDEN:] = first.weight[:, :HIDDEN]
    return model


def _twins(models: list[GraphSurrogate], batch: _Batch) -> np.ndarray:
    return torch_geometric.gin_forward(
        models,
        n_layers=N_LAYERS,
        hidden=HIDDEN,
        tokens=batch.tokens.numpy(),
        edges=batch.edges.numpy(),
        owner=batch.owner.numpy(),
        features=batch.features.numpy(),
    ).forward


@pytest.mark.oracle
def test_pyg_s_gin_reproduces_the_graph_surrogate_on_tied_weights() -> None:
    examples = _tree_examples()
    batch = _Batch(examples)
    models = [_tied(_ours(examples, seed)) for seed in (0, 1)]
    theirs = _twins(models, batch)
    for model, twin in zip(models, theirs, strict=True):
        with torch.no_grad():
            mine = model(batch).numpy()
        assert mine.shape == (len(examples),)
        assert float(mine.std()) > 1e-6  # not a constant, or the pin is empty
        np.testing.assert_allclose(twin, mine, rtol=0.0, atol=1e-12)


@pytest.mark.smoke
def test_on_general_weights_gin_is_a_different_architecture() -> None:
    # GIN sums the node and its neighbours *before* its network, so one
    # weight serves both; ours concatenates them, so two do. The twin built
    # from the node half is then not our model, and the gap is the neighbour
    # half's contribution rather than round-off.
    examples = _tree_examples()
    batch = _Batch(examples)
    general = _ours(examples, 2)
    with torch.no_grad():
        general_out = general(batch).numpy()
    tied = _tied(_ours(examples, 2))
    with torch.no_grad():
        tied_out = tied(batch).numpy()
    theirs = _twins([_ours(examples, 2), tied], batch)
    assert float(np.abs(general_out - theirs[0]).max()) > 1e-6
    assert float(np.abs(tied_out - theirs[1]).max()) < 1e-12


@pytest.mark.oracle
def test_pyg_s_sum_pool_reproduces_the_set_surrogate() -> None:
    # Issue #997: `SetSurrogate` pools before its encoder's last affine map;
    # the twin applies the map to every row and pools with PyG's
    # `global_add_pool`, the textbook order. The same function, summed in
    # another order.
    examples = _tree_examples()
    batch = _Batch(examples)
    torch.manual_seed(0)
    model = SetSurrogate(examples.features.shape[1], examples.tokens[0].shape[1])
    theirs = torch_geometric.set_forward(
        [model],
        hidden=32,
        tokens=batch.tokens.numpy(),
        owner=batch.owner.numpy(),
        features=batch.features.numpy(),
    ).forward[0]
    with torch.no_grad():
        mine = model(batch).numpy()
    assert float(mine.std()) > 1e-6
    np.testing.assert_allclose(theirs, mine, rtol=0.0, atol=1e-11)
