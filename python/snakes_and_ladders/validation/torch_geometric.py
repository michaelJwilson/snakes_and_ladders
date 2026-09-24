"""PyTorch Geometric as an oracle for the graph surrogate's message passing (issue #977).

``GINConv`` with ``eps = 0`` is ``MLP(x + sum_j x_j)``;
:class:`snakes_and_ladders.learn.surrogate.GraphSurrogate`'s layer is
``x + MLP([x, sum_j x_j])``. The two coincide when the first linear map reads
the node and its neighbour sum alike, so on such weights PyG's message
passing, a second implementation of the neighbour sum and the pooling,
reproduces ours. It runs only in ``scripts/torch_geometric.py``, in a
subprocess (issue #322 ran it in-process).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from snakes_and_ladders.validation.runner import run

#: The script this adapter runs.
SCRIPT = "torch_geometric"


@dataclass(frozen=True)
class Forward:
    """The ``GINConv`` twin's output per model, one row each, and what each cost."""

    forward: np.ndarray
    #: Wall seconds of each twin's forward pass alone.
    seconds: np.ndarray


def gin_forward(
    models: Sequence[torch.nn.Module],
    *,
    n_layers: int,
    hidden: int,
    tokens: np.ndarray,
    edges: np.ndarray,
    owner: np.ndarray,
    features: np.ndarray,
) -> Forward:
    """Each ``GraphSurrogate``'s ``GINConv`` twin, run on one batch.

    ``edges`` holds each undirected edge once, as rows of ``tokens``;
    ``owner`` names each row's example, one of ``features``' rows.
    """
    inputs: dict[str, np.ndarray] = {
        "n_models": np.asarray(len(models)),
        "n_features": np.asarray(features.shape[1]),
        "n_token_features": np.asarray(tokens.shape[1]),
        "hidden": np.asarray(hidden),
        "n_layers": np.asarray(n_layers),
        "n": np.asarray(features.shape[0]),
        "tokens": np.ascontiguousarray(tokens, dtype=np.float64),
        "edges": np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2),
        "owner": np.ascontiguousarray(owner, dtype=np.int64),
        "features": np.ascontiguousarray(features, dtype=np.float64),
    }
    for index, model in enumerate(models):
        for key, value in model.state_dict().items():
            inputs[f"{index}:{key}"] = value.detach().numpy()
    out = run(SCRIPT, inputs).outputs
    return Forward(out["forward"], out["case_seconds"])
