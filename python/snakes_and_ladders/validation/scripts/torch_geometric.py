"""PyTorch Geometric's ``GINConv`` twin of ``GraphSurrogate``, on a batch the adapter wrote (issue #977).

Each model arrives as ``GraphSurrogate``'s state, keyed ``<index>:<parameter>``,
with the shared ``n_features``, ``n_token_features``, ``hidden`` and
``n_layers``. The twin keeps the model's embedding and decoder; each layer
becomes a ``GINConv`` with ``eps = 0`` whose network is the layer's MLP with
the first map's node half, since GIN's own neighbour sum supplies the other
half. The batch is ``tokens``, the undirected ``edges`` (one row each),
``owner`` (each row's example) and ``features``, with ``n`` examples.

Outputs ``forward``, one row per model, and ``case_seconds``, the twin's
forward alone per model; the measured seconds are their sum.

With ``model`` set to ``set`` (issue #997) each model is a ``SetSurrogate``
and its twin keeps the encoder and the decoder around PyG's
``global_add_pool``; ``edges`` is then unread.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from snakes_and_ladders.learn.surrogate import GraphSurrogate, SetSurrogate
from snakes_and_ladders.validation.protocol import dump, received, timed


class _Twin(torch.nn.Module):
    """``GraphSurrogate`` in PyG's vocabulary, weights copied from ``ours``."""

    def __init__(self, ours: GraphSurrogate, hidden: int, nn: Any) -> None:
        super().__init__()
        self.embed = ours.embed
        self.decode = ours.decode
        self._pool = nn.global_add_pool
        convs = []
        for layer in ours.layers:
            first, last = layer[0], layer[2]  # type: ignore[index]
            first_map = torch.nn.Linear(hidden, hidden).to(torch.float64)
            last_map = torch.nn.Linear(hidden, hidden).to(torch.float64)
            conv = nn.GINConv(
                torch.nn.Sequential(first_map, torch.nn.SiLU(), last_map), eps=0.0
            )
            # After construction: `GINConv.__init__` re-initializes the network
            # it is handed, so weights copied before it are lost.
            with torch.no_grad():
                first_map.weight.copy_(first.weight[:, :hidden])
                first_map.bias.copy_(first.bias)
                last_map.weight.copy_(last.weight)
                last_map.bias.copy_(last.bias)
            convs.append(conv)
        self.convs = torch.nn.ModuleList(convs)

    def forward(self, batch: dict[str, torch.Tensor], n: int) -> torch.Tensor:
        state = torch.nn.functional.silu(self.embed(batch["tokens"]))
        edges = batch["edges"]
        edge_index = torch.cat([edges, edges.flip(1)]).T.contiguous()
        for conv in self.convs:
            state = state + conv(state, edge_index)
        pooled = self._pool(state, batch["owner"], size=n)
        out: torch.Tensor = self.decode(torch.cat([pooled, batch["features"]], dim=1))
        return out[:, 0]


class _SetTwin(torch.nn.Module):
    """``SetSurrogate`` with PyG's ``global_add_pool`` for its sum."""

    def __init__(self, ours: SetSurrogate, nn: Any) -> None:
        super().__init__()
        self.encode = ours.encode
        self.decode = ours.decode
        self._pool = nn.global_add_pool

    def forward(self, batch: dict[str, torch.Tensor], n: int) -> torch.Tensor:
        pooled = self._pool(self.encode(batch["tokens"]), batch["owner"], size=n)
        out: torch.Tensor = self.decode(torch.cat([pooled, batch["features"]], dim=1))
        return out[:, 0]


def main() -> None:
    """Run each model's twin forward on the batch and write the outputs back."""
    from torch_geometric import nn  # the framework, imported only in this interpreter

    inputs, returned = received()
    hidden = int(inputs["hidden"])
    n = int(inputs["n"])
    batch = {
        "tokens": torch.as_tensor(inputs["tokens"], dtype=torch.float64),
        "edges": torch.as_tensor(inputs["edges"], dtype=torch.int64).reshape(-1, 2),
        "owner": torch.as_tensor(inputs["owner"], dtype=torch.int64),
        "features": torch.as_tensor(inputs["features"], dtype=torch.float64),
    }
    forward, case_seconds = [], []
    kind = str(inputs.get("model", np.asarray("graph")))
    for index in range(int(inputs["n_models"])):
        prefix = f"{index}:"
        ours: torch.nn.Module = (
            SetSurrogate(
                int(inputs["n_features"]),
                int(inputs["n_token_features"]),
                hidden=hidden,
            )
            if kind == "set"
            else GraphSurrogate(
                int(inputs["n_features"]),
                int(inputs["n_token_features"]),
                hidden=hidden,
                n_layers=int(inputs["n_layers"]),
            )
        )
        ours.load_state_dict(
            {
                key.removeprefix(prefix): torch.as_tensor(value)
                for key, value in inputs.items()
                if key.startswith(prefix)
            }
        )
        twin: torch.nn.Module = (
            _SetTwin(ours, nn)  # type: ignore[arg-type]
            if kind == "set"
            else _Twin(ours, hidden, nn)  # type: ignore[arg-type]
        )
        with torch.no_grad():
            out, seconds = timed(lambda: twin(batch, n))  # noqa: B023
        forward.append(out.numpy())
        case_seconds.append(seconds)
    dump(
        returned,
        {"forward": np.stack(forward), "case_seconds": np.asarray(case_seconds)},
        float(sum(case_seconds)),
    )


if __name__ == "__main__":
    main()
