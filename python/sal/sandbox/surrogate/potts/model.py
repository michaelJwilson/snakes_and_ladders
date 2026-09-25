"""The multigrid network: per (site, state) features in, per-site logits over the states out.

**Grids.** The fine grid is the lattice, its operator the coupling-weighted
adjacency from :meth:`~sal.sim.graph.PottsGraph.compressed_adjacency`, so
``A @ x`` at one state is the neighbour field that state would collect. Each
coarser grid is a :func:`~sal.sim.potts.tile_partition` of the one below at
one tile per :data:`COARSENING` sites; its operator carries the bond weight
between two blocks divided by the source block's size.

**Blocks.** Isotropic message passing: ``x + relu(W_s x + W_n (A x) +
W_m mean_q x)``, layer-normalized over channels. The weights act on channels
and are shared across the ``q`` state channels; the mean over states is the
only term mixing them. So relabelling the states permutes the logits, and it
does so bitwise: the mean is summed in sorted order, which a permutation does
not change.

**The V-cycle.** Blocks per grid on the way down, mean-pooling over each tile
to the next grid, then back up by broadcasting to the tile's sites, a
concatenated skip connection and blocks again. A linear head, summed row by
row, gives one logit per (site, state); the per-site softmax over states is the training loss's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import nn

from sal.sim.graph import PottsGraph
from sal.sim.potts import tile_partition

#: Sites per coarse block: a grid of ``n`` sites coarsens to ``n // 10``.
COARSENING = 10

#: Coarse grids below the lattice.
DEPTH = 3

#: Feature channels per (site, state).
CHANNELS = 24

#: Message-passing blocks per grid on each leg of the V-cycle.
BLOCKS_PER_GRID = 2

#: Input features per (site, state); see :meth:`Surrogate.features`.
N_FEATURES = 4


@dataclass(frozen=True)
class Grid:
    """One grid of the hierarchy.

    Parameters
    ----------
    n_sites : int
        Its site count.
    operator : torch.Tensor
        Sparse COO, coalesced, ``(n_sites, n_sites)``, the weighted neighbour sum.
    assign : torch.Tensor | None
        The block of the next coarser grid each site joins, ``int64``; ``None``
        on the coarsest.
    block_sizes : torch.Tensor | None
        Sites per coarse block, ``float32``; ``None`` on the coarsest.
    """

    n_sites: int
    operator: torch.Tensor
    assign: torch.Tensor | None
    block_sizes: torch.Tensor | None


def _operator(
    n: int, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray
) -> torch.Tensor:
    """A coalesced sparse COO ``(n, n)`` float32 operator from its entries.

    COO rather than CSR: PyTorch's CSR support warns that it is in beta, and
    the product per column is the same sum in the same order either way.
    """
    index = torch.as_tensor(np.stack([rows, cols]), dtype=torch.int64)
    values = torch.as_tensor(weights, dtype=torch.float32)
    coo = torch.sparse_coo_tensor(index, values, (n, n), check_invariants=True)
    return coo.coalesce()


def hierarchy(graph: PottsGraph, depth: int = DEPTH, seed: int = 0) -> list[Grid]:
    """The lattice and ``depth`` coarser grids, each a seeded tiling of the one below.

    Returns
    -------
    list[Grid]
        Finest first, ``depth + 1`` grids.
    """
    offsets, neighbours, couplings = graph.compressed_adjacency()
    rows = np.repeat(np.arange(graph.n_nodes), np.diff(offsets))
    cols, weights = np.array(neighbours), np.array(couplings, dtype=float)
    rng = np.random.default_rng(seed)
    current, n = graph, graph.n_nodes
    grids: list[Grid] = []
    operator = _operator(n, rows, cols, weights)
    for _ in range(depth):
        k = max(2, n // COARSENING)
        assign = tile_partition(current, k, rng)
        sizes = np.bincount(assign, minlength=k).astype(np.float64)
        grids.append(
            Grid(
                n,
                operator,
                torch.as_tensor(assign, dtype=torch.int64),
                torch.as_tensor(sizes, dtype=torch.float32),
            )
        )
        a, b = assign[rows], assign[cols]
        crossing = a != b
        pairs, inverse = np.unique(a[crossing] * k + b[crossing], return_inverse=True)
        summed = np.bincount(inverse, weights=weights[crossing])
        rows, cols, weights = pairs // k, pairs % k, summed
        operator = _operator(k, rows, cols, summed / sizes[rows])
        edges = tuple(
            (int(x), int(y)) for x, y in zip(rows, cols, strict=True) if x < y
        )
        current = PottsGraph(n_nodes=k, edges=edges, coupling=(1.0,) * len(edges))
        n = k
    grids.append(Grid(n, operator, None, None))
    return grids


def _neighbour_sum(operator: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``A @ x`` over sites for ``x`` of shape ``(batch, sites, q, channels)``, one sparse product."""
    batch, sites, q, channels = x.shape
    flat = x.permute(1, 0, 2, 3).reshape(sites, batch * q * channels)
    return (operator @ flat).reshape(sites, batch, q, channels).permute(1, 0, 2, 3)


def _state_mean(x: torch.Tensor) -> torch.Tensor:
    """The mean over the state axis, summed in sorted order so a relabelling reproduces it bitwise."""
    return torch.sort(x, dim=2).values.sum(dim=2, keepdim=True) / x.shape[2]


class Block(nn.Module):
    """One isotropic message-passing step; weights on channels, shared across states."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.self_weight = nn.Linear(channels, channels)
        self.neighbour_weight = nn.Linear(channels, channels, bias=False)
        self.state_weight = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
        """``norm(x + relu(W_s x + W_n A x + W_m mean_q x))``.

        Returns
        -------
        torch.Tensor
            ``x``'s shape.
        """
        update = (
            self.self_weight(x)
            + self.neighbour_weight(_neighbour_sum(operator, x))
            + self.state_weight(_state_mean(x))
        )
        out: torch.Tensor = self.norm(x + torch.relu(update))
        return out


def _pool(x: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Mean over each coarse block's sites."""
    assert grid.assign is not None
    assert grid.block_sizes is not None
    batch, _, q, channels = x.shape
    total = x.new_zeros(batch, grid.block_sizes.numel(), q, channels)
    total.index_add_(1, grid.assign, x)
    return total / grid.block_sizes[None, :, None, None]


class Surrogate(nn.Module):
    """The V-cycle over :func:`hierarchy`'s grids, 28,873 parameters at the defaults.

    Parameters
    ----------
    coupling : float
        The lattice's coupling; the field is read in its units.
    channels, depth, blocks_per_grid : int
        The width, the coarse-grid count and the blocks per grid per leg.
    """

    def __init__(
        self,
        coupling: float,
        channels: int = CHANNELS,
        depth: int = DEPTH,
        blocks_per_grid: int = BLOCKS_PER_GRID,
    ) -> None:
        super().__init__()
        self.coupling = coupling
        self.depth = depth
        self.embed = nn.Linear(N_FEATURES, channels)
        self.down = nn.ModuleList(
            nn.ModuleList(Block(channels) for _ in range(blocks_per_grid))
            for _ in range(depth + 1)
        )
        self.up = nn.ModuleList(
            nn.ModuleList(Block(channels) for _ in range(blocks_per_grid))
            for _ in range(depth)
        )
        self.merge = nn.ModuleList(
            nn.Linear(2 * channels, channels) for _ in range(depth)
        )
        self.head = nn.Linear(channels, 1)

    def features(self, field: torch.Tensor) -> torch.Tensor:
        """``h / J``, ``(h - max_q h) / J``, the argmax indicator where the maximum is positive, and one.

        Returns
        -------
        torch.Tensor
            ``(batch, sites, q, N_FEATURES)``.
        """
        top = field.max(dim=2, keepdim=True).values
        return torch.stack(
            [
                field / self.coupling,
                (field - top) / self.coupling,
                (field == top).to(field.dtype) * (top > 0).to(field.dtype),
                torch.ones_like(field),
            ],
            dim=-1,
        )

    def forward(self, field: torch.Tensor, grids: list[Grid]) -> torch.Tensor:
        """Logits per (site, state) for a batch of fields.

        Parameters
        ----------
        field : torch.Tensor
            ``(batch, sites, q)``, float32, in energy units.
        grids : list[Grid]
            :func:`hierarchy` of the fields' lattice, at this model's depth.

        Returns
        -------
        torch.Tensor
            ``(batch, sites, q)``.
        """
        x = self.embed(self.features(field))
        skips: list[torch.Tensor] = []
        for level in range(self.depth + 1):
            for block in cast(nn.ModuleList, self.down[level]):
                x = block(x, grids[level].operator)
            if level < self.depth:
                skips.append(x)
                x = _pool(x, grids[level])
        for level in reversed(range(self.depth)):
            assign = grids[level].assign
            assert assign is not None
            x = self.merge[level](torch.cat([x[:, assign], skips[level]], dim=-1))
            for block in cast(nn.ModuleList, self.up[level]):
                x = block(x, grids[level].operator)
        # The head as a row-wise reduction, not a matrix-vector product: the
        # latter's kernel sums a row in an order that depends on where the row
        # sits, which would break bitwise equivariance.
        logits: torch.Tensor = (x * self.head.weight[0]).sum(dim=-1) + self.head.bias
        return logits


def predict(model: Surrogate, grids: list[Grid], field: np.ndarray) -> np.ndarray:
    """The per-site argmax labelling of one field, without a gradient.

    Returns
    -------
    np.ndarray
        ``int64``, shape ``(n_nodes,)``.
    """
    with torch.no_grad():
        logits = model(torch.as_tensor(field, dtype=torch.float32)[None], grids)
    return np.asarray(logits[0].argmax(dim=-1).numpy(), dtype=np.int64)
