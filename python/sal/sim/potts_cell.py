"""The Potts reference cell: planted domains under a field that misleads (issue #1390).

One instance modelled on a downstream workload, declared at three tiers and
varied one factor at a time from ``stress``. A lattice at ``J`` and ``q``;
contiguous planted domains whose label shares are declared; and a unary whose
top-two margin is ``margin_ratio * J * d`` (``d`` the graph's mean degree),
whose top label is the planted one except on ``mislead`` of sites, where a
uniform wrong label is top and the planted one second, and to which Gaussian
jitter of scale ``jitter`` is added so that ties between labellings are
measure zero. #1378's study built the same cell inline
(``sal.qa.known_ground_states.workload``); the ``stress`` tier reproduces it
bitwise, so the study can move onto this fixture.

**Draw order.** The plane-wave surface, the misled sites, the wrong labels and
the jitter are drawn, in that order, from ``default_rng(seed)``: the order
#1378 drew them in. A k-NN graph's points come from ``default_rng([seed, 1])``
and the forbidden labels from ``default_rng([seed, 2])``, so a variant adding
either leaves every other draw where it was.

**Variants.** ``stress.yaml`` declares ``variants``: a name, and the one
factor that differs from the file's own instance (issue #1390's list).
:meth:`PottsReferenceParams.variant` builds it. Variants are a key of the
``stress`` file rather than problems of their own because each is defined *as*
one factor away from ``stress``: a directory per variant would restate every
other field, and would need a ``ci`` tier the variant has no meaning at.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
from scipy.spatial import cKDTree

from sal.sim.graph import (
    BoundaryCondition,
    PottsGraph,
    boundary_from_declared,
    lattice_graph,
    triangular_lattice_graph,
)
from sal.sim.potts import energy, forbid

_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "graph",
        "shape",
        "boundary",
        "n_states",
        "coupling",
        "shares",
        "margin_ratio",
        "mislead",
        "jitter",
        "planted_digest",
    }
)

#: The graphs a cell is declared on.
GRAPHS = ("triangular", "square", "knn", "disconnected")

#: Plane waves summed into the surface the planting thresholds.
N_WAVES = 8


@dataclass(frozen=True)
class PlantedPotts:
    """One drawn cell: the graph, the field, the planted labelling.

    Parameters
    ----------
    graph : PottsGraph
        The lattice, every coupling ``J``.
    field : np.ndarray
        ``(n_nodes, n_states)`` log-weights; ``-inf`` where a label is
        forbidden.
    planted : np.ndarray
        ``(n_nodes,)`` labels the domains were cut into.
    n_states : int
        ``q``.
    margin : float
        The top-two margin before jitter, ``margin_ratio * J * d``.
    """

    graph: PottsGraph
    field: np.ndarray
    planted: np.ndarray
    n_states: int
    margin: float

    @property
    def planted_energy(self) -> float:
        """The planted labelling's energy: an upper bound on the minimum."""
        return energy(self.graph, self.field, self.planted)

    @property
    def misled(self) -> float:
        """The fraction of sites whose field argmax is not the planted label."""
        return float(np.mean(np.argmax(self.field, axis=1) != self.planted))


def planted_digest(cell: PlantedPotts) -> str:
    """16 hex characters of the SHA-256 of the planted labels, then the field.

    Returns
    -------
    str
    """
    digest = hashlib.sha256(np.ascontiguousarray(cell.planted, dtype=np.int64))
    digest.update(np.ascontiguousarray(cell.field, dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class PottsReferenceParams:
    """The declared cell, drawn by :meth:`instance`.

    Parameters
    ----------
    seed : int
        The seed every draw is made from.
    graph : str
        One of :data:`GRAPHS`.
    shape : tuple[int, int]
        ``(rows, columns)``; ``rows * columns`` sites on every graph.
    boundary : BoundaryCondition
        The lattices' boundary; unused by ``knn``.
    n_states : int
        ``q``.
    coupling : float
        ``J``.
    shares : tuple[float, ...]
        The planted label shares, one per label, summing to 1.
    margin_ratio : float
        The top-two margin over ``J * d``.
    mislead : float
        The fraction of sites whose top label is a wrong one.
    jitter : float
        The scale of the Gaussian noise added to every field entry.
    planted_digest : str
        :func:`planted_digest` of :meth:`instance`, as recorded.
    neighbours : int
        ``k`` of the ``knn`` graph.
    components : int
        Bands of the ``disconnected`` graph, each its own periodic lattice.
    forbidden : float
        The fraction of sites at which one label, neither the planted nor the
        top two, is forbidden.
    variants : Mapping[str, Mapping[str, Any]]
        One-factor variants by name, as the file declares them.
    """

    seed: int
    graph: str
    shape: tuple[int, int]
    boundary: BoundaryCondition
    n_states: int
    coupling: float
    shares: tuple[float, ...]
    margin_ratio: float
    mislead: float
    jitter: float
    planted_digest: str
    neighbours: int = 6
    components: int = 1
    forbidden: float = 0.0
    variants: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    declared: Mapping[str, Any] = field(default_factory=dict, compare=False)
    path: Path = field(default=Path(), compare=False)

    #: The fields :func:`sal.fixtures.load_params` checks are present.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the declared cell, refusing a value no instance can carry.

        Raises
        ------
        ValueError
            If the graph is unknown, the shares are not a distribution over
            ``n_states`` labels, or a fraction lies outside ``[0, 1]``.
        """
        graph = str(declared["graph"])
        if graph not in GRAPHS:
            msg = f"{path}: graph {graph!r} is not one of {list(GRAPHS)}"
            raise ValueError(msg)
        rows, columns = (int(extent) for extent in declared["shape"])
        n_states = int(declared["n_states"])
        shares = tuple(float(share) for share in declared["shares"])
        if len(shares) != n_states or min(shares) <= 0.0:
            msg = f"{path}: shares must be {n_states} positive numbers, got {shares}"
            raise ValueError(msg)
        if abs(sum(shares) - 1.0) > 1e-9:
            msg = f"{path}: shares sum to {sum(shares)}, not 1"
            raise ValueError(msg)
        fractions = {
            name: float(declared.get(name, 0.0)) for name in ("mislead", "forbidden")
        }
        for name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                msg = f"{path}: {name} must lie in [0, 1], got {value}"
                raise ValueError(msg)
        if fractions["forbidden"] > 0.0 and n_states < 3:
            msg = f"{path}: forbidding a label needs n_states >= 3, got {n_states}"
            raise ValueError(msg)
        components = int(declared.get("components", 1))
        if rows % components:
            msg = f"{path}: {components} components do not divide {rows} rows"
            raise ValueError(msg)
        variants = declared.get("variants") or {}
        return cls(
            seed=int(declared["seed"]),
            graph=graph,
            shape=(rows, columns),
            boundary=boundary_from_declared(path, declared["boundary"]),
            n_states=n_states,
            coupling=float(declared["coupling"]),
            shares=shares,
            margin_ratio=float(declared["margin_ratio"]),
            mislead=fractions["mislead"],
            jitter=float(declared["jitter"]),
            planted_digest=str(declared["planted_digest"]),
            neighbours=int(declared.get("neighbours", 6)),
            components=components,
            forbidden=fractions["forbidden"],
            variants={str(name): dict(entry) for name, entry in variants.items()},
            declared=dict(declared),
            path=path,
        )

    @property
    def n_nodes(self) -> int:
        """Sites."""
        return self.shape[0] * self.shape[1]

    def variant(self, name: str) -> PottsReferenceParams:
        """The declared variant ``name``: this cell with its one factor changed.

        Its ``planted_digest`` is the variant's own where it declares one, and
        empty otherwise; a variant declares no variants.

        Raises
        ------
        KeyError
            If no variant of that name is declared; the message lists those
            that are.
        """
        if name not in self.variants:
            msg = f"{self.path}: no variant {name!r}; declares {sorted(self.variants)}"
            raise KeyError(msg)
        changed = {
            **self.declared,
            "planted_digest": "",
            **self.variants[name],
            "variants": {},
        }
        return type(self).from_declared(changed, self.path)

    def _graph(self) -> tuple[PottsGraph, np.ndarray, np.ndarray]:
        """The graph and each site's two surface coordinates."""
        rows, columns = self.shape
        if self.graph == "knn":
            points = np.random.default_rng([self.seed, 1]).random((self.n_nodes, 2))
            _, found = cKDTree(points).query(points, k=self.neighbours + 1)
            pairs = np.sort(
                np.stack(
                    [
                        np.repeat(np.arange(self.n_nodes), self.neighbours),
                        found[:, 1:].ravel(),
                    ],
                    axis=1,
                ),
                axis=1,
            )
            unique = np.unique(pairs, axis=0)
            graph = PottsGraph(
                n_nodes=self.n_nodes,
                edges=tuple((int(i), int(j)) for i, j in unique),
                coupling=(self.coupling,) * len(unique),
            )
            return graph, points[:, 0], points[:, 1]
        build = triangular_lattice_graph if self.graph != "square" else lattice_graph
        band = rows // self.components
        piece = build((band, columns), self.boundary, self.coupling)
        if self.components == 1:
            graph = piece
        else:
            per = band * columns
            edges = tuple(
                (i + part * per, j + part * per)
                for part in range(self.components)
                for i, j in piece.edges
            )
            graph = PottsGraph(
                n_nodes=self.n_nodes,
                edges=edges,
                coupling=(self.coupling,) * len(edges),
            )
        r, c = np.divmod(np.arange(self.n_nodes), columns)
        return graph, r, c

    def instance(self) -> PlantedPotts:
        """Draw the cell from :attr:`seed`, in the order the module states.

        Returns
        -------
        PlantedPotts
        """
        rng = np.random.default_rng(self.seed)
        graph, first, second_axis = self._graph()
        n, q = self.n_nodes, self.n_states
        rows, columns = self.shape
        surface = np.zeros(n)
        for _ in range(N_WAVES):
            kr, kc = rng.integers(1, 4), rng.integers(1, 4)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            if self.graph == "knn":
                argument = kr * first + kc * second_axis
            else:
                # #1378's arithmetic, operation for operation, so the stress
                # tier is its cell bitwise.
                argument = kr * first / rows + kc * second_axis / columns
            surface += np.cos(2.0 * np.pi * argument + phase)
        cuts = np.quantile(surface, np.cumsum(self.shares)[:-1])
        planted = np.searchsorted(cuts, surface).astype(np.int64)
        degree = 2.0 * len(graph.edges) / n
        margin = self.margin_ratio * self.coupling * degree
        values = np.full((n, q), -margin)
        misled = rng.random(n) < self.mislead
        wrong = (planted + rng.integers(1, q, size=n)) % q
        top = np.where(misled, wrong, planted)
        second = np.where(misled, planted, (planted + 1) % q)
        sites = np.arange(n)
        values[sites, second] = 0.0
        values[sites, top] = margin
        values += rng.normal(0.0, self.jitter, size=values.shape)
        if self.forbidden > 0.0:
            draws = np.random.default_rng([self.seed, 2])
            chosen = draws.random(n) < self.forbidden
            score = draws.random((n, q))
            for kept in (top, second, planted):
                score[sites, kept] = -1.0
            allowed = np.ones((n, q), dtype=bool)
            allowed[sites[chosen], np.argmax(score, axis=1)[chosen]] = False
            values = forbid(values, allowed)
        return PlantedPotts(graph, values, planted, q, margin)
