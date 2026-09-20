"""Typed loader for the tree fixture files.

Per ``sim/CLAUDE.md``, a simulated dataset's truth is fully defined in a
yaml: topology and branch lengths (tau), the alphabet size (k), the root
distribution (pi), the seed, the number of sites, and the tolerance a
validation test checks simulated frequencies against. No field defaults
silently -- every one of these must be present in the yaml (root
CLAUDE.md, "Do not introduce silent behavior changes").

``tau`` is written out node by node, or -- where writing it out would bury
the fixture in yaml -- declared as ``{balanced: n, height: h}``, which names
:func:`snakes_and_ladders.sim.tree.balanced_tree`'s two arguments. That
function is deterministic, so the declaration fixes the tree exactly as a
literal one does; what it does not do is carry six hundred lines for a
200-leaf instance (issue #582).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from snakes_and_ladders.sim.tree import Node, balanced_tree

_REQUIRED_FIELDS = frozenset({"seed", "n_sites", "tolerance", "k", "pi", "tau"})


@dataclass(frozen=True)
class SimulationParams:
    """Fully-specified inputs for :func:`snakes_and_ladders.sim.simulate.simulate_alignment`.

    Parameters
    ----------
    tau : Node
        Root of the topology, with branch lengths attached to each non-root
        node.
    k : int
        Number of states.
    pi : np.ndarray
        Root state distribution, shape (k,), summing to 1.
    seed : int
        Seed for ``np.random.default_rng``.
    n_sites : int
        Number of alignment columns to simulate.
    tolerance : float
        Absolute tolerance a validation test checks empirical substitution
        frequencies against the analytic Jukes-Cantor transition
        probabilities within, at ``n_sites``.
    """

    tau: Node
    k: int
    pi: np.ndarray
    seed: int
    n_sites: int
    tolerance: float

    #: The fields :func:`snakes_and_ladders.fixtures.load_params` checks are present before
    #: calling :meth:`from_declared`.
    required_fields: ClassVar[frozenset[str]] = _REQUIRED_FIELDS

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Build the truth from a tree fixture's declared mapping.

        ``declared`` is the mapping
        :func:`snakes_and_ladders.fixtures.load_params` read from ``path``
        with :attr:`required_fields` present; ``path`` names the file in
        every error.

        The parsed, validated parameters.

        Raises
        ------
        ValueError
            If a required field is missing, or ``pi`` does not have shape (k,)
            and sum to 1.
        """
        k = int(declared["k"])
        pi = np.asarray(declared["pi"], dtype=np.float64)
        if pi.shape != (k,):
            msg = f"{path}: pi has shape {pi.shape}, expected ({k},)"
            raise ValueError(msg)
        if not np.isclose(pi.sum(), 1.0):
            msg = f"{path}: pi sums to {pi.sum()}, expected 1.0"
            raise ValueError(msg)

        tau = _tau_from_declaration(declared["tau"], path)

        return cls(
            tau=tau,
            k=k,
            pi=pi,
            seed=int(declared["seed"]),
            n_sites=int(declared["n_sites"]),
            tolerance=float(declared["tolerance"]),
        )


def _tau_from_declaration(raw: Any, path: Path) -> Node:
    """The topology a fixture's ``tau`` names, literal or generated.

    Raises
    ------
    ValueError
        If a generated declaration names neither both of ``balanced`` and
        ``height`` nor a literal node.
    """
    if isinstance(raw, dict) and "balanced" in raw:
        missing = {"balanced", "height"} - set(raw)
        if missing:
            msg = f"{path}: a generated tau needs {sorted(missing)} as well"
            raise ValueError(msg)
        unknown = set(raw) - {"balanced", "height"}
        if unknown:
            msg = f"{path}: a generated tau takes no {sorted(unknown)}"
            raise ValueError(msg)
        return balanced_tree(int(raw["balanced"]), float(raw["height"]))
    return _node_from_dict(raw, branch_length=None)


def _node_from_dict(raw: dict[str, Any], branch_length: float | None) -> Node:
    children = tuple(
        _node_from_dict(child, branch_length=float(child["branch_length"]))
        for child in raw.get("children", [])
    )
    return Node(name=str(raw["name"]), branch_length=branch_length, children=children)
