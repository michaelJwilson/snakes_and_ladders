"""Typed loader for ``simulation_params.yaml`` files.

Per ``sim/CLAUDE.md``, a simulated dataset's truth is fully defined in a
yaml: topology and branch lengths (tau), the alphabet size (k), the root
distribution (pi), the seed, the number of sites, and the tolerance a
validation test checks simulated frequencies against. No field defaults
silently -- every one of these must be present in the yaml (root
CLAUDE.md, "Do not introduce silent behavior changes").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phylo.fixtures import load_declared
from phylo.sim.tree import Node

_REQUIRED_FIELDS = frozenset({"seed", "n_sites", "tolerance", "k", "pi", "tau"})


@dataclass(frozen=True)
class SimulationParams:
    """Fully-specified inputs for :func:`phylo.sim.simulate.simulate_alignment`.

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


def load_simulation_params(path: Path) -> SimulationParams:
    """Load and validate a ``simulation_params.yaml`` file.

    Parameters
    ----------
    path : Path
        Path to the yaml file.

    Returns
    -------
    SimulationParams
        The parsed, validated parameters.

    Raises
    ------
    ValueError
        If a required field is missing, or ``pi`` does not have shape (k,)
        and sum to 1.
    """
    raw = load_declared(path, _REQUIRED_FIELDS)

    k = int(raw["k"])
    pi = np.asarray(raw["pi"], dtype=np.float64)
    if pi.shape != (k,):
        msg = f"{path}: pi has shape {pi.shape}, expected ({k},)"
        raise ValueError(msg)
    if not np.isclose(pi.sum(), 1.0):
        msg = f"{path}: pi sums to {pi.sum()}, expected 1.0"
        raise ValueError(msg)

    tau = _node_from_dict(raw["tau"], branch_length=None)

    return SimulationParams(
        tau=tau,
        k=k,
        pi=pi,
        seed=int(raw["seed"]),
        n_sites=int(raw["n_sites"]),
        tolerance=float(raw["tolerance"]),
    )


def _node_from_dict(raw: dict[str, Any], branch_length: float | None) -> Node:
    children = tuple(
        _node_from_dict(child, branch_length=float(child["branch_length"]))
        for child in raw.get("children", [])
    )
    return Node(name=str(raw["name"]), branch_length=branch_length, children=children)
