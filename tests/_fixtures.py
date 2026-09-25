"""Shared access to the tree fixtures of the problem registry.

One spelling of the fixture location, per `DEV.md`'s Test Layout; the
directory is the registry's (:mod:`sal.sim.fixtures`), so the
suite and the figures agree. The Felsenstein- and Farris-zone trees (issue
#209) live here because small- and large-parsimony tests score the same two.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sal.fixtures import load_params
from sal.sim import fixtures as registry
from sal.sim.params import SimulationParams
from sal.sim.simulator import simulate_tree
from sal.sim.topology import leaf_bipartitions
from sal.sim.tree import Node

#: Where the fixtures live, re-exported from the registry so the suite and the
#: figures cannot disagree about it.
FIXTURES_DIR = registry.FIXTURES_DIR

if TYPE_CHECKING:
    from sal.sim.params import SimulationParams

# The fixtures every backend is exercised against, smallest first.
SMALL_SITES = "tree_jc/ci.yaml"
FOUR_TAXA = "tree_jc/stress.yaml"
EIGHT_TAXA = "tree_jc/release.yaml"


def fixture_path(name: str) -> Path:
    """Absolute path to a named fixture, e.g. ``"tree_jc/stress.yaml"``.

    Raises `FileNotFoundError` on a typo rather than a later parse error.
    """
    path = FIXTURES_DIR / name
    if not path.is_file():
        available = sorted(
            str(p.relative_to(FIXTURES_DIR)) for p in FIXTURES_DIR.rglob("*.yaml")
        )
        msg = f"no fixture {name!r} in {FIXTURES_DIR}; available: {available}"
        raise FileNotFoundError(msg)
    return path


def load_fixture(name: str) -> SimulationParams:
    """Load a named fixture's simulation parameters."""
    return load_params(fixture_path(name), SimulationParams)


def simulated_alignment(
    name: str, n_sites: int | None = None
) -> tuple[SimulationParams, dict[str, np.ndarray]]:
    """A fixture's parameters and the alignment simulated at its own seed.

    ``name`` is a literal or constant at the call site: `tests/_problems.py` reads it.
    """
    params = load_fixture(name)
    dataset = simulate_tree(params, np.random.default_rng(params.seed), n_sites)
    return params, dict(dataset.alignment)


# --- the Felsenstein and Farris zones (issue #209) -------------------------

FOUR_TAXA_LEAVES = ("A", "B", "C", "D")

# A long branch is long enough for convergent change to be common; the
# internal branch is short enough that little signal supports the true split.
# These are the standard proportions for the zone, not values tuned until the
# effect appeared.
ZONE_LONG, ZONE_SHORT, ZONE_INTERNAL = 0.75, 0.02, 0.02


def balanced_four_taxa(
    first: float, second: float, third: float, fourth: float
) -> Node:
    """``((A,B),(C,D))`` with the four pendant lengths; internals ``ZONE_INTERNAL``."""
    return Node(
        "root",
        None,
        (
            Node("i1", ZONE_INTERNAL, (Node("A", first), Node("B", second))),
            Node("i2", ZONE_INTERNAL, (Node("C", third), Node("D", fourth))),
        ),
    )


#: Felsenstein: the long branches are A and C, which are *not* a cherry in
#: the true tree, so grouping them is the mistake convergent change invites.
FELSENSTEIN_ZONE = balanced_four_taxa(ZONE_LONG, ZONE_SHORT, ZONE_LONG, ZONE_SHORT)
#: Farris: the long branches are A and B, which *are* a cherry.
FARRIS_ZONE = balanced_four_taxa(ZONE_LONG, ZONE_LONG, ZONE_SHORT, ZONE_SHORT)
#: The bipartition key of the generating topology, ``AB|CD``.
ZONE_TRUE_SPLIT = leaf_bipartitions(balanced_four_taxa(0.1, 0.1, 0.1, 0.1))
