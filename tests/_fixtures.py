"""Shared access to the `simulation_params.yaml` fixtures.

`DEV.md`'s Test Layout says a fixture shared across modules lives in a
top-level underscore-prefixed module, imported rather than collected. Every
regression and benchmark module that needs a simulation fixture went through
its own copy of the path instead -- under two different names, and with the
benchmark copies reaching back up through ``parent.parent``. One spelling of
the location lives here, so moving the fixtures is a one-line change.

The Felsenstein- and Farris-zone trees (issue #209) live here for the same
reason: the small-parsimony tests and the large-parsimony search are scored
against the same two trees, and each fixed the same tree once before.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from snakes_and_ladders.search.topology import leaf_bipartitions
from snakes_and_ladders.sim.params import load_simulation_params
from snakes_and_ladders.sim.tree import Node

if TYPE_CHECKING:
    from snakes_and_ladders.sim.params import SimulationParams

FIXTURES_DIR = Path(__file__).parent / "regression" / "fixtures"

# The fixtures every backend is exercised against, smallest first.
SMALL_SITES = "simulation_params_small_sites.yaml"
FOUR_TAXA = "simulation_params.yaml"
EIGHT_TAXA = "simulation_params_8taxa.yaml"


def fixture_path(name: str) -> Path:
    """Absolute path to a named fixture.

    Parameters
    ----------
    name : str
        File name of the fixture, e.g. ``"simulation_params.yaml"``.

    Returns
    -------
    Path
        Path to the fixture.

    Raises
    ------
    FileNotFoundError
        If no such fixture exists -- a typo names a file that never loads,
        which would otherwise surface as an unrelated parse error.
    """
    path = FIXTURES_DIR / name
    if not path.is_file():
        available = sorted(p.name for p in FIXTURES_DIR.glob("*.yaml"))
        msg = f"no fixture {name!r} in {FIXTURES_DIR}; available: {available}"
        raise FileNotFoundError(msg)
    return path


def load_fixture(name: str) -> SimulationParams:
    """Load a named fixture's simulation parameters.

    Parameters
    ----------
    name : str
        File name of the fixture.

    Returns
    -------
    SimulationParams
        The parsed, validated parameters.
    """
    return load_simulation_params(fixture_path(name))


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
    """``((A,B),(C,D))`` with the four pendant branch lengths given.

    Parameters
    ----------
    first, second, third, fourth : float
        Pendant branch lengths of ``A``, ``B``, ``C`` and ``D``; both
        internal branches are ``ZONE_INTERNAL``.

    Returns
    -------
    Node
        The rooted tree, root of degree 2.
    """
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
