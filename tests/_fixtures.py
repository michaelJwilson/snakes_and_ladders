"""Shared access to the `simulation_params.yaml` fixtures.

`DEV.md`'s Test Layout says a fixture shared across modules lives in a
top-level underscore-prefixed module, imported rather than collected. Every
regression and benchmark module that needs a simulation fixture went through
its own copy of the path instead -- under two different names, and with the
benchmark copies reaching back up through ``parent.parent``. One spelling of
the location lives here, so moving the fixtures is a one-line change.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from snakes_and_ladders.sim.params import load_simulation_params

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
