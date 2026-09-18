"""The root exports the seams the package declares, and nothing at import time.

Issue #717. Every name in ``__all__`` resolves to the object its submodule
declares --- identity, not a copy --- and a bare ``import snakes_and_ladders``
imports no submodule, so a caller reaching for a graph pays for no ``torch``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import snakes_and_ladders
from snakes_and_ladders import parallel
from snakes_and_ladders.backend import Backend
from snakes_and_ladders.learn.environment import Environment
from snakes_and_ladders.opt.objective import Objective
from snakes_and_ladders.sim.factor_graph import FactorGraph
from snakes_and_ladders.sim.fixtures import fixture
from snakes_and_ladders.sim.graph import PottsGraph


@pytest.mark.structural
@pytest.mark.critical
def test_every_exported_name_is_the_object_its_module_declares() -> None:
    declared = {
        "Backend": Backend,
        "Environment": Environment,
        "FactorGraph": FactorGraph,
        "Objective": Objective,
        "PottsGraph": PottsGraph,
        "fixture": fixture,
        "parallel": parallel,
    }

    assert sorted(snakes_and_ladders.__all__) == sorted(declared)
    for name, expected in declared.items():
        assert getattr(snakes_and_ladders, name) is expected


@pytest.mark.structural
def test_the_bare_import_loads_no_submodule() -> None:
    # Resolved on first use: the root names `learn.environment`, and `learn`
    # imports `torch`, so the check is on the bare import in a fresh process.
    code = (
        "import sys, snakes_and_ladders; "
        "print(sorted(m for m in sys.modules if m.startswith('snakes_and_ladders.')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"


@pytest.mark.edge_case
def test_a_name_outside_the_surface_is_refused() -> None:
    with pytest.raises(AttributeError, match="no attribute 'double'"):
        _ = snakes_and_ladders.double


@pytest.mark.structural
def test_a_submodule_resolves_on_first_use_and_loads_nothing_beside_it() -> None:
    """``sal.sim.tree`` reads as written, and costs ``sim.tree`` alone.

    A fresh process, since the suite has imported everything: after the bare
    import neither ``sim`` nor ``learn`` is loaded; reading ``sal.sim.tree``
    loads ``sim.tree`` and leaves ``learn``, which imports ``torch``, alone.
    """
    code = (
        "import sys\n"
        "import snakes_and_ladders as sal\n"
        "before = sorted(m for m in sys.modules if m.startswith('snakes_and_ladders.'))\n"
        "tree = sal.sim.tree\n"
        "assert tree is sys.modules['snakes_and_ladders.sim.tree'], tree\n"
        "assert sal.sim.tree is tree\n"
        "assert 'snakes_and_ladders.learn' not in sys.modules\n"
        "print(before)\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "[]"


@pytest.mark.edge_case
def test_a_name_that_is_no_submodule_is_refused() -> None:
    """A misspelling raises ``AttributeError``, as an attribute would."""
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = snakes_and_ladders.sim.nope
    with pytest.raises(AttributeError, match="no attribute '_private'"):
        _ = snakes_and_ladders._private
