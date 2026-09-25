"""The root exports the seams the package declares, and nothing at import time.

Issue #717. Every name in ``__all__`` resolves to the object its submodule
declares --- identity, not a copy --- and a bare ``import sal``
imports no submodule, so a caller reaching for a graph pays for no ``torch``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import sal
from sal import parallel
from sal.backend import Backend
from sal.fixtures import Params, load_params
from sal.learn.environment import Environment
from sal.opt.objective import Objective
from sal.sim.factor_graph import FactorGraph
from sal.sim.fixtures import fixture
from sal.sim.graph import PottsGraph
from sal.sim.simulator import Simulator


@pytest.mark.infra
@pytest.mark.critical
def test_every_exported_name_is_the_object_its_module_declares() -> None:
    declared = {
        "Backend": Backend,
        "Environment": Environment,
        "FactorGraph": FactorGraph,
        "Objective": Objective,
        "Params": Params,
        "PottsGraph": PottsGraph,
        "Simulator": Simulator,
        "fixture": fixture,
        "load_params": load_params,
        "parallel": parallel,
    }

    assert sorted(sal.__all__) == sorted(declared)
    for name, expected in declared.items():
        assert getattr(sal, name) is expected


@pytest.mark.infra
def test_the_bare_import_loads_no_submodule() -> None:
    # Resolved on first use: the root names `learn.environment`, and `learn`
    # imports `torch`, so the check is on the bare import in a fresh process.
    code = (
        "import sys, sal; print(sorted(m for m in sys.modules if m.startswith('sal.')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"


@pytest.mark.smoke
def test_a_name_outside_the_surface_is_refused() -> None:
    with pytest.raises(AttributeError, match="no attribute 'double'"):
        _ = sal.double


@pytest.mark.infra
def test_a_submodule_resolves_on_first_use_and_loads_nothing_beside_it() -> None:
    """``sal.sim.tree`` reads as written, and costs ``sim.tree`` alone.

    Fresh process: ``sim.tree`` loads, ``learn`` (which imports ``torch``) does not.
    """
    code = (
        "import sys\n"
        "import sal\n"
        "before = sorted(m for m in sys.modules if m.startswith('sal.'))\n"
        "tree = sal.sim.tree\n"
        "assert tree is sys.modules['sal.sim.tree'], tree\n"
        "assert sal.sim.tree is tree\n"
        "assert 'sal.learn' not in sys.modules\n"
        "print(before)\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "[]"


@pytest.mark.smoke
def test_a_name_that_is_no_submodule_is_refused() -> None:
    """A misspelling raises ``AttributeError``, as an attribute would."""
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = sal.sim.nope
    with pytest.raises(AttributeError, match="no attribute '_private'"):
        _ = sal._private
