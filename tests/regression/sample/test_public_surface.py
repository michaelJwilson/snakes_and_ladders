"""The public Potts sampling surface resolves, is documented, and is what callers import (issue #1143).

Callers outside ``sal`` compose the kernels of :mod:`sal.sample.potts_mcmc`
as well as its drivers. The names they depend on are the package's
``__all__`` and :mod:`sal.sample.schedule`'s, so a rename or a move breaks
a test here rather than a caller silently. Three guards: every declared name
resolves and carries its own docstring; the exported kernels' signatures are
snapshotted, so a signature change is a deliberate edit of this file; and no
module under ``tests/``, ``docs/nb/`` or ``python/sal/validation/`` imports a
name from a ``potts_mcmc`` submodule when the package exports it.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from sal.sample import potts_mcmc, schedule

from tests._paths import REPO_ROOT

#: The modules whose ``__all__`` this issue declares or extends.
SURFACES = [potts_mcmc, schedule]

#: The trees a caller of the surface lives in; a missing one is skipped.
CALLER_TREES = ["tests", "docs/nb", "python/sal/validation"]

#: The prefix of the implementation submodules the package re-exports from.
SUBMODULE_PREFIX = f"{potts_mcmc.__name__}."

#: Each exported kernel's signature, as :func:`inspect.signature` prints it.
#: Annotations are strings under ``from __future__ import annotations``, so
#: the text is the source's and does not depend on the import order. The
#: Wolff kernels gained ``backend``, Rust by default, in #1362.
SIGNATURES = {
    "adjacency_lists": (
        "(offsets: 'np.ndarray', neighbours: 'np.ndarray',"
        " couplings: 'np.ndarray') -> 'AdjacencyLists'"
    ),
    "bond_probability": "(graph: 'PottsGraph', beta: 'float') -> 'np.ndarray'",
    "bond_roots": (
        "(n_nodes: 'int', bonds: 'np.ndarray', *,"
        " backend: 'Backend' = <Backend.RUST: 'rust'>) -> 'np.ndarray'"
    ),
    "heat_bath_labels": (
        "(log_weights: 'np.ndarray', rng: 'np.random.Generator') -> 'np.ndarray'"
    ),
    "sweep_at": (
        "(rows: 'np.ndarray', offsets: 'np.ndarray', neighbours: 'np.ndarray',"
        " couplings: 'np.ndarray', backend: 'Backend')"
        " -> 'Callable[[np.ndarray, np.random.Generator, float], None]'"
    ),
    "swendsen_wang_heat_bath_sweep": (
        "(state: 'np.ndarray', graph: 'PottsGraph',"
        " rows: 'SiteField | np.ndarray', rng: 'np.random.Generator',"
        " beta: 'float' = 1.0, backend: 'Backend' = <Backend.RUST: 'rust'>)"
        " -> 'None'"
    ),
    "wolff_heat_bath_sweep": (
        "(state: 'np.ndarray', rows: 'SiteField | np.ndarray',"
        " offsets: 'np.ndarray', neighbours: 'np.ndarray',"
        " couplings: 'np.ndarray', rng: 'np.random.Generator',"
        " counter: 'ClusterCounter | None' = None,"
        " graph: 'PottsGraph | None' = None, beta: 'float' = 1.0,"
        " lists: 'AdjacencyLists | None' = None,"
        " backend: 'Backend' = <Backend.RUST: 'rust'>) -> 'int'"
    ),
    "wolff_sweep": (
        "(state: 'np.ndarray', rows: 'SiteField | np.ndarray',"
        " offsets: 'np.ndarray', neighbours: 'np.ndarray',"
        " couplings: 'np.ndarray', rng: 'np.random.Generator',"
        " counter: 'ClusterCounter | None' = None,"
        " graph: 'PottsGraph | None' = None, beta: 'float' = 1.0,"
        " root: 'int | None' = None, proposed: 'int | None' = None,"
        " lists: 'AdjacencyLists | None' = None,"
        " backend: 'Backend' = <Backend.RUST: 'rust'>) -> 'int'"
    ),
}


def _own_docstring(value: object) -> str | None:
    """The docstring ``value`` declares, not one inherited or generated.

    A dataclass without a docstring is given its signature as ``__doc__`` and
    a class inherits nothing from ``__doc__`` but may carry its base's
    through :func:`inspect.getdoc`, so a class is read from its own
    ``__dict__`` and a generated ``Name(...)`` is refused.
    """
    if inspect.isclass(value):
        doc = value.__dict__.get("__doc__")
        if isinstance(doc, str) and doc.startswith(f"{value.__name__}("):
            return None
        return doc if isinstance(doc, str) else None
    return getattr(value, "__doc__", None)


@pytest.mark.infra
@pytest.mark.parametrize("module", SURFACES, ids=lambda module: module.__name__)
def test_every_public_name_resolves_and_is_documented(module: ModuleType) -> None:
    names: list[str] = module.__all__

    assert names == sorted(names)
    assert len(set(names)) == len(names)
    undocumented = [
        name
        for name in names
        if not (_own_docstring(getattr(module, name)) or "").strip()
    ]
    assert undocumented == []


@pytest.mark.infra
def test_the_public_kernels_are_exported_with_their_snapshotted_signatures() -> None:
    signatures = {
        name: str(inspect.signature(getattr(potts_mcmc, name))) for name in SIGNATURES
    }

    assert set(SIGNATURES) <= set(potts_mcmc.__all__)
    assert signatures == SIGNATURES


@pytest.mark.infra
@pytest.mark.parametrize("name", ["PottsMove", "cluster_tempering"])
def test_the_public_drivers_and_moves_are_exported(name: str) -> None:
    assert name in potts_mcmc.__all__


@pytest.mark.infra
@pytest.mark.parametrize(
    "name",
    ["parallel_tempering", "adapt_ladder_potts", "TemperedChains"],
)
def test_the_unsupported_tempering_is_not_exported(name: str) -> None:
    # Moved to `sal.sandbox.potts_tempering` by the owner's decision (#1352).
    assert name not in potts_mcmc.__all__
    assert not hasattr(potts_mcmc, name)


@pytest.mark.infra
def test_the_public_schedule_names_callers_use_are_declared() -> None:
    assert {"ConstantTempSchedule", "ScheduleParams", "ScheduleShape"} <= set(
        schedule.__all__
    )


def _sources() -> Iterator[tuple[Path, str]]:
    """Every Python module and notebook code cell under :data:`CALLER_TREES`."""
    for tree in CALLER_TREES:
        root = REPO_ROOT / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            yield path, path.read_text()
        for path in sorted(root.rglob("*.ipynb")):
            cells = json.loads(path.read_text())["cells"]
            for cell in cells:
                if cell["cell_type"] != "code":
                    continue
                # A magic or shell line is not Python; it imports nothing.
                lines = "".join(cell["source"]).splitlines()
                code = "\n".join(
                    line for line in lines if not line.lstrip().startswith(("%", "!"))
                )
                yield path, code


def _submodule_imports(source: str) -> Iterator[tuple[str, str, int]]:
    """``(module, name, line)`` for each ``from sal.sample.potts_mcmc.<sub> import name``."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if module.startswith(SUBMODULE_PREFIX):
                for alias in node.names:
                    yield module, alias.name, node.lineno


@pytest.mark.infra
def test_no_caller_imports_a_public_name_from_a_submodule() -> None:
    public = set(potts_mcmc.__all__)
    reached = [
        f"{path.relative_to(REPO_ROOT)}:{line}: {name} from {module}"
        for path, source in _sources()
        for module, name, line in _submodule_imports(source)
        if name in public
    ]

    assert reached == []


@pytest.mark.infra
def test_the_import_scan_reads_a_multi_line_import() -> None:
    """The guard's parser sees a parenthesized import across lines, aliases included."""
    source = (
        "from sal.sample.potts_mcmc.sweeps import (\n"
        "    GUARD,\n"
        "    bond_probability as probability,\n"
        ")\n"
        "from sal.sample.potts_mcmc import wolff_sweep\n"
    )

    assert list(_submodule_imports(source)) == [
        ("sal.sample.potts_mcmc.sweeps", "GUARD", 1),
        ("sal.sample.potts_mcmc.sweeps", "bond_probability", 1),
    ]
