"""What the module closure must reach before a selection may be trusted.

The closure `infra/baselines.py --changed` selects records on (stamps went
with #490, the record digest with #460). A closure missing a module is a
record a PR moved and nothing recomputed; pinned in both directions. Figures
are compared by bytes at release (#484, `test_release_gate.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sal.inputs import _imported_names, module_closure


@pytest.mark.critical
@pytest.mark.infra
def test_the_closure_follows_imports_transitively(tmp_path: Path) -> None:
    # `first` reaches `sim.tree` only through `shared` and then the package's
    # `__init__`; a closure one level deep would miss a module the recorded
    # numbers were computed from and leave a stale record readable.
    package = tmp_path / "python" / "sal"
    (package / "opt").mkdir(parents=True)
    (package / "sim").mkdir()
    (package / "__init__.py").write_text("")
    (package / "opt" / "__init__.py").write_text("")
    (package / "sim" / "__init__.py").write_text("from . import tree\n")
    (package / "sim" / "tree.py").write_text("LEAVES = 4\n")
    (package / "opt" / "shared.py").write_text("from sal.sim import tree\n")
    (package / "opt" / "first.py").write_text("from sal.opt.shared import tree\n")
    (package / "opt" / "second.py").write_text("import numpy\n")

    first = {path.name for path in module_closure(["sal.opt.first"], tmp_path)}
    second = {path.name for path in module_closure(["sal.opt.second"], tmp_path)}

    assert {"first.py", "shared.py", "tree.py"} <= first
    assert "second.py" not in first
    # `second` carries the two package ``__init__`` files, whose statements
    # its own import runs, and nothing `first` reaches through `shared`.
    assert second == {"second.py", "__init__.py"}


@pytest.mark.critical
@pytest.mark.infra
def test_relative_and_from_imports_resolve_to_absolute_names() -> None:
    names = _imported_names(
        "from . import tree\nfrom .params import load\nimport sal.opt.fit\n",
        "sal.sim.simulate",
    )

    assert {
        "sal.sim",
        "sal.sim.tree",
        "sal.sim.params",
        "sal.sim.params.load",
        "sal.opt.fit",
    } <= names


@pytest.mark.critical
@pytest.mark.infra
def test_a_module_reaching_the_extension_carries_the_rust_sources(
    tmp_path: Path,
) -> None:
    # A kernel change alters what a search computes, and the Python closure
    # alone cannot see it.
    package = tmp_path / "python" / "sal"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "fast.py").write_text("from sal import oxisal\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("")
    (tmp_path / "Cargo.lock").write_text("")

    names = {path.name for path in module_closure(["sal.fast"], tmp_path)}

    assert {"fast.py", "lib.rs", "Cargo.lock"} <= names
