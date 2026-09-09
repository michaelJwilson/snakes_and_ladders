"""What a baseline record's input hash must cover before a read may trust it.

`snakes_and_ladders.inputs` hashed committed figures and notebooks until
issue #490 deleted the stamps; what is left computes the digest a baseline
record carries (issue #401), and the ways that digest can be wrong are the
same three: a closure that misses a module the numbers were computed from, a
hash that moves with where the checkout lives, and a fixture directory whose
new tier the hash does not see. Each is pinned below.

The tests the cache had --- that a changed renderer stales its own figure and
no other, that prose in a shared module stales none, that an unchanged tree
renders nothing --- went with the stamps they refereed. The claim they stood
for is now `infra/release.sh` rendering every figure and comparing bytes
(issue #484), which `tests/regression/test_release_gate.py` pins.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from snakes_and_ladders.inputs import _imported_names, digest, module_closure


@pytest.mark.critical
@pytest.mark.structural
def test_the_closure_follows_imports_transitively(tmp_path: Path) -> None:
    # `first` reaches `sim.tree` only through `shared` and then the package's
    # `__init__`; a closure one level deep would miss a module the recorded
    # numbers were computed from and leave a stale record readable.
    package = tmp_path / "python" / "snakes_and_ladders"
    (package / "opt").mkdir(parents=True)
    (package / "sim").mkdir()
    (package / "__init__.py").write_text("")
    (package / "opt" / "__init__.py").write_text("")
    (package / "sim" / "__init__.py").write_text("from . import tree\n")
    (package / "sim" / "tree.py").write_text("LEAVES = 4\n")
    (package / "opt" / "shared.py").write_text(
        "from snakes_and_ladders.sim import tree\n"
    )
    (package / "opt" / "first.py").write_text(
        "from snakes_and_ladders.opt.shared import tree\n"
    )
    (package / "opt" / "second.py").write_text("import numpy\n")

    first = {
        path.name for path in module_closure(["snakes_and_ladders.opt.first"], tmp_path)
    }
    second = {
        path.name
        for path in module_closure(["snakes_and_ladders.opt.second"], tmp_path)
    }

    assert {"first.py", "shared.py", "tree.py"} <= first
    assert "second.py" not in first
    # `second` carries the two package ``__init__`` files, whose statements
    # its own import runs, and nothing `first` reaches through `shared`.
    assert second == {"second.py", "__init__.py"}


@pytest.mark.critical
@pytest.mark.structural
def test_relative_and_from_imports_resolve_to_absolute_names() -> None:
    names = _imported_names(
        "from . import tree\nfrom .params import load\nimport snakes_and_ladders.opt.fit\n",
        "snakes_and_ladders.sim.simulate",
    )

    assert {
        "snakes_and_ladders.sim",
        "snakes_and_ladders.sim.tree",
        "snakes_and_ladders.sim.params",
        "snakes_and_ladders.sim.params.load",
        "snakes_and_ladders.opt.fit",
    } <= names


@pytest.mark.critical
@pytest.mark.structural
def test_a_module_reaching_the_extension_carries_the_rust_sources(
    tmp_path: Path,
) -> None:
    # A kernel change alters what a search computes, and the Python closure
    # alone cannot see it.
    package = tmp_path / "python" / "snakes_and_ladders"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "fast.py").write_text(
        "from snakes_and_ladders import oxi_snakes_and_ladders\n"
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("")
    (tmp_path / "Cargo.lock").write_text("")

    names = {
        path.name for path in module_closure(["snakes_and_ladders.fast"], tmp_path)
    }

    assert {"fast.py", "lib.rs", "Cargo.lock"} <= names


@pytest.mark.critical
@pytest.mark.structural
def test_the_digest_does_not_depend_on_where_the_checkout_lives(
    tmp_path: Path,
) -> None:
    # Two clones of the same tree must agree, or a record written in one
    # would be refused as stale when read in the other.
    one, two = tmp_path / "one", tmp_path / "two"
    for root in (one, two):
        (root / "a").mkdir(parents=True)
        (root / "a" / "f.txt").write_text("same")

    assert digest([one / "a" / "f.txt"], one, "x") == digest(
        [two / "a" / "f.txt"], two, "x"
    )
    assert digest([one / "a" / "f.txt"], one, "x") != digest(
        [one / "a" / "f.txt"], one, "y"
    )


@pytest.mark.mathematical
def test_a_directory_input_hashes_the_files_under_it(tmp_path: Path) -> None:
    # A fixture named as a problem rather than as one tier is a directory
    # (issue #382). Hashing it as a unit is what makes a tier added to it a
    # change the digest sees; the alternative read the record unchecked.
    problem = tmp_path / "fixtures" / "potts_lattice"
    problem.mkdir(parents=True)
    (problem / "ci.yaml").write_text("seed: 1\n")
    before = digest([problem], tmp_path)

    assert before == digest([problem / "ci.yaml"], tmp_path)

    (problem / "stress.yaml").write_text("seed: 2\n")

    assert digest([problem], tmp_path) != before
