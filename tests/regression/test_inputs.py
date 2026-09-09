"""What the figure cache must guarantee before it is allowed to skip a render.

Skipping a render is only safe if the digest moves whenever the figure could
have, so what is pinned here is not the saving but the ways the cache could
be wrong (issue #372): a changed renderer, fixture or spec that left its
stamp untouched; a change to one figure's inputs that moved another's stamp;
a module reached only through two imports that the closure missed; and a
tree with nothing changed that rendered something anyway.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from snakes_and_ladders.inputs import (
    digest,
    imported_names,
    module_closure,
    read_stamp,
    write_stamp,
)
from snakes_and_ladders.qa import build
from snakes_and_ladders.qa.manifest import (
    CAP_WAIVERS,
    CITED_RENDER_CAP,
    FIGURES,
    FigureSpec,
    cited_stems,
)


def _tree(root: Path) -> tuple[FigureSpec, FigureSpec]:
    """Build a small package with two renderers sharing one module.

    Returns
    -------
    tuple[FigureSpec, FigureSpec]
        A spec reading a fixture through ``shared``, and one reading nothing.
    """
    package = root / "python" / "snakes_and_ladders"
    (package / "qa").mkdir(parents=True)
    (package / "sim").mkdir()
    (package / "__init__.py").write_text("")
    (package / "qa" / "__init__.py").write_text("")
    (package / "sim" / "__init__.py").write_text("from . import tree\n")
    (package / "sim" / "tree.py").write_text("LEAVES = 4\n")
    (package / "qa" / "shared.py").write_text(
        "from snakes_and_ladders.sim import tree\n"
    )
    (package / "qa" / "first.py").write_text(
        "from snakes_and_ladders.qa.shared import tree\n"
    )
    (package / "qa" / "second.py").write_text("import numpy\n")
    fixtures = root / "tests" / "regression" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "params.yaml").write_text("seed: 1\n")
    first = FigureSpec(
        "first",
        "snakes_and_ladders.qa.first",
        ("--params", "tests/regression/fixtures/params.yaml"),
        seconds=1.0,
    )
    second = FigureSpec("second", "snakes_and_ladders.qa.second", (), seconds=1.0)
    return first, second


@pytest.mark.critical
@pytest.mark.structural
def test_the_closure_follows_imports_transitively(tmp_path: Path) -> None:
    # `first` reaches `sim.tree` only through `shared` and then the package's
    # `__init__`; a closure one level deep would miss the module the figure
    # is computed from.
    first, second = _tree(tmp_path)

    names = {path.name for path in first.inputs(tmp_path)}

    assert {"first.py", "shared.py", "tree.py", "params.yaml"} <= names
    assert "second.py" not in names
    # `second` carries the two package ``__init__`` files, whose statements its
    # own import runs, and nothing `first` reaches through `shared`.
    assert {"second.py", "__init__.py"} == {
        path.name for path in second.inputs(tmp_path)
    }


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize(
    ("touch", "addition"),
    [
        ("python/snakes_and_ladders/qa/first.py", "TAXA = 8\n"),
        ("python/snakes_and_ladders/sim/tree.py", "TAXA = 8\n"),
        ("tests/regression/fixtures/params.yaml", "taxa: 8\n"),
    ],
)
def test_a_changed_input_moves_only_its_own_stamp(
    tmp_path: Path, touch: str, addition: str
) -> None:
    # The case the cache exists for, and its mirror: the renderer, a module
    # two imports away, or the fixture changes and `first` is stale, while
    # `second`, which reads none of them, is not.
    first, second = _tree(tmp_path)
    before = first.input_digest(tmp_path), second.input_digest(tmp_path)

    path = tmp_path / touch
    path.write_text(path.read_text() + addition)

    after = first.input_digest(tmp_path), second.input_digest(tmp_path)
    assert after[0] != before[0]
    assert after[1] == before[1]


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize(
    "touch",
    [
        "python/snakes_and_ladders/qa/first.py",
        "python/snakes_and_ladders/sim/tree.py",
    ],
)
def test_a_comment_on_an_input_moves_no_stamp(tmp_path: Path, touch: str) -> None:
    # The other half of the rule above, and the reason the digest is over an
    # AST rather than over bytes: prose in a module a figure reaches is not
    # a figure the code no longer produces (issue #418).
    first, _ = _tree(tmp_path)
    before = first.input_digest(tmp_path)

    path = tmp_path / touch
    path.write_text(f"# what this is for\n{path.read_text()}")

    assert first.input_digest(tmp_path) == before


@pytest.mark.critical
@pytest.mark.structural
def test_a_changed_spec_moves_the_stamp(tmp_path: Path) -> None:
    # The arguments are inputs too: the same renderer over another fixture
    # is another figure.
    first, _ = _tree(tmp_path)
    other = FigureSpec(first.stem, first.module, (*first.arguments, "--k", "3"), 1.0)

    assert other.input_digest(tmp_path) != first.input_digest(tmp_path)


@pytest.mark.critical
@pytest.mark.structural
def test_an_unchanged_tree_renders_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The saving, pinned: with every stamp equal to the current digest, the
    # stale set is empty, and a missing or differing stamp puts a figure back.
    first, second = _tree(tmp_path)
    monkeypatch.setattr(build, "REPO_ROOT", tmp_path)
    figures = tmp_path / "figures"
    figures.mkdir()
    write_stamp(first.stamp(figures), first.input_digest(tmp_path))
    write_stamp(second.stamp(figures), second.input_digest(tmp_path))

    assert build.stale((first, second), figures) == ()

    write_stamp(second.stamp(figures), "0" * 64)
    assert build.stale((first, second), figures) == (second,)
    first.stamp(figures).unlink()
    assert build.stale((first, second), figures) == (first, second)


@pytest.mark.critical
@pytest.mark.structural
def test_the_digest_does_not_depend_on_where_the_checkout_lives(
    tmp_path: Path,
) -> None:
    # Two clones of the same tree must agree, or CI would re-render what a
    # contributor stamped.
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
    # change the stamp sees; the alternative silently skipped the render.
    problem = tmp_path / "fixtures" / "potts_lattice"
    problem.mkdir(parents=True)
    (problem / "ci.yaml").write_text("seed: 1\n")
    before = digest([problem], tmp_path)

    assert before == digest([problem / "ci.yaml"], tmp_path)

    (problem / "stress.yaml").write_text("seed: 2\n")

    assert digest([problem], tmp_path) != before


@pytest.mark.critical
@pytest.mark.structural
def test_relative_and_from_imports_resolve_to_absolute_names() -> None:
    names = imported_names(
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
    # A kernel change alters what a figure computes, and the Python closure
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
def test_a_stamp_round_trips(tmp_path: Path) -> None:
    stamp = tmp_path / "x.inputs"

    assert read_stamp(stamp) is None
    write_stamp(stamp, "abc")
    assert read_stamp(stamp) == "abc"


@pytest.mark.critical
@pytest.mark.structural
def test_every_committed_figure_has_a_stamp() -> None:
    # A figure without a stamp is rendered on every build, which is the cost
    # the cache removes; the release gate's `--all` writes them all.
    missing = [
        spec.stem
        for spec in FIGURES
        if any(build.DEFAULT_OUTPUT_DIR.glob(f"{spec.stem}.*"))
        and not spec.stamp(build.DEFAULT_OUTPUT_DIR).is_file()
    ]
    assert missing == []


@pytest.mark.critical
@pytest.mark.structural
def test_a_cited_figure_renders_inside_the_cap_or_carries_a_waiver() -> None:
    # The per-pull-request build is the sum of its stale cited figures, so a
    # cited figure over the cap is a third of the whole validation budget on
    # its own. One over it is either uncited, and rendered at the release
    # gate as `topology_accuracy` is, or waived with the ticket that owns it.
    cited = cited_stems(*build.DEFAULT_DOCUMENTS)
    over = {
        spec.stem: spec.seconds
        for spec in FIGURES
        if spec.stem in cited and spec.seconds > CITED_RENDER_CAP
    }

    assert set(over) <= set(CAP_WAIVERS), (
        f"cited figures over the {CITED_RENDER_CAP:.0f} s cap without a waiver: {over}"
    )
    assert all(ticket.startswith("#") for ticket in CAP_WAIVERS.values())
    assert set(CAP_WAIVERS) <= {spec.stem for spec in FIGURES}


@pytest.mark.critical
@pytest.mark.structural
def test_every_manifest_entry_states_a_measured_time() -> None:
    assert all(spec.seconds > 0 for spec in FIGURES)


# --- The closure hash, in both directions -----------------------------------
#
# The unit of the hash is a module's AST with docstrings removed, over the
# renderer's whole import closure. The saving is worth nothing on its own: a
# stamp that misses a change publishes a committed figure the code no longer
# produces, and nothing else in the suite would notice. So each test that pins
# a saving has a sibling pinning the fire.
#
# Issue #418 narrowed this further, to the definitions the entry point
# executes. Over the twelve branches merged before 2026-09-09 that walk staled
# the same 26 figures this hash does --- it saved none --- and it cost seven
# fallbacks, so #425 removed it and the tests that pinned them.

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The module every figure renderer imports, and the one whose edits were
#: measured: #394's agent declared two constants here and restamped all 17
#: cited figures then committed, with zero PDF bytes changed.
SHARED = "python/snakes_and_ladders/qa/runner.py"


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """A copy of the package, the fixtures and the Rust sources under ``tmp_path``.

    The measured case is about the real import graph -- thirty modules deep,
    one of them shared by every renderer -- and a synthetic package would only
    pin the analysis against itself. The copy is edited instead of the
    checkout so a failing test cannot leave the tree modified.

    Returns
    -------
    Path
        A repository root the real :data:`FIGURES` can be digested against.
    """
    for source in ("python", "src", "tests/regression/fixtures"):
        shutil.copytree(
            REPO_ROOT / source,
            tmp_path / source,
            ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyc"),
        )
    shutil.copy(REPO_ROOT / "Cargo.lock", tmp_path / "Cargo.lock")
    return tmp_path


def _cited() -> tuple[FigureSpec, ...]:
    """The figures the documents cite, which are the ones a stamp is read for.

    Returns
    -------
    tuple[FigureSpec, ...]
    """
    return build.selected(list(build.DEFAULT_DOCUMENTS), every=False)


def _staled_by(root: Path, edit: tuple[str, str, str]) -> set[str]:
    """The cited figures whose digest an edit moves.

    Parameters
    ----------
    root : Path
        A repository root to edit, not the checkout.
    edit : tuple[str, str, str]
        Repository-relative path, the text to replace, and its replacement.

    Returns
    -------
    set[str]
        The stems that went stale.
    """
    specs = _cited()
    before = {spec.stem: spec.input_digest(root) for spec in specs}
    path = root / edit[0]
    text = path.read_text()
    assert edit[1] in text, f"{edit[0]} no longer contains the text this edits"
    path.write_text(text.replace(edit[1], edit[2], 1))
    return {spec.stem for spec in specs if spec.input_digest(root) != before[spec.stem]}


@pytest.mark.structural
def test_prose_in_a_shared_module_stales_no_figure(repo_copy: Path) -> None:
    # The case the AST hash exists for. `qa/runner.py` is in every cited
    # figure's closure, so before this a reworded docstring there restamped
    # all 19 with zero PDF bytes changed.
    staled = _staled_by(
        repo_copy,
        (
            SHARED,
            "Shared command-line entry point for the QA figure and table scripts.",
            "The shared command-line entry point for the QA scripts.",
        ),
    )

    assert staled == set()


@pytest.mark.structural
def test_a_comment_in_a_shared_module_stales_no_figure(repo_copy: Path) -> None:
    staled = _staled_by(
        repo_copy,
        (
            SHARED,
            "# Closed here rather than in the builder",
            "# Closed here, not there",
        ),
    )

    assert staled == set()


@pytest.mark.structural
def test_an_edit_to_a_shared_module_stales_every_figure_that_imports_it(
    repo_copy: Path,
) -> None:
    # The direction that matters, and the price of the closure hash: an edit
    # to a statement in `qa/runner.py` stales every cited figure, because
    # every one of them imports it. Fewer would be a figure published against
    # code that no longer produces it.
    staled = _staled_by(
        repo_copy,
        (
            SHARED,
            'log.info("wrote %s and %s", written.figure_path, written.caption_path)',
            'log.info("wrote %s then %s", written.figure_path, written.caption_path)',
        ),
    )

    assert staled == {spec.stem for spec in _cited()}


@pytest.mark.structural
def test_the_rust_sources_are_inputs_only_where_the_closure_names_the_extension(
    tmp_path: Path,
) -> None:
    # The closure applied to the kernel: a renderer whose closure names the
    # extension is restamped by a kernel change and one whose closure does not
    # is left alone. The package's `__init__` is empty here so the two cases
    # are distinguishable; in the checkout it re-exports `double`, which puts
    # every renderer on the first side. That over-approximation cost nothing
    # over the twelve branches merged before 2026-09-09, none of which touched
    # `src/` or `Cargo.lock`.
    package = tmp_path / "python" / "snakes_and_ladders"
    (package / "qa").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "qa" / "__init__.py").write_text("")
    (package / "fast.py").write_text(
        "from snakes_and_ladders import oxi_snakes_and_ladders\n\n\n"
        "def run():\n    return oxi_snakes_and_ladders.double(1)\n"
    )
    (package / "qa" / "kernel.py").write_text(
        "from snakes_and_ladders.fast import run\n\n\ndef main():\n    return run()\n"
    )
    (package / "qa" / "pure.py").write_text("def main():\n    return 1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("// one\n")
    (tmp_path / "Cargo.lock").write_text("")
    kernel = FigureSpec("kernel", "snakes_and_ladders.qa.kernel", (), seconds=1.0)
    pure = FigureSpec("pure", "snakes_and_ladders.qa.pure", (), seconds=1.0)
    before = kernel.input_digest(tmp_path), pure.input_digest(tmp_path)

    (tmp_path / "src" / "lib.rs").write_text("// two\n")

    assert kernel.input_digest(tmp_path) != before[0]
    assert pure.input_digest(tmp_path) == before[1]
