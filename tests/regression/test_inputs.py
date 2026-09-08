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


# --- Reachability, in both directions ---------------------------------------
#
# The stamp narrowed from "every module the renderer imports" to "every
# definition it executes" (issue #418, part 3). The saving is worth nothing on
# its own: a stamp that misses a change publishes a committed figure the code
# no longer produces, and nothing else in the suite would notice. So each test
# below that pins a saving has a sibling pinning the fire, and the walk's own
# escape hatch --- a module it cannot narrow --- is asserted rather than
# trusted.

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
def test_no_cited_figure_falls_back_to_a_whole_module() -> None:
    # What makes the counts below mean anything. Each is measured on a walk
    # that narrowed every module it crossed; one renderer falling back to a
    # whole module would restamp on any edit to it and the saving would be
    # reported for a case that is not being exercised.
    reported = {
        spec.stem: spec.reach(REPO_ROOT).fallbacks
        for spec in _cited()
        if spec.reach(REPO_ROOT).fallbacks
    }

    assert reported == {}


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
def test_a_constant_no_renderer_reads_stales_no_figure(repo_copy: Path) -> None:
    # #394's measured case exactly: two parameter constants declared beside
    # the ones the renderers do read.
    declaration = 'LDPC_PARAMS = ParamsArgument("params", load_ldpc_params)'
    staled = _staled_by(
        repo_copy,
        (
            SHARED,
            declaration,
            f'{declaration}\nUNREAD_A = ParamsArgument("a", load_ldpc_params)\n'
            'UNREAD_B = ParamsArgument("b", load_ldpc_params)',
        ),
    )

    assert staled == set()


@pytest.mark.structural
def test_an_edit_to_a_reached_function_stales_every_figure_that_calls_it(
    repo_copy: Path,
) -> None:
    # The direction that matters. `figure_main` is what every figure renderer
    # calls; `table_main` is what only the table renderers call. Each must
    # stale exactly its callers -- fewer is a figure published against code
    # that no longer produces it.
    tables = {
        spec.stem
        for spec in _cited()
        if "table_main"
        in (REPO_ROOT / "python" / Path(*spec.module.split(".")))
        .with_suffix(".py")
        .read_text()
    }
    figures = {spec.stem for spec in _cited()} - tables
    assert tables, "the cited set must contain a table renderer"
    assert figures, "the cited set must contain a figure renderer"

    staled = _staled_by(
        repo_copy,
        (
            SHARED,
            'log.info("wrote %s and %s", written.figure_path, written.caption_path)',
            'log.info("wrote %s then %s", written.figure_path, written.caption_path)',
        ),
    )

    assert staled == figures


@pytest.mark.structural
def test_an_edit_reached_by_one_renderer_stales_only_that_one(
    repo_copy: Path,
) -> None:
    tables = {
        spec.stem
        for spec in _cited()
        if "table_main"
        in (REPO_ROOT / "python" / Path(*spec.module.split(".")))
        .with_suffix(".py")
        .read_text()
    }

    staled = _staled_by(
        repo_copy,
        (
            SHARED,
            'log.info("wrote %s and %s", written.table_path, written.caption_path)',
            'log.info("wrote %s / %s", written.table_path, written.caption_path)',
        ),
    )

    assert staled == tables


def _opaque_tree(root: Path, extra: str) -> FigureSpec:
    """A two-module package whose shared module carries ``extra``.

    ``entry`` calls ``shared.used`` and never names ``shared.unused``, so a
    walk that narrows correctly hashes the first and not the second. ``extra``
    is the line that decides whether it may narrow at all.

    Returns
    -------
    FigureSpec
        A spec whose module is ``entry``.
    """
    package = root / "python" / "snakes_and_ladders"
    (package / "qa").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "qa" / "__init__.py").write_text("")
    (package / "reg.py").write_text("def register(f):\n    return f\n")
    (package / "other.py").write_text("VALUE = 1\n")
    (package / "shared.py").write_text(
        f"{extra}\n\n\ndef used():\n    return 1\n\n\ndef unused():\n    return 2\n"
    )
    (package / "qa" / "entry.py").write_text(
        "from snakes_and_ladders.shared import used\n\n\ndef main():\n    return used()\n"
    )
    return FigureSpec("entry", "snakes_and_ladders.qa.entry", (), seconds=1.0)


#: One line per way a module can reach a definition no name in it points at.
#: Each must put the module in `Reach.fallbacks`; the test below then pins that
#: the fallback restores the guarantee rather than merely reporting a doubt.
OPAQUE_LINES = (
    "from snakes_and_ladders.other import *",
    "import snakes_and_ladders.other",
    "from snakes_and_ladders import other\nX = getattr(other, 'VALUE')",
    "import importlib",
    "def __getattr__(name):\n    return name",
    "X = globals()",
    "from snakes_and_ladders.reg import register\n\n\n@register\ndef hidden():\n    return 3",
)


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize("extra", OPAQUE_LINES)
def test_a_module_the_walk_cannot_narrow_is_named_and_hashed_whole(
    tmp_path: Path, extra: str
) -> None:
    # The guarantee's escape hatch, asserted rather than silent. Each line is
    # a way to call a definition the import graph does not name; the walk must
    # say so and then hash the module whole, so a change to a function nothing
    # references still moves the digest.
    spec = _opaque_tree(tmp_path, extra)
    reach = spec.reach(tmp_path)
    before = spec.input_digest(tmp_path)

    assert [entry.split(":")[0] for entry in reach.fallbacks] == [
        "snakes_and_ladders.shared"
    ]

    shared = tmp_path / "python" / "snakes_and_ladders" / "shared.py"
    shared.write_text(shared.read_text().replace("return 2", "return 22"))

    assert spec.input_digest(tmp_path) != before


#: Constructs that look like the ones above and are not. A walk that fell back
#: on these would report a saving it is not making, which is why they are
#: pinned beside the fallbacks rather than left to the counts to reveal.
BENIGN_LINES = (
    "from importlib import metadata\nX = metadata.version('numpy')",
    "import argparse\n\n\ndef parse(a):\n    return getattr(a, 'flag')",
    "import numpy as np\nX = np.zeros(3)",
    "from dataclasses import dataclass\n\n\n@dataclass\nclass Held:\n    x: int",
)


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize("extra", BENIGN_LINES)
def test_a_lookup_that_cannot_reach_a_definition_is_not_a_fallback(
    tmp_path: Path, extra: str
) -> None:
    # `getattr` on an `argparse` namespace reaches an attribute of that object,
    # never a definition this walk could have missed; reading a version out of
    # `importlib.metadata` imports nothing chosen at run time. Falling back on
    # either would leave `qa/runner.py` --- which does the first --- hashed
    # whole, and the measured case unfixed.
    spec = _opaque_tree(tmp_path, extra)

    assert spec.reach(tmp_path).fallbacks == ()


@pytest.mark.critical
@pytest.mark.structural
def test_a_module_the_walk_can_narrow_charges_only_what_is_reached(
    tmp_path: Path,
) -> None:
    # The saving, and its bound: with nothing opaque in it, the unreached
    # function is not an input and the reached one is.
    spec = _opaque_tree(tmp_path, "CONSTANT = 1")
    before = spec.input_digest(tmp_path)

    assert spec.reach(tmp_path).fallbacks == ()

    shared = tmp_path / "python" / "snakes_and_ladders" / "shared.py"
    shared.write_text(shared.read_text().replace("return 2", "return 22"))
    assert spec.input_digest(tmp_path) == before

    shared.write_text(shared.read_text().replace("return 1", "return 11"))
    assert spec.input_digest(tmp_path) != before


#: Ways one definition reaches another that are not a plain call. Each is a way
#: the walk could under-fire, which is the direction that breaks the guarantee.
INDIRECT_REFERENCES = (
    "HANDLERS = {'a': target}\n\n\ndef used():\n    return HANDLERS['a']()",
    "def used(f=target):\n    return f()",
    "class Used:\n    def go(self):\n        return target()\n\n\ndef used():\n    return Used().go()",
    "def used():\n    def inner():\n        return target()\n    return inner()",
    "class Base:\n    pass\n\n\nclass Used(target):\n    pass\n\n\ndef used():\n    return Used()",
    "def used():\n    return [target() for _ in range(1)]",
)


@pytest.mark.critical
@pytest.mark.structural
@pytest.mark.parametrize("shape", INDIRECT_REFERENCES)
def test_a_definition_reached_indirectly_is_still_an_input(
    tmp_path: Path, shape: str
) -> None:
    # A call is not the only way one definition reaches another: a dispatch
    # table, a default argument, a method, a closure, a base class and a
    # comprehension all do. Each must be followed, or the digest under-fires.
    package = tmp_path / "python" / "snakes_and_ladders"
    (package / "qa").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "qa" / "__init__.py").write_text("")
    (package / "target.py").write_text("class target:\n    VALUE = 1\n")
    (package / "shared.py").write_text(
        f"from snakes_and_ladders.target import target\n\n\n{shape}\n"
    )
    (package / "qa" / "entry.py").write_text(
        "from snakes_and_ladders.shared import used\n\n\ndef main():\n    return used()\n"
    )
    spec = FigureSpec("entry", "snakes_and_ladders.qa.entry", (), seconds=1.0)
    before = spec.input_digest(tmp_path)

    (package / "target.py").write_text("class target:\n    VALUE = 2\n")

    assert spec.input_digest(tmp_path) != before


@pytest.mark.structural
def test_the_rust_sources_are_inputs_only_where_a_renderer_reaches_the_extension(
    tmp_path: Path,
) -> None:
    # The same narrowing, applied to the kernel: a figure that runs no Rust
    # is not restamped by a kernel change, and one that does still is.
    package = tmp_path / "python" / "snakes_and_ladders"
    (package / "qa").mkdir(parents=True)
    (package / "__init__.py").write_text("from .oxi_snakes_and_ladders import double\n")
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

    assert kernel.reach(tmp_path).extension
    assert not pure.reach(tmp_path).extension

    (tmp_path / "src" / "lib.rs").write_text("// two\n")

    assert kernel.input_digest(tmp_path) != before[0]
    assert pure.input_digest(tmp_path) == before[1]
