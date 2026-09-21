"""The directory division is read from the imports, and every edge between directories is declared (issue #830).

Each directory's `CLAUDE.md` states what belongs in it, and an import from one
directory into another is a claim the sentence has to admit. The calling audit
(`docs/reviews/2026-09-20-calling.md`) found six edges no sentence admitted;
#830 moved what they reached. What stays is declared here with its reason, so
the next edge fails a test rather than an audit, and an edge that disappears
is noticed too: a declared reason for an import nobody makes is a stale
reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "python" / "snakes_and_ladders"
DIRECTORIES = ("sim", "likelihood", "opt", "search", "sample", "learn", "qa")

#: The two directories the division does not run through, each against its
#: reason. `sandbox/` is the conserved home: an implementation a measurement
#: declined, kept to referee the one that replaced it, so it reads the live
#: module it is the oracle for and a declaration per read would restate
#: `sandbox/CLAUDE.md` eight times. What holds it is the other direction,
#: which `tests/regression/test_sandbox.py` asserts: no package directory
#: imports it. `scripts/` is the command line, which reaches whatever a
#: command runs.
EXCLUDED = ("sandbox", "scripts")

#: Every directory-to-directory import edge the tree carries, with the
#: sentence that admits it. ``(root)`` is the package root: seams and
#: numerics every directory may read.
ADMITTED: dict[tuple[str, str], str] = {
    (
        "learn",
        "likelihood",
    ): "learn/CLAUDE.md names tree.py and ranking.py as its application exceptions",
    (
        "learn",
        "opt",
    ): "a policy fits by opt.fit and a failure probe spends an opt.budget",
    (
        "learn",
        "sample",
    ): "an environment's moves are the sampler's own, down to the accept step (potts_nd), and a schedule tempers PPO's epsilon",
    (
        "learn",
        "search",
    ): "learn/CLAUDE.md names tree.py and ranking.py as its application exceptions",
    (
        "learn",
        "sim",
    ): "learn/CLAUDE.md names tree.py, ranking.py and potts_nd.py as its application exceptions",
    (
        "likelihood",
        "opt",
    ): "likelihood.objective implements opt.objective.Objective and constrains through opt.constrain: the seam",
    ("likelihood", "sim"): "an evaluator reads the model sim/ declares",
    ("qa", "learn"): "qa renders what every directory computes",
    ("qa", "likelihood"): "qa renders what every directory computes",
    ("qa", "opt"): "qa renders what every directory computes",
    ("qa", "sample"): "qa renders what every directory computes",
    ("qa", "search"): "qa renders what every directory computes",
    ("qa", "sim"): "qa renders what every directory computes",
    (
        "sample",
        "opt",
    ): "hmc, mala and slice sample an opt.objective.Objective: the seam",
    ("sample", "search"): (
        "sample.gibbs.anneal_topology and sample.tempered.tempered_topologies score a topology by "
        "search.infer.score_topology, the fit; the one edge #830 could not cut, and the cycle with "
        "search -> sample it leaves is why score_topology's home is the next question"
    ),
    ("sample", "sim"): "a sampler draws over the graph sim/ declares",
    ("sim", "opt"): (
        "the fixture registry reads the test functions' loader from opt.testfunctions, "
        "whose validation names the objectives; the chain's loader left with #830"
    ),
    ("search", "likelihood"): "a search scores by an evaluator",
    ("search", "opt"): "a search fits by opt.fit and spends an opt.budget",
    (
        "search",
        "sample",
    ): "the ground-state table runs the samplers as rows; projection seeds by annealing; support reads a tempered ensemble",
    ("search", "sim"): "a search moves over the structure sim/ declares",
}


def _directory(path: Path, package: Path = PACKAGE) -> str:
    relative = path.relative_to(package)
    return relative.parts[0] if len(relative.parts) > 1 else "(root)"


def _imported_directories(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names: list[str] = []
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("snakes_and_ladders")
        ):
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [
                alias.name
                for alias in node.names
                if alias.name.startswith("snakes_and_ladders")
            ]
        for name in names:
            parts = name.split(".")
            found.add(
                parts[1]
                if len(parts) > 2 or (len(parts) == 2 and parts[1] in DIRECTORIES)
                else "(root)"
            )
    return found


def edges(package: Path = PACKAGE) -> dict[tuple[str, str], set[str]]:
    """Every ``(from, to)`` directory edge, with the modules that make it.

    ``package`` is a parameter so the reader can be run over a tree whose
    edges are known, which is what the negative control at the end of this
    module does; every caller here reads the package.
    """
    realized: dict[tuple[str, str], set[str]] = {}
    for path in sorted(package.rglob("*.py")):
        if path.name == "__init__.py" or any(part in EXCLUDED for part in path.parts):
            continue
        source = _directory(path, package)
        if source == "(root)":
            continue
        for target in _imported_directories(path):
            if target in DIRECTORIES and target != source:
                realized.setdefault((source, target), set()).add(path.stem)
    return realized


@pytest.mark.critical
@pytest.mark.infra
def test_every_directory_edge_is_declared_and_every_declaration_is_an_edge() -> None:
    realized = edges()
    undeclared = {
        edge: sorted(modules)
        for edge, modules in realized.items()
        if edge not in ADMITTED
    }
    assert undeclared == {}, f"imports no CLAUDE.md sentence admits: {undeclared}"
    stale = sorted(set(ADMITTED) - set(realized))
    assert stale == [], f"declared edges nothing makes: {stale}"


@pytest.mark.critical
@pytest.mark.infra
def test_the_only_cycle_is_the_declared_one() -> None:
    # A directory division is absolute when its import graph is a DAG. One
    # cycle remains and is declared above; a second one is a new defect.
    realized = set(edges())
    cycles = {(a, b) for (a, b) in realized if (b, a) in realized}
    assert cycles == {("sample", "search"), ("search", "sample")}, cycles


@pytest.mark.smoke
def test_sim_imports_no_inference_directory() -> None:
    # `sim/CLAUDE.md`: "Nothing here performs inference". The fixture registry
    # reached into `opt` for two problems' params until #830 moved the chain's;
    # the test functions' loader stays there and is the one admitted read.
    realized = edges()
    assert {edge for edge in realized if edge[0] == "sim"} <= {("sim", "opt")}
    assert realized.get(("sim", "opt"), set()) <= {"fixtures"}


def _write(package: Path, module: str, source: str) -> None:
    """Write one module of a synthetic package, directories and all."""
    path = package / module
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


@pytest.mark.infra
@pytest.mark.smoke
def test_the_edge_reader_finds_an_edge_no_sentence_admits(tmp_path: Path) -> None:
    # The pairing `tests/regression/docs` established, and the failure mode
    # this guard actually has: a reader that finds nothing reports a divided
    # tree, and the assertions above pass on an empty dictionary. So the
    # reader is run over a tree whose edges are known -- `opt -> sim`, which
    # `opt/CLAUDE.md` admits for no module, written in the four import forms
    # the package uses -- and `opt -> search` from the function-local one,
    # which is the form a cycle hides behind -- beside a module that crosses
    # nothing.
    package = tmp_path / "snakes_and_ladders"
    _write(
        package,
        "opt/objective.py",
        "from snakes_and_ladders.sim.graph import PottsGraph\n"
        "from snakes_and_ladders.sim.potts import (\n    site_field,\n)\n"
        "import snakes_and_ladders.numerics\n"
        "def fit() -> None:\n"
        "    from snakes_and_ladders.search.infer import score_topology\n",
    )
    _write(
        package, "sim/graph.py", "from snakes_and_ladders.numerics import logsumexp\n"
    )
    _write(
        package, "sandbox/declined.py", "from snakes_and_ladders.opt.fit import fit\n"
    )

    realized = edges(package)

    assert realized == {("opt", "sim"): {"objective"}, ("opt", "search"): {"objective"}}
    assert {edge for edge in realized if edge not in ADMITTED} == {
        ("opt", "sim"),
        ("opt", "search"),
    }
    # The package root is read as an edge to nowhere, and `sandbox/` is not
    # read at all: the two exclusions above, exercised rather than stated.
    assert _imported_directories(package / "sim" / "graph.py") == {"(root)"}
    assert not any(edge[0] == "sandbox" for edge in realized)


@pytest.mark.infra
@pytest.mark.smoke
def test_the_cycle_reader_finds_a_cycle_on_a_tree_that_has_one(tmp_path: Path) -> None:
    # The same control for the second claim: two directories importing each
    # other are a cycle, and one importing the other is not.
    package = tmp_path / "snakes_and_ladders"
    _write(
        package,
        "search/infer.py",
        "from snakes_and_ladders.sample.gibbs import sweep\n",
    )
    _write(
        package,
        "sample/gibbs.py",
        "from snakes_and_ladders.sim.graph import PottsGraph\n",
    )
    _write(package, "sim/graph.py", "import numpy as np\n")

    realized = set(edges(package))

    assert {(a, b) for (a, b) in realized if (b, a) in realized} == set()

    _write(
        package,
        "sample/tempered.py",
        "from snakes_and_ladders.search.infer import score\n",
    )
    realized = set(edges(package))

    assert {(a, b) for (a, b) in realized if (b, a) in realized} == {
        ("sample", "search"),
        ("search", "sample"),
    }
