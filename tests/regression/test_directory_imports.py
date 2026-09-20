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


def _directory(path: Path) -> str:
    relative = path.relative_to(PACKAGE)
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


def edges() -> dict[tuple[str, str], set[str]]:
    """Every ``(from, to)`` directory edge, with the modules that make it."""
    realized: dict[tuple[str, str], set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "__init__.py" or any(part in EXCLUDED for part in path.parts):
            continue
        source = _directory(path)
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
