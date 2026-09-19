"""The duplications issues #230, #413 and #277 closed, asserted rather than remembered.

A consolidation that nothing enforces is a consolidation with a half-life.
Each of the three below was written between four and twelve times before it
had one home, and each grew *after* the survey that counted it was filed:
edge iteration doubled from six sites to twelve while the ticket waited.

The fourth is a *value* rather than a routine, and its guard reaches wider
than the package: the exact transition a fixture is declared at was written
as a float in a test and again in a notebook cell, and the two agreed by
coincidence of two literals rather than by construction (issue #413). A
closed form computed in three places is the same defect as a function
implemented in three places, so it is checked the same way.

Each guard is paired with a test that the guard fails on a violating input.
That pairing is the discipline `tests/regression/docs` established: a check
that has never been seen to fail is not known to work, and a regex over
source files is exactly the kind that silently matches nothing.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "python" / "snakes_and_ladders"
sys.path.insert(0, str(REPO_ROOT / "infra"))

import appraise_structures  # noqa: E402
import duplication_survey  # noqa: E402

#: Issue #717's rows, pinned at the count on the day each was first measured
#: (2026-09-18, `main` at a5b6fa4) so nothing grows while the eight pull
#: requests land; the pull request that lowers a row lowers its pin. Two of
#: the ticket's numbers were impressions this query corrected: it named 14
#: modules without a docstring and there is one (`scripts/__init__.py`), and
#: nine compiled twins where eight sit beside an oracle.
SLIMMING_BASELINE = {
    "Potts energies of a labelling": 1,
    "site-field broadcasts": 0,
    "annealers": 3,
    "ground-state run_ wrappers": 7,
    "backend enums": 1,
    "Python paths above a compiled kernel": 6,
    "surrogate modules": 4,
    "modules without a docstring": 0,
    "root exports": 7,
    "test modules pinning a twin to its oracle beyond the first": 13,
    # Both rows count the tree rather than a duplicate, so they move when a
    # module lands: `search.balanced`, the one proposal kernel the Potts
    # lattice and the factor graph share, took them from 148 and 1,576
    # (#756). The second moves on a public name too, and #756's cluster moves
    # for a frustrated lattice added twelve without adding a module; issue
    # #755's seam is one public callable, so the row rises by one more. A row
    # that rises states why here or it is a duplicate.
    "flat modules": 149,
    "API-map entries": 1603,
}

#: Issue #755's audit, pinned at the count it was taken on (2026-09-19, this
#: tree): every cluster `infra/appraise_structures.py` reports at three or
#: more members, and how many members each holds. A cluster that grows is a
#: near-duplicate added, and a cluster that appears is a shape nobody
#: decided --- the audit's outcome per row is in `docs/reviews/2026-09-19.md`
#: and a row that moves without that file moving is the audit going stale.
#: The consuming-reference counts are deliberately absent: they move with
#: every unrelated mention of a name, so pinning them would fail for reasons
#: that are not duplication.
CLUSTER_BASELINE = {
    "fields:name": 7,
    "prefix:Exact": 5,
    "role:incidence": 15,
    "suffix:Dataset": 5,
    "suffix:Decoding": 7,
    "suffix:Fit": 6,
    "suffix:Params": 17,
    "suffix:Result": 6,
}

#: State-carrying classes over the whole package, the number the clusters are
#: drawn from.
#: Re-pinned on the merge with `main` 5fe6ef2 (2026-09-19): the audit's 245
#: read 247 there, main having added two state-carrying classes since.
STRUCTURE_BASELINE = 247

#: `enumeration.argmax` outside its own module. Issue #755 folded the three
#: `learn` oracles that enumerated, scored and took the first maximizer onto
#: `enumeration.enumerated_optimum`, so the composition is written once.
#: `likelihood.hmm_paths` keeps its own call and is named here rather than
#: folded: it reads the score vector again for the posterior, so the scores
#: outlive the argmax and `enumerated_optimum`, which returns one score,
#: cannot carry it.
ARGMAX_CONSUMERS = {"likelihood/hmm_paths.py"}

# The consolidated home of each pattern, which legitimately contains it once.
LOGSUMEXP_OWNER = "numerics.py"
TRANSITION_OWNER = "sim/potts.py"
EDGE_ITERATION_OWNER = "sim/graph.py"
ADJACENCY_OWNER = "sim/graph.py"
ENUMERATION_OWNER = "enumeration.py"

PRIVATE_LOGSUMEXP = re.compile(r"^def _logsumexp\(", re.MULTILINE)
OPEN_CODED_EDGES = re.compile(r"zip\(\s*\w+\.edges,\s*\w+\.coupling")
CAP_LITERAL = re.compile(r"^\s*MAX_ENUMERABLE\w* = \d", re.MULTILINE)
SQUARE_TRANSITION = re.compile(r"log\(\s*1(\.0)?\s*\+\s*(np\.|numpy\.|math\.)?sqrt")
#: The annotation every list-of-lists adjacency carries. It is the whole
#: catch rather than half of one: the builders all start from an empty
#: comprehension, which `mypy --strict` refuses without a type, so a copy
#: cannot enter the package unannotated. A `torch.Tensor` coupling does not
#: match, which is deliberate --- `likelihood.surrogate.tree_log_partition`
#: walks a tree whose couplings carry gradients, and the compressed rows are
#: `float64` arrays that would cut the autodiff graph.
NEIGHBOUR_LISTS = re.compile(r"list\[list\[tuple\[int, ?float\]\]\]")
#: Names an environment carried before it was named for its problem. Issue
#: #644 retired the first two and #705 the third, each on the same rule: the
#: problem is `potts`, `hmm` or `tree`, and a class named for its mechanism
#: sends a reader looking for a mechanism. A retired name is guarded rather
#: than remembered --- `Topology` was the *third* spelling of one seam, so the
#: spellings return unless something refuses them.
RETIRED_ENVIRONMENTS = re.compile(
    r"\b(PottsLandscape|StatePathLandscape|TopologyEnvironment)\b"
)

#: Where a caller of the transition may live: the package, the suite and the
#: notebooks. Wider than the package alone, because both copies this guard
#: exists for were outside it.
SEARCHED = (
    PACKAGE,
    REPO_ROOT / "tests",
    REPO_ROOT / "docs" / "nb",
)


def _offenders(pattern: re.Pattern[str], owner: str) -> list[str]:
    """Package files matching ``pattern``, excluding the one that owns it."""
    return _found(pattern, owner, (PACKAGE,), ("*.py",))


def _found(
    pattern: re.Pattern[str],
    owner: str,
    roots: tuple[Path, ...],
    suffixes: tuple[str, ...],
) -> list[str]:
    """Files under ``roots`` matching ``pattern``, excluding the owner."""
    return [
        str(path.relative_to(REPO_ROOT))
        for root in roots
        for suffix in suffixes
        for path in sorted(root.rglob(suffix))
        if not path.as_posix().endswith(owner) and pattern.search(path.read_text())
    ]


@pytest.mark.critical
@pytest.mark.infra
def test_logsumexp_has_one_implementation() -> None:
    # Four copies in two spellings, in `sim.potts`, `opt.potts`,
    # `likelihood.potts` and `likelihood.belief_propagation`. They agreed, so
    # nothing failed; a divergence would have been silent and would have moved
    # a log-partition function rather than raising.
    assert _offenders(PRIVATE_LOGSUMEXP, LOGSUMEXP_OWNER) == []


@pytest.mark.critical
@pytest.mark.infra
def test_no_module_walks_edges_and_couplings_by_hand() -> None:
    # Twelve sites across eight modules zipped the two tuples together. The
    # pairing is an invariant of `PottsGraph`, so it belongs to the class:
    # a consumer that omits `strict=True` truncates to the shorter tuple and
    # silently drops edges from an energy.
    assert _offenders(OPEN_CODED_EDGES, EDGE_ITERATION_OWNER) == []


@pytest.mark.critical
@pytest.mark.infra
def test_the_adjacency_is_built_in_one_place() -> None:
    # Seven builders of one object: `potts_mcmc._adjacency`,
    # `potts_mcmc_rust.flatten_adjacency`, open-coded copies in
    # `ground_state`, `alpha_expansion`, `sim.potts` and
    # `spatio_sequential`, and `PottsGraph.compressed_adjacency` itself --
    # the sixth found by this guard rather than by the survey. The copies
    # agreed, so nothing failed; what they cost was the layout root
    # `CLAUDE.md` names -- a pointer chase and a Python float per neighbour
    # where the compressed rows are a stride (issue #277). `learn/potts.py`
    # keeps its own, without couplings: `learn/` imports no application
    # module, and its call site says so.
    assert _offenders(NEIGHBOUR_LISTS, ADJACENCY_OWNER) == []


@pytest.mark.critical
@pytest.mark.infra
def test_the_enumeration_cap_is_defined_once() -> None:
    # Four thresholds in three units before this: 200_000 configurations,
    # 200_000 paths, 20 nodes, and a docstring-only `n <= 6` that nothing
    # enforced. Enumeration is the oracle nearly every claim here rests on,
    # so how it declines is the one thing that should not vary.
    assert _offenders(CAP_LITERAL, ENUMERATION_OWNER) == []


@pytest.mark.critical
@pytest.mark.infra
def test_the_square_lattice_transition_is_computed_in_one_place() -> None:
    # `J_c = ln(1 + sqrt(q))` was a literal in
    # `tests/regression/search/test_potts_mcmc.py` and again in a
    # `docs/nb/potts_chain.ipynb` cell, each building its own 12x12 lattice
    # at it. The two agreed, so nothing failed; a rounded copy would have
    # moved one of them off the transition silently, and the instance is
    # only interesting *at* it. `snakes_and_ladders.sim.potts` owns the form
    # and `potts_lattice/stress` declares the instance.
    assert (
        _found(SQUARE_TRANSITION, TRANSITION_OWNER, SEARCHED, ("*.py", "*.ipynb")) == []
    )


@pytest.mark.critical
@pytest.mark.infra
def test_no_retired_environment_name_returns() -> None:
    # Three names for the same seam in two years: `PottsLandscape` and
    # `StatePathLandscape` went in #644, `TopologyEnvironment` in #705. The
    # search is the whole repository and not the package alone, because the
    # old names survived longest in the suite and in a notebook cell -- which
    # is where the fourth spelling would come back from.
    assert (
        _found(
            RETIRED_ENVIRONMENTS,
            "test_duplication_guards.py",
            SEARCHED,
            ("*.py", "*.ipynb"),
        )
        == []
    )


@pytest.mark.critical
@pytest.mark.infra
def test_each_guard_fails_on_violating_source() -> None:
    # The guards exercised. Each searches source text, so each passes
    # vacuously if the pattern is wrong -- which is the failure mode a guard
    # over a regex actually has.
    violating = {
        PRIVATE_LOGSUMEXP: "def _logsumexp(values, axis):\n    return values\n",
        OPEN_CODED_EDGES: "for e, c in zip(graph.edges, graph.coupling, strict=True):\n",
        CAP_LITERAL: "MAX_ENUMERABLE_THINGS = 200_000\n",
        NEIGHBOUR_LISTS: "neighbours: list[list[tuple[int, float]]] = []\n",
        # Split so this module is not its own offender: the guard reads the
        # suite, and a literal here would match.
        SQUARE_TRANSITION: "TRANSITION = math.log(1.0 + " + "math.sqrt(3.0))\n",
        # Split for the same reason as the line above.
        RETIRED_ENVIRONMENTS: "landscape = Potts" + "Landscape(graph, field)\n",
    }
    clean = {
        PRIVATE_LOGSUMEXP: "from snakes_and_ladders.numerics import logsumexp\n",
        OPEN_CODED_EDGES: "for edge, coupling in graph.weighted_edges():\n",
        CAP_LITERAL: "from snakes_and_ladders.enumeration import refuse_oversized\n",
        NEIGHBOUR_LISTS: (
            "offsets, neighbours, couplings = graph.compressed_adjacency()\n"
        ),
        SQUARE_TRANSITION: 'print(f"at J_c = ln(1 + sqrt(3)) = {coupling:.4f}")\n',
        RETIRED_ENVIRONMENTS: "environment = TreeEnvironment(alignment, k=4)\n",
    }

    assert [p for p, text in violating.items() if not p.search(text)] == []
    assert [p for p, text in clean.items() if p.search(text)] == []


@pytest.mark.critical
@pytest.mark.infra
def test_the_slimming_rows_hold_at_their_baseline() -> None:
    # Issue #717 is measured by these rows before and after each of its
    # pull requests. A row above its pin is a new duplicate; a row below it
    # is a pull request that landed and did not lower the pin.
    rows = {finding.name: finding for finding in duplication_survey.FINDINGS}
    realized = {name: rows[name].now() for name in SLIMMING_BASELINE}

    assert realized == SLIMMING_BASELINE


@pytest.mark.critical
@pytest.mark.infra
def test_each_slimming_query_counts_a_violating_tree(tmp_path: Path) -> None:
    # The file-and-import queries exercised on a tree built to trip each:
    # a twin beside its oracle, a module without a docstring, a root that
    # exports one name, and two test modules that both pin the twin.
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text('"""Root."""\n\n__all__ = ["one"]\n')
    (package / "kernel.py").write_text('"""The oracle."""\n')
    (package / "kernel_rust.py").write_text("import numpy\n")
    (package / "surrogate.py").write_text('"""A surrogate."""\n')
    (package / "sandbox").mkdir()
    (package / "sandbox" / "declined_rust.py").write_text("x = 1\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    for name in ("test_a.py", "test_b.py"):
        (tests / name).write_text(
            "from pkg import kernel\nfrom pkg.kernel_rust import run\n"
        )
    (tests / "test_c.py").write_text("from pkg import kernel\n")

    assert duplication_survey.flat_modules(package) == 3
    assert duplication_survey.twin_modules(package) == 1
    assert duplication_survey.surrogate_modules(package) == 1
    assert duplication_survey.undocumented_modules(package) == 1
    assert duplication_survey.root_exports(package) == 1
    assert duplication_survey.twin_pins_beyond_the_first(package, tests) == 1

    (package / "kernel_rust.py").write_text('"""Now documented."""\n')
    assert duplication_survey.undocumented_modules(package) == 0


@pytest.mark.critical
@pytest.mark.infra
def test_each_slimming_pattern_matches_its_own_kind() -> None:
    # The regular-expression rows, on one violating and one clean line each.
    patterns = {
        finding.name: re.compile(finding.pattern, re.MULTILINE)
        for finding in duplication_survey.FINDINGS
        if finding.pattern
    }
    violating = {
        "Potts energies of a labelling": "def energy(graph, field, labelling):\n",
        "site-field broadcasts": "def _site_field(graph, values):\n",
        "annealers": "def anneal_potts(graph, field, schedule):\n",
        "ground-state run_ wrappers": "def run_wolff(rung, budget, rng):\n",
        "backend enums": "class CoupledBackend:\n",
    }
    clean = {
        "Potts energies of a labelling": "    energy = energies(graph, f, s)[0]\n",
        "site-field broadcasts": "values = site_field(field, graph.n_nodes)\n",
        "annealers": "schedule = geometric_schedule(1.0, 0.1, 100)\n",
        "ground-state run_ wrappers": "result = METHODS[name](rung, budget, rng)\n",
        "backend enums": "backend = Backend.RUST\n",
    }

    assert [n for n, text in violating.items() if not patterns[n].search(text)] == []
    assert [n for n, text in clean.items() if patterns[n].search(text)] == []


def _imports_argmax(path: Path) -> bool:
    """Whether ``path`` imports ``argmax`` from the enumeration seam.

    By `ast` rather than by a line regex: the one legitimate consumer spells
    the import over five lines, and a guard that a parenthesised import slips
    past is a guard that passes vacuously.
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "snakes_and_ladders.enumeration"
            and any(alias.name == "argmax" for alias in node.names)
        ):
            return True
    return False


@pytest.mark.critical
@pytest.mark.infra
def test_the_enumerated_argmax_is_composed_in_one_place() -> None:
    # `learn.potts.optimum`, `learn.hmm.optimum` and
    # `learn.relaxed.enumerate_optimum` each enumerated, scored and took the
    # first maximizer in full: three bodies, one arithmetic, and three places
    # a tie rule could drift apart from the determinism their docstrings
    # promise. The fourth copy is what this refuses (issue #755).
    realized = {
        str(path.relative_to(PACKAGE))
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.name != "enumeration.py" and _imports_argmax(path)
    }

    assert realized == ARGMAX_CONSUMERS


@pytest.mark.critical
@pytest.mark.infra
def test_the_argmax_query_reads_a_parenthesised_import(tmp_path: Path) -> None:
    # The pairing this module requires. The one legitimate consumer spells
    # the import over five lines, so the failure mode worth exercising is a
    # query that only reads a single-line `from ... import argmax`.
    over_five_lines = tmp_path / "wrapped.py"
    over_five_lines.write_text(
        "from snakes_and_ladders.enumeration import (\n"
        "    MAX_ENUMERABLE_CONFIGURATIONS,\n"
        "    argmax,\n"
        "    configurations,\n"
        ")\n"
    )
    through_the_seam = tmp_path / "folded.py"
    through_the_seam.write_text(
        "from snakes_and_ladders.enumeration import enumerated_optimum\n"
    )

    assert _imports_argmax(over_five_lines)
    assert not _imports_argmax(through_the_seam)


@pytest.mark.critical
@pytest.mark.infra
def test_the_structure_clusters_hold_at_their_audited_size() -> None:
    # Issue #755 decided each cluster against root `CLAUDE.md`'s rule --- one
    # abstraction where it aligns and simplifies several use cases --- and
    # recorded the reason per row. The decision is only worth what the count
    # it was taken at is worth, so the count is asserted rather than cited.
    # The query itself is exercised in `test_structure_survey.py`, which
    # plants a near-duplicate and a shape below the rule and reads both back.
    found = appraise_structures.structures()
    realized = {
        cluster.key: len(cluster.members)
        for cluster in appraise_structures.clusters(found)
    }

    assert len(found) == STRUCTURE_BASELINE
    assert realized == CLUSTER_BASELINE
