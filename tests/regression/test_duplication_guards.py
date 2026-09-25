"""The duplications issues #230, #413 and #277 closed, asserted rather than remembered.

A guard per seam refuses the next copy of a routine, value or store written
four to twelve times (edge iteration grew from six sites to twelve). The value is
the exact transition a fixture declares, once a float in a test and a notebook
(#413). The seams: `MessageSchedule` and `SparseIncidence` (#755, #586), read
from the class and syntax trees, not a regex; no consumer branches on a
schedule's name (#592). The derivations (#863): the repository root, once
counted from each file's depth fifty-three times, and one `PROBLEMS.md`
reader, once nine. Each guard is paired with a failing input; overlap
surveys live in dated reviews (#813).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import _paths
import pytest
import rename_package
from sal.likelihood.schedule import (
    SCHEDULES,
    MessageSchedule,
    MessageScheduleName,
)

from tests._paths import REPO_ROOT

PACKAGE = REPO_ROOT / "python" / "sal"

#: `enumeration.argmax` outside its own module (#755 folded three `learn`
#: oracles onto `enumerated_optimum`). `likelihood.hmm_paths` reuses the score
#: vector for the posterior, which `enumerated_optimum` does not return.
ARGMAX_CONSUMERS = {"likelihood/hmm_paths.py"}

# The consolidated home of each pattern, which legitimately contains it once.
LOGSUMEXP_OWNER = "numerics.py"
TRANSITION_OWNER = "sim/potts.py"
EDGE_ITERATION_OWNER = "sim/graph.py"
ADJACENCY_OWNER = "sim/graph.py"
ENUMERATION_OWNER = "enumeration.py"

#: A private `logsumexp` outside `numerics`. The pattern read `_logsumexp(`
#: exactly and so said nothing about `_logsumexp_last`, the one the tree
#: carries; `\w*` is what a suffix costs it.
PRIVATE_LOGSUMEXP = re.compile(r"^def _logsumexp\w*\(", re.MULTILINE)

#: `message_passing`'s private `logsumexp` is the owner's five operations in
#: order over the last axis, bitwise equal to `numerics.logsumexp`, without the
#: general axis handling; folding it is row R10 of the 2026-09-20 design review.
LOGSUMEXP_INLINES = {"likelihood/message_passing.py"}
OPEN_CODED_EDGES = re.compile(r"zip\(\s*\w+\.edges,\s*\w+\.coupling")
CAP_LITERAL = re.compile(r"^\s*MAX_ENUMERABLE\w* = \d", re.MULTILINE)
SQUARE_TRANSITION = re.compile(r"log\(\s*1(\.0)?\s*\+\s*(np\.|numpy\.|math\.)?sqrt")
#: Every list-of-lists adjacency carries this annotation (`mypy --strict`
#: refuses the empty comprehension without it). A `torch.Tensor` coupling does
#: not match: `tree_log_partition` needs couplings that carry gradients.
NEIGHBOUR_LISTS = re.compile(r"list\[list\[tuple\[int, ?float\]\]\]")
#: Environment names retired for naming a mechanism, not the problem (#644,
#: #705); `Topology` was the third spelling of one seam.
RETIRED_ENVIRONMENTS = re.compile(
    r"\b(PottsLandscape|StatePathLandscape|TopologyEnvironment)\b"
)

#: The module owning the schedule seam, and the base every schedule inherits.
SCHEDULE_OWNER = "likelihood/schedule.py"
SCHEDULE_BASE = "MessageSchedule"

#: The registered schedules: `tree`, `upward`, `downward`, `flooding`,
#: `sequential` and `residual` (#825 added the sixth). `search.ground_state.Entry`
#: shares only the field `name`. A seventh raises this in its own PR.
SCHEDULE_COUNT = 6

#: A consumer branching on a schedule's name: the `if` #592 deleted. Names come
#: from `MessageScheduleName`, so a new schedule is guarded on registration.
#: `message_passing_reference` is the oracle, not a consumer; see below.
SCHEDULE_NAMES = "|".join(str(member) for member in MessageScheduleName)
SCHEDULE_NAME_BRANCH = re.compile(
    r"(?:if|elif|while)\b[^\n]*(?:"
    r"\.name\s*(?:==|!=|in)\s*[^\n]*"
    rf"""(?:MessageScheduleName\.|["'](?:{SCHEDULE_NAMES})["'])"""
    r"|\bis(?: not)?\s+MessageScheduleName\."
    r")"
)

#: `message_passing_reference`'s `is MessageScheduleName.TREE` chooses which
#: order the oracle writes out; the `is` form entered the pattern in #864.
SCHEDULE_NAME_CONSUMERS = {"likelihood/message_passing_reference.py"}
SCHEDULE_NAME_MATCH = re.compile(r"match\s+[^\n]*\b(?:schedule|plan)\w*\.name\s*:")


#: The module owning the compressed layout, and the two calls that build a
#: store through it: `SparseIncidence.from_pairs` and, for a Potts graph,
#: `PottsGraph.compressed_adjacency`, which holds one.
INCIDENCE_OWNER = "incidence.py"

#: Package modules building a store through the seam, as of 048a342 plus
#: `sample/tempered.py` and `sample/annealed.py` (#766); `search/ground_state.py`
#: left with #858; `qa/potts_clusters.py` joined with #1041. A PR adding or
#: removing a consumer moves the pin.
SEAM_CONSUMERS = 12

#: Files this guard does not read, each against the reason, rather than an
#: allow-list nobody can audit. The first two are measurements the
#: 2026-09-19 review records; the third is what this guard found on `main`.
EXCLUDED: dict[str, str] = {
    "ragged.py": (
        "offsets rebuilt per call kept: the Python scan is 39.3 us against "
        "335.11 ms for one Baum-Welch iteration over the same batch, 0.012% "
        "(#677)"
    ),
    "search/maxflow.py": (
        "list of lists kept: a row is 1.95 ms as lists against 3.93 ms flat "
        "with offsets over 16,384 rows of degree six, and the compiled "
        "consumer takes `as_arrays` (#586); `from_arcs` builds through the "
        "seam"
    ),
    "learn/surrogate.py": (
        "found by this guard, to be folded or declined under #755: "
        "`_Batch.__init__` lays the token blocks of a batch end to end and "
        "computes their starts with `np.cumsum`"
    ),
}

#: Object graphs where the relation is the model (`sim.tree.Node`, `Factor`,
#: `Region`, eight more) build no store and trip neither rule. `scipy.sparse`
#: is read as text, since the alias is arbitrary; the package uses `scipy` for
#: `linear_sum_assignment` only. Suite and notebooks are searched too.
FOREIGN_SPARSE = re.compile(r"\b(?:csr|csc|coo)_(?:matrix|array)\s*\(")

#: The validation seams of issue #1010: a goal's median of package runs, a
#: framework test's skip, and a script's timed peak, each written out by
#: hand 38, 24 and 6 times before it had one home.
HAND_MEDIAN = re.compile(r"np\.median\(\s*\[\s*package\(")
HAND_SKIP = re.compile(r"skipif\(\s*not available\(")
HAND_MEASURE = re.compile(r"peaked\(\s*lambda: timed\(")
MEASURE_OWNER = "validation/protocol.py"
#: A fixture's alignment built field by field: its tree, states and root
#: handed to `simulate_alignment` one by one rather than the fixture to
#: `sim.simulator.simulate_tree`, which 104 sites spelled out (issue #1010).
FIXTURE_ALIGNMENT = re.compile(
    r"simulate_alignment\(\s*(?:tau=)?(\w+)\.tau,\s*(?:k=)?\1\.k,\s*(?:pi=)?\1\.pi\b"
)
FIXTURE_ALIGNMENT_OWNER = "sim/simulator.py"
#: A Rust twin imported inside a function by hand: the refusal, the local
#: import and the comment on the cycle it avoids, which eight sites in six
#: modules wrote before `backend.twin` carried them (issue #1010). Indented
#: only: a module-level import of a twin is the cycle and fails at import.
#: A Rust twin is `<sub>.rust.<algorithm>` since #1059.
TWIN_IMPORT = re.compile(
    r"^[ \t]+(?:from sal\.\w+\.rust(?:\.\w+)? import \w+\b"
    r"|import sal\.\w+\.rust\.\w+\b)",
    re.MULTILINE,
)
TWIN_OWNER = "backend.py"

#: Where a caller of the transition may live: the package, the suite and the
#: notebooks. Wider than the package alone, because both copies this guard
#: exists for were outside it.
SEARCHED = (
    PACKAGE,
    REPO_ROOT / "tests",
    REPO_ROOT / "docs" / "nb",
)


def _admitted(path: str, declared: set[str]) -> bool:
    """Whether a match is one of the sites declared, by path below the package."""
    return any(path.endswith(f"sal/{site}") for site in declared)


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


def _members(node: ast.ClassDef) -> set[str]:
    """The attributes and methods one class defines in its own body."""
    found: set[str] = set()
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            found.add(statement.target.id)
        elif isinstance(statement, ast.Assign):
            found |= {
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            }
        elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            found.add(statement.name)
    return found


def _schedule_shaped(source: str) -> dict[str, list[str]]:
    """Classes shaped like a schedule in one source, and the bases each lists.

    Shaped: a name ending ``MessageSchedule``, or both ``guarantee`` and ``name``.
    """
    shaped: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or node.name == SCHEDULE_BASE:
            continue
        if not (
            node.name.endswith(SCHEDULE_BASE) or {"guarantee", "name"} <= _members(node)
        ):
            continue
        shaped[node.name] = [
            base.id if isinstance(base, ast.Name) else base.attr
            for base in node.bases
            if isinstance(base, ast.Name | ast.Attribute)
        ]
    return shaped


def _schedule_classes() -> dict[str, list[str]]:
    """Every schedule-shaped class in the package, read from the source."""
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        found.update(_schedule_shaped(path.read_text()))
    return found


def _names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    """Every name a binding writes to, `self._offsets` reading as `_offsets`."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [
        sub.id if isinstance(sub, ast.Name) else sub.attr
        for target in targets
        for sub in ast.walk(target)
        if isinstance(sub, ast.Name | ast.Attribute)
    ]


def _called(node: ast.AST) -> str:
    """The name a call calls, without its module: `np.cumsum` reads `cumsum`."""
    if not isinstance(node, ast.Call):
        return ""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return node.func.id if isinstance(node.func, ast.Name) else ""


#: The two arrays a compressed store is: the row starts, and the permutation
#: that put the pairs in row-major order. Matched on the name a line binds
#: rather than on the arithmetic, because the arithmetic is four lines and a
#: reader writing a fifth spelling still calls the result `offsets`.
ROW_STARTS = re.compile(r"offsets|indptr")
ROW_INDEX = re.compile(r"\brows?\b|\brow_|_rows?\b")


def _hand_built_stores(source: str) -> list[str]:
    """The compressed-sparse stores one source builds by hand, with their lines.

    `cumsum` to `offsets`/`indptr`, or `lexsort`/`argsort` to `order`; the owner trips both.
    """
    found: set[tuple[int, str]] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _called(node) == "cumsum":
            out = [kw for kw in node.keywords if kw.arg == "out"]
            if out and ROW_STARTS.search(ast.unparse(out[0].value)):
                found.add((node.lineno, "row starts by cumsum"))
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        names = _names(node)
        for call in ast.walk(node.value):
            called = _called(call)
            if called == "cumsum" and any(ROW_STARTS.search(n) for n in names):
                found.add((node.lineno, "row starts by cumsum"))
            if (
                called in ("lexsort", "argsort")
                and any("order" in n for n in names)
                and ROW_INDEX.search(ast.unparse(call))
            ):
                found.add((node.lineno, "pairs sorted into row-major order"))
    return [f"{line}: {what}" for line, what in sorted(found)]


def _builds_through_the_seam(source: str) -> bool:
    """Whether one source calls `SparseIncidence.from_pairs` or the adjacency."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "compressed_adjacency":
            return True
        if node.func.attr == "from_pairs" and ast.unparse(node.func.value).endswith(
            "SparseIncidence"
        ):
            return True
    return False


def _package_sources() -> dict[str, str]:
    """Every package module, keyed by its path below the package root."""
    return {
        path.relative_to(PACKAGE).as_posix(): path.read_text()
        for path in sorted(PACKAGE.rglob("*.py"))
    }


@pytest.mark.critical
@pytest.mark.infra
def test_logsumexp_has_one_implementation() -> None:
    # Four copies in two spellings, in `sim.potts`, `opt.potts`,
    # `likelihood.potts` and `likelihood.belief_propagation`. They agreed, so
    # nothing failed; a divergence would have been silent and would have moved
    # a log-partition function rather than raising.
    found = _offenders(PRIVATE_LOGSUMEXP, LOGSUMEXP_OWNER)

    assert [path for path in found if not _admitted(path, LOGSUMEXP_INLINES)] == []
    # The declaration is an edge and not a wish: the admitted inline is
    # there, so a fold that removes it takes this entry with it.
    assert sorted(found) == sorted(f"python/sal/{p}" for p in LOGSUMEXP_INLINES)


#: The suffixes and roots the rows below search.
_PY, _PY_NB = ("*.py",), ("*.py", "*.ipynb")
_TESTS, _HERE = (REPO_ROOT / "tests",), "test_duplication_guards.py"

#: Each pattern refused outside its one home: the pattern, the path the owner
#: ends with, the roots searched and the suffixes read.
ABSENT: dict[str, tuple[re.Pattern[str], str, tuple[Path, ...], tuple[str, ...]]] = {
    # Twelve sites across eight modules zipped the two tuples; a consumer
    # without `strict=True` truncates to the shorter and drops edges from an
    # energy. The pairing is `PottsGraph`'s invariant.
    "edges by hand": (OPEN_CODED_EDGES, EDGE_ITERATION_OWNER, (PACKAGE,), _PY),
    # Seven builders of one adjacency (issue #277), the sixth found by this
    # guard; each paid a pointer chase and a Python float per neighbour where
    # the compressed rows are a stride. `learn/potts.py` keeps its own,
    # without couplings, since `learn/` imports no application module.
    "neighbour lists": (NEIGHBOUR_LISTS, ADJACENCY_OWNER, (PACKAGE,), _PY),
    # Four enumeration thresholds in three units, one enforced by nothing;
    # enumeration is the oracle most claims rest on.
    "enumeration cap": (CAP_LITERAL, ENUMERATION_OWNER, (PACKAGE,), _PY),
    # `J_c = ln(1 + sqrt(q))` was a literal in a test and a notebook cell; a
    # rounded copy moves an instance off the transition silently.
    "square transition": (SQUARE_TRANSITION, TRANSITION_OWNER, SEARCHED, _PY_NB),
    # Three names for one seam: #644 retired two, #705 the third. The old
    # names survived longest in the suite and a notebook cell.
    "retired environment": (RETIRED_ENVIRONMENTS, _HERE, SEARCHED, _PY_NB),
    # A `csr_matrix` beside `SparseIncidence` is a second layout. The package
    # and notebooks only: `tests/regression/test_incidence.py` holds
    # `scipy.sparse` as the referee of the one layout (#776).
    "scipy sparse": (
        FOREIGN_SPARSE,
        _HERE,
        (PACKAGE, REPO_ROOT / "docs" / "nb"),
        _PY_NB,
    ),
    # Issue #1010: `median_package`, `requires` and `measured`.
    "hand median": (HAND_MEDIAN, "tests/validation/_goals.py", _TESTS, _PY),
    "hand skip": (HAND_SKIP, "tests/_frameworks.py", _TESTS, _PY),
    "hand measure": (HAND_MEASURE, MEASURE_OWNER, (PACKAGE,), _PY),
    # Issue #1010: `simulate_tree(params, rng, n_sites=...)` draws a
    # fixture's alignment; the fields were spelled out 104 times.
    "fixture alignment": (FIXTURE_ALIGNMENT, FIXTURE_ALIGNMENT_OWNER, SEARCHED, _PY),
    # Issue #1010: `backend.twin(name, backend, __name__)` refuses and
    # imports; eight sites in six modules spelled both out.
    "twin by hand": (TWIN_IMPORT, TWIN_OWNER, (PACKAGE,), _PY),
}


@pytest.mark.critical
@pytest.mark.infra
def test_each_guarded_pattern_is_absent_outside_its_owner() -> None:
    found = {name: _found(*row) for name, row in ABSENT.items()}
    assert {name: files for name, files in found.items() if files} == {}


@pytest.mark.critical
@pytest.mark.infra
def test_no_schedule_is_written_outside_the_base() -> None:
    # A class beside the base carries no `guarantee`, `bounded` or
    # `requires_tree`, so `sum_product` would run a bounded schedule on a
    # loopy graph rather than refuse it (issue #755).
    classes = _schedule_classes()

    assert [name for name, bases in classes.items() if SCHEDULE_BASE not in bases] == []
    assert len(classes) == SCHEDULE_COUNT
    # One registration each, under a name `MessageScheduleName` carries: the
    # registry is `resolve`'s authority and the enum is the convenience, so
    # the two disagreeing is a schedule a name cannot reach or a name that
    # reaches none.
    assert sorted(type(schedule).__name__ for schedule in SCHEDULES.values()) == sorted(
        classes
    )
    assert set(SCHEDULES) == {str(member) for member in MessageScheduleName}
    # The runtime view and the source view agree, which is what says the walk
    # read every module a subclass can be defined in.
    assert sorted(cls.__name__ for cls in MessageSchedule.__subclasses__()) == sorted(
        classes
    )


@pytest.mark.critical
@pytest.mark.infra
def test_no_compressed_store_is_built_outside_the_incidence_seam() -> None:
    # One compressed layout, ten modules through it (#755). A copy lacks the
    # contract: stable order in a row, a shared transpose, `from_pairs`'s refusals.
    sources = _package_sources()
    by_hand = {
        path: stores
        for path, source in sources.items()
        if path != INCIDENCE_OWNER and path not in EXCLUDED
        for stores in [_hand_built_stores(source)]
        if stores
    }

    assert by_hand == {}
    # The reading is not vacuous: the owner builds the store this guard is
    # about, and both rules find it there.
    owner = [
        store.split(": ", 1)[1]
        for store in _hand_built_stores(sources[INCIDENCE_OWNER])
    ]
    assert owner == ["row starts by cumsum", "pairs sorted into row-major order"]
    assert (
        len(
            [
                path
                for path, source in sources.items()
                if path != INCIDENCE_OWNER and _builds_through_the_seam(source)
            ]
        )
        == SEAM_CONSUMERS
    )
    # An exclusion is a file and a reason, and a file that has moved takes its
    # reason with it rather than leaving a rule nothing applies to.
    assert [path for path in EXCLUDED if not (PACKAGE / path).exists()] == []
    assert [path for path, why in EXCLUDED.items() if not why] == []


@pytest.mark.critical
@pytest.mark.infra
def test_no_consumer_branches_on_a_schedules_name() -> None:
    # A branch on the name ran `schedule="tree"` as flooding: a plain string
    # equals a `StrEnum` member without being it (#592).
    found = _offenders(SCHEDULE_NAME_BRANCH, SCHEDULE_OWNER)

    assert [
        path for path in found if not _admitted(path, SCHEDULE_NAME_CONSUMERS)
    ] == []
    assert sorted(found) == sorted(f"python/sal/{p}" for p in SCHEDULE_NAME_CONSUMERS)
    assert _offenders(SCHEDULE_NAME_MATCH, SCHEDULE_OWNER) == []


@pytest.mark.critical
@pytest.mark.infra
def test_a_latex_table_has_one_scaffold() -> None:
    # Issue #926: four QA modules opened `tabular` and `array` by hand.
    assert _offenders(TABULAR_LITERAL, TABULAR_OWNER) == []
    # The owner builds the environment from its specification, so it holds
    # no literal; what it holds is the two scaffolds.
    owner = (PACKAGE / TABULAR_OWNER).read_text()
    assert "def booktabs_tabular(" in owner
    assert "def mathjax_array(" in owner


@pytest.mark.critical
@pytest.mark.infra
def test_the_runtime_band_has_one_reader() -> None:
    # Issue #926: `gap_band` and `qa.starts.curve_band` were the same loop
    # written twice; the second is now an adapter onto the first.
    assert _offenders(HELD_BAND, BAND_OWNER) == []
    assert HELD_BAND.search((PACKAGE / BAND_OWNER).read_text())


@pytest.mark.critical
@pytest.mark.infra
def test_iterated_conditional_modes_is_defined_in_one_module() -> None:
    assert _found(ICM_DEFINITION, ICM_OWNER, SEARCHED, ("*.py",)) == []
    assert ICM_DEFINITION.search((PACKAGE / ICM_OWNER).read_text())


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
        SCHEDULE_NAME_BRANCH: '    if plan.name == "flooding":\n',
        SCHEDULE_NAME_MATCH: '    match plan.name:\n        case "tree":\n',
        # Split for the same reason: this module is inside the search.
        FOREIGN_SPARSE: "matrix = csr" + "_matrix((data, (rows, cols)))\n",
        # Unsplit: the pattern is anchored at a line start and every literal
        # in this module is indented, so the module is not its own offender.
        DERIVED_REPO_ROOT: "REPO_ROOT = Path(__file__).resolve().parents[2]\n",
        CATALOGUE_FILE: "CATALOGUE = REPO_ROOT / 'PROBLEMS.md'\n",
        # Split inside the call for the reason the lines above are split: a
        # whole one here makes this module the reader the guard refuses.
        PIPE_SPLIT: "cells = line.split(" + '"|")\n',
        # Unsplit: the guard reads the package, not this suite.
        TABULAR_LITERAL: '            r"\\begin{tabular}{lrrrr}",\n',
        HELD_BAND: (
            '        index = np.searchsorted(times, grid, side="right") - 1\n'
            "        known = index >= 0\n"
            "        held[row, known] = gaps[index[known]]\n"
        ),
        # Split so this module is not its own offender.
        HAND_MEDIAN: "s = np.median(" + '[package("viterbi", inputs).seconds])\n',
        HAND_SKIP: "mark = pytest.mark.skipif(" + 'not available("gco"), reason="")\n',
        HAND_MEASURE: "r, p = peaked(" + "lambda: timed(call))\n",
        FIXTURE_ALIGNMENT: "d = simulate_" + "alignment(p.tau, p.k, p.pi, rng, 9)\n",
        TWIN_IMPORT: "        from sal.likelihood.rust import pruning\n",
        # Unsplit: anchored at a line start, and this literal is indented.
        ICM_DEFINITION: "def iterated_conditional_modes(\n    graph,\n",
    }
    clean = {
        PRIVATE_LOGSUMEXP: "from sal.numerics import logsumexp\n",
        OPEN_CODED_EDGES: "for edge, coupling in graph.weighted_edges():\n",
        CAP_LITERAL: "from sal.enumeration import refuse_oversized\n",
        NEIGHBOUR_LISTS: (
            "offsets, neighbours, couplings = graph.compressed_adjacency()\n"
        ),
        SQUARE_TRANSITION: 'print(f"at J_c = ln(1 + sqrt(3)) = {coupling:.4f}")\n',
        RETIRED_ENVIRONMENTS: "environment = TreeEnvironment(alignment, k=4)\n",
        SCHEDULE_NAME_BRANCH: (
            "    if plan.requires_tree and not graph.is_tree():\n"
            '        msg = f"the {plan.name} schedule is exact only on a tree"\n'
        ),
        SCHEDULE_NAME_MATCH: "    for step in plan.steps(layout):\n",
        FOREIGN_SPARSE: "from sal.incidence import SparseIncidence\n",
        DERIVED_REPO_ROOT: "from tests._paths import REPO_ROOT\n",
        CATALOGUE_FILE: "rows = catalogue.rows()\n",
        PIPE_SPLIT: "cells = catalogue.cells(line)\n",
        TABULAR_LITERAL: 'table = booktabs_tabular("lrrrr", HEADER, rows)\n',
        HELD_BAND: "band = curve_band(curves, grid)\n",
        HAND_MEDIAN: 'seconds = median_package("viterbi", inputs)\n',
        HAND_SKIP: 'pytestmark = requires("blackjax")\n',
        HAND_MEASURE: "result, seconds, peak = measured(call)\n",
        FIXTURE_ALIGNMENT: "data = simulate_tree(params, rng, n_sites=9)\n",
        TWIN_IMPORT: "    if (rust := twin(name, backend, __name__)) is not None:\n",
        ICM_DEFINITION: "from sal.search.icm import iterated_conditional_modes\n",
    }

    assert [p for p, text in violating.items() if not p.search(text)] == []
    assert [p for p, text in clean.items() if p.search(text)] == []

    # The two halves issue #864 widened, which one entry per pattern cannot
    # reach: a suffixed private name, and the identity comparison. Both
    # matched nothing before, so both guards read the tree and reported
    # what they could not see.
    assert PRIVATE_LOGSUMEXP.search("def _logsumexp_last(values):\n    return values\n")
    assert not PRIVATE_LOGSUMEXP.search(
        "def logsumexp_last(values):\n    return values\n"
    )
    assert SCHEDULE_NAME_BRANCH.search("    if schedule is MessageScheduleName.TREE:\n")
    assert SCHEDULE_NAME_BRANCH.search(
        "    if plan is not MessageScheduleName.FLOODING:\n"
    )
    assert not SCHEDULE_NAME_BRANCH.search("    if plan.requires_tree:\n")
    # The twin's other spelling, and the module-level import a twin's own
    # test is free to write (`test_backend_twins_are_reached_by_their_gateway`).
    assert TWIN_IMPORT.search("    import sal.search.rust.maxflow\n")
    assert not TWIN_IMPORT.search("from sal.likelihood.rust import pruning\n")

    # The structural guard on the same discipline, since it cannot be written
    # as a pattern: a sixth schedule beside the base and the same class under
    # it, and `Entry`'s shape, which carries `name` alone and is not one.
    outside = (
        "class LayeredMessageSchedule:\n"
        '    name: str = "layered"\n\n'
        "    @property\n"
        "    def guarantee(self) -> Guarantee:\n"
        "        return Guarantee.APPROXIMATE\n"
    )
    inside = outside.replace(
        "LayeredMessageSchedule:", f"LayeredMessageSchedule({SCHEDULE_BASE}):"
    )

    assert _schedule_shaped(outside) == {"LayeredMessageSchedule": []}
    assert _schedule_shaped(inside) == {"LayeredMessageSchedule": [SCHEDULE_BASE]}
    assert _schedule_shaped("@dataclass\nclass Entry:\n    name: str\n") == {}

    # Structural: row starts by `cumsum` under three names, row-major sorted
    # pairs, and two clean forms (a seam build, an unrelated `cumsum`).
    builds = (
        "degrees = np.bincount(rows, minlength=n)\n"
        "offsets = np.cumsum(degrees)\n"
        "self._indptr = np.cumsum(counts)\n"
        "np.cumsum(degrees, out=starts_offsets[1:])\n"
        "order = np.lexsort((columns, rows))\n"
    )
    through = (
        "offsets, neighbours, couplings = graph.compressed_adjacency()\n"
        "grouped = SparseIncidence.from_pairs(n_rows, n_cols, rows, columns)\n"
        "cumulative = np.cumsum(np.exp(local))\n"
        "rank[np.lexsort((np.arange(n), mean_bit))] = np.arange(n)\n"
    )

    assert _hand_built_stores(builds) == [
        "2: row starts by cumsum",
        "3: row starts by cumsum",
        "4: row starts by cumsum",
        "5: pairs sorted into row-major order",
    ]
    assert _hand_built_stores(through) == []
    assert _builds_through_the_seam(through) is True
    assert _builds_through_the_seam(builds) is False


def _imports_argmax(path: Path) -> bool:
    """Whether ``path`` imports ``argmax`` from the enumeration seam.

    By `ast`: the consumer spells the import over five lines.
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "sal.enumeration"
            and any(alias.name == "argmax" for alias in node.names)
        ):
            return True
    return False


@pytest.mark.critical
@pytest.mark.infra
def test_the_enumerated_argmax_is_composed_in_one_place() -> None:
    # Three `learn` oracles each enumerated, scored and took the first
    # maximizer; the fourth copy is what this refuses (issue #755).
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
        "from sal.enumeration import (\n"
        "    MAX_ENUMERABLE_CONFIGURATIONS,\n"
        "    argmax,\n"
        "    configurations,\n"
        ")\n"
    )
    through_the_seam = tmp_path / "folded.py"
    through_the_seam.write_text("from sal.enumeration import enumerated_optimum\n")

    assert _imports_argmax(over_five_lines)
    assert not _imports_argmax(through_the_seam)


# --- one home per derivation, issue #863 --------------------------------------

#: The root derived from a file's own location. `infra/_paths.py` and
#: `tests/_paths.py` carry it; fifty-three modules once counted their parents.
DERIVED_REPO_ROOT = re.compile(
    r"^REPO_ROOT = Path\(__file__\)\.resolve\(\)\.(?:parents\[\d+\]|parent\.parent)",
    re.MULTILINE,
)

#: The pieces of a `PROBLEMS.md` reader: the file and a pipe split, both
#: needed. A hand-opened LaTeX table: the scaffold is `booktabs_tabular` and
#: `mathjax_array` (issue #926), after four modules wrote it line for line.
TABULAR_OWNER = "qa/figure.py"
TABULAR_LITERAL = re.compile(r"\\begin\{(?:tabular|array)\}")

#: A runtime band read by holding each trial's gap forward to a grid: the
#: one reader is `search.mixture_starts.curve_band` (issue #926), where two
#: copies of the loop sat in `search.mixture_starts` and `qa.starts`.
BAND_OWNER = "search/mixture_starts.py"
HELD_BAND = re.compile(
    r"searchsorted\([^\n]*side=\"right\"\)\s*-\s*1\n(?:.*\n){0,2}\s*held\["
)

#: Single-site descent, defined in `search.icm` alone: it sat in
#: `search.alpha_expansion` as the expansion's baseline until issue #1055
#: moved it, with no alias left behind.
ICM_OWNER = "search/icm.py"
ICM_DEFINITION = re.compile(r"^def iterated_conditional_modes\(", re.MULTILINE)

CATALOGUE_FILE = re.compile(r"PROBLEMS\.md")
PIPE_SPLIT = re.compile(r"\.split\(\"\|\"\)")

#: The reader `test_problem_markers.py` keeps, and says in its own docstring
#: why: a guard that imports the thing it checks agrees with it by
#: construction. One deliberate second reading; `infra/catalogue.py` is the
#: first.
CATALOGUE_READERS = {
    "infra/catalogue.py",
    "tests/regression/test_problem_markers.py",
}

#: Where the repository root is derived, and where the catalogue is parsed.
PATH_OWNERS = {"infra/_paths.py", "tests/_paths.py"}

#: The two sides of the tree a guard over the machinery reads.
MACHINERY = (REPO_ROOT / "infra", REPO_ROOT / "tests")


def _machinery_sources() -> dict[str, str]:
    """Every module under ``infra/`` and ``tests/``, keyed by repository path."""
    return {
        str(path.relative_to(REPO_ROOT)): path.read_text()
        for root in MACHINERY
        for path in sorted(root.rglob("*.py"))
    }


@pytest.mark.critical
@pytest.mark.infra
def test_the_repository_root_is_derived_in_two_places() -> None:
    # Fifty-three modules wrote `parents[n]` with `n` counted from where the
    # file sat (issue #863). The count is the part that cannot be checked by
    # reading: it is right until the file moves, and wrong silently after.
    derived = {
        path
        for path, source in _machinery_sources().items()
        if DERIVED_REPO_ROOT.search(source)
    }

    assert derived == PATH_OWNERS


@pytest.mark.critical
@pytest.mark.infra
def test_the_two_repository_roots_are_one_path() -> None:
    # Two homes, because a test reads the tree it is part of and `infra/`
    # reads the tree it builds; one value, because they are the same tree.
    assert _paths.REPO_ROOT == REPO_ROOT


@pytest.mark.critical
@pytest.mark.infra
def test_the_problem_catalogue_has_one_reader() -> None:
    # Nine readers with nine filters, one of them with no short-row guard, so
    # a row that lost a column was read as a row with a missing key rather
    # than refused (issue #863).
    readers = {
        path
        for path, source in _machinery_sources().items()
        if CATALOGUE_FILE.search(source) and PIPE_SPLIT.search(source)
    }

    assert readers == CATALOGUE_READERS


#: Private cross-module imports, each with its reason (#1010, CLEAN's E):
#: `_submodules` is the lazy-import plumbing; `_HmmObjective` is what
#: `opt.jax.hmm` narrows on until C5 (#1004), re-exported from `opt.hmm`.
PRIVATE_IMPORTS_ADMITTED = {
    ("sal", "_submodules"),
    ("sal.opt.hmm", "_HmmObjective"),
    ("sal.opt.hmm.objectives", "_HmmObjective"),
}


def _private_imports(source: str) -> set[tuple[str, str]]:
    """``(module, name)`` for every ``from sal... import _name`` in ``source``."""
    found: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("sal")
        ):
            found.update(
                (node.module, alias.name)
                for alias in node.names
                if alias.name.startswith("_") and not alias.name.startswith("__")
            )
    return found


@pytest.mark.critical
@pytest.mark.infra
def test_no_module_imports_another_modules_private_name() -> None:
    # Issue #1010 (C4): a name with a leading underscore belongs to its
    # module; another module that needs it gets a public name or a seam.
    crossing = {
        f"{path.relative_to(PACKAGE)}: {module}.{name}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for module, name in _private_imports(path.read_text())
        if (module, name) not in PRIVATE_IMPORTS_ADMITTED
    }
    assert crossing == set()
    assert _private_imports("from sal.sample.hmc import _warm_up\n")
    assert not _private_imports("from sal.sample.chain import run_chain\n")


#: The package's import name before issue #1048, which nothing imports now.
RETIRED = "snakes_and_ladders"  # rename-package: keep


@pytest.mark.critical
@pytest.mark.infra
def test_no_line_names_the_retired_package() -> None:
    # Issue #1048: `infra/rename_package.py` renamed the package, and a line
    # it would still rewrite is an import or a path to a package that is gone.
    # Its keep-list is the exception: records of a path as it was, and the
    # repository's and the distribution's own name.
    assert rename_package.pending(RETIRED, "sal") == []


@pytest.mark.infra
def test_the_rename_guard_reports_a_retired_import() -> None:
    # The failing input: an import of the old name is a line to rewrite, and a
    # name that only contains it, the console script's, is not.
    line, counts = rename_package.rewrite_line(
        "x.py", f"from {RETIRED}.sim import potts  # run_{RETIRED}\n", RETIRED, "sal"
    )

    assert line == f"from sal.sim import potts  # run_{RETIRED}\n"
    assert counts == {"dotted": 1}


@pytest.mark.smoke
def test_the_retired_name_does_not_import() -> None:
    # No alias is kept (issue #1048): the old name fails in a fresh process,
    # where no earlier import can have left it in `sys.modules`.
    code = (
        "import importlib, sal\n"
        "try:\n"
        f"    importlib.import_module({RETIRED!r})\n"
        "except ModuleNotFoundError:\n"
        "    print('refused')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "refused"


#: The backends a twin's subpackage is named for: `<sub>.<backend>.<algorithm>`
#: (issue #1059). `python` is the oracle's own member and names no subpackage.
TWIN_BACKENDS = frozenset({"numba", "rust", "torch", "jax"})

#: Twin imports outside the twin's subpackage, each with its reason. The torch
#: pruning module is the differentiable evaluator a tree fit, a search's
#: cached partials and a learner's ranking tape through; it is no door of
#: `likelihood.pruning`'s float signature, so it has no gateway yet, and which
#: one it gets is an open question on #1059. `sandbox` referees the route it
#: conserved against it, as `tests/` does.
TWIN_IMPORTS_ADMITTED = {
    ("learn/ranking.py", "sal.likelihood.torch.pruning"),
    ("learn/tree.py", "sal.likelihood.torch.pruning"),
    ("qa/backend_agreement.py", "sal.likelihood.rust.pruning"),
    ("qa/backend_agreement.py", "sal.likelihood.torch.pruning"),
    ("qa/opt_branch_recovery.py", "sal.likelihood.torch.pruning"),
    ("sandbox/pruning_burn.py", "sal.likelihood.torch.pruning"),
    ("search/infer.py", "sal.likelihood.torch.pruning"),
}


def _twin_imports(source: str) -> set[str]:
    """Every ``sal.<sub>.<backend>.<algorithm>`` ``source`` imports, at any depth."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names = (
                [f"{node.module}.{alias.name}" for alias in node.names]
                if len(node.module.split(".")) == 3
                else [node.module]
            )
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        for name in names:
            parts = name.split(".")
            if parts[0] == "sal" and len(parts) >= 4 and parts[2] in TWIN_BACKENDS:
                found.add(".".join(parts[:4]))
    return found


def _notebook_code(path: Path) -> str:
    """A notebook's code cells as one module, its magics dropped."""
    cells = json.loads(path.read_text())["cells"]
    return "\n".join(
        line
        for cell in cells
        if cell["cell_type"] == "code"
        for line in "".join(cell["source"]).splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )


@pytest.mark.critical
@pytest.mark.infra
def test_backend_twins_are_reached_by_their_gateway() -> None:
    # Issue #1059: a caller reaches `<sub>.<backend>.<algorithm>` through the
    # gateway `<sub>.<algorithm>` and its `backend=`, so only the twin's own
    # subpackage and `tests/` import it. The package, `infra/` and the
    # notebooks are read.
    crossing = {
        (path.relative_to(PACKAGE).as_posix(), module)
        for path in sorted(PACKAGE.rglob("*.py"))
        for module in _twin_imports(path.read_text())
        if module.split(".")[1] != path.relative_to(PACKAGE).parts[0]
    }
    crossing |= {
        (path.relative_to(REPO_ROOT).as_posix(), module)
        for path in sorted((REPO_ROOT / "infra").rglob("*.py"))
        for module in _twin_imports(path.read_text())
    }
    crossing |= {
        (path.relative_to(REPO_ROOT).as_posix(), module)
        for path in sorted((REPO_ROOT / "docs" / "nb").glob("*.ipynb"))
        for module in _twin_imports(_notebook_code(path))
    }
    assert crossing - TWIN_IMPORTS_ADMITTED == set()
    assert TWIN_IMPORTS_ADMITTED - crossing == set(), "an admitted crossing is gone"
    # Paired inputs: each spelling of a twin import, and two that are not.
    assert _twin_imports("from sal.likelihood.rust import pruning\n") == {
        "sal.likelihood.rust.pruning"
    }
    assert _twin_imports("def f():\n    import sal.search.rust.maxflow\n") == {
        "sal.search.rust.maxflow"
    }
    assert _twin_imports("from sal.opt.jax.hmm import value_and_grad\n") == {
        "sal.opt.jax.hmm"
    }
    assert _twin_imports("from sal.likelihood import pruning\n") == set()
    assert _twin_imports("from sal.backend import Backend\n") == set()
