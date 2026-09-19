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

The fifth is a *seam* rather than a duplication, and what it guards is a
shape rather than a text (issue #755). `likelihood.schedule.MessageSchedule`
is the base the five schedules inherit and five modules call through, so a
sixth written beside it rather than under it would type-check, run, and carry
none of the guarantee its consumers read. That claim is structural, so it is
read from the class tree rather than from a regex; the one half a regex
answers is that no consumer branches on a schedule's name, which is the `if`
issue #592 deleted.

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
from snakes_and_ladders.likelihood.schedule import (
    SCHEDULES,
    MessageSchedule,
    MessageScheduleName,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "python" / "snakes_and_ladders"
sys.path.insert(0, str(REPO_ROOT / "infra"))

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
    "flat modules": 148,
    "API-map entries": 1576,
}

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

#: The module owning the schedule seam, and the base every schedule inherits.
SCHEDULE_OWNER = "likelihood/schedule.py"
SCHEDULE_BASE = "MessageSchedule"

#: The registered schedules, pinned at what the 2026-09-19 audit read from the
#: tree: `tree`, `upward`, `downward`, `flooding` and `sequential`. The audit's
#: `fields:name` cluster has seven members; the seventh,
#: `search.ground_state.Entry`, shares the field `name` and nothing else and is
#: not a schedule, so this pin is five and not seven. A sixth schedule raises
#: it in the pull request that registers it.
SCHEDULE_COUNT = 5

#: A consumer branching on which schedule it holds, by name. The base's
#: methods are the seam --- `message_passing._run` reads `requires_tree`,
#: `bounded`, `guarantee` and `steps`, and `name` only to build a message ---
#: so a branch on the name is the `if` issue #592 deleted, returning. The
#: names come from `MessageScheduleName` rather than from a list here, so a
#: sixth schedule is guarded as soon as it is registered.
#: `likelihood.message_passing_reference` is not a consumer of the base: it
#: takes the enum, implements the two orders that predate the seam and is the
#: oracle the seam is pinned against, which is why the pattern reads `.name`
#: and not the enum members it compares.
SCHEDULE_NAMES = "|".join(str(member) for member in MessageScheduleName)
SCHEDULE_NAME_BRANCH = re.compile(
    r"(?:if|elif|while)\b[^\n]*\.name\s*(?:==|!=|in)\s*[^\n]*"
    rf"""(?:MessageScheduleName\.|["'](?:{SCHEDULE_NAMES})["'])"""
)
SCHEDULE_NAME_MATCH = re.compile(r"match\s+[^\n]*\b(?:schedule|plan)\w*\.name\s*:")

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

    Shaped is read two ways, because a sixth schedule may arrive under either:
    a name ending in ``MessageSchedule``, or the pair `resolve` needs --- a
    ``guarantee`` and a ``name``. `search.ground_state.Entry` carries ``name``
    alone, so it is not shaped and this guard says nothing about it; that is
    the audited `fields:name` cluster's seventh member and the reason the pin
    below is five.
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
def test_no_schedule_is_written_outside_the_base() -> None:
    # Five schedules, one base, five modules calling through it (issue #755).
    # A class written beside the base carries no `guarantee`, no `bounded` and
    # no `requires_tree` the consumers read, and `sum_product` would take the
    # default for each -- a bounded, approximate schedule run on a loopy graph
    # -- rather than refuse it. Read from the class tree and from the registry
    # rather than from a regex, because the claim is what a class *is*.
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
def test_no_consumer_branches_on_a_schedules_name() -> None:
    # The seam is the base's methods. `sum_product` chose between two orders
    # with an `if` until #592, and the defect that surfaced was in that branch:
    # a plain string compares equal to a `StrEnum` member without being it, so
    # `schedule="tree"` ran flooding. A branch on the name brings the shape
    # back one schedule at a time.
    assert _offenders(SCHEDULE_NAME_BRANCH, SCHEDULE_OWNER) == []
    assert _offenders(SCHEDULE_NAME_MATCH, SCHEDULE_OWNER) == []


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
        SCHEDULE_NAME_BRANCH: (
            "    if plan.requires_tree and not graph.is_tree():\n"
            '        msg = f"the {plan.name} schedule is exact only on a tree"\n'
        ),
        SCHEDULE_NAME_MATCH: "    for step in plan.steps(layout):\n",
    }

    assert [p for p, text in violating.items() if not p.search(text)] == []
    assert [p for p, text in clean.items() if p.search(text)] == []

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
