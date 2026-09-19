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

A *seam* rather than a duplication: no module builds a compressed-sparse
store by hand (issue #755). `incidence.SparseIncidence` is the one compressed
layout (#586) and `as_arrays()` the one way across the FFI boundary, so a
second `csr` or `coo` written beside them carries no transpose, no stable
order within a row and no oracle. The claim is structural --- an offsets array
is a `cumsum` over degrees whatever the line is spelled like --- so it is read
from the syntax tree, and only the imported `csr_matrix` spelling is read as a
text.

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

#: The module owning the compressed layout, and the two calls that build a
#: store through it: `SparseIncidence.from_pairs` and, for a Potts graph,
#: `PottsGraph.compressed_adjacency`, which holds one.
INCIDENCE_OWNER = "incidence.py"

#: Package modules building a store through the seam, pinned at what this
#: guard reads on `main` at 048a342: `sim/graph.py`,
#: `sim/ldpc.py`, `sim/factor_graph.py`, `sim/potts.py`, `search/maxflow.py`,
#: `search/potts_mcmc.py`, `search/potts_keyed.py`, `search/ground_state.py`,
#: `search/alpha_expansion.py` and `search/spatio_sequential.py`. The pull
#: request that adds a consumer raises the pin; one that removes the last
#: caller of the seam lowers it to a number a reader can question.
SEAM_CONSUMERS = 10

#: Files this guard does not read, each against the reason, rather than an
#: allow-list nobody can audit. The first two are measurements the
#: 2026-09-19 review records and `infra/appraise_structures.py` prints in
#: `MEASURED`; the third is what this guard found on `main`.
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

#: The object graphs are not excluded because nothing reads them: the rule
#: says nothing about them. Eleven of the `role:incidence` cluster's fifteen
#: members --- `sim.tree.Node`, `sim.factor_graph.Factor`,
#: `sandbox.region_graph.Region` and eight more --- are graphs where the
#: relation *is* the model, which the compressed layout serves rather than
#: replaces (the 2026-09-19 review, `role:incidence`). They build no store, so
#: they trip neither rule below and need no entry above.
#:
#: The one spelling a syntax tree cannot decide: `scipy.sparse` builds the
#: store, and the name it is built under is whatever the import aliased. The
#: package carries `scipy` for `linear_sum_assignment` and nothing else, and a
#: `csr_matrix` here would be a second compressed layout with a second set of
#: conventions. Searched over the suite and the notebooks too, because that is
#: where a reader reaches for one.
FOREIGN_SPARSE = re.compile(r"\b(?:csr|csc|coo)_(?:matrix|array)\s*\(")

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

    Two rules, and each is what `incidence._row_major` does in one line, so the
    owner's own source trips both --- which is the positive control the guard
    asserts rather than a coincidence. A `cumsum` bound to an `offsets` or
    `indptr` name is the row starts; a `lexsort` or `argsort` over a row index
    bound to an `order` name is the COO pair sorted into row-major order.
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
def test_no_compressed_store_is_built_outside_the_incidence_seam() -> None:
    # One compressed layout, ten modules building through it (issue #755).
    # Three wrote the counting sort separately before #586 -- `ParityCheck`
    # by `lexsort` and `searchsorted`, `PottsGraph` per call, `FactorGraph`
    # not at all -- and they agreed, so nothing failed; what a fourth costs
    # is the contract `SparseIncidence` states and a copy does not: a stable
    # order within a row, a transpose sharing one per-entry array, and the
    # refusals `from_pairs` makes at the build.
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
def test_no_module_builds_a_scipy_sparse_store() -> None:
    # `scipy` is carried for `linear_sum_assignment`. A `csr_matrix` beside
    # `SparseIncidence` would be a second compressed layout whose row order,
    # duplicate handling and transpose are somebody else's, and the compiled
    # consumers take `as_arrays`, which it does not have.
    assert (
        _found(
            FOREIGN_SPARSE,
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
        # Split for the same reason: this module is inside the search.
        FOREIGN_SPARSE: "matrix = csr" + "_matrix((data, (rows, cols)))\n",
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
        FOREIGN_SPARSE: "from snakes_and_ladders.incidence import SparseIncidence\n",
    }

    assert [p for p, text in violating.items() if not p.search(text)] == []
    assert [p for p, text in clean.items() if p.search(text)] == []

    # The structural guard on the same discipline, since the claim is what a
    # line computes and not how it is spelled: the row starts by `cumsum`
    # under three names, the pairs sorted into row-major order, and the two
    # clean forms -- a build through the seam, and a `cumsum` that addresses
    # no relation, which is the `restarts`-for-`starts` failure the structure
    # survey exists to avoid.
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
