#!/usr/bin/env python
"""The data structures the package repeats, derived rather than recalled.

Issue #230 surveyed *functions* and #400 *abstractions*; neither surveyed the
types that carry state, and both left a hand-written list --- ``FINDINGS`` in
`infra/duplication_survey.py`, ``PROTOCOLS``/``CONTRACTS`` in the since-deleted
`infra/seams_survey.py`. A hand list re-counts what is known and discovers
nothing, and a gate then exists to enforce that the list is updated, which is
the work the list was meant to save. Issue #586 deleted that one; this is the
argument it was deleted on.

This derives the inventory instead: every class holding state, its fields,
its public methods, the modules importing it, and the *clusters* of classes
that share a shape. A near-duplicate added tomorrow appears here without
anyone remembering to add it.

**It also ranks what each cluster costs.** Root ``CLAUDE.md``'s Runtime
Optimization Opportunities are a checklist, and a survey that reports shape
without cost licenses a merge that tidies three representations into the
slowest of them. Each cluster is reported against the checklist items this
tool can decide from the source --- the layout a structure stores, whether a
derived layout is rebuilt per call, and whether a neighbour walk is a list of
lists --- with the rest left to the profile, which is the only thing that
ranks them.

Run with ``uv run python infra/appraise_structures.py``; ``--json`` writes the
inventory for a test to read rather than re-deriving it. This reports; what
gates is ``tests/regression/test_structure_survey.py``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "python" / "snakes_and_ladders"

#: Below this many members a cluster is a coincidence of naming rather than a
#: shared shape. Two classes sharing one field name share nothing.
CLUSTER_RULE = 3

#: Base classes whose subclasses are somebody else's shape, not this
#: package's, so a cluster over them would report a framework's design.
FOREIGN_BASES = ("Protocol", "Enum", "StrEnum", "IntEnum", "Exception", "Module")


@dataclass(frozen=True)
class Structure:
    """One class that carries state, as the source declares it."""

    module: str
    name: str
    fields: tuple[str, ...]
    methods: tuple[str, ...]
    bases: tuple[str, ...]
    frozen: bool
    consumers: tuple[str, ...] = ()
    layout: str = "unknown"
    notes: tuple[str, ...] = ()

    @property
    def qualified(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass
class Cluster:
    """Structures sharing a shape, and what the shape costs."""

    key: str
    members: list[str] = field(default_factory=list)
    layouts: dict[str, str] = field(default_factory=dict)
    consumers: int = 0
    findings: list[str] = field(default_factory=list)


def _modules(root: Path = PACKAGE_ROOT) -> dict[str, ast.Module]:
    """Every package module under ``root``, dotted, parsed."""
    out: dict[str, ast.Module] = {}
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts) or "__init__"] = ast.parse(path.read_text())
    return out


def _base_names(node: ast.ClassDef) -> tuple[str, ...]:
    names = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
        elif isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
            names.append(base.value.id)
    return tuple(names)


def _is_frozen(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        if call is None:
            continue
        for keyword in call.keywords:
            if keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant):
                return bool(keyword.value.value)
    return False


def _carries_state(node: ast.ClassDef, fields: tuple[str, ...]) -> bool:
    """Whether the class holds data, rather than being a name for behaviour."""
    if any(base in FOREIGN_BASES for base in _base_names(node)):
        return False
    if fields:
        return True
    # A class assigning to `self` in `__init__` carries state even without
    # annotations at class level.
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            for sub in ast.walk(item):
                if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store):
                    return True
    return False


def structures(root: Path = PACKAGE_ROOT) -> list[Structure]:
    """Every state-carrying class declared under ``root``.

    ``root`` is an argument so a test can point the walk at a planted
    near-duplicate and assert it is reported, which is what makes this a
    survey rather than a claim about one.
    """
    found: list[Structure] = []
    trees = _modules(root)
    sources = {
        name: (root / Path(*name.split("."))).with_suffix(".py") for name in trees
    }
    texts = {
        name: (path.read_text() if path.exists() else "")
        for name, path in sources.items()
    }
    for module, tree in trees.items():
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            annotated = tuple(
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            )
            if not _carries_state(node, annotated):
                continue
            methods = tuple(
                sorted(
                    item.name
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                    and not item.name.startswith("_")
                )
            )
            consumers = tuple(
                sorted(
                    other
                    for other, text in texts.items()
                    if other != module and re.search(rf"\b{node.name}\b", text)
                )
            )
            found.append(
                Structure(
                    module=module,
                    name=node.name,
                    fields=annotated,
                    methods=methods,
                    bases=_base_names(node),
                    frozen=_is_frozen(node),
                    consumers=consumers,
                    layout=_layout(annotated),
                    notes=_notes(f"{module}.{node.name}", node, annotated),
                )
            )
    return found


#: How a structure stores an incidence relation, decided from **exact** field
#: names with corroboration rather than from substrings. The first attempt
#: matched substrings and misfiled three structures --- `restarts` contains
#: "starts", `evaluated_columns` and `columns` contain "col" --- which is the
#: failure this survey exists to avoid: measuring names rather than behaviour.
#: An incidence layout therefore needs two fields, not one, because a single
#: array named `offsets` is a coincidence and `offsets` beside `indices` is a
#: design.
_CSR_EXACT = frozenset({"offsets", "indptr", "check_offsets", "variable_offsets"})
_CSR_SUFFIX = "_offsets"
_CSR_PARTNER = frozenset(
    {
        "indices",
        "neighbours",
        "neighbors",
        "columns",
        "edge_variable",
        "edge_check",
        "target",
        "term_column",
        "entry_start",
        "factor_start",
    }
)
#: The **implicit-offsets** spelling: segment lengths beside one flat payload,
#: with the offsets derived rather than stored. `ragged.Ragged` is the case
#: that named it --- the survey ran for months reporting 218 classes and no
#: `Ragged`, because the classifier knew `offsets` and `indptr` and not the
#: third way of writing the same relation (issue #677).
#:
#: The corroboration rule of `_CSR_EXACT` applies here too, and it is what
#: keeps this from over-matching. A payload partner is **required**:
#: `sim.hmm.HmmParams` carries `lengths` and no flat array, because there it
#: declares the shape of a batch rather than addressing one, and a params is
#: not a layout. And `sizes` is deliberately **not** a length field:
#: `sample.potts_mcmc.ClusterCounter.sizes` is a histogram of cluster sizes,
#: `search.ground_state.Rung.sizes` and `sim.potts.SpatioOnlyParams.sizes` are
#: problem sizes, and admitting the name would misfile all three --- the same
#: failure as matching `restarts` for `starts`.
_SEGMENTED_EXACT = frozenset({"lengths"})
_SEGMENTED_SUFFIX = "_lengths"
_SEGMENTED_PARTNER = frozenset({"values", "observations", "data", "flat"})
_COO_EXACT = frozenset({"edges", "edge_variable", "edge_check"})
_COO_PARTNER = frozenset({"coupling", "couplings", "weight", "weights", "capacity"})
_GRAPH_EXACT = frozenset({"children", "nodes", "variables", "factors", "outgoing"})


def _layout(fields: tuple[str, ...]) -> str:
    """The incidence layout a structure stores, or that it stores none."""
    names = set(fields)
    csr = {f for f in names if f in _CSR_EXACT or f.endswith(_CSR_SUFFIX)}
    if csr and (names & _CSR_PARTNER or len(csr) >= 2):
        return "csr"
    segmented = {
        f for f in names if f in _SEGMENTED_EXACT or f.endswith(_SEGMENTED_SUFFIX)
    }
    if segmented and names & _SEGMENTED_PARTNER:
        return "csr"
    coo = names & _COO_EXACT
    if coo and (names & _COO_PARTNER or len(coo) >= 2):
        return "coo"
    if names & _GRAPH_EXACT:
        return "object-graph"
    return "none"


#: Layouts the rule would flag that a measurement has settled, against the
#: reason. A finding is a question, and a question that has been answered
#: with numbers should not be asked again every run; an entry here is not an
#: opinion but a benchmark, named so the next reader can re-run it rather
#: than re-litigate it. The reason replaces the finding, so the survey still
#: reports the layout --- it stops calling it a cost.
#: Keyed ``<qualified>:<kind>`` rather than by the class, because one class
#: can raise two findings and a measurement answers one of them:
#: `FlowNetwork` keeps its list of lists on a benchmark *and* still derives
#: its layout per call, which is open as #642 (issue #677).
MEASURED: dict[str, str] = {
    "ragged.Ragged:offsets": (
        "offsets rebuilt per call kept: the Python scan is 39.3 us on the "
        "600-segment `hmm/ci` batch against 335.11 ms for one Baum-Welch "
        "iteration over it, 0.012%, and the NumPy cumsum that would replace "
        "it saves 6 us of that (2026-09-16, #677). A ratio with no effect "
        "size, which root CLAUDE.md leaves alone"
    ),
    "search.maxflow.FlowNetwork:list-of-lists": (
        "list of lists kept: Dinic is a pure-Python inner loop, where a row "
        "is 1.95 ms as lists, 3.93 ms flat with offsets and 25.63 ms as a "
        "NumPy slice over 16,384 rows of degree six; the compiled consumer "
        "takes the contiguous form from as_arrays (#586)"
    ),
}


def _notes(
    qualified: str, node: ast.ClassDef, fields: tuple[str, ...]
) -> tuple[str, ...]:
    """Cost findings the source decides, per root `CLAUDE.md`'s checklist."""
    out: list[str] = []
    source = ast.unparse(node)
    # A class that derives the layout once under a cache derives it once for
    # every method that reads it, so the per-call finding is about the class
    # and not about each method: a delegating accessor is not a second
    # derivation, and reporting it as one would outlive the fix.
    derived_once = any(
        isinstance(item, ast.FunctionDef)
        and any(
            "cached_property" in ast.unparse(d) or "lru_cache" in ast.unparse(d)
            for d in item.decorator_list
        )
        and any(mark in ast.unparse(item) for mark in ("cumsum", "offsets", "indptr"))
        for item in node.body
    )
    # What decides the finding is whether the *offsets* are stored, not whether
    # the store is compressed at all. An implicit-offsets structure is
    # compressed --- `_layout` now says `csr` for it --- and still rebuilds the
    # offsets on every call, which is the "recompute or store" question
    # unanswered rather than answered. Keying on the layout alone hid that for
    # every `Ragged`-shaped class (issue #677).
    stores_offsets = any(
        name in _CSR_EXACT or name.endswith(_CSR_SUFFIX) for name in fields
    )
    for item in node.body:
        if not isinstance(item, ast.FunctionDef) or derived_once or stores_offsets:
            continue
        # A constructor is not a per-call derivation (issue #642). "Derives X
        # per call" names a cost that repeats on one object: an accessor asked
        # twice pays twice, and storing the result removes the second payment.
        # A `classmethod` that builds an instance runs once per instance it
        # builds, so there is no second payment to remove and no rewrite that
        # would clear the finding short of deleting the constructor. Reported
        # of one, the finding is unanswerable, and `FlowNetwork.from_arcs`
        # carried it on `main` while the survey's *own* next line recorded the
        # measurement that keeps the store it was objecting to --- the two read
        # together as the survey contradicting itself about one class.
        #
        # This narrows what is reported, so `tests/regression/test_structure_survey.py`
        # holds it to a positive control: a class with a real per-call accessor
        # over a non-compressed store is still reported, and the day this stops
        # firing for one, that test fails rather than this going quiet.
        if any(
            isinstance(d, ast.Name) and d.id == "classmethod"
            for d in item.decorator_list
        ):
            continue
        body = ast.unparse(item)
        # A method *reading* `self.offsets` consumes a derivation; the method
        # that computes the scan is the one to report. `Ragged.segments` reads
        # `self.offsets` once and was reported as a second derivation of it,
        # which is the survey measuring a name again (issue #677).
        derives_csr = any(mark in body for mark in ("cumsum", "indptr")) or (
            "offsets" in body and "self.offsets" not in body
        )
        if not derives_csr:
            continue
        if _layout(fields) == "csr":
            out.append(
                MEASURED.get(
                    f"{qualified}:offsets",
                    f"{item.name} derives the offsets per call from a store "
                    "that keeps them implicitly (CLAUDE.md, recompute or store)",
                )
            )
        else:
            out.append(
                f"{item.name} derives a compressed layout per call from a store "
                "that is not compressed (CLAUDE.md, Memory layout; recompute or store)"
            )
    if "list[list[" in source:
        out.append(
            MEASURED.get(
                f"{qualified}:list-of-lists",
                "carries a list of lists where the rule asks for offsets into one "
                "array (CLAUDE.md, Memory layout)",
            )
        )
    return tuple(out)


def clusters(found: list[Structure]) -> list[Cluster]:
    """Structures grouped by shared shape, largest first."""
    by_key: dict[str, Cluster] = defaultdict(lambda: Cluster(key=""))
    for structure in found:
        for key in _keys(structure):
            cluster = by_key[key]
            cluster.key = key
            cluster.members.append(structure.qualified)
            cluster.layouts[structure.qualified] = structure.layout
            cluster.consumers += len(structure.consumers)
            cluster.findings.extend(
                f"{structure.qualified}: {note}" for note in structure.notes
            )
    keep = [c for c in by_key.values() if len(c.members) >= CLUSTER_RULE]
    return sorted(keep, key=lambda c: (-len(c.members), c.key))


def _keys(structure: Structure) -> list[str]:
    """The shapes a structure belongs to: its suffix, and its field signature.

    Two kinds of key, and the difference decides what a reader does with a
    cluster. A ``role:`` key says the members carry the same *relation* and
    differ in how they store it, which is a merge to consider ---
    ``role:incidence`` is the one issue #586 acted on. A ``suffix:`` or
    ``prefix:`` key says only that the names agree, and the names of this
    package agree by convention: ``suffix:Params`` holds fifteen
    per-problem parameter bundles whose whole shared surface is ``seed`` and
    ``tolerance``, each read by one simulator, and nothing calls any of them
    polymorphically. Merging on a name key would be writing a seam the rule
    in root ``CLAUDE.md`` refuses --- a contract belongs where three or more
    modules call *through* it, and ``prefix:Exact`` is reported at five
    members and **zero** consuming references.
    """
    keys: list[str] = []
    for suffix in ("Params", "Result", "Fit", "Decoding", "Dataset", "Enumeration"):
        if structure.name.endswith(suffix):
            keys.append(f"suffix:{suffix}")
    if structure.name.startswith("Exact"):
        keys.append("prefix:Exact")
    if structure.layout in ("csr", "coo", "object-graph"):
        keys.append("role:incidence")
    if structure.fields:
        keys.append("fields:" + ",".join(sorted(structure.fields)[:3]))
    return keys


def unclustered_findings(found: list[Structure], grouped: list[Cluster]) -> list[str]:
    """Cost findings on classes that belong to no cluster.

    A finding was printed only inside a cluster, so one on a class sharing its
    shape with nobody was derived and then dropped --- and that is a second
    blindness, not a display detail: `ragged.Ragged` raised the per-call
    derivation finding for months while its only cluster key was a field
    signature of one member, so nothing printed it (issue #677). The cost of a
    structure does not depend on how many others look like it.
    """
    clustered = {member for cluster in grouped for member in cluster.members}
    return [
        f"{structure.qualified}: {note}"
        for structure in found
        if structure.qualified not in clustered
        for note in structure.notes
    ]


def render(found: list[Structure], grouped: list[Cluster]) -> str:
    lines = [
        f"{len(found)} state-carrying classes; "
        f"{len(grouped)} clusters at {CLUSTER_RULE} or more members",
        "",
    ]
    for cluster in grouped:
        lines.append(
            f"## {cluster.key}  ({len(cluster.members)} members, "
            f"{cluster.consumers} consuming references)"
        )
        for member in sorted(cluster.members):
            lines.append(f"  {cluster.layouts[member]:16s} {member}")
        for finding in sorted(set(cluster.findings)):
            lines.append(f"  ! {finding}")
        lines.append("")
    loose = unclustered_findings(found, grouped)
    if loose:
        lines.append(f"## findings outside every cluster  ({len(loose)})")
        lines.extend(f"  ! {finding}" for finding in sorted(set(loose)))
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the inventory here")
    args = parser.parse_args(argv)

    found = structures()
    grouped = clusters(found)
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "structures": [asdict(s) for s in found],
                    "clusters": [asdict(c) for c in grouped],
                },
                indent=1,
            )
        )
        print(f"wrote {args.json}")
        return 0
    print(render(found, grouped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
