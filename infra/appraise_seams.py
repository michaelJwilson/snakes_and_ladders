#!/usr/bin/env python
"""Every seam the package declares, who implements it, who calls it, what it reaches.

Issue #813, step 1. `infra/gate_new_seams.py` asks one question of a branch:
does a `Protocol` it *adds* have the three consuming modules root `CLAUDE.md`
admits an abstraction for. Nothing asks it of the seams already there, so a
seam that lost its consumers looks the same as one that never had them, and a
problem reached by no seam at all is invisible --- which is the shape #813 is
about: the same intent called differently per problem because no seam spans
them.

This derives the inventory. For every `Protocol` and `ABC` in the package:
the module declaring it, the classes implementing it, the modules naming it,
and the **problems it reaches**, read from `PROBLEMS.md`'s Defines column ---
a seam reaches a problem when a module that *is* that problem names it. A
seam reaching one problem is a seam for one problem, whatever its consumer
count, and that is the reading the consumer rule alone cannot give.

The consumer count and the under-rule reason come from
`infra/gate_new_seams.py` rather than a second implementation: one definition
of what a consumer is, so the gate and this survey cannot disagree.

Run with ``uv run python infra/appraise_seams.py``; ``--json`` writes the
inventory for a test to read rather than re-deriving it. This reports; what
gates is `infra/gate_new_seams.py` on a branch.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import gate_new_seams  # noqa: E402
import problems_tables  # noqa: E402

#: What root `CLAUDE.md` admits an abstraction for, restated from the gate so
#: the two cannot drift.
CONSUMER_RULE = gate_new_seams.CONSUMER_RULE


@dataclass(frozen=True)
class Seam:
    """One declared abstraction, and what the tree does with it."""

    #: Package-relative dotted module declaring it.
    module: str
    #: The class name.
    name: str
    #: ``Protocol`` or ``ABC``: what makes it a seam a reader must satisfy.
    kind: str
    #: Classes implementing it, as ``module.Class``.
    implementers: tuple[str, ...]
    #: Modules naming it, the declaring module excluded. This is the gate's
    #: definition and the one the rule is stated in.
    consumers: tuple[str, ...]
    #: Modules importing anything from the module that declares it. A seam
    #: reached through a registry or an enum of names --- as
    #: `likelihood.schedule.MessageSchedule` is, through
    #: ``MessageScheduleName`` --- is named by one module and called through
    #: by several, and the count above sees only the first.
    importers: tuple[str, ...]
    #: Problems it reaches, by the catalogue's name.
    problems: tuple[str, ...]
    #: The reason its docstring gives for standing under the rule, if any.
    reason: str

    @property
    def earns_it(self) -> bool:
        """Whether the consumer count alone admits it."""
        return len(self.consumers) >= CONSUMER_RULE


def _declared(source: str) -> list[tuple[str, str]]:
    """``(class name, kind)`` for every ``Protocol`` and ``ABC`` in one module."""
    if not source.strip():
        return []
    found: list[tuple[str, str]] = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
            for base in node.bases
        }
        # `Protocol[T]` and `ABC` both arrive as a subscript or a name; the
        # generic form is what `gate_new_seams._protocols` misses, and a
        # generic seam is still a seam (#803 found one it could not see).
        for base in node.bases:
            if isinstance(base, ast.Subscript):
                inner = base.value
                bases.add(
                    inner.id
                    if isinstance(inner, ast.Name)
                    else getattr(inner, "attr", "")
                )
        if "Protocol" in bases:
            found.append((node.name, "Protocol"))
        elif "ABC" in bases:
            found.append((node.name, "ABC"))
    return found


def _importers(module: str, sources: dict[str, str]) -> tuple[str, ...]:
    """Modules importing anything from ``module``, the module itself excluded."""
    tail = module.rsplit(".", 1)[-1]
    found: list[str] = []
    for other, source in sorted(sources.items()):
        if other == module or not source.strip():
            continue
        if f"snakes_and_ladders.{module} import" in source or (
            f"from .{tail} import" in source
        ):
            found.append(other)
    return tuple(found)


def _implementers(name: str, sources: dict[str, str]) -> tuple[str, ...]:
    """Every class inheriting ``name``, as ``module.Class``."""
    found: list[str] = []
    for module, source in sorted(sources.items()):
        if not source.strip():
            continue
        for node in ast.parse(source).body:
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                label = (
                    base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                )
                if label == name:
                    found.append(f"{module}.{node.name}")
    return tuple(found)


def _problems_by_module() -> dict[str, tuple[str, ...]]:
    """Defining module to the problems it defines, from the catalogue."""
    out: dict[str, list[str]] = {}
    for problem, _keys, defines in problems_tables.catalogue_rows():
        for symbol in defines:
            module = symbol.rsplit(".", 1)[0] if symbol[-1:].isupper() else symbol
            out.setdefault(module, []).append(problem)
    return {module: tuple(sorted(set(names))) for module, names in out.items()}


def seams() -> list[Seam]:
    """Every declared seam, with its implementers, consumers and problems."""
    sources = gate_new_seams.source_modules()
    by_module = _problems_by_module()
    found: list[Seam] = []
    for module, source in sorted(sources.items()):
        for name, kind in _declared(source):
            consuming = gate_new_seams.consumers(module, name, sources)
            reached: set[str] = set()
            for consumer in (*consuming, module):
                reached.update(by_module.get(consumer, ()))
            found.append(
                Seam(
                    module=module,
                    name=name,
                    kind=kind,
                    implementers=_implementers(name, sources),
                    consumers=consuming,
                    importers=_importers(module, sources),
                    problems=tuple(sorted(reached)),
                    reason=gate_new_seams.reason_for(module, name, sources),
                )
            )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the inventory here")
    args = parser.parse_args(argv)

    found = seams()
    if args.json:
        args.json.write_text(json.dumps([asdict(s) for s in found], indent=1) + "\n")

    width = max(len(f"{s.module}.{s.name}") for s in found)
    print(
        f"{len(found)} declared seams, the rule at {CONSUMER_RULE} consuming modules\n"
    )
    print(f"{'seam':{width}}  {'kind':8} {'impl':>4} {'cons':>4} {'imp':>4}  problems")
    for seam in sorted(found, key=lambda s: (-len(s.consumers), s.module)):
        label = f"{seam.module}.{seam.name}"
        problems = ", ".join(seam.problems) if seam.problems else "-"
        print(
            f"{label:{width}}  {seam.kind:8} {len(seam.implementers):>4} "
            f"{len(seam.consumers):>4} {len(seam.importers):>4}  {problems[:52]}"
        )

    under = [s for s in found if not s.earns_it]
    print(f"\n{len(under)} under the rule:")
    for seam in under:
        state = "reason given" if seam.reason else "NO REASON"
        print(
            f"  {seam.module}.{seam.name}: {len(seam.consumers)} consumers, "
            f"{len(seam.importers)} importers of its module, {state}"
        )

    spanning = [s for s in found if len(s.problems) >= 2]
    print(
        f"\n{len(spanning)} reach two or more problems; {len(found) - len(spanning)} reach one or none."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
