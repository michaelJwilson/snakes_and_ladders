"""The seams the package has, as a regeneration rather than a recollection.

Issue #400 asked which abstractions the codebase should have. That question
starts from which it has, and an inventory kept by hand is stale the week
after it is written. This tool derives the inventory: every ``Protocol`` the
package declares and every data contract three or more modules share, with
the classes that satisfy each protocol (structurally, by the members the
protocol names), the modules that consume it, and the problem classes it
reaches -- the last read off `PROBLEMS.md`'s catalogue, whose rows name the
symbols each problem is stated in.

The seam rule it reports against (`DEV.md`, Core Development Standards): a
seam earns its place with three or more consumers; one with a single
implementer and no second in sight is a name, not an abstraction.

Run with ``uv run python infra/seams_survey.py --write`` to regenerate
`SEAMS.md`; without ``--write`` it checks the committed file and exits 1 when
stale. ``tests/regression/docs/test_seams_survey.py`` asserts the same.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "python" / "snakes_and_ladders"
LEDGER = REPO_ROOT / "SEAMS.md"
PACKAGE = "snakes_and_ladders"
# Below this many consuming modules a seam is reported as under the rule.
CONSUMER_RULE = 3
# Modules whose import needs the compiled extension or a display; skipped
# from the implementer search, never from the consumer search (which reads
# source text).
SKIP_IMPORT = ("snakes_and_ladders.qa.", "snakes_and_ladders.sandbox")

sys.path.insert(0, str(REPO_ROOT / "infra"))
import problems_tables  # noqa: E402


@dataclass(frozen=True)
class Seam:
    """One abstraction the survey reports.

    Parameters
    ----------
    module : str
        Package-relative module that defines it, dotted.
    name : str
        The class name.
    kind : str
        ``"protocol"`` for a ``typing.Protocol``; ``"contract"`` for a
        dataclass that three or more modules pass between them.
    """

    module: str
    name: str
    kind: str


PROTOCOLS = (
    Seam("opt.objective", "Objective", "protocol"),
    Seam("emissions", "EmissionFamily", "protocol"),
    Seam("emissions", "CountEmissionFamily", "protocol"),
    Seam("opt.schedule", "Schedule", "protocol"),
    Seam("opt.initialize", "Initializer", "protocol"),
    Seam("learn.environment", "Environment", "protocol"),
    Seam("learn.policy", "Policy", "protocol"),
    Seam("learn.policy", "TrainablePolicy", "protocol"),
    Seam("learn.relaxed", "RelaxedObjective", "protocol"),
    Seam("bound", "Surrogate", "protocol"),
    Seam("sim.ldpc", "Channel", "protocol"),
)
CONTRACTS = (
    Seam("sim.factor_graph", "FactorGraph", "contract"),
    Seam("sim.hmm", "HmmParams", "contract"),
    Seam("opt.potts", "PottsParams", "contract"),
    Seam("sim.spatio_sequential", "SpatioSequentialParams", "contract"),
)
SEAMS = PROTOCOLS + CONTRACTS


@dataclass(frozen=True)
class Row:
    """What the survey found for one seam."""

    seam: Seam
    members: tuple[str, ...]
    implementers: tuple[str, ...]
    consumers: tuple[str, ...]
    problems: tuple[str, ...]


def _modules() -> list[str]:
    """Every module in the package, dotted, in a stable order."""
    names = [
        info.name
        for info in pkgutil.walk_packages([str(PACKAGE_ROOT)], prefix=f"{PACKAGE}.")
    ]
    return sorted(names)


def _source_modules() -> dict[str, str]:
    """Package-relative dotted module name to its source text."""
    out: dict[str, str] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts) or "__init__"] = path.read_text()
    return out


def members_of(protocol: type) -> tuple[str, ...]:
    """The names a class must carry to satisfy ``protocol``."""
    attrs: set[str] = set(getattr(protocol, "__protocol_attrs__", set()))
    return tuple(sorted(name for name in attrs if not name.startswith("_")))


def _satisfies(candidate: type, members: tuple[str, ...]) -> bool:
    # A dataclass field without a default is not a class attribute, so the
    # field names count as members too; a protocol member is satisfied by an
    # attribute, a property or a field alike.
    fields = set(getattr(candidate, "__dataclass_fields__", {}))
    annotated = set(getattr(candidate, "__annotations__", {}))
    return all(
        hasattr(candidate, name) or name in fields or name in annotated
        for name in members
    )


def implementers(members: tuple[str, ...]) -> tuple[str, ...]:
    """Classes defined in the package that structurally satisfy the protocol.

    Structural rather than nominal: a class satisfies a ``Protocol`` by its
    members, so the survey checks the members and not an inheritance list,
    which a protocol implementer need not have. Protocols themselves and the
    abstract bases that only declare the members are excluded.
    """
    found: list[str] = []
    for name in _modules():
        if name.startswith(SKIP_IMPORT):
            continue
        module = importlib.import_module(name)
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != name or cls_name.startswith("_"):
                continue
            if getattr(cls, "_is_protocol", False):
                continue
            if inspect.isabstract(cls):
                continue
            if _satisfies(cls, members):
                found.append(f"{name.removeprefix(PACKAGE + '.')}.{cls_name}")
    return tuple(sorted(set(found)))


def consumers(seam: Seam, sources: dict[str, str]) -> tuple[str, ...]:
    """Modules whose source names the seam, other than its own."""
    pattern = re.compile(rf"\b{re.escape(seam.name)}\b")
    return tuple(
        sorted(
            module
            for module, text in sources.items()
            if module != seam.module and pattern.search(text)
        )
    )


def _problem_modules() -> dict[str, set[str]]:
    """Catalogue row name to the modules its symbols live in."""
    out: dict[str, set[str]] = {}
    for problem, symbols in problems_tables.rows():
        out[problem] = {symbol.rsplit(".", 1)[0] for symbol in symbols}
    return out


def problems_reached(
    seam: Seam, implementers_: tuple[str, ...], consumers_: tuple[str, ...]
) -> tuple[str, ...]:
    """Catalogue rows whose modules define, implement or consume the seam."""
    touched = {
        seam.module,
        *consumers_,
        *(impl.rsplit(".", 1)[0] for impl in implementers_),
    }
    return tuple(
        problem for problem, modules in _problem_modules().items() if modules & touched
    )


def survey() -> list[Row]:
    """Every seam with what the package does with it."""
    sources = _source_modules()
    rows: list[Row] = []
    for seam in SEAMS:
        module = importlib.import_module(f"{PACKAGE}.{seam.module}")
        cls = getattr(module, seam.name)
        if seam.kind == "protocol":
            members = members_of(cls)
            impls = implementers(members)
        else:
            members = tuple(
                sorted(
                    f
                    for f in getattr(cls, "__dataclass_fields__", {})
                    if not f.startswith("_")
                )
            )
            impls = ()
        cons = consumers(seam, sources)
        rows.append(
            Row(seam, members, impls, cons, problems_reached(seam, impls, cons))
        )
    return rows


def under_the_rule(row: Row) -> str:
    """The rule's verdict on one seam, as the ledger prints it."""
    if (
        row.seam.kind == "protocol"
        and len(row.implementers) <= 1
        and len(row.consumers) < CONSUMER_RULE
    ):
        return "one implementer, under the rule"
    if len(row.consumers) < CONSUMER_RULE:
        return "under the rule"
    return "earns its place"


def render(rows: list[Row]) -> str:
    """The ledger text."""
    n_problems = len(_problem_modules())
    lines = [
        "# Seams",
        "",
        "The abstractions the package has, read from the package by",
        "`infra/seams_survey.py` (issue #400): every `Protocol` it declares and every",
        "data contract three or more modules share, with the classes that satisfy each",
        "protocol by its members, the modules that consume it, and the problem classes",
        "of `PROBLEMS.md` it reaches. The rule it is read against is in `DEV.md`",
        f"(Core Development Standards): a seam earns its place with {CONSUMER_RULE} or more",
        "consuming modules. Do not edit by hand -- run",
        "`uv run python infra/seams_survey.py --write`.",
        "",
        f"## Protocols ({len(PROTOCOLS)})",
        "",
        "| Seam | Home | Members | Implementers | Consumers | Problems reached | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.seam.kind != "protocol":
            continue
        lines.append(_line(row, n_problems))
    lines += [
        "",
        f"## Data contracts ({len(CONTRACTS)})",
        "",
        "| Seam | Home | Fields | Consumers | Problems reached | Verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.seam.kind != "contract":
            continue
        lines.append(_contract_line(row, n_problems))
    lines.append("")
    return "\n".join(lines)


def _line(row: Row, n_problems: int) -> str:
    impls = ", ".join(f"`{name}`" for name in row.implementers) or "none"
    return (
        f"| `{row.seam.name}` | `{row.seam.module}` | {', '.join(row.members)} | {impls} "
        f"| {len(row.consumers)} | {len(row.problems)} of {n_problems} | {under_the_rule(row)} |"
    )


def _contract_line(row: Row, n_problems: int) -> str:
    return (
        f"| `{row.seam.name}` | `{row.seam.module}` | {', '.join(row.members)} "
        f"| {len(row.consumers)} | {len(row.problems)} of {n_problems} | {under_the_rule(row)} |"
    )


def main(argv: list[str] | None = None) -> int:
    """Regenerate or check the ledger."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="regenerate SEAMS.md")
    args = parser.parse_args(argv)
    text = render(survey())
    if args.write:
        LEDGER.write_text(text)
        print(f"wrote {LEDGER}")
        return 0
    if not LEDGER.exists() or LEDGER.read_text() != text:
        print(
            "SEAMS.md is stale; regenerate with: uv run python infra/seams_survey.py --write"
        )
        return 1
    print("SEAMS.md is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
