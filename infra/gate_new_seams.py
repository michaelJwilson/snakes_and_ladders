"""A protocol the branch adds names the consumers that earn it.

`infra/review_gates.sh` calls this. Root `CLAUDE.md`'s seam rule admits an
abstraction with three or more consuming modules; one below that is kept only
for a reason **stated where it is declared**. Issue #586 deleted `SEAMS.md`
and the hand-written catalogue that used to hold those reasons, on the
argument that a ledger is a copy of the truth: a `Protocol` is discoverable at
import, so the declaration is the record and a second list of declarations is
a list to keep in step.

Two things did not follow the ledger, and live here instead. The consumer
count is a property of the **import graph**, not of the class tree, so
:func:`consumers` computes it. And the reason a seam under the rule is kept
has to be written down somewhere a reader already looks, so it goes in the
declaring class's own docstring behind :data:`REASON`, and this gate reads it
there.

Usage::

    uv run python infra/gate_new_seams.py --base origin/main
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

from _paths import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT / "python" / "snakes_and_ladders"
#: Consuming modules at or above which a seam earns its place, per root
#: `CLAUDE.md`. Below it the seam needs a reason.
CONSUMER_RULE = 3
#: What a docstring writes in front of the reason a seam under the rule is
#: kept. A fixed prefix rather than prose, so the gate reads what a reviewer
#: reads and neither has to trust the other.
REASON = "Under the seam rule:"


def added_protocols(base: str) -> list[tuple[str, str]]:
    """Every ``Protocol`` class the branch adds under ``python/``.

    Parameters
    ----------
    base : str
        The branch the pull request targets.

    Returns
    -------
    list[tuple[str, str]]
        ``(package-relative dotted module, class name)``, sorted.
    """
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD", "--", "python/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    base_names: set[tuple[str, str]] = set()
    head_names: set[tuple[str, str]] = set()
    for name in changed:
        path = Path(name)
        if path.suffix != ".py":
            continue
        module = ".".join(
            path.relative_to("python/snakes_and_ladders").with_suffix("").parts
        )
        module = module.removesuffix(".__init__").removesuffix("__init__")
        head = (REPO_ROOT / path).read_text() if (REPO_ROOT / path).is_file() else ""
        before = subprocess.run(
            ["git", "show", f"{base}:{name}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        head_names |= {(module, found) for found in _protocols(head)}
        base_names |= {
            (module, found)
            for found in _protocols(before.stdout if before.returncode == 0 else "")
        }
    return sorted(head_names - base_names)


def _protocols(source: str) -> set[str]:
    """The names of the ``Protocol`` classes in one module's source.

    Returns
    -------
    set[str]
    """
    if not source.strip():
        return set()
    found: set[str] = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef):
            continue
        for parent in node.bases:
            name = (
                parent.id
                if isinstance(parent, ast.Name)
                else getattr(parent, "attr", "")
            )
            if name == "Protocol":
                found.add(node.name)
    return found


def source_modules() -> dict[str, str]:
    """Package-relative dotted module name to its source text.

    Returns
    -------
    dict[str, str]
    """
    out: dict[str, str] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts) or "__init__"] = path.read_text()
    return out


def consumers(module: str, name: str, sources: dict[str, str]) -> tuple[str, ...]:
    """Modules whose source names the seam, other than the one declaring it.

    Read from the source text rather than from imports, so a module reaching
    the seam through a re-export counts as the consumer it is.

    Parameters
    ----------
    module : str
        Package-relative dotted module declaring the seam.
    name : str
        The class name.
    sources : dict[str, str]
        As :func:`source_modules`.

    Returns
    -------
    tuple[str, ...]
        Sorted dotted module names.
    """
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    return tuple(
        sorted(
            other
            for other, text in sources.items()
            if other != module and pattern.search(text)
        )
    )


def reason_for(module: str, name: str, sources: dict[str, str]) -> str:
    """The reason the class's own docstring states for keeping it under the rule.

    Parameters
    ----------
    module, name, sources
        As :func:`consumers`.

    Returns
    -------
    str
        The rest of the paragraph following :data:`REASON`, or ``""`` where
        the class, its docstring or the marker is absent. One paragraph and
        not the remainder of the docstring, so a reason written above a
        ``Parameters`` section stops where the reason does.
    """
    source = sources.get(module, "")
    if not source.strip():
        return ""
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef) or node.name != name:
            continue
        text = ast.get_docstring(node) or ""
        if REASON not in text:
            return ""
        tail = text.split(REASON, 1)[1]
        return " ".join(tail.split("\n\n", 1)[0].split()).strip()
    return ""


def verdicts(base: str) -> list[str]:
    """The rule's verdict on each protocol the branch adds, when it is a failure.

    Returns
    -------
    list[str]
        One line per protocol the rule does not admit and no reason keeps.
    """
    added = added_protocols(base)
    if not added:
        return []
    sources = source_modules()
    failures: list[str] = []
    for module, name in added:
        consuming = consumers(module, name, sources)
        if len(consuming) >= CONSUMER_RULE:
            continue
        if reason_for(module, name, sources):
            continue
        failures.append(
            f"{module}.{name}: {len(consuming)} consuming modules, "
            f"{CONSUMER_RULE} required, and its docstring states no "
            f'"{REASON}" reason for keeping it'
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    """Report a new protocol the seam rule does not admit.

    Returns
    -------
    int
        ``0`` when every added protocol earns its place or states a reason.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    args = parser.parse_args(argv)
    failures = verdicts(args.base)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
