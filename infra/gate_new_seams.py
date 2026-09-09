"""A protocol the branch adds names the consumers that earn it.

`infra/review_gates.sh` calls this. `DEV.md`'s seam rule admits an abstraction
with three or more consuming modules; one below that is kept only for a stated
reason. `infra/seams_survey.py` already computes both numbers, so this asks it
about the protocols the diff adds rather than recomputing anything: a new
`Protocol` must appear in the survey's catalogue, and its verdict must be that
it earns its place or that a reason was written down.

The catalogue is where the reason lives, so a protocol the survey does not know
fails here -- that is the case this gate exists for, since an unlisted protocol
is invisible to `SEAMS.md` and to the rule.

Usage::

    uv run python infra/gate_new_seams.py --base origin/main
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "infra"))
import seams_survey  # noqa: E402


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


def verdicts(base: str) -> list[str]:
    """The survey's verdict on each protocol the branch adds, when it is a failure.

    Returns
    -------
    list[str]
        One line per protocol the rule does not admit and no reason keeps.
    """
    added = added_protocols(base)
    if not added:
        return []
    rows = {(row.seam.module, row.seam.name): row for row in seams_survey.survey()}
    failures: list[str] = []
    for module, name in added:
        row = rows.get((module, name))
        if row is None:
            failures.append(
                f"{module}.{name} is not in seams_survey.SEAMS, so SEAMS.md and "
                f"the consumer rule cannot see it"
            )
            continue
        verdict = seams_survey.under_the_rule(row)
        if "no reason stated" in verdict:
            failures.append(
                f"{module}.{name}: {len(row.consumers)} consumers, "
                f"{seams_survey.CONSUMER_RULE} required -- {verdict}"
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
