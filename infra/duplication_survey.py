"""Count the duplications issue #230 surveys, by the query that found each.

The survey's numbers were an impression until they were a query. Each finding
below carries the pattern that produces it, so a reader re-runs the survey
rather than trusting a table in a pull request, and so the before-and-after
claim is a measurement rather than a recollection.

Run with ``uv run python infra/duplication_survey.py``. It prints a count per
finding and exits 0 regardless: this reports, it does not gate. What gates is
``tests/regression/test_duplication_guards.py``, which asserts the three
findings this consolidation closed and is tested to fail on a violating input.

Counts are over ``python/snakes_and_ladders`` only. Tests legitimately repeat structure --
a test that shares a helper with the code it checks is testing the helper --
so counting them would report the suite as the defect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "python" / "snakes_and_ladders"


@dataclass(frozen=True)
class Finding:
    """One duplication, and the query that counts it.

    Parameters
    ----------
    name : str
        What is duplicated, as the survey names it.
    pattern : str
        Regular expression counted over every ``.py`` file in the package.
    at_filing : int
        The count when issue #230 was filed, for comparison.
    owner : str
        File that legitimately holds the pattern once, excluded from the
        count. A consolidated helper still matches the query that found the
        duplication, and counting its own definition would report the fix as
        the defect.
    """

    name: str
    pattern: str
    at_filing: int
    owner: str = ""


FINDINGS = (
    Finding(
        "enumerate-shaped functions",
        r"^def (?:enumerate\w*|brute_force\w*|optimum)\(",
        8,
    ),
    Finding("private logsumexp copies", r"^def _logsumexp\(", 4),
    Finding(
        "open-coded edge/coupling zips",
        r"zip\(\s*\w+\.edges,\s*\w+\.coupling",
        6,
        owner="sim/graph.py",
    ),
    Finding(
        "enumeration-cap literals",
        r"^\s*MAX_ENUMERABLE\w* = \d",
        2,
        owner="enumeration.py",
    ),
    Finding("signatures taking a seed", r"seed: int", 12),
    Finding("signatures taking a generator", r"rng: np\.random\.Generator", 10),
    Finding(
        "energy-shaped definitions", r"^\s*def (?:energy|log_weights|relaxed)\(", 5
    ),
)


def count(pattern: str, owner: str = "") -> int:
    """How many times ``pattern`` matches across the package.

    Parameters
    ----------
    pattern : str
        Regular expression, applied per file in multiline mode.
    owner : str
        Package-relative path allowed to hold the pattern, excluded from the
        count. Empty means count everywhere.

    Returns
    -------
    int
        Total matches over every ``.py`` file under ``python/snakes_and_ladders``.
    """
    compiled = re.compile(pattern, re.MULTILINE)
    return sum(
        len(compiled.findall(path.read_text()))
        for path in sorted(PACKAGE.rglob("*.py"))
        if not (owner and path.as_posix().endswith(owner))
    )


def main() -> None:
    """Print one row per finding: name, count at filing, count now."""
    width = max(len(finding.name) for finding in FINDINGS)
    print(f"{'finding':{width}}  {'at filing':>9}  {'now':>5}")
    for finding in FINDINGS:
        now = count(finding.pattern, finding.owner)
        print(f"{finding.name:{width}}  {finding.at_filing:>9}  {now:>5}")


if __name__ == "__main__":
    main()
