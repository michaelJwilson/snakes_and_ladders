"""Hold a pull-request body to the line cap, reading it from standard input.

Issue #521. A ticket and a pull request are read before any code is, and a
body long enough to skim is a body nobody reads: the cap makes the author
choose what survives. Detail that does not fit belongs where it is refereed
--- a measurement in ``STATUS.md``, a sweep in an experiment file --- which is
the ruling issue #458 made for experiments and the reason this reuses their
counter rather than adding a second one.

:func:`~experiments.body_lines` is that counter: it charges non-blank content
lines and excuses headings and HTML comments, so the repository has one meaning
for "how long is this text". The cap it is charged against differs --- ten
lines for an experiment, forty here --- because the two texts carry different
things. ``.github/pull_request_template.md`` costs 24 of the forty once its
instructions are excused, which is what leaves a description room.

Run by the ``PR title names its base branch`` job in
``.github/workflows/ci.yml``, which already has the pull request's payload::

    printf '%s' "$BODY" | python3 infra/check_pr_body.py

Exits 1 with a GitHub error annotation naming the count and the cap. It reads
standard input rather than a file or an argument so a body carrying quotes,
backticks or a leading dash reaches it as written.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `infra/` on `sys.path`, the way `select_tests.py` and the guards reach their
# shared names; this file is run as a script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments import body_lines

#: What a ticket or a pull-request body may carry, in content lines. `DEV.md`
#: states it for a reader and `.github/ISSUE_TEMPLATE/*.yml` for an author;
#: `tests/regression/test_pr_body_cap.py` holds those copies to this one.
BODY_LINE_CAP = 40


def problem(body: str) -> str:
    """The error for a body over the cap; empty where it is within it."""
    count = body_lines(body)
    if count <= BODY_LINE_CAP:
        return ""
    return (
        f"pull-request body is {count} content lines, over the "
        f"{BODY_LINE_CAP}-line cap (DEV.md). Blank lines and headings are not "
        f"charged; move what does not fit to STATUS.md or an experiment file."
    )


def main() -> int:
    """Read the body from standard input and report against the cap."""
    found = problem(sys.stdin.read())
    if found:
        print(f"::error::{found}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
