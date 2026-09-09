#!/usr/bin/env bash
# The generated ledgers: CHECKS.md from the tests' own markers, SEAMS.md from
# the package, and the textbook's applicability tables from PROBLEMS.md and
# the suite. None is committed (issue #425) -- a machine-written file that is
# committed becomes a merge participant, and a conflict between two machine
# writings carries no information to resolve.
#
# The guarantee the committed copy used to give -- the ledger matches the
# tree -- is not weakened by that, because it is now checked rather than
# remembered: `--check` regenerates and fails if a *tracked* file moved, so a
# stale copy that reaches the index fails CI wherever it came from.
#
# Usage: infra/ledgers.sh [--check]
#   (default)  regenerate the three files in place
#   --check    regenerate, then fail if regenerating changed a tracked file
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

check=0
case "${1-}" in
  --check) check=1 ;;
  "") ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

# Every generator writes; none is asked to compare against a committed file,
# since there is none. A generator that cannot write -- a catalogue symbol it
# cannot name, a pairing with no note -- fails here, which is the other half
# of what the old per-file `--check` bought.
uv run python infra/checks_ledger.py --write
uv run python infra/seams_survey.py --write
uv run python infra/problems_tables.py --write

if [ "$check" = 0 ]; then
  exit 0
fi

LEDGERS=(CHECKS.md SEAMS.md docs/tex/generated/problems_tables.tex)

# Two failures, and the second is the first one's cause. `git diff HEAD` and
# not `git status`: an untracked ledger is the expected state and says
# nothing, while a tracked one that regeneration rewrote is exactly the
# staleness this replaces.
moved="$(git diff --name-only HEAD -- "${LEDGERS[@]}")"
if [ -n "$moved" ]; then
  echo "::error::regenerating rewrote a committed ledger: $moved" >&2
  git --no-pager diff --stat HEAD -- $moved >&2
  exit 1
fi
# A committed ledger that happens to be current passes the check above and
# will conflict on the next merge anyway, so it is refused on being tracked
# rather than on being stale.
tracked="$(git ls-files -- "${LEDGERS[@]}")"
if [ -n "$tracked" ]; then
  echo "::error::a generated ledger is committed: $tracked" >&2
  echo "these are written from the tree and must not be tracked (.gitignore, issue #425)" >&2
  exit 1
fi
echo "generated ledgers: regeneration changed no tracked file, and none is tracked"
