#!/usr/bin/env bash
# The generated ledgers: CHECKS.md from the tests' own markers, and the
# textbook's applicability tables from PROBLEMS.md and the suite. Neither is
# committed (issue #425) -- a machine-written file that is committed becomes a
# merge participant, and a conflict between two machine writings carries no
# information to resolve.
#
# SEAMS.md was a third and is gone (issue #586): every Protocol is
# discoverable at import, so the declaration is the record, and the one thing
# the ledger computed that the class tree does not -- the consuming-module
# count -- moved into infra/gate_new_seams.py, where it is used.
#
# And two blocks of files that *are* committed: pyproject.toml's marker list
# and DEV.md's tier table, written from infra/gates.py between marker comments
# (issue #470). These are not ledgers of the tree and do not conflict -- they
# are short, and each is derived from one table a person edits -- so they are
# generated in place and the check below is that regenerating did not move
# them. Only the block is written; the prose around it is hand-written, since
# DEV.md is followed step by step.
#
# The guarantee the committed copy used to give -- the ledger matches the
# tree -- is not weakened by that, because it is now checked rather than
# remembered: `--check` regenerates and fails if a *tracked* file moved, so a
# stale copy that reaches the index fails CI wherever it came from.
#
# Usage: infra/ledgers.sh [--check]
#   (default)  regenerate the two files and the two derived blocks in place
#   --check    regenerate, then fail if regenerating changed a tracked file
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
# Issue #556: the shared `.venv` carries dependencies, not the project, so
# `PYTHONPATH` is the only route to the package and no script can repoint a
# global editable install at its own worktree. Exported here rather than left
# to the caller, so this script operates on the tree it lives in whatever the
# environment says.
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"

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
uv run --no-sync python infra/checks_ledger.py --write
uv run --no-sync python infra/problems_tables.py --write
uv run --no-sync python infra/gates.py --write

if [ "$check" = 0 ]; then
  exit 0
fi

LEDGERS=(CHECKS.md docs/tex/generated/problems_tables.tex)
# The files carrying a generated block. Tracked, and required to be: the block
# is part of a file a person reads and edits around.
DERIVED=(pyproject.toml DEV.md)

# Two failures, and the second is the first one's cause. `git diff HEAD` and
# not `git status`: an untracked ledger is the expected state and says
# nothing, while a tracked one that regeneration rewrote is exactly the
# staleness this replaces. A derived block fails the same way and for the same
# reason -- it was edited where it is read rather than where it is written.
moved="$(git diff --name-only HEAD -- "${LEDGERS[@]}" "${DERIVED[@]}")"
if [ -n "$moved" ]; then
  echo "::error::regenerating rewrote a committed file: $moved" >&2
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
echo "generated: regeneration changed no tracked file, and no ledger is tracked"
