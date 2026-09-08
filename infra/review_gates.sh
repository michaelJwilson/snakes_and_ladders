#!/usr/bin/env bash
# The Definition-of-Done gates a reviewer checks before reading a line, run
# as one script so review starts from a pass/fail table rather than from a
# checklist held in the head (CLAUDE.md, Throughput). Each gate is a fact
# about the branch that CI does not assert or asserts late: the head carries
# the base; the two committed PDFs are the base's unless this is a rebuild;
# a changelog fragment exists; every cited figure's stamp matches the tree;
# the critical tier passes. Reading covers what a script cannot: whether the
# change is the plan on the ticket, and whether the tests pin what they
# claim to.
#
# Usage: infra/review_gates.sh [--base <ref>] [--rebuild]
#   --base     the branch the pull request targets (default origin/main)
#   --rebuild  the pull request is a "Rebuild the documents" one, so the
#              PDFs are expected to differ from the base
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

base="origin/main"
rebuild=0
while [ $# -gt 0 ]; do
  case "$1" in
    --base) base="$2"; shift 2 ;;
    --rebuild) rebuild=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_NO_SYNC=1
log="${SAL_SCRATCH:-${TMPDIR:-/tmp}}/review_gates.log"

failures=()
gate() {
  # Runs one named gate and records its verdict.
  local name="$1"
  shift
  if "$@"; then
    printf '%-44s PASS\n' "$name"
  else
    printf '%-44s FAIL\n' "$name"
    failures+=("$name")
  fi
}

environment_imports_this_checkout() {
  # A worktree that shares an environment with a sibling imports the sibling's
  # code unless PYTHONPATH says otherwise; every gate below would then judge
  # the wrong tree.
  local where
  where="$(uv run python -c 'import snakes_and_ladders as s; print(s.__file__)')" || return 1
  case "$where" in
    "$repo_root"/*) return 0 ;;
    *) echo "  imports $where; sync this worktree or set PYTHONPATH" >&2; return 1 ;;
  esac
}

carries_base() { git merge-base --is-ancestor "$base" HEAD; }

pdfs_are_the_base_s() {
  local changed
  changed="$(git diff --name-only "$base"...HEAD -- docs/paper.pdf docs/textbook.pdf)"
  if [ "$rebuild" = 1 ]; then
    [ -n "$changed" ] || { echo "  a rebuild that changed neither PDF" >&2; return 1; }
  else
    [ -z "$changed" ] || { echo "  changed: $changed" >&2; return 1; }
  fi
}

fragment_exists() { uv run towncrier check --compare-with "$base" >/dev/null; }

stamps_match_the_tree() {
  # Digest-only: nothing is rendered. A stale cited figure means the branch
  # changed an input and did not re-render, or re-rendered and did not
  # commit the stamp.
  uv run python - <<'PY'
import sys
from snakes_and_ladders.qa.build import DEFAULT_DOCUMENTS, DEFAULT_OUTPUT_DIR, selected, stale
specs = selected(list(DEFAULT_DOCUMENTS), every=False)
out = stale(specs, DEFAULT_OUTPUT_DIR)
for spec in out:
    print(f"  stale: {spec.stem}", file=sys.stderr)
sys.exit(1 if out else 0)
PY
}

critical_tier_passes() {
  uv run pytest -m critical -q -p no:cacheprovider >"$log" 2>&1 || { tail -n 15 "$log" >&2; return 1; }
}

echo "review gates against $base at $(git rev-parse --short HEAD)"
gate "environment imports this checkout"     environment_imports_this_checkout
gate "head carries the base"                 carries_base
gate "PDFs are the base's (or a rebuild)"    pdfs_are_the_base_s
gate "changelog fragment exists"             fragment_exists
gate "cited figure stamps match the tree"    stamps_match_the_tree
gate "critical tier passes"                  critical_tier_passes

if [ ${#failures[@]} -gt 0 ]; then
  echo "review gates: ${#failures[@]} failed"
  exit 1
fi
echo "review gates: all passed; read the diff against the plan"
