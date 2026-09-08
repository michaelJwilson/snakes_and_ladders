#!/usr/bin/env bash
# The Definition-of-Done gates a reviewer checks before reading a line, run
# as one script so review starts from a pass/fail table rather than from a
# checklist held in the head (CLAUDE.md, Throughput). Each gate is a fact
# about the branch that CI does not assert or asserts late: the head carries
# the base; the two committed PDFs are the base's unless this is a rebuild;
# a changelog fragment exists; every cited figure's stamp matches the tree;
# the critical tier passes; the tests the branch touched say what checks them
# and run inside the duration cap; a protocol it adds names the consumers the
# seam rule wants; and the generated ledgers are a regeneration of the tree
# rather than a recollection. Reading covers what a script cannot: whether the
# change is the plan on the ticket, and whether the tests pin what they
# claim to. Those two rows are the point of a review and stay with the
# reviewer; a gate that claimed them would be worse than no gate.
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
# A test over this many seconds carries `release` or `stress`, or fails
# (tests/conftest.py, DEV.md). Asserted here as infra/validate.sh asserts it,
# on the reference host and never in CI.
export SAL_DURATION_CAP="${SAL_DURATION_CAP:-10}"
log="${SAL_SCRATCH:-${TMPDIR:-/tmp}}/review_gates.log"

failures=()
started=$(date +%s)
gate() {
  # Runs one named gate, records its verdict, and prints its wall clock. The
  # seconds are output rather than a comment because the budget is a number:
  # `DEV.md` gives the whole table 30 s, above which a check belongs in CI
  # rather than in something a reviewer runs per branch.
  local name="$1"
  shift
  local start elapsed
  start=$(date +%s)
  if "$@"; then
    elapsed=$(( $(date +%s) - start ))
    printf '%-44s PASS  %3ds\n' "$name" "$elapsed"
  else
    elapsed=$(( $(date +%s) - start ))
    printf '%-44s FAIL  %3ds\n' "$name" "$elapsed"
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

changed_tests_are_marked_and_inside_the_cap() {
  # Two facts about the tests the diff touched: each says what refereed it,
  # and none quietly added its minute to the per-pull-request tier. The kind
  # is read from the source; the cap needs the tests run, so the ones the
  # critical row above did not already run are run here -- which is what the
  # marginal cost of this gate is, and it is bounded by the cap itself.
  uv run python infra/gate_changed_tests.py --base "$base" || return 1
  local files
  files="$(uv run python infra/gate_changed_tests.py --base "$base" --files)"
  [ -n "$files" ] || return 0
  # shellcheck disable=SC2086
  uv run pytest $files -m "not critical" -q -p no:cacheprovider --durations=5 \
    >"$log" 2>&1 || { tail -n 20 "$log" >&2; return 1; }
}

new_seams_name_their_consumers() {
  uv run python infra/gate_new_seams.py --base "$base"
}

generated_ledgers_are_current() {
  # CHECKS.md, SEAMS.md and the problem tables are regenerations, and each has
  # a CI job that fails when the committed file disagrees with the tree. Late
  # is the problem: a stale ledger is a re-push, and the three checks together
  # cost less than the round trip.
  local stale=0
  uv run python infra/checks_ledger.py --check >/dev/null 2>>"$log" || stale=1
  uv run python infra/seams_survey.py >/dev/null 2>>"$log" || stale=1
  uv run python infra/problems_tables.py --check >/dev/null 2>>"$log" || stale=1
  [ "$stale" = 0 ] || {
    echo "  a generated ledger is stale; see its --write command in infra/" >&2
    return 1
  }
}

echo "review gates against $base at $(git rev-parse --short HEAD)"
gate "environment imports this checkout"     environment_imports_this_checkout
gate "head carries the base"                 carries_base
gate "PDFs are the base's (or a rebuild)"    pdfs_are_the_base_s
gate "changelog fragment exists"             fragment_exists
gate "cited figure stamps match the tree"    stamps_match_the_tree
gate "critical tier passes"                  critical_tier_passes
gate "changed tests marked, inside the cap"  changed_tests_are_marked_and_inside_the_cap
gate "a new seam names its consumers"        new_seams_name_their_consumers
gate "generated ledgers are current"         generated_ledgers_are_current

total=$(( $(date +%s) - started ))
if [ ${#failures[@]} -gt 0 ]; then
  echo "review gates: ${#failures[@]} failed in ${total}s"
  exit 1
fi
echo "review gates: all passed in ${total}s; read the diff against the plan"
