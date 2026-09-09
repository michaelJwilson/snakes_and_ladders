#!/usr/bin/env bash
# The Definition-of-Done gates a reviewer checks before reading a line, run
# as one script so review starts from a pass/fail table rather than from a
# checklist held in the head (CLAUDE.md, Throughput). Each gate is a fact
# about the branch that CI does not assert or asserts late: the head carries
# the base; the two committed PDFs are the base's unless this is a rebuild;
# a changelog fragment exists; every cited figure's stamp matches the tree;
# the critical tier passes, under the duration cap; the tests the branch
# touched say what checks them; a protocol it adds names the consumers the
# seam rule wants; and the generated ledgers regenerate without rewriting a
# committed one. Reading covers what a script cannot: whether the
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

# NOT A GATE (issue #455). Kept because `infra/release.sh` is where the claim
# is now checked, through `qa.build --all --check`, and because a developer
# may still want the answer before pushing. It stopped being a per-pull-request
# gate because the check is cheap and the fix is not: a merge of `main` moves
# a module in a figure's closure, the stamp goes stale, and the branch pays a
# six-minute render for a figure that re-renders to identical bytes. That was
# eight renders across four branches in one day, and no figure byte moved on
# any of them.
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

changed_tests_say_what_checks_them() {
  # Every test the diff adds or rewrites carries a kind: what refereed it,
  # not when it runs (root CLAUDE.md). Read from the source, so it costs
  # nothing and reports against the diff rather than as a traceback out of
  # the repository-wide guard in the critical tier.
  #
  # The duration half of this row is not here, and the reason is a
  # measurement: running the branch's changed test files took 20 s of a 30 s
  # budget on the branch that added this gate, and they had already been run
  # once by `infra/validate.sh`, which is where SAL_DURATION_CAP is asserted
  # on the reference host. What the gate does get for free is the cap over
  # the critical tier above, which runs under it.
  uv run python infra/gate_changed_tests.py --base "$base"
}

new_seams_name_their_consumers() {
  uv run python infra/gate_new_seams.py --base "$base"
}

generated_ledgers_are_current() {
  # CHECKS.md, SEAMS.md and the problem tables are written from the tree and
  # are not committed (issue #425). Two things can still fail: a generator
  # that no longer runs -- a catalogue symbol it cannot name, a pairing with
  # no note -- and a copy of one that reached the index and has gone stale.
  # CI checks both; here as well because late is the problem, a stale ledger
  # being a re-push and this costing less than the round trip.
  infra/ledgers.sh --check >/dev/null 2>>"$log"
}

echo "review gates against $base at $(git rev-parse --short HEAD)"
gate "environment imports this checkout"     environment_imports_this_checkout
gate "head carries the base"                 carries_base
gate "PDFs are the base's (or a rebuild)"    pdfs_are_the_base_s
gate "changelog fragment exists"             fragment_exists
gate "critical tier passes"                  critical_tier_passes
gate "changed tests say what checks them"    changed_tests_say_what_checks_them
gate "a new seam names its consumers"        new_seams_name_their_consumers
gate "generated ledgers are current"         generated_ledgers_are_current

total=$(( $(date +%s) - started ))
if [ ${#failures[@]} -gt 0 ]; then
  echo "review gates: ${#failures[@]} failed in ${total}s"
  exit 1
fi
echo "review gates: all passed in ${total}s; read the diff against the plan"
