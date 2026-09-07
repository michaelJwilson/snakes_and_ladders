#!/usr/bin/env bash
# Wall clock per test tier, against the budgets DEV.md states.
#
# The budgets are enforced by fixture size rather than by a timing assertion:
# DEV.md forbids ranking performance on CI runners, so a test that failed on
# wall clock would fail for the machine rather than for the change. This
# reports the numbers instead, so a change that moves them is visible in the
# pull request that moves it -- the same footing as `infra/measure_build.sh`.
#
# Run on fixed hardware. State the machine beside any number taken from it.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

measure() {
  local name="$1" budget="$2"
  shift 2
  local start elapsed
  start=$(date +%s)
  "$@" > /dev/null 2>&1
  local status=$?
  elapsed=$(( $(date +%s) - start ))
  printf '%-28s %4ds   budget %4ds   %s\n' "$name" "$elapsed" "$budget" \
    "$( [ "$elapsed" -le "$budget" ] && echo inside || echo OVER )"
  return $status
}

echo "tier                         wall    budget         verdict"
measure "ci (per pull request)" 300 \
  uv run pytest -m "not release and not stress" tests/regression -q
measure "stress (developer)" 600 \
  uv run pytest -m "stress" tests/regression -q
echo
echo "The release gate (plain \`pytest\`) is outside both budgets by design;"
echo "\`infra/release.sh\` runs it."
