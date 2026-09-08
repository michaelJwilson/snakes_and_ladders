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

export UV_NO_SYNC=1
# One process is one core, as the suite runs (tests/conftest.py).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

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

# The critical tier by default; the CI and stress tiers with `--full`. The
# full tiers run the whole suite and, on a shared host, take the host with
# them for the duration (issue #369); measure them on an idle machine and
# state it beside the number.
echo "host: $(getconf _NPROCESSORS_ONLN) cores, load $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || sysctl -n vm.loadavg)"
echo "tier                         wall    budget         verdict"
measure "critical (early gate)" 30 \
  uv run --no-sync pytest -m "critical" tests/regression -q
if [ "${1:-}" = "--full" ]; then
  measure "ci (per pull request)" 300 \
    uv run --no-sync pytest -m "not release and not stress" tests/regression -q
  measure "stress (developer)" 600 \
    uv run --no-sync pytest -m "stress" tests/regression -q
else
  echo "(pass --full for the ci and stress tiers)"
fi

# `infra/validate.sh` against its 300 s budget (issue #372), for the three
# classes of change DEV.md tabulates. Each is simulated by naming the changed
# files to `select_tests.py`; the guards and the selected tests are what
# `validate.sh` would run for that diff, less lint and the critical gate,
# which are the same for every class.
echo
echo "validate.sh selection             wall    budget         verdict"
selection() {
  # Prints the test paths select_tests.py names for the given changed files.
  printf '%s\n' "$@" | uv run --no-sync python infra/select_tests.py --format json \
    | uv run --no-sync python -c 'import json,sys; print(" ".join(json.load(sys.stdin)["paths"]))'
}
for class in "docs-only:docs/tex/paper.tex" \
             "one module (learn/):python/snakes_and_ladders/learn/reinforce.py" \
             "search/-wide:python/snakes_and_ladders/search/infer.py"; do
  name="${class%%:*}"; files="${class#*:}"
  paths="$(selection $files)"
  # shellcheck disable=SC2086
  measure "$name" 300 uv run --no-sync pytest -m "not release and not stress" $paths -q -p no:cacheprovider
done
echo
echo "The release gate (plain \`pytest\`) is outside every budget by design;"
echo "\`infra/release.sh\` runs it."
