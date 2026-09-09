#!/usr/bin/env bash
# Everything a pull request needs before it is pushed, in the order that fails
# fastest, each step timed, the whole under the 300 s budget DEV.md states
# (issue #372). What runs is decided by the diff against the base branch:
# lint and types over the changed Python files; the critical gate; the tests
# infra/select_tests.py names for the diff (the guards alone for a change to
# prose); Sphinx when a docstring or docs/source/ changed; the notebook
# checker, which executes every notebook it is given; the document build when
# docs/tex/ changed, which renders every figure the documents cite. CI runs
# the whole set and is the gate; this is the local answer.
#
# Usage: infra/validate.sh [--base <ref>]     default base: origin/main
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

base="origin/main"
[ "${1:-}" = "--base" ] && base="$2"

# One process is one core: the load average then counts processes, and four
# fit the host. The figure renderers keep their threading; qa.build strips
# these from a render's environment.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# The environment is synced once per worktree (`uv sync --locked`); no step
# here may re-resolve it, which recompiles the Rust extension (issue #369).
export UV_NO_SYNC=1
# A test over this many seconds carries `release`, `stress` or `key`, or fails
# (tests/conftest.py). Asserted here, on the reference host, never in CI.
export SAL_DURATION_CAP="${SAL_DURATION_CAP:-10}"
# The `key` tier is exempt from that cap and held to this one: a key fixture
# is by definition the largest declared instance that fits it (issue #399).
export SAL_KEY_DURATION_CAP="${SAL_KEY_DURATION_CAP:-120}"
# The selected tests are capped, so a selection that grew past the budget is
# reported as such rather than waited for.
selected_cap=180

scratch="${SAL_SCRATCH:-${TMPDIR:-/tmp}/snakes_and_ladders-validate}"
mkdir -p "$scratch"

failures=()
started=$(date +%s)

step() {
  # Runs one named step, prints its wall clock, records a failure.
  local name="$1"
  shift
  local start elapsed
  start=$(date +%s)
  echo "==> $name"
  if "$@"; then
    elapsed=$(( $(date +%s) - start ))
    printf '==> %s: PASS (%ds)\n' "$name" "$elapsed"
  else
    elapsed=$(( $(date +%s) - start ))
    printf '==> %s: FAIL (%ds)\n' "$name" "$elapsed"
    failures+=("$name")
  fi
}

merge_base="$(git merge-base "$base" HEAD)"
# Committed, staged, unstaged and untracked: everything the pull request
# would carry once committed.
changed="$( (git diff --name-only "$merge_base"; git ls-files --others --exclude-standard) | sort -u)"
echo "changed files against $base:"
echo "$changed" | sed 's/^/  /'

changed_python="$(echo "$changed" | grep -E '\.py$' | while read -r f; do [ -f "$f" ] && echo "$f"; done)"
if [ -n "$changed_python" ]; then
  # shellcheck disable=SC2086
  step "ruff check" uv run ruff check $changed_python
  # shellcheck disable=SC2086
  step "ruff format --check" uv run ruff format --check $changed_python
  # The changed files and their tests, not the whole `files` set: mypy over
  # everything is two minutes, and CI keeps the whole set.
  # shellcheck disable=SC2086
  step "mypy --strict (changed)" uv run mypy $changed_python
fi

step "pytest -m critical" uv run pytest -m critical -q -p no:cacheprovider

selection="$(echo "$changed" | uv run python infra/select_tests.py --format json)"
paths="$(echo "$selection" | uv run python -c 'import json,sys; print(" ".join(json.load(sys.stdin)["paths"]))')"
# The tiers this change does not run. `key` is in the list unless the change
# touches the coupled model, the emissions, the fixtures or the Rust crate,
# so a key test costs its two minutes only where it can fail.
markers="$(echo "$selection" | uv run python -c 'import json,sys; print(" and ".join("not " + m for m in json.load(sys.stdin)["deselect"]))')"
if [ -n "$paths" ]; then
  echo "selected: $paths"
  echo "markers: $markers"
  # shellcheck disable=SC2086
  step "selected tests (cap ${selected_cap}s)" timeout "$selected_cap" \
    uv run pytest -m "$markers" $paths -q -p no:cacheprovider --durations=10
else
  echo "==> selected tests: nothing selected"
fi

if echo "$changed" | grep -q '^docs/source/' \
  || git diff -U0 "$merge_base" -- python/ | grep -q '^[+-].*"""'; then
  # `-d` keeps the doctrees between runs, so a second build is seconds.
  step "sphinx-build -W" uv run sphinx-build -b html -d "$scratch/doctrees" \
    docs/source "$scratch/html" -W -q
fi

if echo "$changed" | grep -qE '^(docs/nb/|python/snakes_and_ladders/|tests/regression/fixtures/)'; then
  # Every notebook under docs/nb/, not the ones a digest calls stale: which
  # claims a run verifies is not a hash's decision (issue #480). The six cost
  # the number DEV.md's budget table carries, inside the 300 s.
  step "notebooks" uv run python infra/check_notebooks.py
fi

if echo "$changed" | grep -q '^docs/tex/'; then
  step "documents" infra/build_documents.sh
fi

echo
elapsed=$(( $(date +%s) - started ))
if [ "${#failures[@]}" -eq 0 ]; then
  echo "validate: PASS in ${elapsed}s (budget 300s)"
  exit 0
fi
echo "validate: FAIL in ${elapsed}s -- ${failures[*]}"
exit 1
