#!/usr/bin/env bash
# The local release gate: every check CI runs per PR, plus the
# release-gated scientific tests CI skips (DEV.md's "Release-Gated" budget)
# and the technical-document build. Run before tagging a release; CI's
# per-PR jobs are a subset of this, not a replacement (rust-tests' `cargo
# bench` and the `build`/`audit` jobs are covered by the equivalent checks
# below or are PR-only smoke tests, so are not repeated here).
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

failures=()

run_check() {
  local name="$1"
  shift
  echo "==> ${name}"
  if "$@"; then
    echo "==> ${name}: PASS"
  else
    echo "==> ${name}: FAIL"
    failures+=("${name}")
  fi
}

run_check "ruff check" uv run ruff check .
run_check "ruff format --check" uv run ruff format --check .
run_check "mypy --strict" uv run mypy
run_check "cargo clippy" cargo clippy --locked --all-targets -- -D warnings
run_check "cargo fmt --check" cargo fmt --check
run_check "cargo test" cargo test --locked
# Plain `pytest`, not `-m release`: the latter marker-filters down to only
# release-marked tests, dropping everything `python-tests`' `-m "not
# release"` already covers. DEV.md's "Run the full suite ... with `pytest -m
# release` or plain `pytest`" names both, but only the unfiltered form runs
# the full suite -- see this PR's DEV.md fix.
run_check "pytest (full suite)" uv run pytest --cov=snakes_and_ladders --cov-report=term-missing --cov-fail-under=90
run_check "sphinx-build -W" uv run sphinx-build -b html docs/source docs/_build/html -W
run_check "technical doc" infra/build_technical_doc.sh
# The per-PR build regenerates only what the document cites (issue #154), so
# the figures it does not cite are checked here instead -- the check moves to
# the release gate rather than disappearing. `--check` compares a rebuild
# without overwriting, so a figure that has rotted is reported rather than
# silently refreshed. The clock the figures are rendered against is pinned in
# snakes_and_ladders.qa.build, so this needs no environment of its own.
run_check "QA figures (all, incl. uncited)" \
  uv run python -m snakes_and_ladders.qa.build --all --check

echo
if [ "${#failures[@]}" -eq 0 ]; then
  echo "release gate: PASS (${#failures[@]} failures)"
  exit 0
else
  echo "release gate: FAIL -- ${failures[*]}"
  exit 1
fi
