#!/usr/bin/env bash
# The local release gate: every check CI runs per PR, plus the
# release-gated scientific tests CI skips (DEV.md's "Release-Gated" budget)
# and the document build. Run before tagging a release; CI's
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
# The full documentation build, not an incremental one (issue #485). `-E`
# discards any saved environment and `-a` writes every output, so the run
# re-reads all 134 modules whatever `docs/_build/` holds from the last one: an
# incremental build re-reads only what changed, and a warning in an untouched
# docstring is a warning it never emits. `docs/source/index.rst` autodocuments
# the modules in one document, so there is no cheap incremental form of this to
# gate with -- 32.2 s cold against 8.2 s when nothing changed (#476), which is
# free beside the figure step below. Zero warnings across all 134 modules is
# the current state and `-W` is what holds it there.
run_check "sphinx-build -W (full)" \
  uv run sphinx-build -E -a -b html docs/source docs/_build/html -W
# The generated ledgers are written here rather than read from the tree: none
# is committed (issue #425), and --check fails if a copy of one reached the
# index and a regeneration disagrees with it.
run_check "generated ledgers" infra/ledgers.sh --check
# Every figure, cited and uncited, rendered and compared against the committed
# bytes (issue #484). `--all` ignores the stamps, so no digest, no prediction
# and no skip decides what is checked: the guarantee is that a figure whose
# rendered bytes would change cannot reach a release claiming to be current,
# and rendering all of them is what enforces it. The stamps' measured
# false-positive rate is 100% over 476 decisions (#476), so this is the whole
# mechanism at the release gate rather than the remainder of one.
#
# Before `infra/build_documents.sh`, and that ordering is the check: that
# script renders a stale cited figure *into* docs/tex/figures/, so running it
# first would leave `--check` comparing a rebuild against bytes it had just
# overwritten. A mismatch is a failure, never a silent refresh -- `--check`
# renders into a temporary directory and compares, naming each figure that
# does not reproduce. The clock the figures are rendered against is pinned in
# snakes_and_ladders.qa.build, so this needs no environment of its own.
run_check "QA figures (every figure)" \
  uv run python -m snakes_and_ladders.qa.build --all --check
run_check "documents" infra/build_documents.sh
# The baseline numbers beside each fixture are read per pull request and
# recomputed here (issue #401), the same trade the figures make above: the
# tests that used to compute an enumerated maximum or a hill-climbing rate
# before measuring anything now read one, and the only thing that recomputes
# it is this gate. A cached number no run ever reproduces would be a claim
# with no referee.
run_check "fixture baselines" uv run python infra/baselines.py

echo
if [ "${#failures[@]}" -eq 0 ]; then
  echo "release gate: PASS (${#failures[@]} failures)"
  exit 0
else
  echo "release gate: FAIL -- ${failures[*]}"
  exit 1
fi
