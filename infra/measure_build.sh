#!/usr/bin/env bash
# Times the build, stage by stage, so a claim that it got faster is a pair of
# numbers rather than an impression (issue #154).
#
# Run it on fixed hardware. DEV.md forbids ranking performance on CI runners,
# and that applies here: these numbers set the documented budget, so they must
# come from a machine that does not vary between runs.
#
# Usage: infra/measure_build.sh [--figures-only]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Matches build_technical_doc.sh: matplotlib embeds SOURCE_DATE_EPOCH, so a
# figure timed without it is not the figure the build produces.
export SOURCE_DATE_EPOCH=1735689600
export FORCE_SOURCE_DATE=1

figures_only=0
[ "${1:-}" = "--figures-only" ] && figures_only=1

time_it() {
  # Prints "<name> <seconds>" for one command, discarding its output.
  local name="$1"
  shift
  local start end
  start=$(date +%s.%N)
  "$@" >/dev/null 2>&1 || true
  end=$(date +%s.%N)
  printf '%-24s %8.1f\n' "$name" "$(echo "$end - $start" | bc)"
}

# Rendered into a scratch directory: a timing run must not touch the committed
# figures, which are what the staleness check compares against.
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

echo "=== QA figures (seconds each) ==="
figure_total_start=$(date +%s.%N)
# The manifest is the list, so this cannot fall out of step with what the
# build actually runs -- which is how the old hardcoded list came to
# regenerate figures the document had stopped citing.
while read -r stem; do
  time_it "$stem" uv run python -m snakes_and_ladders.qa.build \
    --output-dir "$scratch" --only "$stem"
done < <(uv run python -m snakes_and_ladders.qa.build --all --list)
figure_total_end=$(date +%s.%N)
printf '%-24s %8.1f\n' "ALL FIGURES" \
  "$(echo "$figure_total_end - $figure_total_start" | bc)"

echo
echo "=== cited-only selection (what a pull request rebuilds) ==="
uv run python -m snakes_and_ladders.qa.build --list | tr '\n' ' '
echo
time_it "cited figures" uv run python -m snakes_and_ladders.qa.build \
  --output-dir "$scratch"

echo
echo "=== LaTeX ==="
for document in paper textbook; do
  time_it "latexmk ($document)" bash -c "
    cd docs/tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir=.. -jobname=$document $document.tex"
done

if [ "$figures_only" -eq 0 ]; then
  echo
  echo "=== test suite ==="
  time_it "pytest -m not release" uv run pytest -m "not release" -q
fi
