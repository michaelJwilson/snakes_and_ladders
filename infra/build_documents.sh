#!/usr/bin/env bash
# Builds the project's documents: regenerates the QA figures and captions they
# cite, then builds docs/paper.pdf and docs/textbook.pdf. Build tooling, not
# science -- it orchestrates snakes_and_ladders.qa and latexmk, and knows nothing about
# topologies, models, or which fixture renders which figure (see this
# directory's CLAUDE.md, and snakes_and_ladders.qa.manifest for the figures themselves).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Both PDFs are committed, so identical inputs must produce identical
# bytes. Both matplotlib (the QA figures below) and pdftex (the LaTeX build)
# honor SOURCE_DATE_EPOCH, embedding it as their PDF /CreationDate instead of
# the current wall-clock time; a fixed constant keeps every rebuild
# byte-identical regardless of when it runs, so a rebuild that changed
# nothing is an empty diff.
# One home for the constant: snakes_and_ladders.qa.build pins it for the figures, and this
# reads it back so latexmk stamps the PDF with the same clock.
#
# `UV_NO_SYNC`: the environment is synced once (`uv sync --locked`), and a
# build must not re-resolve it. Without it every build rebuilt the Rust
# extension, one to two minutes of a core per worktree (issue #369); the
# variable rather than a flag per command so a script this one calls inherits
# it (issue #372).
export UV_NO_SYNC=1
SOURCE_DATE_EPOCH="$(uv run --no-sync python -c \
  'from snakes_and_ladders.qa.build import SOURCE_DATE_EPOCH; print(SOURCE_DATE_EPOCH)')"
export SOURCE_DATE_EPOCH
# SOURCE_DATE_EPOCH alone fixes the PDF's /CreationDate but not \today, which
# reads pdftex's \year/\month/\day primitives -- those follow the wall clock
# unless FORCE_SOURCE_DATE is set. The title page prints \today, so without
# this the rendered first page changes date every day and a rebuild that
# touched nothing is a diff.
export FORCE_SOURCE_DATE=1

# Which figures exist and what renders each one is
# `snakes_and_ladders.qa.manifest`, not a list here: this script had thirteen
# invocations that nothing connected to the document, so when the document
# stopped citing eleven of them the build kept regenerating all thirteen
# (issue #154). Regenerating only what the documents cite makes the cost track
# them. The rest are checked at the release gate, which runs `--all --check`.
#
# Every document is passed, and that is load-bearing rather than tidy: the
# selection is the *union* of what they cite, so leaving one out would stop
# regenerating its figures and fail nothing (issue #249).
#
# Of the cited figures, only those whose inputs changed since their stamp
# are rendered (issue #372); a change confined to docs/tex/ renders nothing.
uv run --no-sync python -m snakes_and_ladders.qa.build \
  --document docs/tex/paper.tex \
  --document docs/tex/textbook.tex \
  --output-dir docs/tex/figures

# The textbook \inputs docs/tex/generated/problems_tables.tex, which is
# written from PROBLEMS.md and the suite and is not committed (issue #425).
# Regenerated here rather than fetched, so the tables the documents typeset
# are the tree's by construction. This generator and not infra/ledgers.sh:
# the tables are what the documents need, and unlike the seams survey this
# imports nothing from the package, so the documents job needs no extra it
# does not already sync. Cheap beside latexmk -- it reads Markdown, a YAML
# file and the test tree.
uv run --no-sync python infra/problems_tables.py --write

for document in paper textbook; do
  (
    cd docs/tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir=.. -jobname="$document" "$document.tex"
  )
done

echo "Built docs/paper.pdf and docs/textbook.pdf"
