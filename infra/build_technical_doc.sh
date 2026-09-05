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
# byte-identical regardless of when it runs, which is what CI's staleness
# check relies on.
# One home for the constant: snakes_and_ladders.qa.build pins it for the figures, and this
# reads it back so latexmk stamps the PDF with the same clock.
SOURCE_DATE_EPOCH="$(uv run python -c \
  'from snakes_and_ladders.qa.build import SOURCE_DATE_EPOCH; print(SOURCE_DATE_EPOCH)')"
export SOURCE_DATE_EPOCH
# SOURCE_DATE_EPOCH alone fixes the PDF's /CreationDate but not \today, which
# reads pdftex's \year/\month/\day primitives -- those follow the wall clock
# unless FORCE_SOURCE_DATE is set. The title page prints \today, so without
# this the rendered first page changes date every day and the staleness check
# below fails on a PR that touched nothing.
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
uv run python -m snakes_and_ladders.qa.build \
  --document docs/tex/paper.tex \
  --document docs/tex/textbook.tex \
  --output-dir docs/tex/figures

for document in paper textbook; do
  (
    cd docs/tex
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir=.. -jobname="$document" "$document.tex"
  )
done

echo "Built docs/paper.pdf and docs/textbook.pdf"
