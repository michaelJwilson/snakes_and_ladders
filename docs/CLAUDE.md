# docs/

The technical document (`tex/`), the API documentation (`source/`), and the
worked notebooks (`nb/`). All three are generated artifacts whose output is
committed, CI regenerates each of them and compares against what is committed,
and that is what these rules are about: an artifact held to that contract has
to come out the same on another machine, which constrains what it may say.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle: the numbers behind each rule live with the code that
produces them or in `STATUS.md`.

## What lives here

`tex/` is the document, its bibliography, and the figures, tables and captions
`snakes_and_ladders.qa` generates; a table ships as a fragment the document includes rather
than as an image, so it matches the surrounding type. `source/` is Sphinx,
built from the docstrings. `nb/` is one notebook per problem class, each
running the application from a seeded fixture to a learned policy against
oracles the regression suite already establishes.

The build script regenerates the figures the document cites and then runs
`latexmk`; nothing else invokes `latexmk`. The rebuild is partial by design —
what the document does not cite is regenerated and compared at the release
gate instead, so every committed figure is still checked against the code
that produced it.

## The reader

Root `CLAUDE.md`'s **Expected Reader** states the formatting contract — what
belongs in the body, what belongs in the appendix, and the register to write
in. It is not restated here.

## Local rules

- **Regenerate an artifact; never edit one.** A figure, a table fragment and
  a caption are outputs of the script that produced them. Editing any by hand
  breaks the guarantee the whole arrangement exists for: that what the
  document shows cannot drift from what was measured.

- **The document reads captions and never restates them.** A caption says
  what a figure shows; the body says why it is there. A paragraph describing
  a figure is restating a string it does not own, and the two will diverge.

- **A published number must survive a rebuild on another machine.** Only a
  quantity *continuous* in its inputs may be quoted, because CI byte-compares
  the rebuilt artifact. A rank statistic over an optimizer's output is the
  standing example: perturbing the scores in their last digits moves it, and a
  number that unstable was never a measurement. Before quoting a computed
  value, perturb its inputs and check that what is printed does not change.

- **Machine-dependent numbers stay out.** `DEV.md` forbids ranking
  performance on CI hardware, so a timing belongs in a benchmark. A caption
  gives the structural reason instead — "a full optimization against a single
  pruning pass", not a pair of millisecond figures.

- **A setting that looks global is global.** Scope a typesetting switch to
  the macro that needs it. The failure mode is output-only: the source stays
  correct and reads correctly in review while every rendered line is wrong.

- **A tool that exits zero can still have failed.** `latexmk` returns success
  on a broken reference, so the log is checked instead — for duplicate labels
  as well as undefined ones, since a grep for one sails past the other.

- **A stated invariant with no test is a defect waiting for its next
  archaeologist.** The index claiming to cover every module drifted three
  times before a millisecond test closed it, because the documentation build
  fails on a broken entry and never on an absent one. Where a document claims
  coverage, a test asserts it.

- **Every clock the build can read is pinned.** Creation dates and `\today`
  are separate switches and both are set, in the module that renders the
  artifacts rather than by each caller: a comparison run without them reports
  everything stale, which is indistinguishable from the rot the comparison
  exists to detect.

- **A notebook is under the same contract as the document**, and the
  comparison is over what a cell *printed*. A rendered image embeds metadata
  that is not stable across library builds, so a figure is checked only for
  still being produced; the printed numbers carry the claim.

- **Regenerate a notebook with the checker's own writer**, never by hand and
  never with a second tool. A regenerator that executes a notebook differently
  writes what the checker then rejects, and one that leaves per-cell
  timestamps in buries the change in a diff of clock values.

- **A notebook's Further Work section is load-bearing.** Each names, with its
  issue number, what the notebook could not demonstrate because the feature is
  not built. A notebook that quietly omitted the unbuilt half would read as a
  complete tour of an incomplete repository, and no re-execution check would
  catch it.

## Boundaries

These rules cover how the artifacts are built and kept true, not what they
say; the science is the application modules' business. `qa/CLAUDE.md` owns the
writer's side of the caption contract, and the split is at the file boundary:
`qa/` writes captions, `docs/` reads them.
