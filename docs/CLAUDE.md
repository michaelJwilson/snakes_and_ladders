# docs/

The paper and the textbook (`tex/`), the API documentation (`source/`), and
the worked notebooks (`nb/`). All are generated artifacts whose output is
committed. CI regenerates the figures and notebooks whose inputs changed (a
stamp beside each says) and compares them against what is committed; the
PDFs it builds without comparing, since a rebuild ticket's pull request
commits those and they lag their sources between (`DEV.md`). An artifact held
to the comparison must come out the same on another machine, which constrains what it may say.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle: the numbers behind each rule live with the code that
produces them or in `STATUS.md`.

## What lives here

`tex/` is two documents, their shared bibliography, notation and preamble, the
hand-drawn sketches the problem statements input, and the figures, tables and
captions `snakes_and_ladders.qa` generates; a table ships as a fragment a
document includes rather than as an image, so it matches the surrounding type.
`source/` is Sphinx, built from the docstrings. `nb/` is one notebook per
problem class, each running the application from a seeded fixture to a learned
policy against oracles the regression suite already establishes.

The build script regenerates the cited figures whose inputs changed and then
runs `latexmk` per document; nothing else invokes `latexmk`. `DEV.md` holds
the selection and the stamps in full. The principle is that a partial rebuild
*moves* a check and never removes one: at the release gate every figure is
regenerated and compared, whatever a document cites and whatever a stamp
said.

Root `CLAUDE.md`'s **Expected Reader** states the formatting contract — what
belongs in the body, what belongs in the appendix, and the register to write
in. It is not restated here either.

## Local rules

- **Regenerate an artifact; never edit one.** A figure, a table fragment and a
  caption are outputs of the script that produced them, and editing any by hand
  breaks the guarantee the arrangement exists for: that what the document shows
  cannot drift from what was measured. A hand-drawn sketch is not one of them.

- **The document reads captions and never restates them.** A caption says what
  a figure shows and the body why it is there; a paragraph describing a figure
  restates a string it does not own, and the two will diverge.

- **A problem statement carries the same five parts, and labels each.** The
  model, the sizes it is supported at, the model as a factor graph with a
  hand-drawn sketch in its own file, the algorithms cited from the appendix,
  and the validation, one referee at a time with what it does *not* establish.
  Labels rather than headings make that checkable — `sec:<p>:model`,
  `par:<p>:sizes`, an `\input{<p>_figure}` defining `fig:<p>:sketch`,
  `sec:<p>:validation`, one `\ref{alg:...}` — since a section missing a part
  reads as complete and only a guard says otherwise.

- **A published number must survive a rebuild on another machine.** Only a
  quantity *continuous* in its inputs may be quoted, because CI byte-compares
  the rebuilt artifact. A rank statistic over an optimizer's output is the
  standing example: perturbing the scores in their last digits moves it, and a
  number that unstable was never a measurement.

- **Machine-dependent numbers stay out.** `DEV.md` forbids ranking performance
  on CI hardware, so a timing belongs in a benchmark; a caption gives the
  structural reason instead — "a full optimization against a single pruning
  pass", not a pair of millisecond figures.

- **A setting that looks global is global.** Scope a typesetting switch to
  the macro that needs it; the failure mode is output-only, the source reading
  correctly in review while every rendered line is wrong.

- **A tool that exits zero can still have failed.** `latexmk` returns success
  on a broken reference, so the log is checked — for duplicate labels as well
  as undefined ones, a grep for one sailing past the other.

- **A stated invariant with no test is a defect waiting for its next
  archaeologist.** The index claiming to cover every module drifted three
  times before a millisecond test closed it: the build fails on a broken entry
  and never on an absent one. Where a document claims coverage, a test says so.

- **Code cites a label, never a title or a number.** A section title is broken
  by the next retitle and an equation number by the next equation; a `\label`
  survives both and the build resolves it. Every label a docstring or test
  cites is asserted to exist, the code having once cited nine that never did.

- **Every clock the build can read is pinned.** Creation dates and `\today`
  are separate switches, both set in the module that renders the artifacts
  rather than by each caller, so a rebuild that changed nothing is an empty
  diff and not a page of new dates.

- **A document that states an algorithm names no code.** A formulation, a
  recursion or an invariant is true whatever implements it, and a module path
  inside one makes the text stale the next time a file moves. The paper cites
  the textbook for every formulation and the textbook cites nothing here; a
  test asserts it, since the rule is easy to keep and easy to forget.

- **One definition per symbol, shared by every document.** Notation lives in
  one file both inputs, so a symbol cannot mean two things in two places; a
  document needing a new one adds it there rather than defining its own.

- **A notebook is under the same contract as the document**, and the
  comparison is over what a cell *printed*: a rendered image embeds metadata
  that is not stable across library builds, so a figure is checked only for
  still being produced. Regenerate one with the checker's own writer, never by
  hand and never with a second tool — one that executes it differently writes
  what the checker rejects, and one that leaves per-cell timestamps in buries
  the change.

- **A notebook's Further Work section is load-bearing, so its shape is
  checked.** Each names, with its issue number, what the notebook could not
  demonstrate. Omitting the unbuilt half reads as a complete tour of an
  incomplete repository, and re-execution cannot catch it — a markdown cell
  has no output — so the checker fails a notebook whose last cell is not that
  section, or whose line names no ticket.

## Boundaries

These rules cover how the artifacts are built and kept true, not what they say;
the science is the application modules' business. `qa/CLAUDE.md` owns the
writer's side of the caption contract: `qa/` writes captions, `docs/` reads them.
