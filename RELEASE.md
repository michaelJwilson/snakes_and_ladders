# Cutting a release

[DEV.md](DEV.md) covers making a change. This file covers cutting a release:
what the cut is taken against, the steps in order, who runs each, and roughly
what each costs. `infra/release.sh` is the gate;
[`.github/ISSUE_TEMPLATE/release.yml`](.github/ISSUE_TEMPLATE/release.yml) is the
ticket that drives it. `CLAUDE.md` remains authoritative.

**Every number here is a rough upper bound, rounded up**, so a reader can judge
the hours a release commits them to before starting. **None of them is a
baseline**: no later run is measured against a figure in this file. A number
that supports a claim lives in `STATUS.md` beside that claim, and a number that
supports a decision lives on the ticket that measured it. Most were taken on a
4-core development host shared by several agents, which is no longer how it runs
(`DEV.md`, issue #521), so round up again rather than down.

## What a release is cut against

**One commit of `main`, in one sitting.** The release notes then describe a tree
that exists.

**Not a union of open branches.** 0.5.0 was cut that way on 2026-09-09 — eight
open branches merged in stack order — and was abandoned before a pull request
opened: while its gate ran, four of the eight landed and `main` moved beneath
it. A release cut against a union is stale the moment part of that union lands
(issue #468).

**Not `dev`.** The release template's first precondition assumes `dev` is `main`
plus the open pull requests. It is not: 104 commits `main` does not have, and
270 of `main`'s it does not (2026-09-09, issue #489).

**A merge into the release branch now conflicts on prose alone.** It used to
conflict on generated files — at the 0.5.0 cut, 30 figure-stamp and baseline
conflicts across four merges resolved with no figure's bytes moving and no
baseline's value changing — and both classes have since gone: issue #490 deleted
every figure stamp and issue #460 the baseline digest. Where a file is a list
rather than a generated artefact — a package `__init__`, a module `CLAUDE.md`,
`STATUS.md` — both sides are kept in the file's existing order, and where both
sides *deleted* a line, as `TICKETS.md`'s branches do, the union is of the
deletions. Two hazards a merge carries that the tree does not show: a textual
merge of two branches' bibliography entries can leave an entry's closing brace
missing, which `infra/check_citations.py` walks for (issue #503); and concurrent
branches write colliding experiment numbers that `git` cannot see, three
branches having written `docs/experiments/007` (issue #491).

## Who runs what

Time is half the commitment; the other half is which steps wait on a person.

| Step | Runs it |
| --- | --- |
| The ledger reading, the audit, the gate, the version bump, `towncrier build`, opening the pull request | Whoever carries the ticket |
| Adding the `release` label once the gate passes | Maintainer |
| Branch-protection settings, including a required check's name after a job rename | **Maintainer only** |
| Merging the release pull request | Maintainer |
| `git tag v<version> && git push origin v<version>`, and publishing the GitHub release from that tag | Maintainer |

A branch-protection entry is a repository setting: an agent can read the check
names a pull request reports and say whether they match the required set, and
nothing more. A mismatch is reported, never waited for.

## Preconditions

Each is checked, not assumed. One that does not hold is recorded with what
follows from it, rather than waited for. The verdicts below are the current
answers (issue #468).

| Precondition | Verdict |
| --- | --- |
| `dev` is `main` plus the open pull requests | **Not met.** Diverged both ways, 104 and 270 commits (issue #489). No release has been built from `dev` |
| The previous release's tag exists | **Not met, and recorded rather than waited for.** `git tag -l` is empty; the repository has never carried a tag, through 0.3.0 and 0.4.0 alike. The baseline an audit reads against is the `[0.3.0]` section built into `CHANGELOG.md` on 2026-09-03 |
| Every required check reports under the name branch protection lists | **Met**, confirmed three ways: `main` took three merges on 2026-09-09 after issue #377's rename landed; #463 reported ten check runs including `Documents (paper and textbook)`, all green, with `mergeable_state: clean`; and the same job later reported `skipped` on a pull request without holding it up (issue #503) |
| The textbook's applicability tables are regenerated (`infra/problems_tables.py --write`) | **Met** at the 0.5.0 cut |
| The checks ledger and the experiment index are regenerated (`infra/ledgers.sh`, `infra/experiments.py`) | **Met** at the 0.5.0 cut; `infra/ledgers.sh` rewrote nothing tracked |

A required check that reports `skipped` is a check in name only: GitHub counts
`skipped` as a pass, so `Documents (paper and textbook)` — now a push-to-`main`
job — should leave the required set. It blocks nothing while it stays, the
opposite failure to a retired name, which blocks everything.

## The steps

### 0. Read the ledgers

Every `TICKETS.md` bullet names the issue that carries it, and every notebook's
**Further Work** line names one. A bullet or line whose issue has closed either
describes work that landed, and goes, or work that remains with nothing behind
it, and is re-pointed at a ticket filed then. Twelve such bullets and nine such
lines were found at the 0.5.0 audit (issue #400). This is a reading, not a
script: it needs the issue tracker and a judgement.

### 1. The audit

The release template's sections, answered in the release pull request rather
than on the ticket: roadmap progress per milestone, taken from `STATUS.md` with
the pull request that moved it rather than re-derived, and an edit to
`ROADMAP.md` where a milestone was reached that it does not describe — in the
document's existing tone, never a rewrite; the consistency audit over
`CLAUDE.md`, `DEV.md`, `README.md`, `INSTALL.md`, `ROADMAP.md`, this file and
`docs/tex/` against the code; one box per problem statement in
`docs/tex/textbook.tex`, counting the textbook's sections rather than the
template's boxes; and the framework table, in which every hand-rolled
implementation with an equivalent is classified `Validation`, `Extension` or
`Replacement` under `infra/CLAUDE.md`'s rule. Ten frameworks were measured
between 0.3.0 and the 0.5.0 audit and ten declined, so that table came out
all-`Validation`.

A count a worked-in document restates — the test tiers, the flat module counts,
the figure counts — is re-measured on the audit host and dated, or the audit
records why it was not. An unticked box is an answer.

**State the CI evidence covering what is being released, rather than implying
it.** At the 0.5.0 cut two branches carried ten green checks each that no longer
meant anything: the runs predated their being stacked, and the workflow's
`pull_request: branches: [main]` trigger means nothing will replace them.

The audit's fixes go in the release pull request. What it surfaces beyond them
is filed from the task template, sized to be its own ticket and labelled
`blocked`, never folded in; six were filed from the 0.5.0 audit.

### 2. Run the gate

`infra/release.sh` runs every per-pull-request CI check plus what CI skips per
pull request. It runs every step regardless of earlier failures and prints a
pass/fail summary; a non-zero exit means at least one step failed. Nothing has
to be reserved first — the host lock is gone (issue #521) — but the gate holds
the machine for as long as it runs.

**Budget hours, not minutes; two sinks are most of it.**

| Step | Rough upper bound |
| --- | --- |
| `pytest`, every tier, with the coverage gate | **Over an hour**, and never yet run to completion. **The largest sink** |
| `qa.build --all --check`, every figure rendered and compared | **Around 10 minutes** |
| `infra/build_documents.sh` | **Around 10 minutes**, renders included |
| — the two above between them | **The second sink**: the figure work, paid twice, and the part that grows with the manifest |
| `cargo clippy`, `cargo fmt`, `cargo test`, each also `--features sandbox` | **A few minutes** on a cold `cargo` cache |
| `ruff`, `mypy --strict`, `sphinx-build -E -a -W`, `infra/ledgers.sh --check`, `infra/baselines.py` | **Under a minute** each |

The suite is the sink because the gate adds the `release`, `stress` and `key`
tiers to a CI tier already over twenty minutes on its own. The one attempt on
2026-09-09 ran for over an hour, reached 46% with nothing failing, and was
stopped when `main` moved beneath it. Report that run as pass/fail: on a host
this size a timing states the host rather than the tree.

**Stopping the gate is sometimes right.** No amount of a run makes a stale cut
fresh, and the run holds the host while it lasts.

**Two steps rebuild in full rather than predicting what to rebuild**, and both
predictions failed before they were removed. The figure step passes `--all`,
selecting the whole manifest rather than what a document cites; the stamps that
once narrowed it are deleted, on a false-positive rate of 100% over 476
decisions (issues #476, #490). It also passes `--check`, making a mismatch a
failure naming the figure rather than a silent refresh, and runs **before**
`infra/build_documents.sh`, which renders a stale cited figure into
`docs/tex/figures/` and would otherwise supply the bytes the comparison is
against. The Sphinx step is `-E -a` for the same reason: an incremental build's
verdict depends on what `docs/_build/` holds from an earlier branch, while a
release claims all 134 modules are clean. That autodoc records each module as a
dependency, so an incremental build does re-read a *changed* docstring, is
measured rather than assumed — with issue #448's `:cite:` role reintroduced,
the incremental form failed too. `tests/regression/test_release_gate.py` pins
both flags and the ordering.

**The gate renders every figure twice, and both passes are in the table
above.** `qa.build --all --check` compares a rebuild against the committed
bytes without overwriting, and `infra/build_documents.sh` then renders every
figure the two documents cite — since issue #492 the whole manifest — into
`docs/tex/figures/` so `latexmk` has them. The stamps that used to let the
second pass skip most of that work are deleted (issue #490), so a reader
budgeting the gate counts the figure work twice. Whether the second pass can
read the first's output is issue #484's ordering constraint in reverse, not
settled here.

**The figure pass is minutes, and its cost is concentrated.** Two of the
twenty-three figures are half of it and most of the rest are a few seconds each;
`snakes_and_ladders.qa.manifest` carries the per-figure numbers and is the
source, since a guard already reads them (`CITED_RENDER_CAP`). One thing to
expect rather than compute: four `opt_*` figures degrade by up to ~80x on a busy
host (issue #477), where everything else stays near its declared cost.
`snakes_and_ladders.qa.build` strips `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
and `MKL_NUM_THREADS` from a render's environment, so a figure running a
multi-threaded reduction degrades superlinearly on an oversubscribed host.

### 3. Bump the version

Edit `[package].version` in `Cargo.toml` — the single version source
(`CLAUDE.md`), and nowhere else — then run `cargo build` so `Cargo.lock`'s
`oxi_snakes_and_ladders` entry picks the new version up, and commit both.
`maturin` reads the Python package version from that same field
(`dynamic = ["version"]` in `pyproject.toml`), so nothing else is edited.

**A version whose changelog section exists is already spent.** `0.1.0`'s section
was built into `CHANGELOG.md` before the repository was tagged and fragments
accumulated after it; running `towncrier build` at that same version writes a
second section rather than extending the first. So bump past whatever the top
section of `CHANGELOG.md` already carries. `infra/release.sh` does not check
this — the 0.5.0 audit made the check by hand — and issue #146 records the gap;
issue #493 asks the gate to refuse it.

**0.4.0 is never spent, and the gap is deliberate.** `Cargo.toml` reads `0.3.0`
and the top section is `[0.3.0] - 2026-09-03`. The 0.4.0 audit happened, under
issue #358; its *cut* was held, so no `[0.4.0]` section was written and no tag
was pushed. The fragments from that window are consumed into the next section
built, so `CHANGELOG.md` stepping from `[0.3.0]` to a later version is that held
cut and not a missing release. Which version the next cut carries is issue
#522's.

### 4. Build the changelog

Run `uv run towncrier build --version <version>` from the repository root. It
consumes every fragment in `changelog.d/`, deletes them, and inserts a dated
`## [<version>]` section into `CHANGELOG.md`; `changelog.d/README.md` gives a
fragment's form, and CI's `towncrier check` is what makes one exist per
user-visible change. Commit the result. The consumed set belongs to the tree the
cut is taken from: a section built on one tree and landed on another describes
neither.

### 5. Open the pull request, then tag and publish

The pull request carries the version bump, the built changelog and the audit's
fixes, and says at the top which commit of `main` it was cut from and what the
gate returned on that tree, a step the gate did not reach included. A maintainer
adds the `release` label once the gate has passed, merges, then tags the merge
commit and publishes a GitHub release from that tag with the new `CHANGELOG.md`
section as its body. No tag has ever been pushed here, so this step is
unexercised.

### If the cut is abandoned

Land what does not depend on which tree the release is cut from — the audit's
documentation corrections and the template fixes — as its own pull request, and
re-cut later. Each correction is re-checked against current `main` rather than
carried over: of seven corrections salvaged from the 0.5.0 cut, one had already
landed elsewhere and re-applying it would have fought a restructure, and one was
damage the union merge itself had created (issue #468).

## What a release does not do

* **It does not rebuild the committed PDFs.** A pull request changes `docs/tex/`
  and never `docs/paper.pdf` or `docs/textbook.pdf`; the `lint` job and
  `infra/review_gates.sh` both refuse one that does. The rebuild is its own pull
  request, from a **Rebuild the documents** ticket
  (`.github/ISSUE_TEMPLATE/documents.yml`, issue #483). A release ships the PDFs
  its base carried.
* **It does not re-execute the notebooks.** `infra/check_notebooks.py` runs one
  kernel per notebook over the six under `docs/nb/`; the `notebooks` CI job runs
  it, `infra/release.sh` does not.
* **It does not run the audits.** `pip-audit` and `cargo audit` are the `audit`
  CI job's, network-bound, and skipped there on a cache hit when the lockfiles
  are unchanged.
* **It does not change the process, `infra/release.sh`, or any budget.** A
  release documents and exercises what is, including where it is slow.

## Two retracted numbers

Both were true of something and false of what they were attached to. Neither may
return.

* **"~6 minutes per figure, 21 minutes worst case."** A whole re-stamp pass's
  total read as one render (issue #476). It made the figure step look
  hours-scale when it is minutes, and hours-scale is what argues for predicting
  which figures to render — the prediction that measured 100% false positives.
* **"The review-gate table is measured at 30 s."** 30 s is the *budget*. The
  readings that coincided with it were around 40 s on a busy host (issues #469,
  #500). The table has since been read on a quiet one, inside its budget, and
  `DEV.md` carries that figure.
