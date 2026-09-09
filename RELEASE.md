# Cutting a release

[DEV.md](DEV.md) covers making a change. This file covers cutting a release:
what the cut is taken against, the steps in order, who can run each, and what
each costs. `infra/release.sh` is the gate;
[`.github/ISSUE_TEMPLATE/release.yml`](.github/ISSUE_TEMPLATE/release.yml) is
the ticket that drives it. `CLAUDE.md` remains authoritative: where it and this
file disagree, it wins.

**The costs here are a snapshot, not a promise.** Each is dated, names its
source, and says whether the host was quiet. They come from one 4-core
development host, and the exclusive host lock is not honoured in practice
(issue #496): during one exclusive measurement 2.68 cores were busy of which
1.75 were foreign, and a `with_lock measure` request has queued for its full
1,800 s wait and timed out while three worktrees rendered figures and ran
`pytest` (issue #431). A number taken under contention is an upper bound and
says so.

## What a release is cut against

**One commit of `main`, in one sitting.** The release notes then describe a tree
that exists.

**Not a union of open branches.** 0.5.0 was cut that way on 2026-09-09 — eight
open branches merged in stack order onto `main` at `3c8ab8e` — and was abandoned
before a pull request was opened: while its gate ran, four of those eight landed
and `main` moved to `f1f2c81`. A release cut against a union is stale the moment
part of that union lands, and its notes then describe a tree nobody has (issue
#468; the divergence was priced in advance on issue #467). That is the cost of
the strategy, not of the run.

**Not `dev`.** The template's first precondition assumes `dev` is `main` plus the
open pull requests. It is not: 104 commits `main` does not have, and 270 of
`main`'s it does not (2026-09-09, issue #489).

**A merge into the release branch regenerates; it does not re-render.** Where the
branch takes a merge, the figure stamps and the fixture baselines are rewritten
from the merged tree (`infra/baselines.py --write`, `snakes_and_ladders.inputs`)
rather than the figures re-rendered: at the 0.5.0 cut, 30 stamp and baseline
conflicts across four merges resolved with no figure's bytes moving and no
baseline's value changing — only the recorded digests. Where a file is a list
rather than a generated artifact — a package `__init__`, a module `CLAUDE.md`,
`STATUS.md` — both sides are kept in the file's existing order, and where both
sides *deleted* a line, as `TICKETS.md`'s branches do, the union is of the
deletions. Two hazards a merge carries that the tree does not otherwise show: a textual merge of two branches'
bibliography entries can leave an entry's closing brace missing, which
`infra/check_citations.py` now walks for (issue #503) and a reader caught by hand
before it did; and concurrent branches write colliding experiment numbers that
`git` cannot see, three branches having written `docs/experiments/007` (issue
#491).

## Who does what

| Step | Runs it |
| --- | --- |
| The ledger reading, the audit, the gate, the version bump, `towncrier build`, the pull request | Whoever carries the ticket |
| Adding the `release` label once the gate passes | Maintainer |
| Branch-protection settings, including a required check's name after a job rename | **Maintainer only** |
| Merging the release pull request | Maintainer |
| `git tag v<version> && git push origin v<version>`, and publishing the GitHub release from that tag | Maintainer |

A branch-protection entry is a repository setting, so an agent can read the check
names a pull request reports and say whether they match the required set, and
nothing more. A mismatch is reported, never waited for.

## Preconditions

Each is checked, not assumed. One that does not hold is recorded, with what
follows from it, rather than waited for. Verdicts as of 2026-09-09 (issue #468),
which are the current answers and not a template:

| Precondition | Verdict |
| --- | --- |
| `dev` is `main` plus the open pull requests | **Not met.** Diverged both ways, 104 and 270 commits (issue #489). No release has been built from `dev` |
| The previous release's tag exists | **Not met, and recorded rather than waited for.** `git tag -l` is empty; the repository has never carried a tag, through 0.3.0 and 0.4.0 alike. The baseline an audit reads against is instead the `[0.3.0]` section built into `CHANGELOG.md` on 2026-09-03 |
| Every required check reports under the name branch protection lists | **Met**, confirmed three times independently: `main` took three merges on 2026-09-09 after issue #377's rename landed on 2026-09-08; #463 reported ten check runs including `Documents (paper and textbook)`, all green, with `mergeable_state: clean`; and the same job later reported `skipped` on a pull request without holding it up (issue #503). No owner-only settings change is outstanding. `DEV.md` recorded the opposite until this file took the precondition over |
| The textbook's applicability tables are regenerated (`infra/problems_tables.py --write`) | **Met** at the 0.5.0 cut; the committed table was current |
| The checks ledger and the experiment index are regenerated (`infra/ledgers.sh`, `infra/experiments.py`) | **Met** at the 0.5.0 cut; `infra/ledgers.sh` rewrote nothing tracked |

One required check that reports `skipped` is a check in name only: GitHub counts
`skipped` as a pass, so `Documents (paper and textbook)` — now a push-to-`main`
job — should leave the required set. It blocks nothing while it stays, which is
the opposite failure to a retired name, which blocks everything.

## The steps

### 0. Read the ledgers

Every `TICKETS.md` bullet names the issue that carries it, and every notebook's
**Further Work** line names one. A bullet or line whose issue has closed either
describes work that landed, and goes, or work that remains with nothing behind
it, and is re-pointed at a ticket filed then. Twelve such bullets and nine such
lines were found at the 0.5.0 audit (issue #400). This is a reading, not a
script: it needs the issue tracker and a judgement on whether the work landed.

### 1. The audit

The template's sections, answered in the release pull request rather than on the
ticket: roadmap progress per milestone, taken from `STATUS.md` with the pull
request that moved it rather than re-derived, and an edit to `ROADMAP.md` where
a milestone was reached that it does not describe — in the document's existing
tone, never a rewrite; the consistency audit over
`CLAUDE.md`, `DEV.md`, `README.md`, `INSTALL.md`, `ROADMAP.md`, this file and
`docs/tex/` against the code; one box per problem statement in
`docs/tex/textbook.tex` — count the sections rather than the boxes, since the
template's checklist has been the shorter of the two — whose content is judged
per statement and whose shape
`tests/regression/test_problem_statements_complete.py` already checks; and the
framework table, in which every hand-rolled implementation with an equivalent is
classified `Validation`, `Extension` or `Replacement` under `infra/CLAUDE.md`'s
rule that a framework is a referee until a measurement says it beats what it
fronts. Ten frameworks were measured between 0.3.0 and the 0.5.0 audit and ten
declined, so that table came out all-`Validation`; saying so with the numbers is
the result, not an empty section.

A count a worked-in document restates — the test tiers, the flat module counts,
the figure counts — is re-measured on the audit host and dated, or the audit
records why it was not. An unticked box is an answer.

**State the CI evidence covering what is being released, rather than implying
it.** At the 0.5.0 cut two branches carried ten green checks each that no longer
meant anything: the runs predated their being stacked, and the workflow's
`pull_request: branches: [main]` trigger means nothing will replace them.

The audit's fixes go in the release pull request. What it surfaces beyond them is
filed from the task template, sized to be its own ticket and labelled `blocked`,
never folded in; six were filed from the 0.5.0 audit.

### 2. Run the gate

`infra/release.sh` runs every per-pull-request CI check plus what CI skips per
pull request. It runs every step regardless of earlier failures and prints a
pass/fail summary; a non-zero exit means at least one step failed. Run it under
`infra/locks.sh`'s `with_lock measure`, which asks for the host exclusively —
and read the caveat at the top of this file, because asking is not getting. In
order, with what each costs:

| Step | Cost |
| --- | --- |
| `ruff check`, `ruff format --check`, `mypy --strict` | Under a minute between them; reported as pass/fail at the 0.5.0 run because the host was at load 5–7 |
| `cargo clippy --locked --all-targets -D warnings`, `cargo fmt --check`, `cargo test --locked` | Likewise; both `--features sandbox` variants passed beside them, which is what proves the gated route still compiles |
| `pytest` (every tier, coverage gate) | **~1,100 s** for the CI tier alone, uncontended (1,098 s over 2,121 tests, 2026-09-09, issue #455); the gate adds the `release`, `stress` and `key` tiers on top, and no complete reading exists — see below |
| `sphinx-build -E -a -W` over all 134 modules | **32.2 s** cold, against 8.2 s when nothing changed (issue #451; the step and its flags are issue #485) |
| `infra/ledgers.sh --check` | **7–9 s**, from the review gate's ledger row, which writes rather than compares and so costs about 1 s more (issues #425, #469); read at load 5 to 10, so an upper bound |
| `qa.build --all --check`, every figure against the committed bytes | **431.8 s** declared over the manifest's 23 entries; **~500 s measured** for 18 of them, at 1.1–1.4x declared even at load 8–10 (issue #477) |
| `infra/build_documents.sh` | **15.8 s** with the figure stamps current, up to **305.8 s** with one stale (issue #433), of which citation integrity is **12.0 ms** of work and 53–62 ms of wall clock including interpreter start on a quiet host, and 55 ms at load 9.4 — the check is bounded by file reading rather than by the host (issue #503) |
| `infra/baselines.py` | Unmeasured at the gate. The recomputation it holds cost 8.3 s of uniform rollouts, 7.5 s of maximum-likelihood fits and 1.9 s of exact expected returns per pull request before it moved here (issue #401) |

**Two steps rebuild in full rather than predicting what to rebuild**, and both
predictions failed before they were removed. The figure step passes `--all`,
which ignores the stamps — their false-positive rate is 100% over 476 decisions
(issue #476) — and `--check`, which makes a mismatch a failure naming the figure
rather than a silent refresh. It runs **before** `infra/build_documents.sh`,
because that script renders a stale cited figure into `docs/tex/figures/` and
would otherwise supply the very bytes the comparison is against. The Sphinx step
is `-E -a` for the same reason in miniature: an incremental build's verdict
depends on what `docs/_build/` holds from an earlier branch, while a release
claims all 134 modules are clean. `tests/regression/test_release_gate.py` pins
both flags and the ordering.

**The figure pass is minutes, and its cost is concentrated.**
`topology_accuracy` at 124.0 s and `rl_tree_policy` at 101.4 s are 52.2% of the
431.8 s the manifest declares between them; the median figure is 5.0 s. Sizing
the step at the declared sum is close for 18 of the 23, but **four `opt_*`
figures blow out 8.0x, 8.5x, 24.3x and 79.0x under contention** (issue #477).
That is not generic load — everything else at the same load stayed near 1.2x.
`snakes_and_ladders.qa.build` strips `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
and `MKL_NUM_THREADS` from a render's environment, so a figure that runs a
multi-threaded reduction degrades superlinearly on an oversubscribed host. Budget
the step at roughly 500 s of render on an idle host, plus whatever those four
cost there, which no measurement yet says.

**The full suite is the step without a complete reading.** The one attempt on
2026-09-09 ran past 3,500 s against the CI tier's 1,100 s baseline and was
stopped at 46%, with zero `FAILED` and zero `ERROR`, when `main` moved beneath
it. Plan for the gate to hold the host for hours, and report the run as pass/fail
unless the host was quiet: at that load a timing states the host, not the tree.

**Stopping the gate is sometimes right.** No amount of a run makes a stale cut
fresh, and the run holds the host while it lasts.

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
this — the 0.5.0 audit made the check by hand, reading two files — and issue
#146 records the gap; issue #493 asks the gate to refuse it.

**0.4.0 is never spent, and the gap is deliberate.** `Cargo.toml` reads `0.3.0`
and the top section is `[0.3.0] - 2026-09-03`. The 0.4.0 audit happened, under
issue #358; its *cut* was held, so no `[0.4.0]` section was ever written and no
tag was ever pushed. The fragments from that window are consumed into the next
section built, which is 0.5.0's. A reader who finds `CHANGELOG.md` stepping from
`[0.3.0]` to `[0.5.0]` is looking at that held cut and not at a missing release.

### 4. Build the changelog

Run `uv run towncrier build --version <version>` from the repository root. It
consumes every fragment in `changelog.d/` (129 on 2026-09-09), deletes them, and
inserts a dated `## [<version>]` section into `CHANGELOG.md`; see
`changelog.d/README.md` for a fragment's form, and note that CI's `towncrier
check` is what makes one exist per user-visible change. Commit the result. The
consumed set belongs to the tree the cut is taken from: a section built on one
tree and landed on another describes neither.

### 5. Open the pull request, then tag and publish

The pull request carries the version bump, the built changelog and the audit's
fixes, and says at the top which commit of `main` it was cut from and what the
gate returned on that tree, a step the gate did not reach included. A maintainer
adds the `release` label once the gate has passed, merges, then tags the merge
commit and publishes a GitHub release from that tag with the new `CHANGELOG.md`
section as its body. No tag has ever been pushed here, so this
step is documented and unexercised.

### If the cut is abandoned

Land what does not depend on which tree the release is cut from — the audit's
documentation corrections and the template fixes — as its own small pull request,
and re-cut later. Each correction is re-checked against current `main` rather
than carried over: of seven corrections salvaged from the 0.5.0 cut, one had
already landed elsewhere and re-applying it would have fought a restructure, and
one was damage the union merge itself had created and no defect on `main` at all
(issue #468).

## What a release does not do

* **It does not rebuild the committed PDFs.** A pull request changes
  `docs/tex/` and never `docs/paper.pdf` or `docs/textbook.pdf`; the `lint` job
  and `infra/review_gates.sh` both refuse one that does. The rebuild is its own
  pull request, from a **Rebuild the documents** ticket
  (`.github/ISSUE_TEMPLATE/documents.yml`, issue #483). A release therefore ships
  the PDFs its base carried, and they lag the sources by design.
* **It does not re-execute the notebooks.** `infra/check_notebooks.py` runs one
  kernel per notebook over the six under `docs/nb/`; the `notebooks` CI job runs
  it, and `infra/release.sh` does not.
* **It does not run the audits.** `pip-audit` and `cargo audit` are the `audit`
  CI job's, network-bound, and skipped there on a cache hit when the lockfiles
  are unchanged.
* **It does not change the process, `infra/release.sh`, or any budget.** A
  release documents and exercises what is, including where what is, is slow.

## Two retracted numbers

Both were true of something and false of what they were attached to. Neither may
return.

* **"~6 minutes per figure, 21 minutes worst case."** That was a whole re-stamp
  pass's total read as one render (issue #476). It made the figure step look
  hours-scale when it is minutes, and hours-scale is what argues for predicting
  which figures to render — the prediction that measured 100% false positives.
* **"The review-gate table is measured at 30 s."** 30 s is the *budget*. The only
  readings of the eight-row table are 39 s and 43 s, both at load 8–10 with the
  exclusive lock unavailable, so both are upper bounds; whether the table is
  inside its budget on an idle host is unmeasured (issues #469, #500). A
  measurement that coincides with a budget is still not that budget.
