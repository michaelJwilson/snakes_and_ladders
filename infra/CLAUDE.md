# infra/

CI/CD, the agentic workflow, and experiment tracking. No scientific application
references live here; everything that efficiently developed the science and supports
proof that it is valid does.

`select_tests.py` decides which tests a pull request needs and what to measure
coverage over, from the files it changed. It reads the import graph rather than
a list of dependents, and answers "everything" for any change it cannot
attribute to one module — a lockfile, a shared fixture, this directory. A
selection that guesses narrowly is a test that silently did not run, so the
unsafe answer is the one that looks like a saving. Nothing bounds that answer:
a bound and the fixed timeout that reads it cannot coexist, because the
timeout then cancels the pull requests the bound refused and a cancelled job
is indistinguishable from a failing one.

A cache decides *when* work is redone; it never decides whether a check runs.
The two read the same way in a script and differ in what a wrong answer costs:
a skipped rebuild is redone next time, and a skipped check is a claim nobody
made. What bounds an expensive check is a stated budget and a tier, never a
hash — and a cache is kept only while a measurement says it saves something,
because one that is wrong on every decision it makes costs the renders it
predicted and buys nothing. `DEV.md` carries what each check costs and what it
buys.

A worktree is created by `new_worktree.sh` rather than by a convention in a
brief. A step a person repeats is a step a person skips, and each of the ones
it replaces had already been skipped.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle: `DEV.md` holds the CI budget, the repository settings and
the CI jobs, including the size caps that keep an exact oracle available, and
is not restated here.

A run is read back from its log, so every line an entry point writes carries
the elapsed time and the phase the run was in; libraries emit through the
standard logger and never configure it, entry points configure it once.
`DEV.md` shows the line and names the module.

A measurement against a baseline that is not in an experiment file with its
commit and its budget is an anecdote; the Markdown under `docs/experiments/`
is the ledger, a run store is its source, and the index is generated from
the files, never edited.

A recorded number is compared to a recomputation of it at the precision the
number has, never bitwise. A count, an enumeration and a rate over seeded
draws reproduce exactly and are held to that; a quantity an iterative
optimiser produced does not, because the ordering of its floating-point
reduction belongs to the host, and holding it to exact equality fails a
change that changed nothing --- whereupon the reflex is to rewrite the
record, which accepts whatever ran that day. The tolerance is therefore
stated beside the number, derived from a measurement of the spread and not
chosen, relative and keyed to the lowest precision in the comparison (root
`CLAUDE.md`), and tight enough that a change worth catching still fails.

A framework is adopted the way a backend is: it fronts a hot path once a
measurement says it beats the implementation there, and that implementation
moves to `python/snakes_and_ladders/sandbox/` and referees it from there
(`sandbox/CLAUDE.md`). Until that measurement exists a framework is a
referee, not a replacement — a second implementation our numbers are pinned
against, reached through an adapter that leaves the interface it wraps
unchanged. Every such package lives in the `frameworks` extra, and every test
that needs one skips without it, so the core install carries none of them.

## Tickets and the approval flow

- **Priorities.** `high` runs immediately; `medium` runs at the next
  scheduled slot when tokens refresh; `low` runs outside 09:00–17:00
  Princeton time and is the default.
- **Tag every ticket by submodule**, so batches stage against the roadmap
  rather than sprawling across it.
- **`/approve` by a maintainer** opens a pull request, and may do so
  unattended provided the PR implements a plan already posted to the thread.
  A plan that turns out to be flawed gets a revised plan posted in the
  thread, not a silent correction.
- **Record the branch before the first commit.** Once a branch is created for
  an approved plan, the first thing posted is a single issue comment naming
  the branch (and, once opened, the PR number) — before any further commit is
  pushed. A session that then fails, is interrupted, or is deferred
  mid-implementation leaves a ticket that already points at the in-flight
  branch, instead of requiring a search across open branches and PRs to find
  it.
- **A plan has a required shape**, stated in `ROADMAP.md` §0.2 and not
  restated here: a plan that does not carry its own validation is a proposal
  to find out later whether the work was right.
- **A plan is subject to the Writing Style**, like everything else written
  here. Referenced, not restated: root `CLAUDE.md` holds it.
- **One implementation agent per four cores**, the document build counted as
  a process, and an agent waits for its own long job in the foreground: a job
  it detaches ends its turn and nothing resumes it (issue #369).
