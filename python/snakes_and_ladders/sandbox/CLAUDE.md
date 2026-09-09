# sandbox/

The oracle home. A hand-rolled implementation that a framework has replaced
on a hot path moves in here, because the replacement is pinned against it and
a deleted oracle is a claim with no referee (issue #322). It is the place root
`CLAUDE.md`'s oracle rule — every accelerated path keeps its reference
implementation — puts a reference once the accelerated path is a library
rather than a backend of our own.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle.

## Local rules

- **An implementation moves in when a measured adoption lands, not before.**
  The framework has to beat the implementation on the hot path by the margin
  root `CLAUDE.md` sets, both numbers in the pull request, and the move leaves
  nothing behind but the adapter that fronts the framework. Its regression
  tests move with it and keep passing: they are what the adapter is pinned
  against.
- **Only `tests/` and `snakes_and_ladders.qa` import from here.** Never
  `sim/`, `likelihood/`, `opt/`, `search/` or `learn/`, asserted by
  `tests/regression/test_sandbox.py`. A hot path that reaches back into its
  own oracle has not been replaced, and a test that pins a framework against
  a copy the framework's caller also runs pins nothing.
- **Nothing is deleted.** An oracle that is slow is still the oracle; an
  oracle nothing tests against is a gap to file as a ticket, not a file to
  remove.
- **Not re-exported from the package root**, per root `CLAUDE.md`'s Package
  Surface rule. A caller that wants an oracle names it.

## What is here

Nothing yet. The frameworks issue #322 adopted arrived as adapters and as
referees — a Gymnasium adapter over the unchanged `learn.Environment`
protocol, `rustworkx` conversions beside the generators they are checked
against, TorchRL and PyTorch Geometric as unit oracles — and none of them
replaced an implementation on a hot path.

The candidates the ticket named have since been measured, and every one of
them lost to the implementation it would have displaced (#388, #389, #390;
`STATUS.md` carries the numbers and
`docs/experiments/008-frameworks-on-three-hot-paths.md` the tables). That is
the rule above working rather than failing: an empty oracle home is what a
repository looks like when its own code is the faster implementation, and the
alternative — moving a reference out to make room for something slower — is
the outcome the measurement exists to prevent.
