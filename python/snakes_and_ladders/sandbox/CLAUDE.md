# sandbox/

The oracle home. A hand-rolled implementation that a framework has replaced
on a hot path moves in here, because the replacement is pinned against it and
a deleted oracle is a claim with no referee (issue #322). It is the place root
`CLAUDE.md`'s oracle rule — every accelerated path keeps its reference
implementation — puts a reference once the accelerated path is a library
rather than a backend of our own. A route that lost the measurement it was
built for moves in for the same reason read the other way round: the numbers
that declined it are a claim, and the code is what lets someone run them
again.

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
- **A declined route moves in only if it was finished.** Clean, scoped to a
  question that was posed, and refereed by tests that still pass — an
  abandoned attempt is not conserved, it is dropped. What is conserved is the
  comparison, so the tests come too and the experiment file keeps the numbers.
- **What it costs the default build, it does not cost.** A conserved route
  carries its own dependencies behind a build flag, off by default, so the
  wheel, the per-pull-request jobs and a developer's install pay nothing for a
  route that lost. `DEV.md` names the flag and where it is compiled.
- **Something has to build it on a schedule.** A route nothing compiles stops
  compiling the first time a neighbouring API moves, and the conserved
  comparison is then unrunnable exactly when someone wants to re-run it, which
  is the failure this directory exists to prevent. A flag no gate turns on is
  a deletion with extra steps, and the honest answer in that case is to delete
  the code.
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

Two declined routes, each answering a question its ticket posed and losing on
the answer. `tropical`, the tropical Grassmannian relaxation of topology
search (issue #408), refereed by `tests/regression/test_sandbox_tropical.py`.
And `pruning_burn`, `burn`'s taped gradient over the Felsenstein recursion
(issue #449), whose Rust half is behind the `sandbox` Cargo feature;
`docs/experiments/007-pruning-gradient-routes.md` carries its numbers and
`tests/regression/likelihood/test_pruning_burn.py` referees it, skipping where
the feature is off.

No framework has moved in yet. The ones issue #322 adopted arrived as adapters
and as referees — a Gymnasium adapter over the unchanged `learn.Environment`
protocol, `rustworkx` conversions beside the generators they are checked
against, TorchRL and PyTorch Geometric as unit oracles — and none of them
replaced an implementation on a hot path. The candidates the ticket named are
the Python `search.maxflow` Dinic behind a library minimum cut,
`learn.surrogate.GraphSurrogate`'s message passing behind PyTorch Geometric,
and `learn.ppo`'s advantage estimate and clipped loss behind TorchRL; each
moves in with the measurement that justifies it, and `TICKETS.md` carries
them as follow-ups.
