# sandbox/

Clean, well-scoped solutions to problems that were posed, kept off the hot
path. Two things arrive here and they are the same thing seen from either
side: a hand-rolled implementation a framework replaced, kept because the
replacement is pinned against it and a deleted oracle is a claim with no
referee (issue #322); and a candidate a measurement declined, kept because
the question it answers was well posed and the answer is a result. It is the
place root `CLAUDE.md`'s oracle rule — every accelerated path keeps its
reference implementation — puts a reference once the accelerated path is a
library rather than a backend of our own.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle.

## Local rules

- **An implementation moves in when a measurement lands, not before.** The
  framework has to beat the implementation on the hot path by the margin root
  `CLAUDE.md` sets, or the candidate has to have failed to; both numbers in
  the pull request either way, and the move leaves nothing behind but the
  adapter that fronts the winner. Regression tests move with it and keep
  passing: they are what the surviving path is pinned against.
- **What moves in is a solution, not the work that produced it.** Complete
  enough to read as an implementation rather than as scaffolding. Probes,
  harnesses, half-written experiments and code kept because deleting it feels
  wasteful do not qualify: they are deleted, and what survives is the
  measurement they produced, in the pull request and wherever the result is
  recorded.
- **The problem it answers has to be well posed, and saying so is the
  author's job.** The module's first paragraph states the problem, the
  measured answer and that it is not on a hot path. Where a candidate cannot
  be expressed at the interface it was supposed to replace, restate the
  problem at the level where it can be and say so there — never a wrapper
  that pretends.
- **Only `tests/` and `snakes_and_ladders.qa` import from here.** Never
  `sim/`, `likelihood/`, `opt/`, `search/` or `learn/`, asserted by
  `tests/regression/test_sandbox.py`. A hot path that reaches back into its
  own oracle has not been replaced, and a test that pins a framework against
  a copy the framework's caller also runs pins nothing.
- **Nothing is deleted, once it is here.** An oracle that is slow is still
  the oracle; one nothing tests against is a gap to file as a ticket, not a
  file to remove. Scaffolding is outside the rule: it never moves in.
- **Not re-exported from the package root**, per root `CLAUDE.md`'s Package
  Surface rule. A caller that wants an oracle names it.

## What is here

`pruning_problem.PruningProblem`, the declined answer to "cross the FFI
boundary once per search rather than once per pass" — a `#[pyclass]` holding
the alignment a pruning pass reads. It is here on the second of the two
readings above: the question was well posed and the answer is a result, not
an adoption. `STATUS.md` carries the measurement and the module's first
paragraph states it; its regression tests pin the handle bitwise against
`likelihood.pruning_rust`, which is unchanged and remains the backend.

The frameworks issue #322 adopted arrived as adapters and as referees — a
Gymnasium adapter over the unchanged `learn.Environment` protocol,
`rustworkx` conversions beside the generators they are checked against,
TorchRL and PyTorch Geometric as unit oracles — and none of them
replaced an implementation on a hot path. The candidates the ticket named are
the Python `search.maxflow` Dinic behind a library minimum cut,
`learn.surrogate.GraphSurrogate`'s message passing behind PyTorch Geometric,
and `learn.ppo`'s advantage estimate and clipped loss behind TorchRL; each
moves in with the measurement that justifies it, and `TICKETS.md` carries
them as follow-ups.
