# sandbox/

The home of the second solution. Two answers to one stated problem, and the
one that is not on the hot path lives here, because a comparison with only
one side left is a claim with no referee (issue #322). The two need not be a
library and our own code: a method declined against a simpler method is the
same case. It is the place root `CLAUDE.md`'s oracle rule — every accelerated
path keeps its reference implementation — puts a reference once the choice is
between two solutions rather than between two backends.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle.

## Local rules

- **What moves in is a solution, not the work that produced it.** A clean,
  well-scoped answer to a problem that was posed, complete enough to read as
  an implementation rather than as scaffolding. Probes, harnesses,
  half-written experiments and code kept because deleting it feels wasteful
  do not qualify: they are deleted, and what survives is the measurement they
  produced, in the pull request and wherever the result is recorded.
- **The problem it answers has to be well posed, and saying so is the
  author's job.** Where a candidate cannot be expressed at the interface it
  was supposed to replace, the honest move is to restate the problem at the
  level where it can be, and to say that in the first paragraph — not to
  write a wrapper that pretends. A solution to a question nobody can state is
  not a second implementation, it is a file.
- **Whichever side lost moves in, and it moves in when the measurement
  lands.** A framework that beat the implementation on the hot path by the
  margin root `CLAUDE.md` sets leaves that implementation here; a framework
  that lost stays here itself. Both numbers go in the pull request either
  way, and the hot path keeps exactly one implementation.
- **A decline is kept as code, not as a paragraph.** The losing side is what
  makes the comparison re-runnable, and a library's next version moves the
  numbers that declined it. So its tests and its benchmark move in with it
  and keep running beside the winner's: a decline nothing re-measures has
  become an assertion about a version nobody is holding.
- **Only `tests/` and `snakes_and_ladders.qa` import from here.** Never
  `sim/`, `likelihood/`, `opt/`, `search/` or `learn/`, asserted by
  `tests/regression/test_sandbox.py`. A hot path that reaches back into its
  own referee has not chosen between the two, and a test that pins a
  framework against a copy the framework's caller also runs pins nothing.
- **Nothing is deleted.** A referee that is slow is still the referee; one
  nothing tests against is a gap to file as a ticket, not a file to remove.
- **Not re-exported from the package root**, per root `CLAUDE.md`'s Package
  Surface rule. A caller that wants a referee names it.

## What is here

Three framework fronts, each measured against the implementation it would
have replaced on a hot path and declined on that measurement (issue #390 for
PyTorch Geometric over `learn.surrogate.GraphSurrogate`, #389 for `rustworkx`
over the Swendsen-Wang labelling, #388 for `scipy.sparse.csgraph` over the
ground-state minimum cut). Each module's docstring names the experiment that
declined it, the ratio and the host, and each arrives with the test that
referees it, so the comparison is re-runnable when a library's next version
moves the numbers rather than settled by a paragraph.

`tropical` is the same rule against no framework at all: the tropical
Grassmannian relaxation of topology search, declined against neighbor joining
(#408). A method declined against a simpler method is the case the header
above names, and it is why the rule is stated over two solutions rather than
over a library and our own code.

The frameworks issue #322 adopted arrived instead as adapters and as referees
on paths nobody proposed replacing — a Gymnasium adapter over the unchanged
`learn.Environment` protocol, `rustworkx` conversions beside the generators
they are checked against — and those stay where they are used.
