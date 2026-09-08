# likelihood/

Evaluators: what a model says about data, on a tree, a graph or a chain. This
is the hottest path in the project, since every proposed move costs at least
one evaluation and search proposes many.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle: the numbers behind each rule live with the code that
produces them, and the module docstrings say which.

## What lives here

The pruning recursion and its backends, an exact and an approximate evaluator
for the Potts MRF, Fitch and Sankoff parsimony as a second criterion over the
same topology, path enumeration for the HMM, one sum-product over any factor graph
that is pinned to each of those where they overlap, the enumeration oracle for
the coupled model with its E step and field, forward–backward as an
evaluator, a parity-check decoder specialised from that sum-product and
held to it, and the adapter that presents any of them to `snakes_and_ladders.opt`'s fitting interface. That adapter lives here and not in
`opt/`, because `opt/` may import no application module — the dependency runs
application to infrastructure, never back.

## Local rules

- **The reference implementation is the oracle and it stays.** Every
  accelerated backend is pinned against it. Deleting the slow path to "clean
  up" removes the only thing that says the fast path is right.

- **Correctness comes from brute force, not from another backend.** Direct
  marginalization over the hidden states is the test. Two backends agreeing
  proves nothing if both are wrong.

- **Cross-device agreement is a relative tolerance keyed on the lowest
  precision in the comparison.** It is relative because a log-likelihood is a
  sum over sites, so an absolute bound fixed at one problem size does not
  transfer to another; it is keyed on the lowest precision because one bound
  loose enough for `float32` would let a broken `float64` backend pass.
  `device.py` states both tolerances and the measurements they are derived
  from, and every comparison reads them from there rather than retyping them.

- **A boundary cost hides behind the fixture's size, never behind the
  algorithm's.** A backend measured only where the existing fixtures sit
  reports its kernel. A cost paid per element crossing into it grows with the
  problem instead, so it is invisible at a fixture chosen for a different
  reason and decisive at the scale the roadmap declares. Every accelerated
  path is therefore measured at both, because the two answer different
  questions and this repository has had them disagree.

- **A backend is accepted or rejected against that tolerance, never adjusted
  until it matches.** A discrepancy inside the bound is not a bug and must not
  be "fixed"; one outside it is not a tolerance to loosen.

- **An approximate evaluator states which regime carries its correctness.**
  Where it is exact, equality against enumeration is asserted. Where it is
  approximate, the deviation is *reported*: asserting agreement would assert
  something false, and asserting only that it ran is the coverage theatre root
  `CLAUDE.md` forbids. What may be asserted in the approximate regime is
  structure fixed independently of this implementation — a limit where the
  approximation is exact, or an ordering physics predicts.

- **A density is not a probability, and only one of them is bounded.** The
  evidence of a model over a countable support is a probability, so its
  logarithm is at most zero; the evidence of one over the reals is a density
  and carries no such bound. A check written against the first fails on
  correct code under the second, so what may be asserted is keyed on the
  support the model declares, never assumed from the discrete case that
  happened to come first.

- **Refuse rather than return an unconverged number.** A quantity read off
  iterations that never settled is not an estimate of anything, and a caller
  cannot tell it from one that is. The same reasoning refuses any parameter
  setting that makes convergence vacuous rather than achieved.

- **An oracle whose cost scales like the thing it referees stops refereeing.**
  Enumeration is exponential in problem size; a second exact method that is
  exponential in some smaller dimension reaches instances enumeration cannot,
  and is itself pinned twice — against enumeration where both fit, and by
  reduction to a case with a closed form.

- **A criterion may be here to be wrong.** A method this repository does not
  advocate earns its place where its failure is a *theorem* rather than a
  defect. Its correctness test is then paired: it must fail where theory says
  it fails **and** succeed where theory says it succeeds, because an
  implementation that is simply broken fails both and the first result alone
  cannot tell the two apart.

- **Two decodings of one model are different answers, not approximations of
  each other.** A maximum over paths and a per-site maximum of marginals
  differ, and the second can return something the model assigns no path to.
  They agree on almost every fixture, which is exactly why a decoder that
  computes one and reports the other survives a suite with no case where they
  diverge; the fixture that separates them is the test.

- **Rescaling must stay differentiable, and the scaling factor is a
  constant.** Partial likelihoods underflow, so they are rescaled with the log
  of the scaling accumulated separately, and that transformation sits inside
  the autodiff graph. The factor itself does not: it cancels between the
  division and the log it is added to, so it carries no derivative, and
  differentiating it buys a backward pass that computes zero. What referees
  this is the gradient of the rescaled path against the unrescaled one, not an
  inspection of the graph.

- **Memoize on the canonical form.** A topology has many spellings; keying a
  cache on a raw string silently recomputes what has already been scored.

- **Fit only what is estimable.** Where a model has an exactly flat direction,
  the parameters along it are confounded and the fit reports the combination
  that is identified rather than the parts that are not. Recovering a part
  afterwards is a drawing convention, never an estimate.

- **Differentiable backends keep structure and parameters apart.** Branch
  lengths reach the autodiff backend as a tensor, never read back off the
  topology object, so gradients flow through the tensor and never through
  Python floats baked into a structure.

- **A surrogate carries its claim, and the claim is checked.** A bound is
  proved in the textbook and certified against the exact value on every
  structure an oracle scores, with no violation allowed; a learned predictor
  claims a coverage and is certified to it. A cheap value that claims nothing
  ranks, and is never reported as the evaluation it stands in for.
