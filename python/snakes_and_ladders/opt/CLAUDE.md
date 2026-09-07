# opt/

Fits continuous parameters by gradient methods. The interface is
model-agnostic by construction: the same machinery serves phylogenetic
trees, HMMs and the Potts model (issue #63), and the phylogenetic case is
one instance of it rather than its author.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle: the numbers behind each rule live with the code that
produces them or in `STATUS.md`, and the module docstrings say which.

## What lives here

`objective.py` is the interface — an unconstrained parameter vector, a
differentiable scalar to minimize, and a map back to named constrained
parameters. `constrain.py` holds the constraint maps every instance shares.

`fit.py` holds the optimizer and the interval machinery: L-BFGS with a
strong-Wolfe line search, and standard errors from the observed Fisher
information pushed through the constraint map by the delta method. The
intervals live here rather than in a test because a standard error computed
in a test is not available to the next instance.

`potts.py` and `hmm.py` are **reference instances**, not applications: a
1-D Potts chain in an external field and a discrete HMM, each with an exact
independent oracle for its own objective. They exist so the interface is
tested against something that is not a tree. The phylogenetic instance
belongs with them, as another instance.

Truth types and data generation live in `snakes_and_ladders.sim`; a reference instance
here imports the type and fits, but draws no data. Issue #186 tracks the last
instance still holding its own generator.

## Framework

**PyTorch**, per root `CLAUDE.md`. What that means here: constraints and the
optimizer below are written against its autograd, and the MPS backend is the
Apple Silicon path the memory requirement in `ROADMAP.md` assumes.

## Local rules

- **`converged` is a statement about the gradient, never about the global
  minimum.** A run satisfying the first-order condition reports convergence
  wherever it stopped, and on a multimodal surface that is routinely a local
  minimum. A result resting on a single fit of such a surface says so, or
  reports a multi-start rate instead.

- **A test function is not a likelihood, so it has no observed information.**
  The Hessian at its minimum is a curvature rather than an information matrix,
  and an interval built from it carries no meaning. Issue #122 covers the
  general form: an interval is refused where the quantity it would summarize
  does not exist.

- **An acceptance rate is not a diagnostic on its own.** An integrator step
  too large biases a posterior's *spread* downward while leaving its mean
  right and its acceptance rate healthy, because divergent trajectories are
  rejected preferentially in the tails. The energy error tracks that and the
  acceptance rate does not, so a sampler here reports both.

- **A negative log-likelihood is not a log posterior.** Reading a bare
  likelihood as a density is a posterior under an improper flat prior, which
  for most models is not normalizable, and no diagnostic inside a sampler can
  notice. `hmc.WithGaussianPrior` makes the prior an explicit declaration by
  the caller rather than an assumption by the sampler.

- **No application imports.** Nothing here may import from `snakes_and_ladders.sim`,
  `snakes_and_ladders.likelihood` or `snakes_and_ladders.search`. This is asserted by
  `tests/regression/test_opt_objective.py`, not left to review: a single
  convenience import turns the abstraction back into a phylogenetics-specific
  optimizer, and neither `ruff` nor `mypy` would notice.
- **Constraints by construction, not by projection.** Branch lengths through
  a log or softplus map, the root distribution through a softmax, rate
  parameters positive through a log map. An optimizer that has to be stopped
  from leaving the feasible set will eventually leave it.
- **Gauge-fix, or a fitted parameter has no value.** Where a
  reparameterization leaves the likelihood unchanged, the direction along it
  is not estimable: the observed information is singular and an interval is
  undefined. The constraint map removes the freedom rather than the optimizer
  tolerating it.
- **Finite differences are the derivative test that matters here.** Root
  `CLAUDE.md` requires the check, and this is the module where a wrong
  derivative surfaces. The comparison is relative to the gradient's norm:
  entrywise relative fails wherever an entry is exactly zero, and absolute
  does not transfer across data sizes.
- **Every threshold is relative, inside the optimizer too.** Convergence is
  the gradient's norm against the objective's own magnitude. A library's
  absolute stopping tolerance halts a summed log-likelihood long before its
  gradient is small relative to the objective, so those are switched off
  rather than trusted — the same failure a test would have with an absolute
  bound, one layer down.
- **A symmetric starting point can be a stationary point.** Where a model's
  parameters are exchangeable, the symmetric point has an exactly zero
  gradient in the exchangeable block and a fit started there never leaves it,
  while appearing to make progress. Starting points break the symmetry
  deterministically, and a test pins the reason.
- **Recovery is the acceptance test.** Fit simulated data with known
  parameters and require the confidence intervals to cover the truth at the
  nominal rate. A likelihood that increases proves the optimizer runs, not
  that the model is right. Where a model has an exact symmetry — permuting an
  HMM's hidden states leaves its likelihood unchanged — recovery is stated up
  to that symmetry, and the alignment is part of the test.

## Discrete moves are outside the interface

A discrete move changes the *structure* — a different topology, chain length
or state count — and so changes what the parameter vector means and how long
it is. It cannot be a step inside a fit over a fixed-length vector: it
constructs a **new** `Objective`. The loop that proposes moves owns that
construction and calls `fit` per candidate.

This is a seam, not a feature to build here. An optimizer that owned the
outer loop would have to know what a move is, which is the model knowledge
this module exists to exclude.
