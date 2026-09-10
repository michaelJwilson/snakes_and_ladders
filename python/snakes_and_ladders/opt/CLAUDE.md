# opt/

Fits continuous parameters by autodiff/analytic gradient methods. The interface is
model-agnostic by construction: the same machinery serves all supported problems.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and all related work — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local.

## Assumed frameworks

PyTorch.

## Local rules

- **`converged` is a statement about the gradient, never about the global
  minimum.**

- **Where a likelihood is unbounded, refuse rather than clamp.**

- **An interval is a statement about a maximum, and about nothing else.** Not
  the route: an EM fit, a gradient fit and the best of many starts get the same
  interval through one door, so every objective inverts its constraint map. Not
  a point that is no maximum: an unconverged fit, a parameter at a bound and a
  test function's curvature are refused — issue #122's general form.

- **An acceptance rate is not a diagnostic, and adaptation is a warm-up.** A
  step too large biases a posterior's *spread* downward while its mean and
  acceptance look right, so a sampler reports the energy error too. A step or
  mass adapted toward a target is set in a discarded warm-up, opted into and
  reported on the result; the draws are a fixed-parameter chain.

- **A negative log-likelihood is not a log posterior.** Reading a bare
  likelihood as a density is a posterior under an improper flat prior, which
  for most models is not normalizable, and no diagnostic inside a sampler can
  notice.

- **No application imports.** 

- **Constraints by construction, not by projection.**

- **Gauge-fix, or a fitted parameter has no value.**

- **Finite differences are one derivative test that matters here.**

- **Every threshold is relative, inside the optimizer too.**

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

## Discrete moves are in search/
