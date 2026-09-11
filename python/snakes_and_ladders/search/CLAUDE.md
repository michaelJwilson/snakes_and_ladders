# search/

Discrete optimization: move sets over discrete structures to be optimized, 
exact and approximate solvers, samplers, and the agents that choose among moves.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and related work — e.g. every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local.

## Local rules

- **Discrete tests run where an exact oracle reaches.** Exhaustive enumeration
  is the reference, so the sizes are chosen to keep it available; a move test
  past that size proves nothing extra and costs the budget `DEV.md` sets.

- **Neighbourhood generators are verified against counts.** Where a closed
  form for the neighbour count exists the test uses it, and otherwise
  exhaustive enumeration.

- **Connectivity is tested, not assumed** — and reaching every structure is a
  different claim from finding the best one. Only exhaustive search or a
  sound bound gives the second.

- **Every move set states whether it is complete**, in which sense, and what
  it costs per step. A surrogate decides what is fitted, never what is reported.

- **A budget is counted in evaluations and seconds.**

- **A structure is scored at most once per search**, keyed on a canonical form.

- **Truth is a terminal penalty, never a training signal.** An agent that can
  see the answer during training learns to look it up.

- **A move's features are what it already computes without a fit, and none
  may be constant across the neighbourhood.** The softmax cancels a constant
  (`learn/CLAUDE.md`), so a distance from the state under NNI is excluded.

- **A cheap objective is a different surface, not a noisy estimate.**

- **A surface that does not totally order its candidates cannot be measured
  by a statistic that assumes it does.**

- **A fixture whose answer is trivial measures nothing.**

- **An approximation with a bound states the bound and measures the gap.**

- **A certificate states what it actually certifies.**

- **A construction error in a reduction does not break loudly.**

- **A sampler is validated by the distribution it converges to, never by
  inspection.**

- **A goodness-of-fit test must be thinned, and the thinning is part of the
  test.**

- **A sweep must not stop on a state-dependent condition.**

- **A cluster move in an external field needs an accept step**, and in a
  *per-site* field that step sums the difference over the cluster's own
  members rather than scaling one shared difference by `|C|` (#551).

- **A structural statistic measured at finite temperature does not referee a
  ground state.** `potts_spots` records a size tilt and a per-class occupancy
  from thermal draws; a ferromagnet's ground state matches neither, and the
  tilt is not even monotone in temperature. A minimizer is refereed by an
  exact optimum where one exists and by a bound where one does not, never by
  a sampler's summary statistic.
