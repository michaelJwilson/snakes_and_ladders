# search/

Discrete optimization: move sets over discrete structures to be optimized,
exact and approximate solvers, and samplers. The agents that choose among moves
are `learn/`'s, and issue #779 moved the last of them out of here.

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

- **A move offered to a learner is the move this package runs, wrapped.**
  `potts_keyed` parameterizes what the action names and writes out the `T = 0`
  limit `beta = 1 / 0` cannot express; it implements no kernel of its own, so
  there is no second physics to keep in step (#706). Above zero the wrapper is
  pinned bitwise against the sweep it delegates to.

- **A cluster move is judged by its cluster as well as by its law.** A
  construction can be exact and still buy nothing: where the cluster is the
  lattice the move is a global symmetry, and where the pair's overlap defect
  percolates an exchange between replicas carries them nowhere new. So a
  cluster move reports its mean cluster size beside its chi-square, and an
  instance on which it percolates is reported as measured rather than left to
  the reader to infer from a null result (#756).

- **A structural statistic measured at finite temperature does not referee a
  ground state.** `spatio_only` records a size tilt and a per-class occupancy
  from thermal draws; a ferromagnet's ground state matches neither, and the
  tilt is not even monotone in temperature. A minimizer is refereed by an
  exact optimum where one exists and by a bound where one does not, never by
  a sampler's summary statistic.

- **A ladder placed from a measurement is placed from its noise too.** The
  feedback criterion redistributes rungs on a round-trip statistic, so a
  warm-up too short to resolve that statistic places them where the noise
  fell: at 400 sweeps per measurement the placed ladder ranged from 1,469 to
  5,357 recorded sweeps per round trip against the geometric ladder's 1,690,
  and at 1,000 it landed at 1,280 either way. A placement reports the warm-up
  it was read from, and a criterion is compared against another criterion at
  equal rungs and equal warm-up or not at all (#756).
