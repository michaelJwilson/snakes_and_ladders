# search/

Discrete optimization: move sets over discrete structures to be optimized, and
exact and approximate solvers. The samplers are `sample/`'s (#777) and the agents
that choose among moves are `learn/`'s (#779).

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and every docstring, comment and commit message in this
module. It is referenced here, never restated. What follows is local.

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

- **A budget is counted in the work it spends, and the seconds are reported
  beside it.** Evaluations for a search, decisions for a learner
  (`learn/CLAUDE.md`), gradients for a sampler (`opt/CLAUDE.md`): the unit is
  what the method spends, since two methods spending different things are not
  ranked by either column alone. Wall-clock is reported and never the budget
  --- it is a property of the host, and this rule read "evaluations and
  seconds" until #803, which made it say the opposite of `learn/`'s.

- **A structure is scored at most once per search**, keyed on a canonical form.

- **A cheap objective is a different surface, not a noisy estimate.**

- **A surface that does not totally order its candidates cannot be measured
  by a statistic that assumes it does.**

- **A fixture whose answer is trivial measures nothing.**

- **An approximation with a bound states the bound and measures the gap.**

- **A certificate states what it actually certifies.**

- **A construction error in a reduction does not break loudly.**

- **A move offered to a learner is the move this package runs, wrapped.**
  `potts_keyed` parameterizes what the action names and writes out the `T = 0`
  limit `beta = 1 / 0` cannot express; it implements no kernel of its own, so
  there is no second physics to keep in step (#706). Above zero the wrapper is
  pinned bitwise against the sweep it delegates to.
