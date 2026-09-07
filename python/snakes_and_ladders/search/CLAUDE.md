# search/

Discrete optimization: move sets over structures, exact and approximate
solvers, samplers, and the agents that choose among moves.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle: the numbers behind each rule live with the code that
produces them or in `STATUS.md`, and the module docstrings say which.

## What lives here

Topology neighbourhoods and the searches that walk them; an exact two-state
ground-state solver by minimum cut and its extension to any label count by
expansion moves; the same model read from its NP-hard side as Max-Cut, with a
relaxation and a certificate; Monte Carlo move sets for the Potts lattice and
the statistics that judge them; the outer inference loop; and this
application's instance of `snakes_and_ladders.learn`'s environment protocol.

Two seams run through it and neither may be reversed. A discrete move changes
the structure being fitted, so the loop builds a new objective rather than
stepping inside a fit — which is why this module may import `snakes_and_ladders.likelihood`
and `snakes_and_ladders.opt` while neither may import it. And the environment lives here
rather than beside the estimator that consumes it, because `learn/` may import
no application module.

## Local rules

- **Discrete tests run where an exact oracle reaches.** Exhaustive
  enumeration is the reference, so the sizes are chosen to keep it available;
  a move test past that size proves nothing extra and costs the budget
  `DEV.md` sets.

- **Neighbourhood generators are verified against counts.** Where a closed
  form for the neighbour count exists the test uses it, and otherwise
  exhaustive enumeration.

- **Connectivity is tested, not assumed** — and reaching every structure is a
  different claim from finding the best one. Only exhaustive search or a
  sound bound gives the second.

- **Every move set states whether it is complete**, in which sense, and what
  it costs per step.

- **A budget is counted in evaluations, never in seconds.** `DEV.md` forbids
  ranking performance on CI hardware, and a wall-clock budget makes a result
  depend on the machine that produced it rather than on its seed. The
  evaluation is also the only unit that matters: generating candidates is
  orders of magnitude cheaper than scoring one.

- **A structure is scored at most once per search**, keyed on a canonical form
  that is independent of how it was spelled, because overlapping
  neighbourhoods otherwise pay the dominant cost twice.

- **Truth is a terminal penalty, never a training signal.** An agent that can
  see the answer during training learns to look it up.

- **A cheap objective is a different surface, not a noisy estimate of the
  expensive one.** Substituting one for the other is licensed by *measuring*
  that they agree where it matters — the argmax — not by a high correlation,
  and the comparison is rerun whenever the fixture changes.

- **A surface that does not totally order its candidates cannot be measured
  by a statistic that assumes it does.** Where candidates tie to within the
  optimizer's convergence, their relative order is not a property of the
  model, so a rank statistic over them is unstable across machines and is not
  a measurement. Report something continuous in the scores.

- **Where a problem crosses from exact to NP-hard, the boundary is refused
  rather than approximated.** A solver that is exact under a stated condition
  raises when the condition fails, because a wrong answer on a structured
  instance is indistinguishable from a right one at any size worth solving.

- **A fixture whose answer is trivial measures nothing.** A uniform field
  makes a ferromagnetic ground state constant; a bipartite graph makes its
  maximum cut every edge. Both admit a solver broken in ways only a harder
  instance exposes, and a benchmark built on either is measuring its own
  fixture.

- **An approximation with a bound states the bound and measures the gap.**
  The bound is the claim that holds at every size; the realized ratio is
  measured beside it and is not quoted as if it were the guarantee.

- **A certificate states what it actually certifies.** Where a guarantee
  assumes a sub-problem is solved to optimality and this repository solves it
  approximately, the certificate is optimistic and the symptom is asserted
  rather than glossed — a case where an exact solve could not produce the
  value that comes back. Where an exact oracle reaches, the realized ratio is
  measured against the true optimum instead, and that is the number to trust.

- **A construction error in a reduction does not break loudly.** A mis-costed
  cut yields a labelling that is merely *worse*, which is indistinguishable
  from a hard instance. Enumeration alone does not catch it: what does is a
  **reduction** — a regime where the general construction must reproduce a
  simpler one already validated, exactly. Any new reduction needs one.

- **A sampler is validated by the distribution it converges to, never by
  inspection.** At an enumerable size the exact distribution is available, so
  a move set is tested by goodness-of-fit against it at a declared
  significance and chain length. A chain that visibly moves is what a sampler
  with a broken accept step also does.

- **A goodness-of-fit test must be thinned, and the thinning is part of the
  test.** Successive sweeps are not independent draws, so run on every sweep
  the test rejects a *correct* sampler. Move sets doing different amounts of
  work per sweep need different thinning, or the comparison rejects whichever
  was thinned less.

- **A sweep must not stop on a state-dependent condition.** Each step
  preserves the target distribution; composing a *number* of them chosen from
  the outcome does not, and the resulting bias favours exactly the states that
  triggered the stop.

- **A cluster move in an external field needs an accept step.** The bond
  construction is exact at zero field only and knows nothing about the field
  term, so without the correction the sampler runs, looks right, and converges
  to the wrong distribution. Every distributional test here therefore runs
  with a field as well as without.
