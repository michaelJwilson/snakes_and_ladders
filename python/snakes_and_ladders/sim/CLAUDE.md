# sim/

Generation: data drawn from a declared model under a declared generator, with
the truth that produced it retained beside it. Nothing here performs inference.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file too — and every docstring, comment and commit message
in this module. It is referenced here, never restated. What follows is local,
and is principle: the numbers behind each rule live with the code that
produces them or in `STATUS.md`, and the module docstrings say which.

## What lives here

Substitution models on a tree; a general undirected graph carrying a per-edge
coupling, the lattices and random graphs built as constructed cases of it, and
the samplers that draw spin configurations on them; hidden state paths and
their emissions; the package's single source of Newick; and `canonical.py`,
the problem instances whose answer is known from outside this repository.

## Local rules

- **A generator, never a seed.** Randomness enters as a generator object
  passed in. Seeding inside a call makes every draw of an ensemble identical,
  which looks like a passing test over many draws and is one draw. That
  mistake has been made here, so the signature is what prevents it.

- **Validate against the analytic result, never against our own likelihood.**
  Where a model has a closed form the technical document states it, and the
  simulator is checked against that. A validation test states its tolerance
  and runs across a range of sizes, because the tolerance is a Monte Carlo
  bound and a single size does not exercise it.

- **A drawn ensemble reaches structures a hand-built fixture does not.** Where
  a property should hold across a *class* of objects, draw the class. The
  boundary cases between a module's code paths are the ones no author thinks
  to write down, and an ensemble finds them.

- **No asymptotic result is testable at the sizes enumeration allows.**
  Threshold and limit results hold as the problem grows; at the sizes an exact
  oracle can referee they say nothing. Check the property *per draw* against
  the oracle, and report the limit for orientation rather than asserting it.

- **A fixture is admitted on two clauses, and both are load-bearing.** Its
  answer must be known from *outside* this repository — a closed form, a
  published result, or an enumeration sharing no code with what it tests — and
  more than one module must consume it. The first stops the suite filling with
  cases that only re-test what already passes. The second decides what belongs
  in `canonical.py` rather than in one module's own file.

- **A known energy is not a known optimum.** A constructed state whose energy
  is known bounds the optimum and survives past the size enumeration reaches;
  claiming it *is* the optimum makes every result built on it wrong. State the
  weaker claim, and measure where the stronger one fails.

- **A fixture proposed as hard is hard only once measured.** Difficulty is a
  property of the instance *and* the baseline, so a construction is not a hard
  case until a baseline has failed on it. Where the measurement says otherwise,
  the finding is recorded and the fixture keeps whatever narrower job it does.

- **Frustration is what an odd cycle buys.** A two-state antiferromagnet wants
  every edge to disagree, which is possible exactly when the graph is
  2-colourable, so every bipartite lattice is unfrustrated and useless as a
  hard case. A geometry containing odd cycles is what makes the ground state
  non-trivial, and where a counting argument closes exactly it fixes that
  ground state at every size.

- **A degenerate fixture proves nothing, and degeneracy is easy to build by
  accident.** A symmetric construction can tie at the optimum, and a test
  distinguishing two answers then measures a tie-break rather than the
  difference it claims. Pin the margin, not only the answer.
