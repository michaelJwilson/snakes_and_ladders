# sim/

Generation: data drawn from a declared model under a declared generator, with
the truth that produced it retained beside it. Nothing here performs inference.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and every docstring, comment and commit message in this
module. It is referenced here, never restated. What follows is local.

## Local rules

- **A problem that draws data is simulated one way**, `(params, rng) -> dataset`,
  registered under its model's name beside the loader that reads it; a model
  with no simulator says why in the same registry (#829).

- **A generator, never a seed.** Randomness enters as a generator object
  passed in. Seeding inside a call makes every draw of an ensemble identical,
  which looks like a passing test over many draws and is one draw. That
  mistake has been made here, so the signature is what prevents it.

- **Validate against analytic results.**

- **A drawn ensemble reaches structures a hand-built fixture does not.** Where
  a property should hold across a *class* of objects, draw the class. The
  boundary cases between a module's code paths are the ones no author thinks
  to write down, and an ensemble finds them.

- **A supported instance is a fixture, never a literal.** A problem the
  repository supports declares its instance as a fixture, one per size tier, and
  every caller --- test, figure, notebook, experiment --- reads it from there.

- **A declaration and a measurement are two files.** What a *reference*
  algorithm achieves on an instance --- an enumerated optimum, the rate hill
  climbing reaches it from seeded starts --- is a measurement: written by a
  tool, moving when the code that computes it moves, and worth nothing once
  the tree has moved under it. It lives in `<tier>.baseline.json` beside
  `<tier>.yaml`, carrying the seed and budget that produced each number and
  the library versions it was computed against, so a regenerated number never
  rewrites a hand-written instance and a diff says which of the two moved.
  **How closely a recomputation has to agree is part of the measurement**, and
  is recorded with it: a counted or enumerated number is reproduced exactly, a
  number an iterative optimiser produced is held to a stated relative
  tolerance, because the ordering of its floating-point reduction belongs to
  the host and not to the tree (issue #527).
  
  **A record is refereed by recomputing it, never by a hash of the tree it
  came from**: a hash re-keys on every edit in a wide closure and says only
  that the tree moved, where a recomputation says whether the number did.
  `DEV.md` carries what that cost and bought (issue #460).

- **A fixture is admitted on two clauses, and both are load-bearing.** Its
  answer must be known from *outside* this repository — a closed form, a
  published result, or an enumeration sharing no code with what it tests. 

- **The Potts model is one adjacency, one energy and one conditional, and
  they live here.** `PottsGraph.compressed_adjacency` is the only neighbour
  structure --- derived once per graph through
  `sal.incidence.SparseIncidence`, the layout `ParityCheck`
  and `FactorGraph` also hold, and handed to every caller read-only ---
  `energies` the only scorer, `heat_bath_log_weights` the only
  conditional; six modules restated one of the three before issue #277. They
  are in `sim/` because `search/` may import `sim/` and `sim/` may not import
  back, so anything the simulator needs has to be reachable from here. The
  *loops* around the conditional stay separate --- stepped in time,
  vectorized across chains, compiled --- because they are separate.

- **A known energy is not a known optimum.**

- **A fixture proposed as hard is hard only once measured.** Difficulty is a
  property of the instance *and* the baseline, so a construction is not a hard
  case until a baseline has failed on it. Where the measurement says otherwise,
  the finding is recorded and the fixture keeps whatever narrower job it does.

- **A degenerate fixture proves nothing, and degeneracy is easy to build by
  accident.** A symmetric construction can tie at the optimum, and a test
  distinguishing two answers then measures a tie-break rather than the
  difference it claims. Pin the margin, not only the answer.
