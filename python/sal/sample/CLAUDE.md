# sample/

Samplers, annealers and tempering: the chains that draw from a distribution and
the schedules that carry them between distributions. Issue #777 moved them here
from `opt/` and `search/`, where the rules for one kind of sampler were stated
in one file and the rules for the other in the other. An optimizer is judged by
the optimum it reaches and a sampler by the distribution it converges to, which
is why the two directories are two. Each module's docstring names what it holds.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and every docstring, comment and commit message in this
module. It is referenced here, never restated. What follows is local.

## Assumed frameworks

NumPy, PyTorch for the continuous samplers that take a gradient, the Rust
sweeps behind `potts_mcmc`. A sampler that reads values alone evaluates
through `opt.objective.energy_of` on arrays and draws from a
`np.random.Generator` (#1011). A module that takes no derivative imports no
torch: it takes array-likes and converts a tensor on entry, found by its
`detach` method, so a `Chain`'s draws reach it without the module importing
their type.

## Local rules

- **An acceptance rate is not a diagnostic, and adaptation is a warm-up.** A
  step too large biases a posterior's *spread* downward while its mean and
  acceptance look right, so a sampler reports the energy error too. A step or
  mass adapted toward a target is set in a discarded warm-up, opted into and
  reported on the result; the draws are a fixed-parameter chain. An annealing
  run is not a stationary chain, so a step re-tuned along it is a heuristic
  and says so.

- **A cost unit belongs to a sampler, not to the comparison.** A gradient is
  an objective evaluation and a backward pass through the same tape, so a
  method that spends gradients and one that spends evaluations are not ranked
  by either column alone. Report each in the unit it spends, name the unit,
  and settle the ranking on what all of them spend.

- **An annealer or a tempering is a `Step` on the shared loop** (#1218). An
  entry point builds its step and reads the result. A step carries what it
  scored so no loop re-scores a point, and charges its move in its own unit.
  A loop that does not fold is named in `loop`'s docstring with the reason.

- **Annealing converges by its polisher, never by its schedule (#1363).**
  The schedule runs its full defined length with no window and no extra
  hold; the polish runs from the best state to its own criterion, and the
  result's `termination` is the polisher's. No polish is the unpolished run,
  bitwise.

- **Python is not the step caller where the step is cheaper than the call.**
  A loop that crosses into a compiled kernel once per step pays an
  interpreter iteration and a crossing per step; where the step is a few
  sites, that is the run. Such a loop runs whole in the compiled backend,
  one crossing per run, and the Python loop stays as the oracle it is
  pinned to (#1368).

- **A tuned step is chosen by a named criterion, and the criterion is not
  the outcome.** A criterion can rank first the step that carries the
  starts out of the truth's basin (#1219). A tuned step is judged against
  the outcome it serves before it replaces a measured one, and a pilot
  shorter than the run cannot see where the run goes (#1251).

- **A sampler is validated by the distribution it converges to, never by
  inspection.**
- **A goodness-of-fit test must be thinned, and the thinning is part of the
  test.**
- **A sweep must not stop on a state-dependent condition.**
- **A cluster move in an external field needs an accept step or an exact
  draw**, and in a *per-site* field either sums over the cluster's own
  members rather than scaling one shared difference by `|C|` (#551, #1142).

- **A cluster move is judged by its cluster as well as by its law.** A
  construction can be exact and still buy nothing: where the cluster is the
  lattice the move is a global symmetry, and where the pair's overlap defect
  percolates an exchange between replicas carries them nowhere new. So a
  cluster move reports its mean cluster size beside its chi-square, and an
  instance on which it percolates is reported as measured (#756).

- **A move that is exact and does not mix stays, and says so.** A chain
  reports the limit on its own result and ends `NOT_MIXING` rather than
  leaving it to the reader (#1316); `STATUS.md` records each measured limit
  (#1314).

- **A structural statistic measured at finite temperature does not referee a
  ground state.** A minimizer is refereed by an exact optimum where one
  exists and by a bound where one does not, never by a sampler's summary
  statistic.

- **A ladder placed from a measurement is placed from its noise too.** A
  warm-up too short to resolve the round-trip statistic places rungs where
  the noise fell (`STATUS.md`, #756). A placement reports the warm-up it was
  read from, and a criterion is compared against another at equal rungs and
  equal warm-up or not at all.

- **An algorithm family has one home: its explicit call.** A generic call is
  a `match` on the family's enum onto one explicit call per member, offers
  every input to every member, hands each only what it reads and refuses a
  control a member does not take (`relabel`, #1305; `schedule.ramp`, #1333).
  A test holds each member to its explicit call and a case. A candidate that
  does not fit the family is declined to `sal.sandbox` (#1352).

- **An option that changes the run defaults off.** A tuning option left at
  its default leaves the run bitwise unchanged, and a parallel pilot owns
  its own generator so a thread pool returns the serial result bitwise
  (#1337).

- **A schedule or ladder is given or tuned, never defaulted (#1317, #1337).**
  `"auto"` requires its tuning object beside it and is refused without one
  before any draw; a given schedule draws nothing for tuning.

- **One field behaviour on every Potts entry point (#1317).** Every entry
  point takes a move set and a recolouring, and a move refuses a recolouring
  it has no form for by name rather than ignoring it.
  `tests/regression/sample/test_potts_recolour.py` fails an entry point
  without both.

- **A default is the law-preserving move, not the old one (#1323).** Where a
  default changes the law a caller's pins recorded, that caller passes the
  old move explicitly and its pins hold bitwise.
