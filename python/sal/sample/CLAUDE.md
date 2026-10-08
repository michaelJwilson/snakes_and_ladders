# sample/

Samplers, annealers and tempering: the chains that draw from a distribution and
the schedules that carry them between distributions. Issue #777 moved them here
from `opt/` (continuous: `hmc`, `langevin`, `slice`, `schedule`) and `search/`
(discrete: `potts_mcmc`, `gibbs`, `balanced`, `potts_keyed`; tempering and
annealing: `tempered`, `annealed`; diagnostics: `statistics`), where the rules
for one kind of sampler were stated in one file and the rules for the other in
the other. `metropolis` (#1006) joined the continuous samplers, `tune` chooses a step from a pilot over a grid (#1219), `declared` reads the
kernel an objective names through `supported_gradient` and the operators a compiled chain runs (#1220), and `sample.hmc.jax` runs a
chain on an objective whose energy is a JAX function (#1008). An optimizer is judged by the optimum it reaches and a sampler by
the distribution it converges to, which is why the two directories are two.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and every docstring, comment and commit message in this
module. It is referenced here, never restated. What follows is local.

## Assumed frameworks

NumPy, PyTorch for the continuous samplers that take a gradient, the Rust
sweeps behind `potts_mcmc`. A sampler that reads values alone evaluates
through `opt.objective.energy_of` on arrays and draws from a
`np.random.Generator`; `metropolis` is the first (#1011). A module that
takes no derivative imports no torch: `relabel` takes array-likes and
converts a tensor on entry, found by its `detach` method, so a `Chain`'s
draws reach it without the module importing their type.

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

- **An annealer or a tempering is a `Step` on the shared loop.** `loop.anneal`
  walks a schedule and `loop.temper` runs the exchange (#1218); an entry
  point builds its step and reads the result. A step carries what it scored
  so no loop re-scores a point, and charges its move in its own unit. A loop
  that does not fold --- a walker along the ladder, a re-estimated state ---
  is named in `loop`'s docstring with the reason.

- **Annealing converges by its polisher, never by its schedule (#1363).**
  The schedule runs its full defined length, steps or a `Budget`, with no
  window and no extra hold; a `polish` then runs from the final state to its
  own criterion (an ICM fixed point, a gradient tolerance), and the result's
  `termination` is the polisher's. `spent` holds both, `polish_spent` the
  polish's part. `polish=None` is the unpolished run, bitwise.

- **Python is not the step caller where the step is cheaper than the call.**
  A loop that crosses into a compiled kernel once per step pays an
  interpreter iteration and a crossing per step; where the step is a few
  sites, that is the run. Such a loop runs whole in the compiled backend,
  one crossing per run, and the Python loop stays as the oracle it is
  pinned to (#1368).

- **A tuned step is chosen by a named criterion, and the criterion is not
  the outcome.** `tune_step` ranks a pilot's candidates by the ESJD per
  gradient, the lowest energy reached or the polished gap; on #1195's HMM
  starts the first two ranked the largest step first, which carried the
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
  fell --- at one warm-up the placed ladder's recorded sweeps per round trip
  spanned a factor of three and at a longer one both criteria landed
  together, which `STATUS.md` records under #756. A placement reports the
  warm-up it was read from, and a criterion is compared against another
  criterion at equal rungs and equal warm-up or not at all.

- **A relabelling algorithm has one home: its explicit call (#1305).**
  `relabel.relabel` is a `match` on `RelabelMethod` onto `relabel.stephens`
  and one call per member, the module's function of that name. The generic
  call offers every input to every method and hands each only what it reads;
  a loop control a method does not take is refused.
  `tests/regression/sample/test_relabel.py` holds each member to one case and
  each pair of calls to bitwise-equal relabellings.

- **A ramp has one home: its explicit call (#1333).** `schedule.ramp` is a
  `match` on `ScheduleShape` onto `ramp.<shape>`, and every shape makes some
  `g(T)` linear in some `s(k)` with exact endpoints; `ScheduleParams.build`
  goes through the same `match`. A schedule that reads the chain, or a
  measured `sigma_E(T)`, is not a ramp; the two #1333 built are declined and
  conserved in `sal.sandbox.adaptive_schedules` (#1352).
  `tests/regression/sample/test_schedule_ramps.py` holds each member to an
  explicit call, a case and a `tune_schedule` candidate.
- **Schedule pilot options default off (#1337).** `ScheduleTuning`'s
  `racing`, `common` and `polish` leave a default run bitwise unchanged; under
  `common` each pilot still owns its generator, a deep copy of the round's one
  stream, so a thread pool returns the serial result bitwise.
  `tests/regression/sample/test_schedule_racing.py` holds racing to a
  hand-run halving and the polished rank to the polish applied by hand.
- **A tuned ladder is `"auto"` beside a `LadderTuning` (#1337).**
  `cluster_tempering` resolves it through `tune.resolve_ladder` onto
  `schedule.adapt_ladder`, on
  pilots drawn from one child spawned first; a given ladder draws nothing
  for it. `tests/regression/sample/test_ladder_tuning.py` holds the band.

**One field behaviour on every Potts entry point (#1317).** Every entry point
takes `move` as a move set applied in order and `recolour`: a cluster's
label is drawn by `Recolour.HEAT_BATH` or proposed by `Recolour.UNIFORM`,
resolved once by `move_set`. A move with no heat-bath form refuses
`HEAT_BATH` by name rather than ignoring it. A new entry point without both
fails `tests/regression/sample/test_potts_recolour.py`.

**The defaults are the law-preserving move, not the old one (#1323).** The
default `recolour=Recolour.PER_MOVE` is `HEAT_BATH` for Wolff and
Swendsen-Wang and `UNIFORM` for Niedermayer, ghost-spin and label-directed,
which have no heat-bath form. Composition is read from the type: a bare
`WOLFF` or `SWENDSEN_WANG` runs with a single-site Gibbs sweep per step
(`moves.composed`), a sequence runs exactly as given. Why: the uniform
proposal freezes under a per-site field (#1314: 1 of 320 accepted), and a
cluster move alone is not ergodic under a forbidden label. The move before
#1323 is `move=[PottsMove.WOLFF], recolour=Recolour.UNIFORM`. `step_visits`
sums over the set, so a composed step is charged both sweeps.
`potts_keyed.cluster_moves` keeps keyed Wolff uniform under `PER_MOVE` (its
action names the label) and composes nothing (an arm applies one move). A
caller whose pins record a measured result (`search.ground_state`'s arms)
passes the old move explicitly.

**A schedule is given or tuned, never defaulted (#1317).** Every annealed
Potts entry point takes its schedule (`schedule`, or `betas` for a ladder of
inverse temperatures) as a `TempSchedule` or `"auto"`; `"auto"` requires a
`ScheduleTuning` beside it and is refused without one before any draw, as
`step_size="auto"` requires `StepTuning`. `tune.tune_schedule` is the one
search: one pilot per candidate through `sal.parallel`, each on its own
spawned generator, so a thread pool returns the serial result bitwise.

## Measured mixing limits

A move whose law enumeration pins and whose chain still does not mix on an
instance, as measured; a row is not a defect and the move stays.

A chain reports the limit on its own result (#1316): `PottsChain` carries
`acceptance`, `largest_cluster_share` and `ess` per observable (energy, then
each label's occupancy), and ends `Stop.NOT_MIXING` under `ESS_FLOOR`.
`sample_potts_starts` runs the ordered, drawn and equilibrated starts and
reads split R-hat against `RHAT_THRESHOLD`. On the row below's instance the
ordered chain accepts 1 of 320 moves and every R-hat is 30.4 to 95.9.

| Move | Instance | Measured | Issue |
| --- | --- | --- | --- |
| Niedermayer | 64 x 64 periodic, q = 3, field N(0, 1) per (site, label), 1.5 beta_c, from the ordered start | the cluster holds 4,070 to 4,084 of 4,096 sites, so the transposition moves a field sum of order sigma sqrt(N); 0.0% to 0.3% of 320 moves accepted over three seeds, occupancy (0, 0, 1) against Glauber's (0.299, 0.313, 0.388). From a Glauber-equilibrated start: clusters of 485 to 503 sites, 0.3% to 3.0% accepted. The 2 x 3 law passes (p = 0.133) | #1314 |
