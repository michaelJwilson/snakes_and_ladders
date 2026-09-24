# sample/

Samplers, annealers and tempering: the chains that draw from a distribution and
the schedules that carry them between distributions. Issue #777 moved them here
from `opt/` (continuous: `hmc`, `langevin`, `slice`, `schedule`) and `search/`
(discrete: `potts_mcmc`, `gibbs`, `balanced`, `potts_keyed`; tempering and
annealing: `tempered`, `annealed`; diagnostics: `statistics`), where the rules
for one kind of sampler were stated in one file and the rules for the other in
the other. `metropolis` (#1006) joined the continuous samplers, `declared` names the
energy families and operators a compiled chain runs, and `hmc_jax` runs a
chain on an objective whose energy is a JAX function (#1008). An optimizer is judged by the optimum it reaches and a sampler by
the distribution it converges to, which is why the two directories are two.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and every docstring, comment and commit message in this
module. It is referenced here, never restated. What follows is local.

## Assumed frameworks

NumPy, PyTorch for the continuous samplers, the Rust sweeps behind
`potts_mcmc`.

## Local rules

- **An acceptance rate is not a diagnostic, and adaptation is a warm-up.** A
  step too large biases a posterior's *spread* downward while its mean and
  acceptance look right, so a sampler reports the energy error too. A step or
  mass adapted toward a target is set in a discarded warm-up, opted into and
  reported on the result; the draws are a fixed-parameter chain.

- **A cost unit belongs to a sampler, not to the comparison.** A gradient is
  an objective evaluation and a backward pass through the same tape, so a
  method that spends gradients and one that spends evaluations are not ranked
  by either column alone. Report each in the unit it spends, name the unit,
  and settle the ranking on what all of them spend.

- **A sampler is validated by the distribution it converges to, never by
  inspection.**

- **A goodness-of-fit test must be thinned, and the thinning is part of the
  test.**

- **A sweep must not stop on a state-dependent condition.**

- **A cluster move in an external field needs an accept step**, and in a
  *per-site* field that step sums the difference over the cluster's own
  members rather than scaling one shared difference by `|C|` (#551).

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
