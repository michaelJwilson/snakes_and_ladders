# STATUS

What has landed against `ROADMAP.md`, how it was established, and the pull
request that carries it. Read at version `0.3.0`.

This file is a ledger against the roadmap, not a project board. Open work lives
in GitHub issues, and its titles are collected in `TICKETS.md`. A milestone is
recorded here as **landed** only where an independent oracle pins the claim; a
capability that runs but is checked against nothing is recorded as **not
started**, on the terms §0.4 sets.

## Summary

The checks each row rests on are listed, per test, in `CHECKS.md`, generated
from the suite; `PROBLEMS.md` names the code behind each problem class.

| Roadmap item | Status | Evidence | Key PRs |
| --- | --- | --- | --- |
| §0 Development loop | Landed | Ten required checks; committed PDF byte-compared and every notebook re-executed on each PR | [#49](https://github.com/michaelJwilson/snakes_and_ladders/pull/49), [#57](https://github.com/michaelJwilson/snakes_and_ladders/pull/57), [#72](https://github.com/michaelJwilson/snakes_and_ladders/pull/72), [#92](https://github.com/michaelJwilson/snakes_and_ladders/pull/92), [#102](https://github.com/michaelJwilson/snakes_and_ladders/pull/102), [#151](https://github.com/michaelJwilson/snakes_and_ladders/pull/151) |
| 1.1 Simulation & ground truth | Trees, the HMM and Potts (1-D chain plus general N-D lattice/MRF) landed as first-class simulators | Simulated substitution frequencies against the closed-form JC probabilities; GTR reproduces JC to machine precision; HMM state and emission marginals against brute-force path enumeration; Potts single-site and pair marginals against exhaustive enumeration at 3-state 3x3 and 2-state 4x4 | [#58](https://github.com/michaelJwilson/snakes_and_ladders/pull/58), [#64](https://github.com/michaelJwilson/snakes_and_ladders/pull/64), [#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115), [#120](https://github.com/michaelJwilson/snakes_and_ladders/pull/120), [#182](https://github.com/michaelJwilson/snakes_and_ladders/pull/182), [#190](https://github.com/michaelJwilson/snakes_and_ladders/pull/190) |
| 1.2 Likelihood & energy engine | CPU landed (NumPy, PyTorch, Rust); belief propagation landed with two exact oracles; GPU dispatch not started | Worst relative deviation 4.0e-14 against brute-force marginalization across three backends and four site counts spanning a factor of 30 | [#66](https://github.com/michaelJwilson/snakes_and_ladders/pull/66), [#74](https://github.com/michaelJwilson/snakes_and_ladders/pull/74), [#81](https://github.com/michaelJwilson/snakes_and_ladders/pull/81), [#112](https://github.com/michaelJwilson/snakes_and_ladders/pull/112), [#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148) |
| 1.3 Continuous optimization | Landed for trees, the HMM, the 1-D Potts chain and the 2-D lattice; posterior sampling landed beside the curvature-based intervals | Gradients against central differences; 95% intervals cover truth at the nominal rate over 60 replicates; the lattice fitted against an enumerated normalizer, coverage 157/160 at 100 samples and 153/160 at 400 and 1600 | [#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115), [#116](https://github.com/michaelJwilson/snakes_and_ladders/pull/116), [#119](https://github.com/michaelJwilson/snakes_and_ladders/pull/119), [#120](https://github.com/michaelJwilson/snakes_and_ladders/pull/120) |
| 1.4 Move sets & classical baselines | Trees landed; Potts cluster updates landed; the exact-baseline family landed — minimum cut, alpha expansion with its proved bound, and Max-Cut with a certificate; Viterbi not started | NNI and SPR neighbour counts exhaustively verified at `n = 5..8`; hill climbing reaches the enumerated optimum from 12 of 12 starts; the two-state ground state exact against enumeration over 36 shape-coupling-field combinations | [#82](https://github.com/michaelJwilson/snakes_and_ladders/pull/82), [#127](https://github.com/michaelJwilson/snakes_and_ladders/pull/127), [#128](https://github.com/michaelJwilson/snakes_and_ladders/pull/128), [#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148), [#212](https://github.com/michaelJwilson/snakes_and_ladders/pull/212) |
| 2.1 RL formulation & deployment | Estimator and both environments landed; a trained tree policy not started | Enumerated gradient against finite differences at 1.5e-11 relative; learned policy 86.6% against greedy's 80.2% on the Potts landscape, 8 of 8 seeds | [#135](https://github.com/michaelJwilson/snakes_and_ladders/pull/135), [#137](https://github.com/michaelJwilson/snakes_and_ladders/pull/137), [#139](https://github.com/michaelJwilson/snakes_and_ladders/pull/139) |
| 2.2 Curriculum learning | Not started | — | — |
| 2.3 Empirical validation | Not started | — | — |
| 2.4 Tracking, ablations & leaderboard | Not started | — | — |
| Stage 3 Research extensions | Gumbel-softmax relaxation of Potts and HMM states landed; the tropical Grassmannian half not started, and blocked on an oracle | Relaxation exact at every corner to 1e-11; estimator bias 0.598 to 0.036 as `tau` falls 2.0 to 0.1, standard deviation 0.165 to 3.39 over 20000 draws; deterministic ascent 18/40 against greedy's 5/40, McNemar `p = 0.00098` | [#225](https://github.com/michaelJwilson/snakes_and_ladders/pull/225) |

## §0 The Development Loop

The loop described in `ROADMAP.md` §0 is in force rather than aspirational.
Blank issues are disabled and filings route through the task template
([#57](https://github.com/michaelJwilson/snakes_and_ladders/pull/57)); the pull-request
template carries the Definition of Done, the benchmark table, the realized
tolerance table, and the deferred-work section
([#49](https://github.com/michaelJwilson/snakes_and_ladders/pull/49),
[#86](https://github.com/michaelJwilson/snakes_and_ladders/pull/86),
[#89](https://github.com/michaelJwilson/snakes_and_ladders/pull/89)); labels are generated
from `.github/labels.yml` by a workflow, so the taxonomy cannot drift from the
documents that describe it.

Ten required checks gate a merge, and three of them do work no reviewer can
do by inspection: the technical-document job rebuilds only the QA figures
the documents under `docs/tex/` cite, comparing the rest at the release gate instead
([#157](https://github.com/michaelJwilson/snakes_and_ladders/pull/157)), and fails a pull
request whose rebuilt `docs/paper.pdf` or `docs/textbook.pdf` differs from the committed one
([#72](https://github.com/michaelJwilson/snakes_and_ladders/pull/72)); the notebooks job
re-executes every notebook under `docs/nb/`, fails one whose Further work
section is missing or names no ticket
([#278](https://github.com/michaelJwilson/snakes_and_ladders/issues/278)), and fails one whose printed
output has moved; and the coverage floor cannot be lowered to pass a change. Cost is managed rather than absorbed:
benchmarks run only when the diff touches code they measure, and the
release-gated suite is excluded per pull request — measured at 138 s over 540
tests against 989 s for the full suite
([#159](https://github.com/michaelJwilson/snakes_and_ladders/pull/159)).

Two releases have been cut under the procedure, each from a Release ticket
gated on `infra/release.sh`: `0.1.0`
([#102](https://github.com/michaelJwilson/snakes_and_ladders/pull/102)) and `0.2.0`
([#151](https://github.com/michaelJwilson/snakes_and_ladders/pull/151)). Each ran the
consolidation audit the template drives, and `0.2.0`'s found real defects — a
categorical sampler duplicated three times, two copies missing the clamp the
third had, so a probability row summing to `1 - 4e-16` could return a category
past the end of the alphabet.

**Between `0.2.0` and `0.3.0`, six pull requests refined the loop and its
record; no roadmap milestone moved.** `ROADMAP.md` was restructured around the
development loop and the three problem classes, and `STATUS.md` and
`TICKETS.md` were introduced as the ledger and backlog this section and
`TICKETS.md` now are
([#152](https://github.com/michaelJwilson/snakes_and_ladders/pull/152),
[#153](https://github.com/michaelJwilson/snakes_and_ladders/pull/153)). The thirteen QA
scripts were routed through one `snakes_and_ladders.qa.runner` rather than each carrying
its own argument parsing and figure-closing boilerplate
([#156](https://github.com/michaelJwilson/snakes_and_ladders/pull/156)), and
`snakes_and_ladders.qa.manifest` now states which figure renders each output so a build can
select a subset rather than regenerate all thirteen
([#157](https://github.com/michaelJwilson/snakes_and_ladders/pull/157)). The regression
suite was split by submodule and its documented budget corrected after being
found stale
([#159](https://github.com/michaelJwilson/snakes_and_ladders/pull/159)). Every module
`CLAUDE.md` now points at the Writing Style section instead of restating it
([#158](https://github.com/michaelJwilson/snakes_and_ladders/pull/158)), and a generated
plan's required shape — 2–5 validated steps ending in an Open Questions
section — is stated in `ROADMAP.md` §0.2, `DEV.md`, and `infra/CLAUDE.md`
alike, alongside the rule that decides which documents may repeat detail
([#164](https://github.com/michaelJwilson/snakes_and_ladders/pull/164)).

## Milestone 1.1 — Simulation & Ground Truth Engine

**Phylogenetics: landed.** A `k`-state Jukes-Cantor simulator generates an
alignment and the ancestral tree in Newick from a typed
`simulation_params.yaml`, retaining the parameters that produced them
([#58](https://github.com/michaelJwilson/snakes_and_ladders/pull/58)). Simulated
substitution frequencies are validated against the closed-form JC transition
probabilities within a yaml-declared Monte Carlo tolerance across several site
and taxon counts. Newick counting, validation and state-labelled serialization
are the package's single source of that functionality
([#64](https://github.com/michaelJwilson/snakes_and_ladders/pull/64)).

The general time-reversible model landed with the fitting work that needed it
([#120](https://github.com/michaelJwilson/snakes_and_ladders/pull/120)): Jukes-Cantor has no
free parameters, so there was nothing to recover without it. It is validated by
reduction — equal exchangeabilities with a uniform `π` reproduce the
Jukes-Cantor rate matrix and its closed-form transition probabilities to
machine precision.

**Potts: 1-D chain plus a general N-D lattice/MRF simulator.** The 1-D chain
in an external field still exists as an `opt` reference instance with an
exact transfer-matrix oracle ([#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115)),
and appears again as a `learn` environment. `snakes_and_ladders.sim.graph.PottsGraph`
now generalizes it to an arbitrary undirected graph with a per-edge
coupling, and `snakes_and_ladders.sim.potts.simulate_potts` samples on it — exactly, by
the same backward-message recursion, when the graph is a 1-D open chain, and
by single-site Gibbs (heat-bath) MCMC otherwise — with an N-D lattice a
constructed case of the general graph rather than a second code path
([#190](https://github.com/michaelJwilson/snakes_and_ladders/pull/190), closing #170,
superseding the
sampling half of #149). `snakes_and_ladders.opt.potts.simulate_chains` cannot import
`snakes_and_ladders.sim` under `opt/CLAUDE.md`'s "no application imports" rule, so it
keeps its own copy of the exact recursion rather than delegating to the new
one — a duplication [#186](https://github.com/michaelJwilson/snakes_and_ladders/issues/186)
tracks resolving, by moving `PottsParams` into `snakes_and_ladders.sim.potts` the way
#171 moved the HMM's truth type. No fitting, cluster updates, or evaluator
on the general graph yet (issues #172, #174).

**HMMs: a first-class simulator.** `snakes_and_ladders.sim.hmm` draws a hidden state path
and an observation sequence jointly from a declared `(pi, A, B)`, retaining
the path alongside the data on the footing the tree simulator already has
([#182](https://github.com/michaelJwilson/snakes_and_ladders/pull/182), closing
[#171](https://github.com/michaelJwilson/snakes_and_ladders/issues/171)). The generator
embedded in `snakes_and_ladders.opt.hmm` — which validated only against brute-force path
enumeration for the fitting objective's own use
([#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115)) — is deleted; `opt`
now imports the truth type from `sim` and draws no data itself. Validated
against brute-force enumeration for the per-position state and emission
marginals, self-normalized importance sampling against the exact path
posterior for one realized observation, and the transition matrix's own
stationary distribution for long-run occupancy.

**Emission families: what a state emits, separated from how it is fitted.**
`snakes_and_ladders.emissions` holds the interface — draw, score, re-estimate —
with the categorical matrix one implementation of it and a univariate Gaussian
the second; the simulator, the forward recursion, Baum-Welch, path enumeration
and the state aligner all go through it
([#228](https://github.com/michaelJwilson/snakes_and_ladders/issues/228)). Every
categorical test in the suite passes against the family without its assertions
being rewritten, which is what says the refactor did not alter a model already
validated.

The Gaussian case is where the discrete assumption stops holding, and both
consequences are pinned rather than discovered later. The evidence is a
*density*, so `log P(observations) <= 0` fails on correct code and no test
asserts it. And the likelihood has **no maximum**: a state's mean on one
observation with its variance going to zero diverges, at exactly `log 10` nats
per tenfold narrowing, which a test exhibits. The variance floor is derived
from the data rather than chosen — `s**2 / n**2`, the nearest-neighbour
spacing below which a state has collapsed onto a point — and reaching it is a
**refusal**, since a clamped fit returns normally and reports an interval
around a degenerate optimum ([#122](https://github.com/michaelJwilson/snakes_and_ladders/issues/122)).

**The identifiable regime is measured, not assumed.** Two states one common
standard deviation apart are nearly the same state, and the fit says so.
Coverage of the 95% Wald intervals over 24 replicates of 240 observations,
against the separation of the emitting means:

| separation | intervals covering | rate | replicates with no interval at all |
| --- | --- | --- | --- |
| 0.5 | 18/28 | 0.643 | 17/24 |
| 1.0 | 46/56 | 0.821 | 10/24 |
| 2.0 | 82/88 | 0.932 | 2/24 |
| 3.0 | 90/96 | 0.938 | 0/24 |
| 4.0 | 93/96 | 0.969 | 0/24 |
| 6.0 | 92/96 | 0.958 | 0/24 |

Recorded as `docs/experiments/002-hmm-gaussian-interval-coverage.md`.

Coverage reaches nominal from two standard deviations of separation upward and
degrades below it, seen twice over: the intervals that exist under-cover, and
most replicates produce **no interval at all** — the observed information is
too ill-conditioned to invert, which is what an unidentifiable model looks
like from inside a fit. Those replicates are counted rather than dropped,
since excluding them unannounced would select for the well-behaved samples.
Reported as a finding: a fixture using only well-separated means would have
been testing the fixture.

**Count emissions: the dispersion axis, bracketed.** Four count families join
the seam — binomial below equidispersion, Poisson exactly at it, negative
binomial and beta-binomial above
([#229](https://github.com/michaelJwilson/snakes_and_ladders/issues/229),
[#260](https://github.com/michaelJwilson/snakes_and_ladders/issues/260)). The
bracketing is the point: an interface exercised only by overdispersed families
has never been asked whether it assumes overdispersion somewhere. Each referees
the neighbour it is a limit of, so the oracles come from outside this
repository rather than from a second call into it: `BetaBinomial(n, 1, 1)` is
the discrete uniform and `Binomial(1, p)` is Bernoulli, both to 1e-14 or
better; the negative binomial approaches Poisson as `r -> inf` and the
beta-binomial approaches the binomial as `a + b -> inf`, both at the `O(1/x)`
rate the truncation predicts, with `r` times the deviation measured at 18.75,
18.88, 18.94, 18.97 and 18.98 across `r` from 500 to 8000 — converging rather
than drifting, which is what makes the extrapolated tolerance legitimate.

**Two of the four have an M step that is an optimization**, which is what
tests whether the seam is real: `reestimate` now returns what its M step did,
not only what it produced. **Both solves bracket rather than step**, and that
was a measurement, not a preference. Newton on the negative binomial's
weighted score converges for moderate `r` and, on a near-Poisson sample,
overshoots in `log r` and underflows to zero — arriving as a domain error
rather than a bad answer; safeguarded bisection reaches `|score| / weight` of
1e-14 to 1e-16 in 45 evaluations and agrees with a 40001-point grid search to
zero relative difference. Minka's fixed point for the beta-binomial is
monotone but linearly convergent: at a true concentration of 120 it was still
moving in the third decimal after 500 iterations and returned that as though
it were an estimate. Alternating bisection in `(p, a + b)` settles in 3 to 9
iterations at a residual of exactly zero. An M step that does not settle is
refused, per `likelihood/CLAUDE.md`.

**The flat-likelihood hazard is the mirror of the Gaussian's, and it is
measured.** Where a Gaussian likelihood is *unbounded* as a variance falls, a
count likelihood goes *flat* as the dispersion rises toward its Poisson or
binomial limit. The bound is derived on the same construction for both count
families — the dispersion at which the overdispersion the model is for falls
below the sampling noise on measuring it, `mu sqrt(W/2)` for the negative
binomial and `(n-1) sqrt(W/2)` for the beta-binomial. Coverage of the 95%
intervals over 16 replicates of 480 counts, against the true dispersion:

| true `r` | intervals covering | rate | replicates with no interval at all |
| --- | --- | --- | --- |
| 0.5 | 44/48 | 0.917 | 4/16 |
| 1.0 | 60/64 | 0.938 | 0/16 |
| 2.0 | 58/64 | 0.906 | 0/16 |
| 5.0 | 59/64 | 0.922 | 0/16 |
| 20.0 | 58/64 | 0.906 | 0/16 |
| 100.0 | 32/36 | 0.889 | 7/16 |

**The finding is not the coverage column.** Coverage sits near nominal at every
dispersion; what degrades is *whether an interval exists*, and it degrades at
**both** ends — a heavy tail at `r = 0.5`, a flat likelihood at `r = 100`.
A non-identified count model announces itself as a singular information matrix,
not as an interval in the wrong place, which is the opposite of what the
Gaussian case showed and is why both were measured rather than one assumed
from the other.

**A Gaussian mixture: the emission seam with the Markov chain removed.**
`snakes_and_ladders.sim.mixture` draws component labels and observations
jointly, retaining the label so a clustering has something to be checked
against; `snakes_and_ladders.opt.mixture` fits
([#262](https://github.com/michaelJwilson/snakes_and_ladders/issues/262)). Its
component M step **is** `GaussianEmission.reestimate`, called with
responsibilities where an HMM passes state posteriors, and a test asserts the
two produce identical numbers on the same responsibilities rather than merely
similar ones — the evidence that the seam extracted from an HMM was not shaped
by one. The unbounded likelihood transfers unchanged with it: a component
collapsed onto a single observation is refused, not clamped.

**The coupled spatio-sequential model has a simulator and an exact oracle**
([#300](https://github.com/michaelJwilson/snakes_and_ladders/issues/300), #290
part 2). `sim.spatio_sequential` declares the truth — a spatial graph with
`J >= 0`, `M` classes, `K` states, `S` positions, `beta`, the circulant
self-transition `t`, one initial distribution and one emission family per
class — and draws labels by the single-site heat bath at `beta` with no field,
chains by `Pi_m` and the circulant transition, and observations by the class's family
at the node's class and the position's state, under one generator; labels may
be planted for recovery studies. `likelihood.spatio_sequential` sums
`eq:joint` over every assignment of the canonical instance (a 2x2 open
lattice, `M = K = 2`, `S = 6`: 65,536 joint states) for the evidence, the
label posterior, the per-class state posterior and the state posterior given
a labelling that part 3's forward–backward is pinned to. The oracle is pinned
two ways that share no code: its evidence equals the sum over labellings of
the per-class forward recursion to a relative gap of 0.0 on three draws, and
its written-out joint equals the factor graph's log-density on 50 random
assignments to 1.4e-14. The simulator is held to what it composes by
chi-square at 0.001 over 400 draws — labellings against the enumerated Potts
prior, transitions against `t`, first states against `Pi_m`, and symbol
counts per (class, state) against the families' tables — and the label
posterior recovers planted labels on 42 of 48 nodes at `S = 6`.

## Milestone 1.2 — Differentiable Likelihood & Energy Engine

**Felsenstein pruning: three CPU backends, one oracle.** Vectorized NumPy is
the reference, with per-node rescaling accumulated in log space
([#66](https://github.com/michaelJwilson/snakes_and_ladders/pull/66)); differentiable
PyTorch takes branch lengths as a tensor separate from the topology so
`torch.autograd` differentiates through them
([#74](https://github.com/michaelJwilson/snakes_and_ladders/pull/74)); Rust implements the
same recursion behind PyO3
([#81](https://github.com/michaelJwilson/snakes_and_ladders/pull/81)). Every one is pinned
against an independent brute-force marginalizer rather than against another
backend: worst relative deviation 4.0e-14 across all three and four site counts
spanning a factor of 30
([#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148)). The pulley
principle and rescaled/unrescaled agreement are checked besides.

**The Rust backend returned nothing at the declared scale, and now returns
2.5x.** Measured against the NumPy oracle end to end, it was **1.8x** at 10
taxa by 1,000 sites and **1.00x** at 200 by 11,000 — the top of the range
`ROADMAP.md` §1.2 declares. Two causes, and both were invisible in the
benchmark cells that existed, which sat at 4 and 8 taxa. The binding took
nested Python lists, so PyO3 built one integer object per observed state: 2.2
million of them at the top of the range, a cost growing with `n x L` while the
kernel's advantage does not. And the recursion held every node's partial
likelihood for the whole computation, 140 MB where the oracle needs 21. The
alignment now crosses as one borrowed array and a partial is released when its
parent has consumed it, giving **1.7x** at 20 taxa by 11,000 sites and
**2.5x** at 200, with peak memory down to 22.6 MB. The backend is still under
the 3x bar a CPU port is now held to: the remaining cost is the kernel itself,
which loops over states scalar-wise where the oracle reaches BLAS, and the
site-parallel recursion is the data-parallel case the roadmap sends to the GPU
rather than to a second CPU port.

**The memory requirement is settled.** The simulator retains every node's
states, `(2n - 1) x L x 8` bytes, and pruning retains one partial likelihood
per open node, `(2n - 2) x L x k x 8` on a caterpillar — the deepest tree on
its leaves, so the worst case, and the topology at which `O(n x L x k)` is
tight rather than loose. At 100 taxa by 11,000 sites that is **87.2 MB**, and
at the declared maximum of 1,000 by 11,000 it is **879 MB**: a factor of **20**
inside the 16 GB requirement. A balanced tree of the same size costs strictly
less, which is what makes the figure a bound.

Both terms are published as arithmetic over the arrays' shapes rather than as a
sampled peak, because a `tracemalloc` figure does not survive a change of
machine — an earlier draft published one and CI regenerated a different table.
The regression suite carries the measurement instead, pinning every published
cell against the allocator's own count: realized agreement **2.5%** at the
smallest cell and **0.3%** at the two larger ones, against a 5% band.

**Device dispatch: declared, CPU-only.** Selection prefers CUDA, then
Metal/MPS, then CPU, and the cross-device tolerance is stated where the
roadmap promised it — relative, and keyed on the lowest precision in the
comparison: `1e-11` with `float64` on both sides, `1e-6` where either side is
`float32`, since Metal cannot do `float64`
([#112](https://github.com/michaelJwilson/snakes_and_ladders/pull/112)). Both figures are
derived from measured agreement, and the `float32` bound is exercised on CPU so
runners without an accelerator still check it. The CUDA and Metal paths
themselves are not implemented.

**Parsimony landed, and it is here to be wrong.** Fitch's algorithm scores a
topology beside the likelihood, pinned against exhaustive enumeration over
internal-node labellings — equality, not a tolerance, since the score is an
integer. Its purpose is the Felsenstein zone, where four taxa with two long
branches placed non-adjacently make parsimony *statistically inconsistent*:
convergent change on the long branches is cheaper to explain by grouping them
than by the true topology, so more data does not help. Measured over 12
replicates, parsimony recovered the true topology **0 of 12 times at 200,
1000 and 5000 sites** while likelihood went 10/12, 12/12, 12/12.

The Farris zone is the control that makes that interpretable: move the same
two long branches to be adjacent and parsimony is right **12 of 12 at every
site count**, while likelihood needs more data — 4/12, 6/12, 10/12. An
implementation that were simply broken would fail both zones, and one result
alone cannot tell the two apart. This is the repository's first fixture where
a named method's failure is a theorem rather than a defect.

**Belief propagation is now measured over an ensemble, not three fixtures.**
`snakes_and_ladders.sim.graph.erdos_renyi_graph` draws `G(n, p)` beside `lattice_graph`,
and BP is checked per draw against enumeration. Over 60 sparse draws, 106
across two ensembles were acyclic and BP was exact on every one — worst
relative deviation 3.7e-15 in `log Z` and 4.9e-13 in the marginals, inside
the `1e-11` float64 bound. The ensemble also reaches what no committed
fixture did: **104 of 120 draws carried an isolated vertex**, the case sitting
on the boundary between the general message-passing loop and the edgeless
special case.

**A correction to what was expected.** #214 proposed asserting that the
deviation on a cyclic draw sits well below the lattice's, on the
locally-tree-like argument. It does not, at this scale: measured over 14
cyclic draws the relative deviation ran 2.7e-04 to 7.4e-03, median 3.6e-03,
against the lattice's peak of 5.2e-03 — comparable, not better. At `n <= 10` a
single cycle is a large fraction of the graph, and the locally-tree-like
argument is asymptotic. The deviation is reported; nothing claims BP is more
accurate on a random graph than on a lattice at these sizes.

**Three canonical fixtures, each consumed by more than one module.**
`snakes_and_ladders.sim.canonical` holds instances whose answer comes from outside this
repository, admitted on two clauses stated in `sim/CLAUDE.md`: the answer must
be independently known, and more than one module must consume it
([#209](https://github.com/michaelJwilson/snakes_and_ladders/issues/209)).

| Fixture | Known from | Consumed by |
| --- | --- | --- |
| Triangular Ising antiferromagnet | a double count that closes exactly: `N` of `3N` edges agree in any ground state, at every size | `sim` builds, `likelihood.potts` enumerates, `search.max_cut` optimizes, `search.potts_mcmc` refuses |
| Planted Viana-Bray spin glass | the planted state's energy, an upper bound on the ground state past enumeration | `sim` builds, `search.alpha_expansion` scores |
| Ambiguous-emission HMM | enumeration of all `2**5` paths | `sim` builds, `likelihood.hmm_paths` decodes both ways |

**The triangular ground-state energy is known at every size.** `3N` edges,
`2N` triangles, each triangle needs one agreeing edge because a 3-cycle is not
2-colourable, each edge lies in two triangles — so at least `N` edges agree,
and enumeration attains that at `N = 9, 12, 16`. With `coupling = -|J|` the
ground-state energy is `|J| * N` exactly. Alpha expansion's factor of 2
([#207](https://github.com/michaelJwilson/snakes_and_ladders/pull/207)) and Goemans-Williamson's
0.87856 ([#215](https://github.com/michaelJwilson/snakes_and_ladders/pull/215)) are the only
other discrete claims here that survive past enumeration, and both are bounds
rather than answers.

**Wannier's residual entropy is reported, never asserted.** The constant
0.3231 per site is a thermodynamic limit. Measured: 0.4153 at `N = 9`, 0.3516
at `N = 12`, 0.2336 at `N = 16` — not close, and not monotone, because a `4x4`
torus is incommensurate with the three-sublattice ground state. The exact
degeneracies (42, 68, 42) are asserted instead. This is the same discipline
[#214](https://github.com/michaelJwilson/snakes_and_ladders/pull/214) arrived at for graph
threshold results.

**The planted spin glass does not replace #177, and this is measured rather
than assumed.** It was proposed as the instance whose difficulty scales, after
[#198](https://github.com/michaelJwilson/snakes_and_ladders/pull/198) measured random-restart
greedy solving #177's tree at 1.000. Against 20-restart iterated conditional
modes at `n = 100` and mean degree 4, descent lands on the planted energy
below frustration 0.2 (mean gap +0.30 at 0.00, -0.12 at 0.05, +0.12 at 0.10,
-0.25 at 0.15) and beats it above (-8.2 at 0.20, -25.7 at 0.30), where the
planted state is no longer near-optimal and is a weak reference. Raising
connectivity does not open a window: at mean degree 12 and frustration 0.05
descent matches the planted energy exactly on every instance. What the fixture
does supply is a **known-energy reference past the size enumeration reaches**,
which nothing else in the repository has. The search for an instance no
baseline solves stays open, and `TICKETS.md` now says so.

**Viterbi and posterior decoding can now be told apart.** Neither decoder is
implemented, but the fixture that separates them is: on `ambiguous_hmm` the
Viterbi path is `(0,0,0,0,0)` — unique, 0.3033 nats clear of the runner-up —
while posterior decoding returns `(0,1,0,1,0)`, the observations themselves,
with every marginal above 0.6256. That posterior sequence is the **5th** most
likely path of 32, 0.6066 nats behind the Viterbi path. A decoder that
computes one and reports the other passes every fixture where they agree,
which is most of them.

**Potts and HMM evaluators: partial.** The 1-D transfer matrix and the HMM
forward recursion exist, each with its exact oracle. Sum-product belief
propagation over a general `PottsGraph` now joins them, with the 2-D strip
transfer matrix as the oracle for the regime enumeration cannot reach
([#206](https://github.com/michaelJwilson/snakes_and_ladders/pull/206)). A forward-backward
routine exposed outside Baum-Welch's internals is still not built.

**What belief propagation is claimed to do, and what it is not.** On a
tree it is exact, and that is where the correctness claim sits: `log Z` agrees
with exhaustive enumeration to 2.0e-15 relative and the single-site marginals
to 2.8e-13, inside `likelihood/CLAUDE.md`'s `1e-11` `float64` bound. On a
loopy lattice it is approximate, so nothing asserts agreement — the deviation
from the exact strip transfer matrix is reported as a measurement, and it is
1.7e-15 at zero coupling, 1.1e-03 at `J = 0.5`, and peaks at 5.2e-03 at
`J = 0.875` on a 6x4 open strip in three states. That peak is the result worth
having: the exact `q`-state transition on a square lattice is at
`J_c = ln(1 + sqrt(q)) = 1.005` for `q = 3`, so the Bethe approximation is
worst where the correlations it neglects are longest-ranged, and it recovers
on both sides. Messages that do not settle raise rather than returning a
number.

**The three evaluators are one algorithm on one structure, and the
structure now exists** ([#290](https://github.com/michaelJwilson/snakes_and_ladders/issues/290),
part 1). `sim.factor_graph` holds variables, factors as log tables, adapters
from a tree, a Potts graph, a hidden Markov chain and the coupled
spatio-sequential model, and the Forney normal form; `likelihood.message_passing`
runs sum-product and max-product over it with a tree schedule that is exact
and refused off a tree, and a damped flooding schedule that is the Bethe
approximation. Each shape is pinned to the evaluator that predates it and
shares no code with it: on the six-node Potts tree `log Z` and every marginal
agree with enumeration to 1e-14 relative; on four chains the evidence and
posteriors agree with the path enumeration to 1e-13 and max-product returns
the enumerated Viterbi path with its joint; per-site sum-product on the
four-taxon fixture sums to pruning's log-likelihood to 1e-13; on the 3x3 open
lattice flooding reproduces `belief_propagation`'s Bethe free energy to 1e-9
and its beliefs to 1e-8; the coupled adapter's `log_density` equals `eq:joint`
written out on all 4,096 joint assignments of a three-node, two-class,
two-state, length-two instance; and the Forney form gives the same marginals
as the bipartite one. The generality has a measured price: 10x the forward
recursion on a 200-step, four-state chain (73 ms against 7 ms) and 57x
`belief_propagation` on an 8x8 lattice (1.04 s against 18 ms), a table per
factor and a dictionary per message against a recursion that knows its shape.
The specialised evaluators therefore stay; the factor graph is the structure
for the model none of them can express.

**Forward–backward is an evaluator** ([#306](https://github.com/michaelJwilson/snakes_and_ladders/issues/306),
closing #173). `likelihood.forward_backward` returns the evidence, the
position posteriors and the pairwise posteriors of one chain in the log
domain, and a forward-filter backward-sample draw of the path; it is pinned
against the path enumeration on four chains to 1e-12 and the E step of the
coupled model built on it against the enumerated conditional posterior to
1e-11. Baum–Welch keeps its own recursion for the gradient it needs.

**The tree was audited against the runtime-optimization opportunities, and
five of the ten lines had a measurement behind them**
([#287](https://github.com/michaelJwilson/snakes_and_ladders/issues/287)). Profiling
first (`tests/benchmarks/profile_hotpaths.py` plus a Potts-side probe) ranked
the Python-level loops by self time, and the top of the ranking was not where a
reader would guess: the SPR neighbourhood at 30 taxa spent 28 of 29 seconds
building a `Node` tree and unioning `frozenset`s per candidate *to
deduplicate*; a 6-taxon hill climb charged 8,730 scalar transition-matrix
calls to 970 likelihood evaluations; a fifth of a REINFORCE run was Python
arithmetic per action. Each change is pinned to the code it replaces, and no
default sampler path moved:

| rule | finding | change | pin | realized |
| --- | --- | --- | --- | --- |
| profile first | five loops above 20% of their run's self time | the five below | — | — |
| layout | adjacency as a list of lists in every kernel | `PottsGraph.compressed_adjacency()`, one contiguous CSR | equal to the list adjacency in order, 4 graphs | exact |
| compiled backends | descent sweep 6 µs/site in Python | `numba` kernel, default for `iterated_conditional_modes` | labelling and energy **bitwise**, 6 seeds | **7x** at 32x32, 3 labels |
| FFI boundary | annealing and tempering on the Python sweep | `backend=Backend.RUST` runs the extension's sweep on the same uniforms | per-replica chi-square, both backends | tempering **5.8x** at 100 nodes, **7.8x** at 32x32; annealing 2.3x (the per-sweep energy now dominates) |
| inlining / vectorization | one `P(t)` call per child per node | every branch's matrix in one call | `torch.equal` per branch for JC; GTR to 2.9e-13 (torch's batched `matrix_exp` is a different kernel); JC log-likelihood bitwise | hill climb **3.99 s to 2.32 s**; one 20-taxa evaluation unchanged at 12 ms |
| inlining / vectorization | `_deltas` per action, 126,000 calls per 50 updates | `features` as one NumPy pass | `array_equal` to `_deltas`, 200 states, deviation 0.0 | REINFORCE **1.51 s to 1.00 s** |
| profile first (algorithmic) | tree built and split-set unioned per SPR candidate | dedup on leaf bitmasks from the adjacency; tree built only for a new key | same neighbours in the same order as the definition, n = 5, 7, 9; key equals `leaf_bipartitions` on all 105 six-leaf topologies | **1.41 s to 0.59 s** at 30 taxa |
| cache, branches, allocation, double buffering | no measurement separating them from the above | none | — | recorded as not measured |

Two things the table does not say. The Rust-backed tempering chain agreed with
the Python one **draw for draw** on the enumerable instance — identical
p-values and exchange acceptances — which is the case `potts_mcmc_rust.py`
says cannot be relied on, and it is not: one draw across a threshold moved by
an ulp would part them, so the backend stays opt-in. And the batched
transition matrices leave the JC log-likelihood bitwise unchanged while the
gradient moves by 1.5e-11 absolute, autograd summing the same terms in a
different order; the finite-difference check that pins the gradient is
unaffected.

**Bounds with proofs, certified rather than trusted**
([#308](https://github.com/michaelJwilson/snakes_and_ladders/issues/308)). A
surrogate carries its claim — lower bound, upper bound or point prediction —
and `certify` holds it to the claim on every structure an oracle can score.
Two one-pass bounds bracket a topology's maximized log-likelihood: one pruning
evaluation at non-negative least-squares lengths from below, and
`sum_s log pi(x_1s) - F log k` from above, where `F` is the Fitch score — a
vertex argument on the multilinear Jukes–Cantor site likelihood that the
suite checks by enumerating all 256 vertices at five taxa and finding the
bound attained on every site. Over the 15 topologies of the five-taxon
fixture at 1,200 sites both hold with no violation; the plug-in bound's worst
gap is 6.7 nats and its mean 2.9, at 2.4 ms against 254 ms for the fit; the
parsimony bound costs 0.1 ms and its gap is 2.2 nats per site, so it is a
bound and not an estimate. The entrywise-limit bound the plan proposed was
derived and dropped: it sits above zero on every fixture, and the parsimony
bound dominates it. For a lattice, the naive mean-field bound and the
tree-reweighted spanning-tree bound (one tree per edge, uniform) sandwich
`log Z` within 0.049 and 0.028 nats per node on average over 40 random
open lattices (worst 0.078 and 0.052), both differentiable in the field and
couplings and both agreeing with central differences to 1e-6 relative; from
them a bracket on the ground-state energy that on the same lattices at
`beta = 3` sits 0.028 nats per node below the enumerated minimum. The
proofs are Appendix B of the textbook.

## Milestone 1.3 — Continuous Optimization via Autodiff

**The interface is model-agnostic, and that is measured rather than asserted.**
An `Objective` is an unconstrained parameter vector, a differentiable scalar,
and a map back to named constrained parameters
([#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115)). Four instances now
run against it unchanged — the Potts chain, the HMM, branch lengths on a fixed
topology, and the GTR substitution model — and none required a change to
`snakes_and_ladders.opt`. A test asserts the module imports nothing from `snakes_and_ladders.sim`,
`snakes_and_ladders.likelihood` or `snakes_and_ladders.search`, so the separation cannot decay by
convenience import.

**The optimizer is now pinned to minimizers known in closed form, not only to
likelihood surfaces.** Every earlier test of `fit` measured a statistical
property — the first-order condition, coverage at the nominal rate, agreement
with Baum-Welch — under which an optimizer that stops early and a parameter
that is weakly identified look identical. Three standard test functions
separate them: Rosenbrock is reached to `1e-11` of its analytic minimizer at
2, 3 and 5 dimensions, the autodiff gradient matches the hand-written closed
form exactly on all three functions, and all four of Himmelblau's equal minima
are reachable, each from its own basin.

The third is a measurement that constrains what may be claimed elsewhere.
On Rastrigin, over 200 starts drawn uniformly from the standard `+/-5.12`
domain, a single L-BFGS fit reached the global minimum **0 times**; restricted
to `+/-2` it reached it in 4%. Every one of those runs reported `converged`,
because every one satisfied the first-order condition. `converged` is a
statement about the gradient and says nothing about global optimality, and any
result resting on a single fit of a multimodal surface has to say so.

**The lattice is fitted, against an exact normalizer.** `log Z` is
enumerated over all 19,683 configurations of a 3-state 3x3 lattice rather than
approximated, so the fitted optimum is checked against a brute-force scan of
the likelihood instead of against the optimizer's own convergence, and the
enumerated normalizer reduces to `snakes_and_ladders.opt.potts.log_partition`'s transfer
matrix on a chain to machine precision. Interval coverage over 40 replicates
is 157/160 at 100 samples, 153/160 at 400 and 153/160 at 1600 — approaching
the nominal rate from above and settling, as the Potts chain does.

That closes the requirements row. It also leaves the hidden Markov model's
half of the same row less settled than the committed coverage figure reads:
its 45/48 = 0.938 at 150 sequences and 91/96 = 0.948 at 2400 both sit within
one binomial standard error of 0.95 (0.032 and 0.022 respectively), so the
under-coverage the caption describes is not distinguishable from sampling
noise at those replicate counts. The two identified causes are real — an
emission fitted near zero, and the post-selection cost of aligning the hidden
states — but stating a sample size at which nominal coverage begins to hold
would need more replicates than the figure runs, and none is claimed here.

**Intervals now have a second, non-asymptotic source.** Hamiltonian Monte
Carlo samples the posterior over any `Objective`, so an interval can be a
quantile rather than a curvature estimate at the mode. The integrator is
pinned where it is exact before anything statistical is claimed: it is
reversible to `1e-15`, and its energy error is second order in the step size,
measured at a ratio of exactly 4.00 across four halvings at fixed trajectory
length. The chain is then checked against two references that are not
samplers -- an analytic Gaussian, and the Potts chain's own two-dimensional
posterior integrated on a grid, which it matches to 0.005 in the mean and 10%
in the spread. On that fixture the Laplace standard error agrees with the
posterior's to 15%, which is the expected outcome for a well-identified
two-parameter model and is what would make a disagreement elsewhere
informative.

**A step size too large biases the spread while the acceptance rate looks
healthy**, and that is why `HmcChain` reports the per-proposal energy error.
Measured against quadrature: at a step of 0.020 the acceptance rate was 0.982
and the posterior standard deviation was 12% low, because divergent
trajectories are rejected preferentially in the tails. Acceptance rate does
not detect it; `max |dH|` tracks it monotonically.

**A fourth-order integrator lands, and loses.** Yoshida's (1990) triple jump
joins leapfrog as a selectable symplectic integrator, both expressed as
compositions of the same kick-drift-kick sub-step so there is one
implementation rather than two
([#266](https://github.com/michaelJwilson/snakes_and_ladders/issues/266)). The
orders are measured rather than claimed, as the ratio by which halving the
step divides the energy error: leapfrog realizes 3.999, 4.000, 4.000, 4.000,
4.000 against a predicted 4, and Yoshida 16.310, 16.077, 16.019, 16.005,
16.001 against a predicted 16 — converging rather than drifting, which is what
makes it an order and not a coincidence at one step size.

**It is slower anyway, and the mechanism is worth recording.** A higher-order
method pays where the step is limited by *accuracy*; here it is limited by
*stability*. Yoshida's middle sub-step runs backwards in time with
`|w0| = 1.70` times the nominal step, so its stability limit in the step size
is about 0.59 of leapfrog's — measured at 0.0333 against 0.0500, a ratio of
1.50 against the 1.70 the coefficient predicts. With three force evaluations
per step on top, the order advantage is spent twice over. At equal
acceptance on the Potts posterior, leapfrog reaches 0.855 at **21** gradient
evaluations per trajectory while Yoshida needs **91** to reach 0.975 and
accepts *nothing* at 61; on the analytic Gaussian it is 3 against 7. The
default does not move.

**Where a fit starts is now the caller's to choose, and multi-start is
measured rather than assumed.** `Objective.initial()` was already the seam;
what went through it was one fixed constant per objective.
`snakes_and_ladders.opt.initialize` adds the objective's own start, a
deterministic perturbation, and random restarts from a passed-in generator, and
`fit_from` reports every fit and their spread rather than only the best --
returning one answer for a surface with four basins is the failure the
abstraction is about.

The measurement says multi-start is a tool for a particular shape of surface
and not a general improvement. On Himmelblau, four equal minima, a single fixed
start reaches exactly **one** basin however often it is run and four random
restarts reach all **four**. On Rastrigin, roughly `10**n` local minima each
satisfying the first-order condition, sixteen restarts reach the global minimum
**2 times in 30** against **0 in 30** from one start -- sixteen times the cost
for a success rate still near zero, and widening the draw does not help
(the same 2 in 30 at scale 4.0 as at 2.0), because the obstacle is the density
of the minima and not the reach of the proposal. Held to eight fits by
`opt.budget.compare` ([#281](https://github.com/michaelJwilson/snakes_and_ladders/issues/281))
it is 0 of 10 either way.

No default changes on that evidence. Every number below was produced from the
objective's own start and still is.

**Fitting and intervals.** L-BFGS with a strong-Wolfe line search, convergence
judged on the gradient relative to the objective's own magnitude, and
confidence intervals from the observed Fisher information pushed through the
constraint map by the delta method
([#116](https://github.com/michaelJwilson/snakes_and_ladders/pull/116)). Validation is
parameter recovery, not convergence: the Potts chain's 95% intervals cover the
truth at exactly the nominal rate over 60 replicates, and the HMM's gradient
fit is cross-checked against Baum-Welch — an independent algorithm sharing no
optimizer, parameterization or constraint map with it.

**The phylogenetic instance.** Branch lengths are recovered within their
intervals on both the unrooted and rooted fixtures, and exchangeabilities and
`π` alongside them
([#119](https://github.com/michaelJwilson/snakes_and_ladders/pull/119),
[#120](https://github.com/michaelJwilson/snakes_and_ladders/pull/120)). Two properties fell
out of doing it: the two branches below a rooted root are estimable only as
their sum, so they are fitted as one parameter and reported summed; and the
GTR model's three normalizations are gauges rather than conventions, each
removing an exactly flat direction that would otherwise leave every parameter
without an interval. The roadmap's sub-second gradient update at `n = 100` is
now measured — 203 ms at 1000 sites — rather than assumed.

**k-means++ lands, and buys nothing the fit can use.** The first initializer
that reads its objective's data
([#262](https://github.com/michaelJwilson/snakes_and_ladders/issues/262)), and
the case [#251](https://github.com/michaelJwilson/snakes_and_ladders/issues/251)
built the `Initializer` protocol for. **The protocol needed no change.** A
data-dependent strategy turns out to be model-*specific* rather than
protocol-incompatible: it takes an `Objective` like every other initializer and
refuses the ones whose parameter vector it cannot interpret, which is why it
lives beside the mixture rather than with the model-free strategies.

Measured against its published guarantee — Arthur & Vassilvitskii (2007) bound
the expected seeding cost at `8 (ln k + 2)` times optimal, and in one dimension
the optimal clustering is computable exactly by dynamic programming over
contiguous runs, so the bound has something to be checked against. On three
components six standard deviations apart, over 200 replicates: mean cost ratio
**2.91** against a bound of **24.79**, worst draw 14.41. Uniform seeding
realizes **11.03** mean and a worst draw of **58.08**, outside the k-means++
guarantee.

**And none of that reaches the likelihood, which is the finding.** EM reaches
the same optimum from either seeding on that mixture — 200/200 from k-means++,
195/200 from uniform — and from the objective's own quantile start too. Harder
fixtures do not reverse it, they make both fail: at five components 1.5
standard deviations apart neither seeding reached the best optimum found in 200
draws, and at five with unequal weights uniform reached it 9 times in 150
against k-means++'s 3 — noise, in the direction opposite to the one a default
change would need. So **no default moves**; k-means++ lands as a strategy a
caller may choose, at a cost of one objective evaluation (271 us of seeding
against 260 us per evaluation at 4000 points).

**An interval at a fit, whatever produced the fit.** The observed information
is a property of an objective *at a point*, not of the route that reached it,
but until now only a gradient fit could ask for one — expectation-maximization
works in the model's own parameters and never builds an unconstrained vector,
so the half of the fits with an independent oracle reported a point estimate
and nothing else
([#268](https://github.com/michaelJwilson/snakes_and_ladders/issues/268)).
Every objective now inverts its own constraint map, exactly: the round trip
`constrain(theta_from(named))` returns its input to between 0 and 4.4e-16
across all eight likelihoods, and bitwise for the three closed-form test
functions and the Gaussian-prior wrapper. `fit(include_intervals=True)` and
`fit_from(include_intervals=True)` attach the interval to the fit — the latter
to the best start only, one Hessian rather than one per start — and refuse it
at an unconverged point rather than report a curvature that is not an
information; off, they compute nothing and return `None`, so a fit inside a
search loop costs what it did.

**The check that costs nothing.** The gradient fit and Baum-Welch share the
model and nothing else, and converge to the same optimum — log-likelihoods
within 3.6e-9 relative. So their intervals must agree, and they do to **0.31%**,
the width of the flat ridge EM approaches slowly. A broken round trip fails
that loudly. The EM-derived intervals then cover truth at **243/264 = 0.920**
over 12 replicates, with 1 of 12 reaching the boundary and contributing none.

**The refusals survive, which is the part that mattered.** An interval from a
Hessian is a statement about a maximum, and a Gaussian component at its
variance floor is not one: the likelihood is unbounded there, the information
is not positive definite, and the new entry point refuses exactly as the old
one does — checked against the healthy point beside it, since a guard that
refused everything would pass a refusal-only test.

**And the comparison `hmc.py` promised is now complete, in both regimes.** A
version already existed — a raw Hessian in unconstrained coordinates against
grid quadrature. The missing halves are the delta-method interval on the
parameters a person names, against a chain, where the approximation is exact
and where it is not. On the analytic Gaussian the Laplace interval equals
`sqrt(diag(covariance))` to 1e-8 and the chain's spread matches it to **0.04%
and 0.47%** at 4000 draws, so agreement is asserted. On the Potts posterior the
sampled spread is **1.057, 1.031 and 1.036** times the Laplace one: slightly
optimistic, the expected direction for a mildly non-Gaussian posterior, and
reported rather than asserted away.

## Milestone 1.4 — Discrete Move Sets & Classical Baselines

**NNI and SPR: landed and counted.** Both neighbourhoods sit behind one
`Topology -> Iterator[Topology]` interface and are verified exhaustively
against `2(n - 3)` and `2(n - 3)(2n - 7)` at `n = 5..8`, over every distinct
topology, with neighbour validity, symmetry and NNI-in-SPR containment
cross-checked ([#82](https://github.com/michaelJwilson/snakes_and_ladders/pull/82)).

**Hill climbing, with an oracle that settles the question.** `infer` climbs
over either neighbourhood, fitting the continuous parameters of every candidate
([#127](https://github.com/michaelJwilson/snakes_and_ladders/pull/127)). Exhaustive
enumeration of unrooted topologies gives search quality an independent
reference below 8 taxa, so "did it find the best tree" has an answer
([#128](https://github.com/michaelJwilson/snakes_and_ladders/pull/128)): on the 6-taxon
fixture both move sets reach the enumerated maximum and recover the generating
topology from all 12 starting points, at a median of 14 candidate fits for NNI
against 48 for SPR. Budgets are counted in candidate fits rather than seconds,
so a run reproduces from its seed, and a topology is scored at most once per
search, keyed on its leaf bipartitions. The fit is the only unit worth
counting: one candidate fit measures 213 ms against 22 us to generate an
entire NNI neighbourhood, a factor of about 10 000.

**A discrete result now states how sure it is**
([#270](https://github.com/michaelJwilson/snakes_and_ladders/issues/270)).
`search.support` reports three quantities and names which: the
*neighbourhood* weight of the returned tree among itself and its neighbours
under a move set, with the margin over the best neighbour beside it; the
*enumerated* weight over every topology, the exact flat-prior posterior over
maximized likelihoods where `(2n-5)!!` fits; and Felsenstein's *bootstrap*,
per internal split, over site-resampled searches. The first is held to the
second: equal to 1e-9 at four taxa under NNI, where the neighbourhood is the
whole space, and at five taxa never smaller than it on any of the 15
topologies, with the best tree's NNI margin equal to its enumerated margin.
The bootstrap is held to its definition — a frequency over the returned
topology's internal splits, reproducible from its generator — and gives the
generating splits support of at least 0.8 at 1,000 sites. The exact weight is
calibrated: over 24 NNI searches on the five-taxon fixture at 30 to 300
sites, the fraction of returned trees equal to the generating one does not
fall from one support bin to the next. Each weight is over *maximized*
likelihoods under a flat prior over topologies and is named so, not called a
posterior; a tempered ensemble over topologies, which would give a marginal
one, does not exist.

**The coupled model is fitted, and the finding is about the start, not the
move** ([#306](https://github.com/michaelJwilson/snakes_and_ladders/issues/306),
#290 parts 3 to 6). `search.spatio_sequential` runs block-coordinate ascent
on `log p(x, l | theta)` with the chains marginalized: an E step per class,
an M step through each family's `reestimate` with `Pi_m` and the shared `t`
in closed form, and a label block that is the ground state of a Potts model
in the external field the posterior defines — by alpha expansion, by
single-site descent, or by an annealed Wolff move in the field whose best
visited labelling is taken only when it improves the joint. The M-step
identity holds through autograd on a Gaussian instance to 1e-10 relative;
the joint is non-decreasing across every block for every solver; and with
the parameters at the truth the label block reaches the enumerated MAP
labelling from the planted labels on 5 of 6 draws of the canonical instance
for every solver. Past enumeration, on a planted 10x10 lattice with weak
emissions, the label problem alone is easy — 0.98 accuracy up to permutation
with the parameters known — and every cold start freezes:

| start, then ten blocks | alpha expansion | single-site descent | annealed Wolff |
| --- | --- | --- | --- |
| uniform labels, true parameters | 0.66 | 0.78 | 0.68 |
| `Graph_BurnIn++` (thirty annealed steps), then alpha expansion | 0.97 | — | — |

The cluster move does not escape what descent freezes into; the trap is the
parameters, which a cold EM collapses before the labels can separate them.
The annealed start — labels nearly free while the emissions are fitted to
what the data alone supports, the prior tightening as the classes separate,
seeded by `Emission_Mixture++` (k-means++ under the family's own negative
log-density, exactly k-means++ under a squared distance) — is what recovers
the labels. Both are recorded, and the escape claim is retracted for this
instance.

**One Gibbs sampler and one annealer serve every problem, over the factor
graph** ([#309](https://github.com/michaelJwilson/snakes_and_ladders/issues/309)).
`search.gibbs` runs a heat-bath sweep over any `FactorGraph` — a variable's
conditional is the product of the factors touching it — tempered by a
schedule for annealing, with an exact block move for a chain-shaped subset
by forward filter and backward sample, and a Metropolis move over tree
topologies on the fitted likelihood. Each instance is held to the
distribution it targets by chi-square at 0.001: the 2x2 Potts lattice in a
field against enumeration; the five-step chain against the enumerated path
posterior, for the single-site sweep and for the block move, whose every
draw is independent; the four-taxon tree at one site against sum-product's
exact marginals; the coupled model's labels against the enumerated
posterior. The sweep copies the Potts kernel's arithmetic — one uniform per
variable, a search of the cumulative conditional — and agrees with it draw
for draw on 2,000 of 2,000 sweeps; annealing reaches the triangular
antiferromagnet's closed-form ground state on 6 of 6 seeds, as `anneal_potts`
does at the same schedule. At temperature one the topology move's visits
over the 15 five-taxon topologies match the flat-prior weight over fitted
likelihoods, the quantity #270 enumerates, and annealed to 0.02 it reaches
the enumerated best from 6 of 6 random starts with every topology fitted
once. The price of generality is smaller than expected: twenty sweeps of an
8x8 three-state lattice take 19.1 ms over factor tables against 15.0 ms for
the Python Potts sweep and 5.9 ms for the Rust one, 1.3x and 3.2x, because
the Python kernel already pays a NumPy call per site. The specialised kernels
stay the default for the Potts lattice.

**What carries between neighbours, and what does not.** A branch is now
identified by the leaf split it induces rather than by a node name, so a
neighbour that shares all but a few branches with its parent starts its fit
from the parent's lengths
([#289](https://github.com/michaelJwilson/snakes_and_ladders/issues/289)). The
warm fit reaches the cold optimum — worst relative gap 5.3e-12 in
log-likelihood over the 90 SPR neighbours of an eight-taxon tree — and the
search's answer does not move. What it saves is smaller than the ticket
hoped, and in one measurement negative: over those 90 neighbours the warm
fits spent 6,053 likelihood evaluations against 5,145 cold, a single
neighbour fit costs the same 49 either way, and only a refit of the *same*
topology from its own lengths drops to 14 — L-BFGS spends its evaluations on
the branches the move changed, not on the ones it kept. The larger saving is
lazy scoring: one cached likelihood evaluation at the warm lengths ranks a
neighbourhood and only the top `K` candidates are fitted, the accepted move
always in full. Subtree partials are cached keyed on the subtree's structure
and lengths, and a cached partial equals a recomputed one bitwise, so the
ranking evaluation is the same arithmetic in the same order. Over four
random starts on the eight-taxon fixture at 2,000 sites, budget 400 candidates,
counted in what the search reports:

| move set | run | same optimum as cold | fits | likelihood evaluations |
| --- | --- | --- | --- | --- |
| NNI | cold | 4/4 | 68.8 | 5,061 |
| NNI | warm | 4/4 | 68.8 | 3,660 |
| NNI | warm, `lazy_top=3` | 4/4 | 26.5 | 1,354 |
| NNI | warm, `lazy_top=1` | 4/4 | 9.5 | 493 |
| SPR | cold | 4/4 | 336.8 | 24,423 |
| SPR | warm | 4/4 | 339.5 | 22,677 |
| SPR | warm, `lazy_top=3` | 4/4 | 13.8 | 1,032 |
| SPR | warm, `lazy_top=1` | **2/4** | 5.2 | 579 |

Warm starts alone are worth 28% on NNI and 7% on SPR. Lazy scoring at
`K = 3` reaches the cold optimum on every start at 3.7x fewer evaluations on
NNI and 24x fewer on SPR; at `K = 1` it holds on NNI and loses half the SPR
starts, because the cheap surface ranks an SPR neighbourhood poorly — the
fitted best sits at lazy rank one in 6 of 6 NNI neighbourhoods and 1 of 6 SPR
neighbourhoods. So warm starts are the default, `lazy_top` is opt-in with
`K` chosen per move set from this table, the budget stays in candidates
scored, and `Inference` reports fits and likelihood evaluations beside it.
RAxML's three-branch local optimization is not built; its gap to the full
optimum is the measurement that would license it.

**The accuracy requirement's first half is met.** Normalized Robinson-Foulds
distance from the inferred to the generating topology is met at the 0.05 bound
from 125 sites upward, with 8 of 8 replicates recovering the topology exactly at
2000 sites against 5 of 8 at 60
([#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148)). The normalizer
counts internal splits only: every tree over the same leaves induces all the
trivial ones, and including them would shrink every distance by a
taxon-count-dependent factor and silently weaken the bound.

**Potts cluster updates landed, validated by the distribution they converge
to** ([#212](https://github.com/michaelJwilson/snakes_and_ladders/pull/212)). Swendsen-Wang
and Wolff run beside single-site heat bath behind one interface. Correctness is
a chi-square goodness-of-fit against the exact Boltzmann distribution at an
enumerable size, at a significance of 0.001, for all three move sets with and
without an external field; the worst p-value over 36 runs spanning six seeds
was 0.0145. That the test has the power it claims is itself pinned: replacing
the field accept step with an unconditional recolouring is rejected at p = 0.0.

Two errors that a test asserting only that the chain ran would have missed are
recorded here because they are the ones this ticket existed to catch. The
field accept step is the first: Wolff's cluster construction alone does not
preserve detailed balance in a field, and without the Metropolis correction on
`|C| * (h_new - h_old)` the sampler runs and converges to the wrong
distribution. The second was a Wolff sweep sized to match the others by
running clusters until their cumulative size reached the site count --- a
state-dependent stopping rule, which biased an aligned two-site chain to 0.384
per aligned state against an exact 0.334.

The reason to have them, measured at the exact transition
`J_c = ln(1 + sqrt(q))` on an open lattice, as energy autocorrelation time
normalized to sites touched:

| extent | single-site | Swendsen-Wang | Wolff |
| --- | --- | --- | --- |
| 8 | 3.27 | 2.56 | 2.71 |
| 12 | 6.89 | 3.91 | 3.04 |
| 16 | 9.74 | 4.33 | 3.68 |
| 24 | 10.37 | 4.86 | 5.01 |

Single-site slows by 3.2x between extent 8 and 24 while both cluster
algorithms slow by roughly 1.9x, so the gap is 2.1x at extent 24 and widening.
That understates the asymptotic separation: these lattices are small and their
boundary is open, both of which soften the transition. Recorded as
`docs/experiments/001-potts-cluster-autocorrelation.md`.

**An exact ground state landed, and it is the repository's first optimum that
is proved rather than enumerated.** For two states with every coupling
non-negative the Ising energy is submodular, so a minimum cut finds its global
minimum in polynomial time. Every other discrete claim here rests on
exhaustive enumeration and therefore stops at about twenty sites; this does
not, so a heuristic past that point finally has something to be checked
against.

Validated three ways, because enumeration alone would inherit the same cap:
against enumeration where it fits, at **exact equality** over 36
shape-coupling-field combinations; against two analytic corners at sizes far
past it — zero field gives an aligned state at `-J |E|`, zero coupling gives
`argmax` per site; and by the max-flow min-cut theorem as a self-check, the
flow value equalling the capacity of the cut residual reachability induces.

A Rust kernel (`src/maxflow.rs`) runs **26-32x** faster than the NumPy
reference measured on its own, and **6.3-10.7x** as a caller sees it. #220
attributed the difference to the Python lists crossing the FFI boundary by
copy, the gap #202 closed for the categorical sampler, and deferred the fix;
[#336](https://github.com/michaelJwilson/snakes_and_ladders/issues/336)
applied it, passing `float64` and `int64` buffers through `rust-numpy` and
returning the configuration as an array, and measured that the copy was not
the term. Minimum of 20 or more rounds, in ms, on square lattices with a
random per-node field:

| Extent | NumPy reference | Kernel, lists (#220) | Kernel, buffers (#336) | Caller (#220) | Caller (#336) | `energy()` |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 4.47 | 0.142 | 0.138 | 0.733 | 0.711 | 0.575 |
| 32 | 22.0 | 0.835 | 0.836 | 3.30 | 3.22 | 2.60 |
| 64 | 180 | 7.15 | 6.93 | 17.3 | 16.9 | 9.94 |

The boundary copy was 0.03-0.2 ms of a 0.7-17 ms call and removing it moved
the caller-visible number by under 3%. What a caller pays for is
`snakes_and_ladders.search.maxflow.energy`, which scores the returned
configuration edge by edge in Python and is 59-81% of the wrapper's time; it
is the oracle's function and is left as it is, so a caller wanting the
kernel's speedup takes the configuration from the extension and scores it
itself. Output is unchanged: the configuration is
equal element by element to the reference's and to the previous binding's at
extents 16, 32 and 64, and the energy is bitwise equal. The reference stays as
the oracle. The port also removes a fragility: the Python blocking flow
recurses to the depth of the level graph and needs `setrecursionlimit` raised
past a few thousand nodes, while the Rust one uses an explicit stack.

The boundary is refused rather than approximated. A negative coupling is
NP-hard and raises; more than two states is alpha expansion (#207), which
takes this as its inner solver.

**Alpha expansion landed, with the repository's first proved approximation
bound.** `k`-state MAP is NP-hard, so the exact cut above stops at two labels;
alpha expansion recovers the general case as a sequence of binary cuts, each
of which the exact solver handles unchanged. For a metric pairwise term its
local minimum is within `2 c_max / c_min` of the global one — exactly 2 for a
uniform Potts coupling.

That bound matters because of what it is *not*: belief propagation reports a
measured deviation, the samplers report a distribution, and enumeration stops
at nine sites. A bound holds at every size, so a result can be checked where
the algorithm actually runs. Measured at `3x3` with three labels over 40 runs,
alpha expansion found the global optimum **39 times** and recovered 99.554% of
the achievable improvement in the one miss — far inside the bound, which is
not tight and is not expected to be.

Where the move set earns its complexity is past enumeration. At `3x3` alpha
expansion and single-site descent are indistinguishable, both finding the
optimum in 31 of 32 runs between them; at `8x8` with four labels, expansion
beat the best of eight single-site descents on every trial, by 1.8 to 11.0 in
energy.

**Two construction errors, and what caught them.** The first draft swapped the
cut's terminal capacities and mis-costed the auxiliary nodes. Neither broke
loudly — both produce a labelling that is merely worse, which is
indistinguishable from a hard problem — and both passed every enumeration test
at `3x3`. The **reduction** caught them: at two labels one expansion is exact,
so it must reproduce the minimum cut energy for energy, and it was failing by
up to 2.55.

**Max-Cut landed as the other side of the same model.** Maximizing the weight
of separated edges *is* minimizing the energy with every coupling negative,
which is exactly the NP-hard side of the boundary the exact cut refuses to
cross. A Goemans-Williamson relaxation gives a run a certificate where
enumeration cannot reach: measured on random graphs with triangles at 12, 16
and 18 nodes the rounded cut reached the enumerated optimum **every time**,
and the computable certificate `value / relaxation` came out 0.95 to 0.98
against a guarantee of 0.87856.

**The certificate is weaker than the theorem, and the repository says so.**
Goemans-Williamson assumes the semidefinite program is solved to optimality;
there is no SDP solver here, so it is solved approximately by Burer-Monteiro
gradient ascent in `torch` rather than by taking a dependency. The value
returned can therefore sit *below* the relaxation's optimum, which makes a
ratio measured against it optimistic. That is asserted rather than glossed: on
a complete bipartite graph, whose maximum cut is exactly `|E|`, the ratio
comes out slightly **above 1** — impossible for an exact solve, and the
measurable evidence of what the certificate does and does not cover.

**Temperature is one object, and it lives where all three consumers can reach
it.** `snakes_and_ladders.opt.schedule` carries the schedules — constant, linear,
geometric, cosine, each mirroring its `torch.optim.lr_scheduler` counterpart
and checked against it to 1e-12 (1e-10 for the cosine, whose torch form is a
recursion) — with both endpoints reached *exactly* at the declared steps, and
a step past the end refused rather than clamped
([#267](https://github.com/michaelJwilson/snakes_and_ladders/issues/267)). The
Potts sampler takes a temperature as model scaling, which the model makes an
exact statement: the tempered energies equal the energies over `T` with a
deviation of **0.0**, and every move set's chain at `T = 2` and `T = 0.5` in a
field passes the chi-square against `exp(-E/T)` enumerated from the unscaled
model (p-values 0.016 to 0.89 at the 0.001 significance). The Hamiltonian
sampler takes it as the momentum's variance — the tempered dynamics are the
untempered ones in rescaled time, so the integrator is untouched — and on the
analytic Gaussian a chain at `T` is the chain at 1 with its deviations scaled
by `sqrt(T)` **draw for draw to 1e-10**. At `T = 1` every operation is the
identity bitwise, and the 31 existing HMC and 13 Potts tests pass untouched.

**Annealing is the sampler on a schedule, and the first instance is a wash.**
`anneal_potts` and `hmc.anneal` run one sweep or one proposal per schedule
step and return the best state seen. On the 9×9 periodic triangular
antiferromagnet, whose ground-state energy is a closed form, geometric
annealing from `T = 2` to `0.05` over 200 sweeps reaches it **20/20** against
single-site descent's **2/20** and a constant `T = 1` control's **7/20** — the
schedule, not the wandering. But descent converges in 2.6 sweeps, so the same
200 sweeps buy 78 restarts, and the best of 78 also reaches it 20/20. On
Rastrigin, measured at equal *objective evaluations* with a counting wrapper:
at 14,400 evaluations annealed Hamiltonian proposals plus a polishing fit reach
the global basin **6/20**, and 101 random-restart fits on the same budget
**10/20**; at 2,900 evaluations it is 0/20 against 1/20. Restarts win on the
continuous surface. Neither is a default.

**Parallel tempering, and the instance where restarts lose.** Replicas at
fixed temperatures exchange configurations on `(β_i − β_j)(E_i − E_j)`, each
replica on its own spawned generator from one seed. The oracle is the one the
samplers already have: with exchanges on, every replica passes the chi-square
against `exp(-E/T_r)` enumerated from the unscaled model (p 0.024 to 0.70,
exchange acceptance 0.78 and 0.57), and the paired negative case — an exchange
that omits the energy term — is caught at p = 0.0 on every replica. Then the
comparison the plan asked for, at **400 sweeps per method** on the planted
Viana–Bray spin glass, against the best energy any method found over 12
instances:

| instance | restarts of descent (100 × ≤4 sweeps) | annealing (1 × 400) | tempering (4 × 100) |
| --- | --- | --- | --- |
| 60 sites, degree 4, frustration 0.2 | 5/12, mean gap 0.75 | **12/12** | **12/12** |
| 60 sites, degree 4, frustration 0.35 | 5/12, gap 1.00 | 9/12, gap 0.50 | 9/12, gap 0.25 |
| 100 sites, degree 6, frustration 0.3 | 2/12, gap 2.58 | 7/12, gap 1.08 | **8/12**, gap 0.50 |

Every method beats the planted energy on every instance, as frustration
predicts. **The plan's prediction that tempering would be hard to justify at
these sizes is retracted**: on the one class of instance the roadmap needs —
frustrated, past enumeration — the tempered methods beat restarts at equal
budget and tempering carries the smallest gap. The triangular antiferromagnet
was too easy to show it; the glass is not. The first row is now held equal
by `opt.budget.compare`
([#281](https://github.com/michaelJwilson/snakes_and_ladders/issues/281)):
with the utility's own streams and the best any method found as the
reference, tempering **12/12**, annealing 10/12 with a mean gap of 0.17, and
restarts of descent 4/12 with a mean gap of 0.75, every method at or below
the planted energy. The five-component mixture comparison waits on the
mixture branch (#263) landing and belongs to the same utility.

**The single-site sweep has a Rust backend, beside the oracle.** Issue #232
profiled it as the one place a Python-level loop dominates -- one interpreter
iteration per site per sweep, five NumPy calls to move one spin. The port runs
**77x** the Python sweep at 64 nodes and **108x** at 1,024, and unlike the
pruning backend the ratio *rises* with size: there is no per-element
marshalling to grow against it, since the adjacency crosses once in
compressed-row form and the uniforms cross as one array.

Nothing switches to it. `f64::exp` agrees with NumPy's to within a unit in the
last place rather than exactly, and `searchsorted` is a threshold, so one draw
across a boundary that moved by 1 ulp sends the two chains apart permanently.
Replacing the oracle would move every autocorrelation figure above, every
committed notebook output that reads a chain, and the goodness-of-fit fixtures'
chain lengths. Agreement is therefore distributional: the Rust chain is scored
against the exact enumerated Boltzmann distribution at 2x2, with a field and
without, at the significance and thinning the Python sweep is held to -- and a
chain drawn under no field is rejected against the with-field truth, which is
what says the test has the power it claims.

**Not built:** Viterbi decoding, and iterated conditional modes over HMM state
paths (`snakes_and_ladders.search.alpha_expansion` carries a lattice ICM as its baseline,
which is a different object). Single-flip local search over the Potts chain exists as an RL
environment, not as a classical baseline suite.

**A surrogate ranks the SPR neighbourhood the lazy score could not**
([#308](https://github.com/michaelJwilson/snakes_and_ladders/issues/308)). The
one-evaluation lazy score of #289 put the fitted best at rank one in 1 of 6
SPR neighbourhoods; on the same six neighbourhoods of the eight-taxon fixture
at 1,000 sites the plug-in bound puts it first in 6 of 6 and the parsimony
bound in 5 of 6, at 1.7 ms and 0.1 ms per candidate against 206 ms per fit.
`infer(..., lazy_top=1, surrogate=)` ranks by any surrogate and fits only the
top candidate. Over four random SPR starts at budget 400: the full search
reaches its optimum from 4/4 at 312 fits and 20,718 likelihood evaluations;
`lazy_top=1` alone reaches it from 2/4 at 5.5 fits; ranked by the plug-in
bound, the parsimony bound, or a learned predictor, 4/4 at 5 fits and 275
evaluations — 62 times fewer fits and 75 times fewer evaluations for the same
answer. Learned predictors read the bound features and are trained on the
gap above the plug-in bound, so a poor fit falls back to the bound: a linear
model, a deep MLP, a Deep Sets model over branch tokens, a one-block
attention model and a graph network over the tree all reach held-out R^2 at
or above 0.999 on 15-topology neighbourhoods of the five-taxon fixture
(16 alignments, split 10/3/3 by alignment) and 0.98 on the two held-out SPR
neighbourhoods at eight taxa, ranking the fitted best first on every held-out
neighbourhood; the three token models return the same value for a tree with
its children shuffled, to 1e-13. The curriculum 5 → 6 taxa measures what
ROADMAP §2.2 predicts: zero-shot at six taxa the set model falls to R^2 0.68
and recovers to 0.94 after transfer; the MLP holds 0.92 zero-shot and 0.95
transferred. On lattices the models predict the gap above the mean-field
bound with R^2 0.996–0.999 held out (2×2 to 2×4), transfer zero-shot to
3×4 and 4×6 at 0.99, and the ground-state energy is learned exactly because
alpha expansion, one of the features, reaches it on every small lattice.
A calibrated bound is a rate claim: at nominal coverage 0.9 the lower bound
held on 100% of 45 held-out examples and the upper on 80%, so the claim
transfers on one side and not the other with three calibration alignments,
and `certify` at the stated rate is what says which.

## Milestone 2.1 — RL Agent Formulation & Deployment

**The estimator is pinned to a closed form, not to a training curve**
([#135](https://github.com/michaelJwilson/snakes_and_ladders/pull/135)). With a finite
action set and horizon the expected return is exact by trajectory enumeration,
and its gradient follows by differentiating it. That oracle carries every
claim: the enumerated gradient agrees with central finite differences to
1.5e-11 relative, the sampled estimator with the enumerated gradient to
9.9e-03 over 6000 episodes, and a myopic variant crediting each action with
only its own reward is rejected at 71%. A score-function estimator with a sign
error is wrong by a factor and still trains, which is why the sampled return is
a diagnostic here rather than a result.

**Learning is demonstrated where it can be refereed.** On the Potts landscape
the reward decomposes exactly into the two features the policy scores, which
puts hill climbing *inside* the policy class as the weight vector proportional
to `(J, 1)`. The learned policy reaches the enumerated optimum from 86.6% of
the 81 starts against greedy's 80.2%, in 8 of 8 training seeds — a statement
about learning rather than about two unrelated algorithms. Recorded as
`docs/experiments/003-potts-chain-reinforce-vs-greedy.md`.

**The phylogenetic environment exists, and the reward it can afford is
measured** ([#137](https://github.com/michaelJwilson/snakes_and_ladders/pull/137)). A state
is a topology, an action an NNI or SPR neighbour, the reward the improvement in
log-likelihood. Two reward models are implemented and the comparison between
them is the deliverable: fitting branch lengths per candidate costs 113.7 ms
against 352 us at fixed known parameters, a factor of 323, and only the second
makes an episode affordable. The substitution is validated rather than assumed
([#139](https://github.com/michaelJwilson/snakes_and_ladders/pull/139)): the two surfaces
score the generating topology highest, agree on the best of all 105 topologies,
and correlate at 0.9568, holding agreement on the best topology across a 50-fold
range of the fixed branch length with correlation never below 0.8719.

Measuring that recorded a property of the fitted surface worth having in
writing: it does not totally order topologies. Many candidates share a
maximized log-likelihood to within the optimizer's convergence, because the
branch distinguishing them fits to zero and the tree collapses to the same
polytomy — so a rank correlation moves by up to 0.04 under a perturbation of
one part in 1e9 and is not a measurement.

**The tree policy has now been trained, and the result is negative.** On the
7-taxon fixture where NNI hill climbing reaches the enumerated maximum from
only 24 of 50 starts, a trained policy reaches it on 0.485 of episodes against
greedy's 0.480 at a matched per-episode budget — +0.005, standard deviation
0.014 over 16 training seeds, 8 ahead, sign test p = 1.0. Two properties of
the environment bound it, and both are measured rather than argued. An episode
terminates when no move improves, and all 945 topologies contain 10 such
states, so every run — 50 of 50 greedy, 800 of 800 learned — ends at one: the
agent selects which local optimum to enter, and cannot leave one. And the
policy ranks moves by a single feature, the improvement a move buys, which
places hill climbing inside the policy class as a temperature. What would
change the answer is named rather than hoped for: a richer feature set
(`TICKETS.md` §2.1) and accepted-worsening steps (Stage 3, stochastic escape).

**Escape is now built, and it moved the comparison's baseline rather than its
result.** An episode can run past a local optimum, and an epsilon-greedy
policy can take the worsening move that leaves one: escape from one of the 9
non-global optima rises from 0.111 at `epsilon = 0` to 0.883 at `0.4`, and
matched-budget success from a random start rises from 0.560 to 0.908 against
the 0.480 a single greedy run reaches. But random-restart hill climbing
reaches the enumerated maximum from **every** start at the same 60-decision
budget: greedy stops after about four decisions, so 60 buys roughly fifteen
restarts, and with the global basin covering about 48% of starting topologies
`1 - 0.52**15` is indistinguishable from 1. The baseline Milestone 2.1 has to
beat on this fixture is therefore 1.000, not the 0.480 the tree-policy
comparison above was stated against, and nothing measured here beats it. The
fixture separates a single greedy run from the optimum; it does not separate
anything from restarts, and a fixture that does is what the next comparison
needs.

**All three problem classes are now MDPs.** `snakes_and_ladders.learn.Environment` had
one instance, a 1-D Potts chain, which is the same position `snakes_and_ladders.opt` was in
before four instances made its model-agnosticism a measurement rather than an
assertion. It now carries the Potts landscape over an arbitrary graph — the
chain is the one-dimensional case of the same class, not a second one — and
the hidden Markov state path, whose objective is a decoding problem rather
than an energy. Both are pinned against the enumerated estimator oracle
carried over unchanged from the chain, and against exhaustive enumeration of
their own state spaces: 19,683 configurations for a 3-state 3x3 lattice, 729
paths for a 3-state sequence of six. Neither takes an application type, so
`snakes_and_ladders.learn` still imports nothing from `snakes_and_ladders.sim`, `snakes_and_ladders.likelihood` or
`snakes_and_ladders.search`, and a test asserts it.

## Milestone 2.4 — Experiment Tracking, Ablations & Leaderboard

**The ledger has a record format before it has a run store**
([#314](https://github.com/michaelJwilson/snakes_and_ladders/issues/314)). An
experiment is a file under `docs/experiments/`, written from a template: the
commit, the feature under test, the fixture and its size tier, the methods
compared at one budget over shared seeds, the results, the finding, and the
tickets it filed. `infra/experiments.py` validates every file against the
template's fields, vocabularies and sections and generates the index that is
the leaderboard, and a guard runs it per pull request. Three measured
comparisons this file already stated are the first entries — the cluster
updates' autocorrelation at the transition, the Gaussian-emission interval
coverage against separation, and REINFORCE against greedy on the Potts chain —
and this file now cites them. The Aim run store (#75) is part 2, behind the
dependency's approval; until then the Results section is typed from the
measurement and names the script that produced it.

## Stage 3 — Research Extensions

**Only the half with an oracle is built.** `ROADMAP.md`'s differentiable-search
bullet names two relaxations. Potts configurations and HMM state paths are
enumerable, so the exact optimum, the exact expected score and the exact
gradient are all computable and "does the relaxation find what discrete search
finds" is falsifiable. Tree topologies at any interesting size are not, so the
tropical Grassmannian half is not started and `TICKETS.md` records that it is
blocked on an oracle rather than on effort
([#211](https://github.com/michaelJwilson/snakes_and_ladders/issues/211)).

**The relaxation is an extension, checked at every corner.** Over every
configuration of an enumerable instance the relaxed score equals the discrete
one to `1e-11` relative, for both spaces. The HMM check crosses a module
boundary — `snakes_and_ladders.learn` may not import `snakes_and_ladders.likelihood`, so
`RelaxedHmmPath.discrete` and `snakes_and_ladders.likelihood.hmm_paths.path_log_probability`
are independent implementations — and the relaxed objective's enumerated
optimum is the Viterbi path.

**One identity carries the result, and its boundary is not what it looks
like.** For a multilinear objective under a factorized `q`,
`E_q[score] = score(q)` exactly: the relaxed form at the marginals *is* the
expected discrete score. Two plausible statements of the limit are false and
are refuted by tests — it is not that the model must be a chain, and it is not
that terms must be pairwise. What breaks it is a term using one site twice,
since `E[X**2] = E[X]` for an indicator; measured, 1.000 against 0.557. It
follows that the relaxed maximum is attained at a vertex, so the relaxation
introduces **no optimum the discrete problem lacks** — everything a relaxed
search loses is lost to local optima of the ascent.

**The estimator bias is measured, not assumed.** Against the exact gradient
over 20000 draws, scaled by the largest exact component:

| `tau` | soft bias (SEM) | soft sd | straight-through bias (SEM) | ST sd |
| --- | --- | --- | --- | --- |
| 2.00 | 0.5975 (0.0012) | 0.165 | 0.5620 (0.0027) | 0.382 |
| 1.00 | 0.3373 (0.0034) | 0.487 | 0.3233 (0.0051) | 0.723 |
| 0.50 | 0.1400 (0.0076) | 1.077 | 0.1418 (0.0090) | 1.272 |
| 0.20 | 0.0475 (0.0157) | 2.220 | 0.0502 (0.0165) | 2.340 |
| 0.10 | 0.0356 (0.0240) | 3.392 | 0.0380 (0.0246) | 3.472 |

Bias falls by a factor of 17 while the standard deviation rises by a factor of
21, so no temperature is good at both. Straight-through's bias matches the soft
estimator's within error and its variance is higher at every temperature, so on
this problem it buys nothing.

**The sampling is what costs, not the relaxation.** Against single-flip hill
climbing over 40 shared seeds, on an antiferromagnetic chain whose optimum
needs coordinated flips:

| Method | Reached the optimum | McNemar |
| --- | --- | --- |
| Greedy hill climbing | 5/40 | — |
| Deterministic relaxation | 18/40 | `p = 0.00098` |
| Soft Gumbel-softmax | 11/40 | `p = 0.18` |
| Straight-through | 11/40 | `p = 0.18` |
| Annealed soft, 0.5 to 0.05 | 11/40 | `p = 0.18` |

The deterministic ascent — which the identity licenses, so it is not a
shortcut — is significantly better than the baseline; adding Gumbel noise gives
that up for a tie, and annealing does not recover it. It is also 15% cheaper
per run: 43.6 ms against 50.1 ms for 100 gradient steps. Three of four variants
tie, and that is reported as a tie, per the precedent
[#193](https://github.com/michaelJwilson/snakes_and_ladders/pull/193) set.

**The budgets are not the same unit and no claim is made that they are.**
Greedy stops at a local maximum after 3.5 decisions on average at 14 discrete
evaluations each; the relaxation takes gradient steps and evaluates no discrete
configuration until the end. What is matched is the restart count and the
seeds. The advantage is not bought with the larger budget: the relaxation
already wins at 25 gradient steps (15/40, `p = 0.0064`).

**The comparison fixture is not the repository's own.** `potts_params.yaml` has
`J = 0.75 > 0`, so its optimum is `argmax(h)` repeated and every method finds
it. That is the third time a fixture has been too easy to separate methods —
after [#177](https://github.com/michaelJwilson/snakes_and_ladders/issues/177),
[#198](https://github.com/michaelJwilson/snakes_and_ladders/pull/198) and #209's planted
spin glass — and it is why the baseline is now run before any claim is made.

**Not built:** the tropical Grassmannian relaxation; the relaxation on the
Potts *lattice*, where the identity holds but nothing has been measured; and
any joint optimization of structure alongside continuous parameters, which is
what the roadmap bullet ultimately asks for.

## §1.2 Requirements Ledger

| Requirement | Status |
| --- | --- |
| Phylogenetic RF ≤0.05 against simulated truth | **Met**, from 125 sites upward ([#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148)) |
| Potts/HMM parameter recovery within 95% intervals | **Met** for the 1-D chain, the discrete HMM ([#116](https://github.com/michaelJwilson/snakes_and_ladders/pull/116)) and the 2-D lattice — realized 0.981 at 100 samples and 0.956 at 400 and 1600, over 40 replicates each |
| Precise state-sequence decoding | **Not started** — no Viterbi decoder (issue #175) |
| Parity with exact oracles on small `n` | **Met** for tree search against exhaustive enumeration ([#128](https://github.com/michaelJwilson/snakes_and_ladders/pull/128)) |
| Parity with IQ-TREE 2 / RAxML-NG on large `n` | **Not started**; the tools are not in the environment (issue #126) |
| `O(n×L×k)` memory inside 16 GB / 24 GB | **Met**: 87.2 MB at 100 taxa by 11,000 sites on the worst-case (caterpillar) topology, 879 MB at the declared maximum — a factor of 20 inside 16 GB, with each figure pinned against the allocator to within 2.5% |
| CUDA, Metal/MPS and CPU dispatch | **CPU only**; selection logic landed ([#112](https://github.com/michaelJwilson/snakes_and_ladders/pull/112)), accelerator paths not implemented |
| Declared cross-device tolerance, not bitwise | **Met**: `1e-11` relative in `float64`, `1e-6` where either side is `float32` ([#112](https://github.com/michaelJwilson/snakes_and_ladders/pull/112)) |

## §1.3 The Technical Document

`docs/tex/` now spans all three problem classes rather than the phylogenetic
application alone: the abstract, methods and appendices state the Potts
Hamiltonian and the HMM decoding problem beside the substitution model, and the
Reference Taxonomy appendix routes the literature by concern. It was cut down
in `14d32d6` from the academic-letter structure of
[#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148) to an
eight-page specification, and has since grown back toward the shape §1.3 asks
for section by section, as recorded below.

Thirteen QA scripts render the figures, each committing a figure with a caption
naming the seed, the sizes and the model that produced it, and `docs/CLAUDE.md`
states the rules that keep a CI-regenerated artifact true
([#140](https://github.com/michaelJwilson/snakes_and_ladders/pull/140)). The two
documents cite seven of them — the textbook the simulated tree and the
Jukes–Cantor curves, the paper the worked simulation, the backend agreement,
parameter recovery, interval coverage and the topology search — and the
per-pull-request build regenerates only those; the remaining six are checked at
the release gate
([#157](https://github.com/michaelJwilson/snakes_and_ladders/pull/157)).

Measured against §1.3's required contents: the model formulations are present
for all three classes, and since
[#274](https://github.com/michaelJwilson/snakes_and_ladders/issues/274) every
equation and algorithm the code cites is stated in the textbook under a label —
the Jukes–Cantor closed form and its normalization, site independence, pruning
and the root marginalization, forward simulation, the forward recursion, the
belief-propagation message and the Bethe free energy, the episode return and the
REINFORCE estimator, and the cross-device tolerance — and
`tests/regression/test_document_labels.py` fails on a citation no document
resolves. The old document had labelled none of them, so nine citations had
never resolved and two named equation numbers from a numbering that no longer
existed. What remains at the level of a statement rather than a derivation is
the pruning and forward–backward recursions; absent entirely are the
branch-and-bound bounds and their proofs, no such bound being implemented.
Three framed placeholders stand in for the RL learning curve, the comparison
against classical software, and hardware scaling — none of which is measured,
and each labelled as a placeholder rather than drawn with invented data.

**The textbook is now one document at one standard**
([#298](https://github.com/michaelJwilson/snakes_and_ladders/issues/298)). Every
problem section carries the same four parts — the model, the model as an
instance of the factor graph of `sec:factor-graph`, the algorithm as the
instance of `eq:sum-product` or of the optimization it is, and the property
that pins it — and the notation table states the factor-graph symbols once
with the identification each section's classical symbol makes. The discrete
solvers that were one sentence each are sections with a labelled equation, a
citation, a regime and a pin: ground states as cuts (`eq:cut-energy`,
`eq:gw`), alpha expansion and its bound (`eq:alpha-expansion`), the heat bath
and the cluster moves with the field accept step (`eq:heat-bath`,
`eq:cluster-accept`), and temperature, annealing and tempering with the
exchange ratio (`eq:exchange`). The hidden Markov section states the backward
pass, the posterior and Viterbi as max-product (`eq:posterior`,
`eq:viterbi`). The coupled spatio-sequential model of #290 has its own section
(`sec:coupled`): the ticket's Forney-style figure, the spatial and chain
priors, the block-coordinate estimator with the M-step identity and the
external field (`eq:coupled-m-step`, `eq:external-field`,
`eq:label-ground-state`), and both algorithms (`alg:wolff-field`,
`alg:graph-burnin`), with a paragraph stating which part is built and which
is planned. Four derivations the main text depends on — the Bethe fixed
point, detailed balance for a cluster move in a field, the exchange ratio, the
delta method — sit in an appendix cited from the point of use. The document
is 21 pages; `texlive-pictures` joins the CI TeX install for the figure.

## What Is Not Claimed

- That a learned policy beats hill climbing on trees. It has now been
  measured on a fixture where hill climbing demonstrably fails, and it does
  not: 0.485 of episodes reach the enumerated maximum against greedy's 0.480,
  a difference of +0.005 with a standard deviation of 0.014 over 16 training
  seeds, 8 of them ahead, at an exact two-sided sign test of p = 1.0. The
  policy does train — an untrained one reaches the maximum on 0.018 — so this
  is a tie rather than a failure to learn. The environment is what bounds it,
  in two ways stated in §2.1 below.
- Any comparison against established software. IQ-TREE 2 and RAxML-NG are not
  installed, and no statement anywhere in the repository compares against them.
- Runtime scaling. Benchmarks are not ranked on CI hardware, so timings live in
  the benchmark suite on fixed hardware rather than in a committed figure.
- Rate variation across sites, and GPU dispatch. Both are specified in
  `docs/tex/` and neither is built.
