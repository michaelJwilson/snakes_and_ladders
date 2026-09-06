# STATUS

What has landed against `ROADMAP.md`, how it was established, and the pull
request that carries it. Read at version `0.3.0`.

This file is a ledger against the roadmap, not a project board. Open work lives
in GitHub issues, and its titles are collected in `TICKETS.md`. A milestone is
recorded here as **landed** only where an independent oracle pins the claim; a
capability that runs but is checked against nothing is recorded as **not
started**, on the terms §0.4 sets.

## Summary

| Roadmap item | Status | Evidence | Key PRs |
| --- | --- | --- | --- |
| §0 Development loop | Landed | Nine required checks; committed PDF byte-compared and every notebook re-executed on each PR | [#49](https://github.com/michaelJwilson/phylo/pull/49), [#57](https://github.com/michaelJwilson/phylo/pull/57), [#72](https://github.com/michaelJwilson/phylo/pull/72), [#92](https://github.com/michaelJwilson/phylo/pull/92), [#102](https://github.com/michaelJwilson/phylo/pull/102), [#151](https://github.com/michaelJwilson/phylo/pull/151) |
| 1.1 Simulation & ground truth | Trees, the HMM and Potts (1-D chain plus general N-D lattice/MRF) landed as first-class simulators | Simulated substitution frequencies against the closed-form JC probabilities; GTR reproduces JC to machine precision; HMM state and emission marginals against brute-force path enumeration; Potts single-site and pair marginals against exhaustive enumeration at 3-state 3x3 and 2-state 4x4 | [#58](https://github.com/michaelJwilson/phylo/pull/58), [#64](https://github.com/michaelJwilson/phylo/pull/64), [#115](https://github.com/michaelJwilson/phylo/pull/115), [#120](https://github.com/michaelJwilson/phylo/pull/120), [#182](https://github.com/michaelJwilson/phylo/pull/182), [#190](https://github.com/michaelJwilson/phylo/pull/190) |
| 1.2 Likelihood & energy engine | CPU landed (NumPy, PyTorch, Rust); belief propagation landed with two exact oracles; GPU dispatch not started | Worst relative deviation 4.0e-14 against brute-force marginalization across three backends and four site counts spanning a factor of 30 | [#66](https://github.com/michaelJwilson/phylo/pull/66), [#74](https://github.com/michaelJwilson/phylo/pull/74), [#81](https://github.com/michaelJwilson/phylo/pull/81), [#112](https://github.com/michaelJwilson/phylo/pull/112), [#148](https://github.com/michaelJwilson/phylo/pull/148) |
| 1.3 Continuous optimization | Landed for trees, the 1-D Potts chain and the HMM; Potts lattice not started | Gradients against central differences; 95% intervals cover truth at the nominal rate over 60 replicates | [#115](https://github.com/michaelJwilson/phylo/pull/115), [#116](https://github.com/michaelJwilson/phylo/pull/116), [#119](https://github.com/michaelJwilson/phylo/pull/119), [#120](https://github.com/michaelJwilson/phylo/pull/120) |
| 1.4 Move sets & classical baselines | Trees landed; Potts cluster updates landed; Viterbi not started | NNI and SPR neighbour counts exhaustively verified at `n = 5..8`; hill climbing reaches the enumerated optimum from 12 of 12 starts | [#82](https://github.com/michaelJwilson/phylo/pull/82), [#127](https://github.com/michaelJwilson/phylo/pull/127), [#128](https://github.com/michaelJwilson/phylo/pull/128) |
| 2.1 RL formulation & deployment | Estimator and both environments landed; a trained tree policy not started | Enumerated gradient against finite differences at 1.5e-11 relative; learned policy 86.6% against greedy's 80.2% on the Potts landscape, 8 of 8 seeds | [#135](https://github.com/michaelJwilson/phylo/pull/135), [#137](https://github.com/michaelJwilson/phylo/pull/137), [#139](https://github.com/michaelJwilson/phylo/pull/139) |
| 2.2 Curriculum learning | Not started | — | — |
| 2.3 Empirical validation | Not started | — | — |
| 2.4 Tracking, ablations & leaderboard | Not started | — | — |
| Stage 3 Research extensions | Not started | — | — |

## §0 The Development Loop

The loop described in `ROADMAP.md` §0 is in force rather than aspirational.
Blank issues are disabled and filings route through the task template
([#57](https://github.com/michaelJwilson/phylo/pull/57)); the pull-request
template carries the Definition of Done, the benchmark table, the realized
tolerance table, and the deferred-work section
([#49](https://github.com/michaelJwilson/phylo/pull/49),
[#86](https://github.com/michaelJwilson/phylo/pull/86),
[#89](https://github.com/michaelJwilson/phylo/pull/89)); labels are generated
from `.github/labels.yml` by a workflow, so the taxonomy cannot drift from the
documents that describe it.

Nine required checks gate a merge, and three of them do work no reviewer can
do by inspection: the technical-document job rebuilds only the QA figures
`docs/tex/main.tex` cites, comparing the rest at the release gate instead
([#157](https://github.com/michaelJwilson/phylo/pull/157)), and fails a pull
request whose rebuilt `docs/draft.pdf` differs from the committed one
([#72](https://github.com/michaelJwilson/phylo/pull/72)); the notebooks job
re-executes every notebook under `docs/nb/` and fails one whose printed
output has moved; and the coverage floor cannot be lowered to pass a change. Cost is managed rather than absorbed:
benchmarks run only when the diff touches code they measure, and the
release-gated suite is excluded per pull request — measured at 138 s over 540
tests against 989 s for the full suite
([#159](https://github.com/michaelJwilson/phylo/pull/159)).

Two releases have been cut under the procedure, each from a Release ticket
gated on `infra/release.sh`: `0.1.0`
([#102](https://github.com/michaelJwilson/phylo/pull/102)) and `0.2.0`
([#151](https://github.com/michaelJwilson/phylo/pull/151)). Each ran the
consolidation audit the template drives, and `0.2.0`'s found real defects — a
categorical sampler duplicated three times, two copies missing the clamp the
third had, so a probability row summing to `1 - 4e-16` could return a category
past the end of the alphabet.

**Between `0.2.0` and `0.3.0`, six pull requests refined the loop and its
record; no roadmap milestone moved.** `ROADMAP.md` was restructured around the
development loop and the three problem classes, and `STATUS.md` and
`TICKETS.md` were introduced as the ledger and backlog this section and
`TICKETS.md` now are
([#152](https://github.com/michaelJwilson/phylo/pull/152),
[#153](https://github.com/michaelJwilson/phylo/pull/153)). The thirteen QA
scripts were routed through one `phylo.qa.runner` rather than each carrying
its own argument parsing and figure-closing boilerplate
([#156](https://github.com/michaelJwilson/phylo/pull/156)), and
`phylo.qa.manifest` now states which figure renders each output so a build can
select a subset rather than regenerate all thirteen
([#157](https://github.com/michaelJwilson/phylo/pull/157)). The regression
suite was split by submodule and its documented budget corrected after being
found stale
([#159](https://github.com/michaelJwilson/phylo/pull/159)). Every module
`CLAUDE.md` now points at the Writing Style section instead of restating it
([#158](https://github.com/michaelJwilson/phylo/pull/158)), and a generated
plan's required shape — 2–5 validated steps ending in an Open Questions
section — is stated in `ROADMAP.md` §0.2, `DEV.md`, and `infra/CLAUDE.md`
alike, alongside the rule that decides which documents may repeat detail
([#164](https://github.com/michaelJwilson/phylo/pull/164)).

## Milestone 1.1 — Simulation & Ground Truth Engine

**Phylogenetics: landed.** A `k`-state Jukes-Cantor simulator generates an
alignment and the ancestral tree in Newick from a typed
`simulation_params.yaml`, retaining the parameters that produced them
([#58](https://github.com/michaelJwilson/phylo/pull/58)). Simulated
substitution frequencies are validated against the closed-form JC transition
probabilities within a yaml-declared Monte Carlo tolerance across several site
and taxon counts. Newick counting, validation and state-labelled serialization
are the package's single source of that functionality
([#64](https://github.com/michaelJwilson/phylo/pull/64)).

The general time-reversible model landed with the fitting work that needed it
([#120](https://github.com/michaelJwilson/phylo/pull/120)): Jukes-Cantor has no
free parameters, so there was nothing to recover without it. It is validated by
reduction — equal exchangeabilities with a uniform `π` reproduce the
Jukes-Cantor rate matrix and its closed-form transition probabilities to
machine precision.

**Potts: 1-D chain plus a general N-D lattice/MRF simulator.** The 1-D chain
in an external field still exists as an `opt` reference instance with an
exact transfer-matrix oracle ([#115](https://github.com/michaelJwilson/phylo/pull/115)),
and appears again as a `learn` environment. `phylo.sim.graph.PottsGraph`
now generalizes it to an arbitrary undirected graph with a per-edge
coupling, and `phylo.sim.potts.simulate_potts` samples on it — exactly, by
the same backward-message recursion, when the graph is a 1-D open chain, and
by single-site Gibbs (heat-bath) MCMC otherwise — with an N-D lattice a
constructed case of the general graph rather than a second code path
([#190](https://github.com/michaelJwilson/phylo/pull/190), closing #170,
superseding the
sampling half of #149). `phylo.opt.potts.simulate_chains` cannot import
`phylo.sim` under `opt/CLAUDE.md`'s "no application imports" rule, so it
keeps its own copy of the exact recursion rather than delegating to the new
one — a duplication [#186](https://github.com/michaelJwilson/phylo/issues/186)
tracks resolving, by moving `PottsParams` into `phylo.sim.potts` the way
#171 moved the HMM's truth type. No fitting, cluster updates, or evaluator
on the general graph yet (issues #172, #174).

**HMMs: a first-class simulator.** `phylo.sim.hmm` draws a hidden state path
and an observation sequence jointly from a declared `(pi, A, B)`, retaining
the path alongside the data on the footing the tree simulator already has
([#182](https://github.com/michaelJwilson/phylo/pull/182), closing
[#171](https://github.com/michaelJwilson/phylo/issues/171)). The generator
embedded in `phylo.opt.hmm` — which validated only against brute-force path
enumeration for the fitting objective's own use
([#115](https://github.com/michaelJwilson/phylo/pull/115)) — is deleted; `opt`
now imports the truth type from `sim` and draws no data itself. Validated
against brute-force enumeration for the per-position state and emission
marginals, self-normalized importance sampling against the exact path
posterior for one realized observation, and the transition matrix's own
stationary distribution for long-run occupancy.

## Milestone 1.2 — Differentiable Likelihood & Energy Engine

**Felsenstein pruning: three CPU backends, one oracle.** Vectorized NumPy is
the reference, with per-node rescaling accumulated in log space
([#66](https://github.com/michaelJwilson/phylo/pull/66)); differentiable
PyTorch takes branch lengths as a tensor separate from the topology so
`torch.autograd` differentiates through them
([#74](https://github.com/michaelJwilson/phylo/pull/74)); Rust implements the
same recursion behind PyO3
([#81](https://github.com/michaelJwilson/phylo/pull/81)). Every one is pinned
against an independent brute-force marginalizer rather than against another
backend: worst relative deviation 4.0e-14 across all three and four site counts
spanning a factor of 30
([#148](https://github.com/michaelJwilson/phylo/pull/148)). The pulley
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
([#112](https://github.com/michaelJwilson/phylo/pull/112)). Both figures are
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
`phylo.sim.graph.erdos_renyi_graph` draws `G(n, p)` beside `lattice_graph`,
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
`phylo.sim.canonical` holds instances whose answer comes from outside this
repository, admitted on two clauses stated in `sim/CLAUDE.md`: the answer must
be independently known, and more than one module must consume it
([#209](https://github.com/michaelJwilson/phylo/issues/209)).

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
([#207](https://github.com/michaelJwilson/phylo/pull/207)) and Goemans-Williamson's
0.87856 ([#215](https://github.com/michaelJwilson/phylo/pull/215)) are the only
other discrete claims here that survive past enumeration, and both are bounds
rather than answers.

**Wannier's residual entropy is reported, never asserted.** The constant
0.3231 per site is a thermodynamic limit. Measured: 0.4153 at `N = 9`, 0.3516
at `N = 12`, 0.2336 at `N = 16` — not close, and not monotone, because a `4x4`
torus is incommensurate with the three-sublattice ground state. The exact
degeneracies (42, 68, 42) are asserted instead. This is the same discipline
[#214](https://github.com/michaelJwilson/phylo/pull/214) arrived at for graph
threshold results.

**The planted spin glass does not replace #177, and this is measured rather
than assumed.** It was proposed as the instance whose difficulty scales, after
[#198](https://github.com/michaelJwilson/phylo/pull/198) measured random-restart
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
([#206](https://github.com/michaelJwilson/phylo/pull/206)). A forward-backward
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

## Milestone 1.3 — Continuous Optimization via Autodiff

**The interface is model-agnostic, and that is measured rather than asserted.**
An `Objective` is an unconstrained parameter vector, a differentiable scalar,
and a map back to named constrained parameters
([#115](https://github.com/michaelJwilson/phylo/pull/115)). Four instances now
run against it unchanged — the Potts chain, the HMM, branch lengths on a fixed
topology, and the GTR substitution model — and none required a change to
`phylo.opt`. A test asserts the module imports nothing from `phylo.sim`,
`phylo.likelihood` or `phylo.search`, so the separation cannot decay by
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

**Fitting and intervals.** L-BFGS with a strong-Wolfe line search, convergence
judged on the gradient relative to the objective's own magnitude, and
confidence intervals from the observed Fisher information pushed through the
constraint map by the delta method
([#116](https://github.com/michaelJwilson/phylo/pull/116)). Validation is
parameter recovery, not convergence: the Potts chain's 95% intervals cover the
truth at exactly the nominal rate over 60 replicates, and the HMM's gradient
fit is cross-checked against Baum-Welch — an independent algorithm sharing no
optimizer, parameterization or constraint map with it.

**The phylogenetic instance.** Branch lengths are recovered within their
intervals on both the unrooted and rooted fixtures, and exchangeabilities and
`π` alongside them
([#119](https://github.com/michaelJwilson/phylo/pull/119),
[#120](https://github.com/michaelJwilson/phylo/pull/120)). Two properties fell
out of doing it: the two branches below a rooted root are estimable only as
their sum, so they are fitted as one parameter and reported summed; and the
GTR model's three normalizations are gauges rather than conventions, each
removing an exactly flat direction that would otherwise leave every parameter
without an interval. The roadmap's sub-second gradient update at `n = 100` is
now measured — 203 ms at 1000 sites — rather than assumed.

## Milestone 1.4 — Discrete Move Sets & Classical Baselines

**NNI and SPR: landed and counted.** Both neighbourhoods sit behind one
`Topology -> Iterator[Topology]` interface and are verified exhaustively
against `2(n - 3)` and `2(n - 3)(2n - 7)` at `n = 5..8`, over every distinct
topology, with neighbour validity, symmetry and NNI-in-SPR containment
cross-checked ([#82](https://github.com/michaelJwilson/phylo/pull/82)).

**Hill climbing, with an oracle that settles the question.** `infer` climbs
over either neighbourhood, fitting the continuous parameters of every candidate
([#127](https://github.com/michaelJwilson/phylo/pull/127)). Exhaustive
enumeration of unrooted topologies gives search quality an independent
reference below 8 taxa, so "did it find the best tree" has an answer
([#128](https://github.com/michaelJwilson/phylo/pull/128)): on the 6-taxon
fixture both move sets reach the enumerated maximum and recover the generating
topology from all 12 starting points, at a median of 14 candidate fits for NNI
against 48 for SPR. Budgets are counted in candidate fits rather than seconds,
so a run reproduces from its seed, and a topology is scored at most once per
search, keyed on its leaf bipartitions. The fit is the only unit worth
counting: one candidate fit measures 213 ms against 22 us to generate an
entire NNI neighbourhood, a factor of about 10 000.

**The accuracy requirement's first half is met.** Normalized Robinson-Foulds
distance from the inferred to the generating topology is met at the 0.05 bound
from 125 sites upward, with 8 of 8 replicates recovering the topology exactly at
2000 sites against 5 of 8 at 60
([#148](https://github.com/michaelJwilson/phylo/pull/148)). The normalizer
counts internal splits only: every tree over the same leaves induces all the
trivial ones, and including them would shrink every distance by a
taxon-count-dependent factor and silently weaken the bound.

**Potts cluster updates landed, validated by the distribution they converge
to** ([#212](https://github.com/michaelJwilson/phylo/pull/212)). Swendsen-Wang
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
boundary is open, both of which soften the transition.

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

A Rust kernel (`src/maxflow.rs`) runs **28-34x** faster than the NumPy
reference measured on its own, and **6.6-10.6x** as a caller sees it; the
difference is the list marshalling crossing the FFI boundary, which is the
same gap #202 closes for the categorical sampler and is deferred to it rather
than solved twice. The reference stays as the oracle. The port also removes a
fragility: the
Python blocking flow recurses to the depth of the level graph and needs
`setrecursionlimit` raised past a few thousand nodes, while the Rust one uses
an explicit stack.

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

**Not built:** Viterbi decoding, and iterated conditional modes over HMM state
paths (`phylo.search.alpha_expansion` carries a lattice ICM as its baseline,
which is a different object). Single-flip local search over the Potts chain exists as an RL
environment, not as a classical baseline suite.

## Milestone 2.1 — RL Agent Formulation & Deployment

**The estimator is pinned to a closed form, not to a training curve**
([#135](https://github.com/michaelJwilson/phylo/pull/135)). With a finite
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
about learning rather than about two unrelated algorithms.

**The phylogenetic environment exists, and the reward it can afford is
measured** ([#137](https://github.com/michaelJwilson/phylo/pull/137)). A state
is a topology, an action an NNI or SPR neighbour, the reward the improvement in
log-likelihood. Two reward models are implemented and the comparison between
them is the deliverable: fitting branch lengths per candidate costs 113.7 ms
against 352 us at fixed known parameters, a factor of 323, and only the second
makes an episode affordable. The substitution is validated rather than assumed
([#139](https://github.com/michaelJwilson/phylo/pull/139)): the two surfaces
score the generating topology highest, agree on the best of all 105 topologies,
and correlate at 0.9568, holding agreement on the best topology across a 50-fold
range of the fixed branch length with correlation never below 0.8719.

Measuring that recorded a property of the fitted surface worth having in
writing: it does not totally order topologies. Many candidates share a
maximized log-likelihood to within the optimizer's convergence, because the
branch distinguishing them fits to zero and the tree collapses to the same
polytomy — so a rank correlation moves by up to 0.04 under a perturbation of
one part in 1e9 and is not a measurement.

**All three problem classes are now MDPs.** `phylo.learn.Environment` had
one instance, a 1-D Potts chain, which is the same position `phylo.opt` was in
before four instances made its model-agnosticism a measurement rather than an
assertion. It now carries the Potts landscape over an arbitrary graph — the
chain is the one-dimensional case of the same class, not a second one — and
the hidden Markov state path, whose objective is a decoding problem rather
than an energy. Both are pinned against the enumerated estimator oracle
carried over unchanged from the chain, and against exhaustive enumeration of
their own state spaces: 19,683 configurations for a 3-state 3x3 lattice, 729
paths for a 3-state sequence of six. Neither takes an application type, so
`phylo.learn` still imports nothing from `phylo.sim`, `phylo.likelihood` or
`phylo.search`, and a test asserts it.

**The lattice is fitted, against an exact normalizer.** `log Z` is
enumerated over all 19,683 configurations of a 3-state 3x3 lattice rather than
approximated, so the fitted optimum is checked against a brute-force scan of
the likelihood instead of against the optimizer's own convergence, and the
enumerated normalizer reduces to `phylo.opt.potts.log_partition`'s transfer
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

## §1.2 Requirements Ledger

| Requirement | Status |
| --- | --- |
| Phylogenetic RF ≤0.05 against simulated truth | **Met**, from 125 sites upward ([#148](https://github.com/michaelJwilson/phylo/pull/148)) |
| Potts/HMM parameter recovery within 95% intervals | **Met** for the 1-D chain, the discrete HMM ([#116](https://github.com/michaelJwilson/phylo/pull/116)) and the 2-D lattice — realized 0.981 at 100 samples and 0.956 at 400 and 1600, over 40 replicates each |
| Precise state-sequence decoding | **Not started** — no Viterbi decoder (issue #175) |
| Parity with exact oracles on small `n` | **Met** for tree search against exhaustive enumeration ([#128](https://github.com/michaelJwilson/phylo/pull/128)) |
| Parity with IQ-TREE 2 / RAxML-NG on large `n` | **Not started**; the tools are not in the environment (issue #126) |
| `O(n×L×k)` memory inside 16 GB / 24 GB | **Met**: 87.2 MB at 100 taxa by 11,000 sites on the worst-case (caterpillar) topology, 879 MB at the declared maximum — a factor of 20 inside 16 GB, with each figure pinned against the allocator to within 2.5% |
| CUDA, Metal/MPS and CPU dispatch | **CPU only**; selection logic landed ([#112](https://github.com/michaelJwilson/phylo/pull/112)), accelerator paths not implemented |
| Declared cross-device tolerance, not bitwise | **Met**: `1e-11` relative in `float64`, `1e-6` where either side is `float32` ([#112](https://github.com/michaelJwilson/phylo/pull/112)) |

## §1.3 The Technical Document

`docs/tex/` now spans all three problem classes rather than the phylogenetic
application alone: the abstract, methods and appendices state the Potts
Hamiltonian and the HMM decoding problem beside the substitution model, and the
Reference Taxonomy appendix routes the literature by concern. It is an eight-page
specification, cut down in `14d32d6` from the academic-letter structure of
[#148](https://github.com/michaelJwilson/phylo/pull/148), and it is the shape
the document is in rather than the shape §1.3 asks for.

Thirteen QA scripts run in the build, each committing a figure with a caption
naming the seed, the sizes and the model that produced it, and `docs/CLAUDE.md`
states the rules that keep a CI-regenerated artifact true
([#140](https://github.com/michaelJwilson/phylo/pull/140)). The document
currently includes two of them — the worked simulation example and the backend
agreement — so eleven committed figures are rebuilt by CI but cited nowhere.

Measured against §1.3's required contents: the model formulations are present
for all three classes, at the level of a statement rather than a derivation.
Absent are the derivations of pruning, belief propagation and forward-backward;
the branch-and-bound bounds and their proofs, no such bound being implemented;
and the parameter-recovery and convergence evidence, which exists as committed
QA figures but is no longer included. Three framed placeholders stand in for
the RL learning curve, the comparison against classical software, and hardware
scaling — none of which is measured, and each labelled as a placeholder rather
than drawn with invented data.

## What Is Not Claimed

- That a learned policy beats hill climbing on trees. The 6-taxon fixture
  cannot support the claim in either direction, because greedy already reaches
  the enumerated optimum from every start. Separating a policy from greedy
  needs a problem harder than exhaustive enumeration can referee, so the oracle
  that validates the search cannot validate the agent replacing it
  (issues #177 and #178).
- Any comparison against established software. IQ-TREE 2 and RAxML-NG are not
  installed, and no statement anywhere in the repository compares against them.
- Runtime scaling. Benchmarks are not ranked on CI hardware, so timings live in
  the benchmark suite on fixed hardware rather than in a committed figure.
- Rate variation across sites, and GPU dispatch. Both are specified in
  `docs/tex/` and neither is built.
