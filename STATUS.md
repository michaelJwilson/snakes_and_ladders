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
| §0 Development loop | Landed | Eight required checks; committed PDF byte-compared on every PR | [#49](https://github.com/michaelJwilson/phylo/pull/49), [#57](https://github.com/michaelJwilson/phylo/pull/57), [#72](https://github.com/michaelJwilson/phylo/pull/72), [#92](https://github.com/michaelJwilson/phylo/pull/92), [#102](https://github.com/michaelJwilson/phylo/pull/102), [#151](https://github.com/michaelJwilson/phylo/pull/151) |
| 1.1 Simulation & ground truth | Trees and the HMM landed as first-class simulators; Potts 1-D only | Simulated substitution frequencies against the closed-form JC probabilities; GTR reproduces JC to machine precision; HMM state and emission marginals against brute-force path enumeration | [#58](https://github.com/michaelJwilson/phylo/pull/58), [#64](https://github.com/michaelJwilson/phylo/pull/64), [#115](https://github.com/michaelJwilson/phylo/pull/115), [#120](https://github.com/michaelJwilson/phylo/pull/120), [#182](https://github.com/michaelJwilson/phylo/pull/182) |
| 1.2 Likelihood & energy engine | CPU landed (NumPy, PyTorch, Rust); GPU dispatch not started; belief propagation not started | Worst relative deviation 4.0e-14 against brute-force marginalization across three backends and four site counts spanning a factor of 30 | [#66](https://github.com/michaelJwilson/phylo/pull/66), [#74](https://github.com/michaelJwilson/phylo/pull/74), [#81](https://github.com/michaelJwilson/phylo/pull/81), [#112](https://github.com/michaelJwilson/phylo/pull/112), [#148](https://github.com/michaelJwilson/phylo/pull/148) |
| 1.3 Continuous optimization | Landed for trees, the 1-D Potts chain and the HMM; Potts lattice not started | Gradients against central differences; 95% intervals cover truth at the nominal rate over 60 replicates | [#115](https://github.com/michaelJwilson/phylo/pull/115), [#116](https://github.com/michaelJwilson/phylo/pull/116), [#119](https://github.com/michaelJwilson/phylo/pull/119), [#120](https://github.com/michaelJwilson/phylo/pull/120) |
| 1.4 Move sets & classical baselines | Trees landed; cluster updates and Viterbi not started | NNI and SPR neighbour counts exhaustively verified at `n = 5..8`; hill climbing reaches the enumerated optimum from 12 of 12 starts | [#82](https://github.com/michaelJwilson/phylo/pull/82), [#127](https://github.com/michaelJwilson/phylo/pull/127), [#128](https://github.com/michaelJwilson/phylo/pull/128) |
| 2.1 RL formulation & deployment | Estimator and both environments landed; a trained tree policy not started | Enumerated gradient against finite differences at 1.5e-11 relative; learned policy 86.6% against greedy's 80.2% on the Potts landscape, 8 of 8 seeds | [#135](https://github.com/michaelJwilson/phylo/pull/135), [#137](https://github.com/michaelJwilson/phylo/pull/137), [#139](https://github.com/michaelJwilson/phylo/pull/139) |
| 2.2 Curriculum learning | Not started | — | — |
| 2.3 Empirical validation | Not started | — | — |
| 2.4 Tracking, ablations & leaderboard | Not started | — | — |
| Stage 3 Research extensions | Gumbel-softmax relaxation of Potts and HMM states landed; the tropical Grassmannian half not started, and blocked on an oracle | Relaxation exact at every corner to 1e-11; estimator bias 0.598 to 0.036 as `tau` falls 2.0 to 0.1, standard deviation 0.165 to 3.39 over 20000 draws; deterministic ascent 18/40 against greedy's 5/40, McNemar `p = 0.00098` | [#225](https://github.com/michaelJwilson/phylo/pull/225) |

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

Eight required checks gate a merge, and two of them do work no reviewer can
do by inspection: the technical-document job rebuilds only the QA figures
`docs/tex/main.tex` cites, comparing the rest at the release gate instead
([#157](https://github.com/michaelJwilson/phylo/pull/157)), and fails a pull
request whose rebuilt `docs/draft.pdf` differs from the committed one
([#72](https://github.com/michaelJwilson/phylo/pull/72)); the coverage floor
cannot be lowered to pass a change. Cost is managed rather than absorbed:
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

**Potts: 1-D only.** A Potts chain in an external field exists as an `opt`
reference instance with an exact transfer-matrix oracle
([#115](https://github.com/michaelJwilson/phylo/pull/115)), and appears again
as a `learn` environment. The N-D lattice and general MRFs the milestone
specifies are not built (issue #149).

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

**Potts and HMM evaluators: partial.** The 1-D transfer matrix and the HMM
forward recursion exist, each with its exact oracle. Belief propagation, the
2-D transfer matrix, and a forward-backward routine exposed outside
Baum-Welch's internals are not built.

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

**Not built:** Swendsen-Wang and Wolff cluster updates, Viterbi decoding, and
iterated conditional modes over state paths. Single-flip local search over the
Potts chain exists as an RL environment, not as a classical baseline suite.

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

## Stage 3 — Research Extensions

**Only the half with an oracle is built.** `ROADMAP.md`'s differentiable-search
bullet names two relaxations. Potts configurations and HMM state paths are
enumerable, so the exact optimum, the exact expected score and the exact
gradient are all computable and "does the relaxation find what discrete search
finds" is falsifiable. Tree topologies at any interesting size are not, so the
tropical Grassmannian half is not started and `TICKETS.md` records that it is
blocked on an oracle rather than on effort
([#211](https://github.com/michaelJwilson/phylo/issues/211)).

**The relaxation is an extension, checked at every corner.** Over every
configuration of an enumerable instance the relaxed score equals the discrete
one to `1e-11` relative, for both spaces. The HMM check crosses a module
boundary — `phylo.learn` may not import `phylo.likelihood`, so
`RelaxedHmmPath.discrete` and `phylo.likelihood.hmm_paths.path_log_probability`
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
[#193](https://github.com/michaelJwilson/phylo/pull/193) set.

**The budgets are not the same unit and no claim is made that they are.**
Greedy stops at a local maximum after 3.5 decisions on average at 14 discrete
evaluations each; the relaxation takes gradient steps and evaluates no discrete
configuration until the end. What is matched is the restart count and the
seeds. The advantage is not bought with the larger budget: the relaxation
already wins at 25 gradient steps (15/40, `p = 0.0064`).

**The comparison fixture is not the repository's own.** `potts_params.yaml` has
`J = 0.75 > 0`, so its optimum is `argmax(h)` repeated and every method finds
it. That is the third time a fixture has been too easy to separate methods —
after [#177](https://github.com/michaelJwilson/phylo/issues/177),
[#198](https://github.com/michaelJwilson/phylo/pull/198) and #209's planted
spin glass — and it is why the baseline is now run before any claim is made.

**Not built:** the tropical Grassmannian relaxation; the relaxation on the
Potts *lattice*, where the identity holds but nothing has been measured; and
any joint optimization of structure alongside continuous parameters, which is
what the roadmap bullet ultimately asks for.

## §1.2 Requirements Ledger

| Requirement | Status |
| --- | --- |
| Phylogenetic RF ≤0.05 against simulated truth | **Met**, from 125 sites upward ([#148](https://github.com/michaelJwilson/phylo/pull/148)) |
| Potts/HMM parameter recovery within 95% intervals | **Met** for the 1-D chain and the discrete HMM ([#116](https://github.com/michaelJwilson/phylo/pull/116)); lattice outstanding |
| Precise state-sequence decoding | **Not started** — no Viterbi decoder |
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

- That a learned policy beats hill climbing on trees. A fixture that could
  settle it now exists — 7 taxa, internal branches an order of magnitude
  shorter than the pendant ones, where NNI hill climbing reaches the
  enumerated maximum from 24 of 50 seeded starts and stops at a genuine local
  optimum on the other 26 — but no policy has been trained on it and no
  budget-matched comparison has been run (issue #178). The 6-taxon fixture
  cannot support the claim in either direction, because greedy reaches the
  enumerated optimum from every start there.
- Any comparison against established software. IQ-TREE 2 and RAxML-NG are not
  installed, and no statement anywhere in the repository compares against them.
- Runtime scaling. Benchmarks are not ranked on CI hardware, so timings live in
  the benchmark suite on fixed hardware rather than in a committed figure.
- Rate variation across sites, and GPU dispatch. Both are specified in
  `docs/tex/` and neither is built.
