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

## §1.2 Requirements Ledger

| Requirement | Status |
| --- | --- |
| Phylogenetic RF ≤0.05 against simulated truth | **Met**, from 125 sites upward ([#148](https://github.com/michaelJwilson/phylo/pull/148)) |
| Potts/HMM parameter recovery within 95% intervals | **Met** for the 1-D chain and the discrete HMM ([#116](https://github.com/michaelJwilson/phylo/pull/116)); lattice outstanding |
| Precise state-sequence decoding | **Not started** — no Viterbi decoder |
| Parity with exact oracles on small `n` | **Met** for tree search against exhaustive enumeration ([#128](https://github.com/michaelJwilson/phylo/pull/128)) |
| Parity with IQ-TREE 2 / RAxML-NG on large `n` | **Not started**; the tools are not in the environment (issue #126) |
| `O(n×L×k)` memory inside 16 GB / 24 GB | **Not measured**; deterministic and reportable, but no figure exists |
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
