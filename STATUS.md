# STATUS

What has landed against `ROADMAP.md`, how it was established, and the pull
request that carries it. Read at version `0.5.0`, the release
[#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376) audits. The cut of `0.4.0` is deferred and the
repository still carries no tag, so the release template's tag precondition is
recorded here as not met rather than waited for; `Cargo.toml` reads `0.3.0`,
which is the version the last built `CHANGELOG.md` section carries.

This file is a ledger against the roadmap, not a project board. Open work lives
in GitHub issues, and its titles are collected in `TICKETS.md`. A milestone is
recorded here as **landed** only where an independent oracle pins the claim; a
capability that runs but is checked against nothing is recorded as **not
started**, on the terms §0.4 sets.

## Summary

The checks each row rests on are listed, per test, in `CHECKS.md`, generated
from the suite by `infra/ledgers.sh` and not committed (issue #425);
`PROBLEMS.md` names the code behind each problem class.

| Roadmap item | Status | Evidence | Key PRs |
| --- | --- | --- | --- |
| §0 Development loop | Landed, and its record widened since 0.3.0: the problem catalogue and checks ledger, the experiment ledger, the run logger, kind markers on every test, one fixture API, citation-driven figure builds with label guards, notebook re-execution, the frameworks extra as referees, and the parallelism seam measured as a negative result | Ten required checks; committed PDFs byte-compared and every notebook re-executed on each PR; `0.3.0` was built into `CHANGELOG.md` on 2026-09-03 and never tagged, so `0.4.0` tags both (#358) | [#49](https://github.com/michaelJwilson/snakes_and_ladders/pull/49), [#57](https://github.com/michaelJwilson/snakes_and_ladders/pull/57), [#72](https://github.com/michaelJwilson/snakes_and_ladders/pull/72), [#92](https://github.com/michaelJwilson/snakes_and_ladders/pull/92), [#102](https://github.com/michaelJwilson/snakes_and_ladders/pull/102), [#151](https://github.com/michaelJwilson/snakes_and_ladders/pull/151), [#205](https://github.com/michaelJwilson/snakes_and_ladders/pull/205), [#243](https://github.com/michaelJwilson/snakes_and_ladders/pull/243), [#248](https://github.com/michaelJwilson/snakes_and_ladders/pull/248), [#285](https://github.com/michaelJwilson/snakes_and_ladders/pull/285), [#294](https://github.com/michaelJwilson/snakes_and_ladders/pull/294), [#316](https://github.com/michaelJwilson/snakes_and_ladders/pull/316), [#318](https://github.com/michaelJwilson/snakes_and_ladders/pull/318), [#353](https://github.com/michaelJwilson/snakes_and_ladders/pull/353), [#355](https://github.com/michaelJwilson/snakes_and_ladders/pull/355), [#375](https://github.com/michaelJwilson/snakes_and_ladders/pull/375), [#379](https://github.com/michaelJwilson/snakes_and_ladders/pull/379), [#380](https://github.com/michaelJwilson/snakes_and_ladders/pull/380) |
| 1.1 Simulation & ground truth | Trees, the HMM under six emission families, Potts (1-D chain plus general N-D lattice/MRF and `G(n, p)`), the Gaussian mixture, the coupled spatio-sequential model, three canonical fixtures with outside answers, and the LDPC code landed as first-class simulators | Simulated substitution frequencies against the closed-form JC probabilities; GTR reproduces JC to machine precision; HMM state and emission marginals against brute-force path enumeration; the count families reach their Poisson and binomial limits at the `O(1/x)` rate; Potts single-site and pair marginals against exhaustive enumeration at 3-state 3x3 and 2-state 4x4; the coupled simulator held to what it composes by chi-square at 0.001 over 400 draws; every encoded codeword satisfies `H c = 0` at `n = 24`, `96`, `510`, and a codeword's channel ratios are the zero word's up to sign on every realization | [#58](https://github.com/michaelJwilson/snakes_and_ladders/pull/58), [#64](https://github.com/michaelJwilson/snakes_and_ladders/pull/64), [#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115), [#120](https://github.com/michaelJwilson/snakes_and_ladders/pull/120), [#182](https://github.com/michaelJwilson/snakes_and_ladders/pull/182), [#190](https://github.com/michaelJwilson/snakes_and_ladders/pull/190), [#223](https://github.com/michaelJwilson/snakes_and_ladders/pull/223), [#224](https://github.com/michaelJwilson/snakes_and_ladders/pull/224), [#259](https://github.com/michaelJwilson/snakes_and_ladders/pull/259), [#261](https://github.com/michaelJwilson/snakes_and_ladders/pull/261), [#263](https://github.com/michaelJwilson/snakes_and_ladders/pull/263), [#302](https://github.com/michaelJwilson/snakes_and_ladders/pull/302), [#356](https://github.com/michaelJwilson/snakes_and_ladders/pull/356) |
| 1.2 Likelihood & energy engine | CPU landed (NumPy, PyTorch, Rust at 2.5x the oracle at 200 taxa by 11,000 sites); belief propagation with two exact oracles; Fitch and Sankoff parsimony; one factor graph with sum-product and max-product over it, Viterbi included; forward–backward as an evaluator; certified bounds and learned surrogates; the LDPC decoder pinned to the general sum-product and to enumeration; two runtime audits; GPU dispatch not started (#280) | Worst relative deviation 4.0e-14 against brute-force marginalization across three backends and four site counts spanning a factor of 30; max-product returns the enumerated Viterbi path on four chains; flooding on the 8x8 lattice within 1.8x of belief propagation after the audit; decoder posteriors within 4.6e-11 of the general flooding on six loopy codes and 1.9e-13 of enumeration on a cycle-free one; the 19,998-bit (3,6) code brackets the erasure threshold 0.4294 between 0.42 and 0.44 | [#66](https://github.com/michaelJwilson/snakes_and_ladders/pull/66), [#74](https://github.com/michaelJwilson/snakes_and_ladders/pull/74), [#81](https://github.com/michaelJwilson/snakes_and_ladders/pull/81), [#112](https://github.com/michaelJwilson/snakes_and_ladders/pull/112), [#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148), [#219](https://github.com/michaelJwilson/snakes_and_ladders/pull/219), [#247](https://github.com/michaelJwilson/snakes_and_ladders/pull/247), [#296](https://github.com/michaelJwilson/snakes_and_ladders/pull/296), [#307](https://github.com/michaelJwilson/snakes_and_ladders/pull/307), [#317](https://github.com/michaelJwilson/snakes_and_ladders/pull/317), [#343](https://github.com/michaelJwilson/snakes_and_ladders/pull/343), [#346](https://github.com/michaelJwilson/snakes_and_ladders/pull/346), [#354](https://github.com/michaelJwilson/snakes_and_ladders/pull/354), [#356](https://github.com/michaelJwilson/snakes_and_ladders/pull/356) |
| 1.3 Continuous optimization | Landed for trees, the HMM, the Potts chain and lattice, and the mixture; an interval at any fit, whatever produced it; posterior sampling by leapfrog and Yoshida, tempered and adapted; initializers, multi-start, k-means++ and, since #373, the tree's two data-driven starts — neighbor joining on pairwise distances and the closest tree of the Hadamard conjugation (#364); closed-form test functions; the mixture at equal evaluations through the budget utility | Gradients against central differences; 95% intervals cover truth at the nominal rate over 60 replicates; the lattice fitted against an enumerated normalizer, coverage 157/160 at 100 samples and 153/160 at 400 and 1600; integrator orders realized at 4.000 and 16.001; the adapted chain's acceptance 0.650 pooled over 20 seeds at a target of 0.65; restarts reach the mixture's best-known optimum from 7/40 starts against tempering's 4/40 (`p = 0.549`) and annealing's 1/40 (`p = 0.031`); neighbor joining returns every branch of a known tree to 1e-12 at 20 and 50 taxa, and no start separates on the fit at five and six taxa, every one converging in 22 to 27 evaluations (experiment 006) | [#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115), [#116](https://github.com/michaelJwilson/snakes_and_ladders/pull/116), [#119](https://github.com/michaelJwilson/snakes_and_ladders/pull/119), [#120](https://github.com/michaelJwilson/snakes_and_ladders/pull/120), [#256](https://github.com/michaelJwilson/snakes_and_ladders/pull/256), [#263](https://github.com/michaelJwilson/snakes_and_ladders/pull/263), [#269](https://github.com/michaelJwilson/snakes_and_ladders/pull/269), [#271](https://github.com/michaelJwilson/snakes_and_ladders/pull/271), [#272](https://github.com/michaelJwilson/snakes_and_ladders/pull/272), [#303](https://github.com/michaelJwilson/snakes_and_ladders/pull/303), [#345](https://github.com/michaelJwilson/snakes_and_ladders/pull/345), [#348](https://github.com/michaelJwilson/snakes_and_ladders/pull/348), [#352](https://github.com/michaelJwilson/snakes_and_ladders/pull/352), [#373](https://github.com/michaelJwilson/snakes_and_ladders/pull/373) |
| 1.4 Move sets & classical baselines | Trees landed, with warm starts, lazy scoring and support; large parsimony; Potts cluster updates; the exact-baseline family — minimum cut, alpha expansion with its proved bound, Max-Cut with a certificate; schedules, annealing and parallel tempering with an opt-in Rust sweep; one Gibbs sampler and annealer over any factor graph, and a tempered ensemble over labellings, decodings and topologies; the coupled model fitted; Viterbi and posterior decoding landed; iterated conditional modes over HMM paths not started (#176) | NNI and SPR neighbour counts exhaustively verified at `n = 5..8`; hill climbing reaches the enumerated optimum from 12 of 12 starts and large parsimony from every start at five and six taxa; the two-state ground state exact against enumeration over 36 shape-coupling-field combinations; on the planted glass at equal sweeps tempering 12/12 against annealing 10/12 and restarts 4/12; the tempered weight within 0.039 of enumeration on every seed of 20 | [#82](https://github.com/michaelJwilson/snakes_and_ladders/pull/82), [#127](https://github.com/michaelJwilson/snakes_and_ladders/pull/127), [#128](https://github.com/michaelJwilson/snakes_and_ladders/pull/128), [#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148), [#212](https://github.com/michaelJwilson/snakes_and_ladders/pull/212), [#220](https://github.com/michaelJwilson/snakes_and_ladders/pull/220), [#221](https://github.com/michaelJwilson/snakes_and_ladders/pull/221), [#222](https://github.com/michaelJwilson/snakes_and_ladders/pull/222), [#255](https://github.com/michaelJwilson/snakes_and_ladders/pull/255), [#272](https://github.com/michaelJwilson/snakes_and_ladders/pull/272), [#284](https://github.com/michaelJwilson/snakes_and_ladders/pull/284), [#289](https://github.com/michaelJwilson/snakes_and_ladders/pull/289), [#304](https://github.com/michaelJwilson/snakes_and_ladders/pull/304), [#307](https://github.com/michaelJwilson/snakes_and_ladders/pull/307), [#310](https://github.com/michaelJwilson/snakes_and_ladders/pull/310), [#346](https://github.com/michaelJwilson/snakes_and_ladders/pull/346), [#350](https://github.com/michaelJwilson/snakes_and_ladders/pull/350) |
| 2.1 RL formulation & deployment | The estimator, the Potts, hidden-path and tree environments, a critic, an actor–critic, PPO and a PUCT planner landed, each pinned to enumeration; a tree policy trained on the fixture hill climbing fails, a tie over one feature and ahead of greedy over the seven-column set (#349); not yet measured against restarts | Enumerated gradient against finite differences at 1.5e-11 relative; on the Potts chain REINFORCE 86.6%, PPO 96.3% and the planner 92.6% at 8.3 evaluations per episode against greedy's 80.2% at 48; on the 7-taxon fixture the single feature reaches 0.487 against greedy's 0.480 (sign test `p = 0.79`) and the full set 0.796, ahead on 16 of 16 seeds (`p = 3.05e-5`), while restarts reach 1.000 at the same budget | [#135](https://github.com/michaelJwilson/snakes_and_ladders/pull/135), [#137](https://github.com/michaelJwilson/snakes_and_ladders/pull/137), [#139](https://github.com/michaelJwilson/snakes_and_ladders/pull/139), [#192](https://github.com/michaelJwilson/snakes_and_ladders/pull/192), [#193](https://github.com/michaelJwilson/snakes_and_ladders/pull/193), [#198](https://github.com/michaelJwilson/snakes_and_ladders/pull/198), [#320](https://github.com/michaelJwilson/snakes_and_ladders/pull/320), [#349](https://github.com/michaelJwilson/snakes_and_ladders/pull/349), [#355](https://github.com/michaelJwilson/snakes_and_ladders/pull/355) |
| 2.2 Curriculum learning | Started: the surrogate curriculum from 5 to 6 taxa and from 3x3 to 4x6 lattices; weight transfer for a policy and batched rollout not started | Zero-shot at six taxa the set surrogate falls to `R^2` 0.68 and recovers to 0.94 after transfer, the MLP holds 0.92 and reaches 0.95; lattice surrogates transfer zero-shot at 0.99 | [#317](https://github.com/michaelJwilson/snakes_and_ladders/pull/317) |
| 2.3 Empirical validation | The budget utility and the exact paired test landed, and six budget-matched comparisons are recorded; no empirical alignment and no external tool ([#126](https://github.com/michaelJwilson/snakes_and_ladders/issues/126)) | Every comparison at one budget over shared seeds with McNemar's exact test: the glass, Rastrigin, the mixture, the relaxation against greedy, the cluster updates at the transition, and the tree's starts at equal evaluations | [#303](https://github.com/michaelJwilson/snakes_and_ladders/pull/303), [#348](https://github.com/michaelJwilson/snakes_and_ladders/pull/348) |
| 2.4 Tracking, ablations & leaderboard | The experiment ledger, its generated index and the run logger landed; six experiments recorded, each capped at a ten-line body (#458); the Aim run store not started ([#75](https://github.com/michaelJwilson/snakes_and_ladders/issues/75)) | Every file under `docs/experiments/` validated against the template and the cap per pull request, and this file cites the files rather than restating them | [#316](https://github.com/michaelJwilson/snakes_and_ladders/pull/316), [#318](https://github.com/michaelJwilson/snakes_and_ladders/pull/318) |
| Stage 3 Research extensions | Gumbel-softmax relaxation of Potts and HMM states landed; the tropical Grassmannian half landed, refereed by enumeration and the Hadamard closed form, and not shown to beat a classical baseline, so it is conserved in `sandbox/`; learned surrogates rank a neighbourhood with exact re-scoring of the top candidates; stochastic escape by epsilon-greedy landed | Gumbel-softmax exact at every corner to 1e-11 and deterministic ascent 18/40 against greedy's 5/40, McNemar `p = 0.00098`; the tropical relaxation exact at every corner to 3.8e-16 relative, four-point violation of the Hadamard metric under 1e-12, ascent 8/8 at five and six taxa and 7/8 at eight against the enumerated maximum, and neighbor joining reaching it at no gradient steps; a surrogate-ranked SPR search reaches its optimum from 4/4 starts at 5 fits against 312; escape from a local optimum rises from 0.111 at `epsilon = 0` to 0.883 at 0.4 | [#198](https://github.com/michaelJwilson/snakes_and_ladders/pull/198), [#225](https://github.com/michaelJwilson/snakes_and_ladders/pull/225), [#317](https://github.com/michaelJwilson/snakes_and_ladders/pull/317) |

## Progress Since the 0.4.0 Audit

What moved between [#375](https://github.com/michaelJwilson/snakes_and_ladders/pull/375), the 0.4.0 audit this one starts from,
and this release, per milestone and with the pull request that carries it. A
milestone not named here did not move.

| Roadmap item | What moved | Pull request |
| --- | --- | --- |
| §0 Development loop | The two documents are named the paper and the textbook, `infra/build_documents.sh` replaces the script named after the retired artifact, and a guard fails any live file naming the retired artifact | [#379](https://github.com/michaelJwilson/snakes_and_ladders/pull/379) (#377) |
| §0 Development loop | Validation is one command under five minutes: an inputs stamp beside every committed figure and notebook, so a build renders only what changed; guard-only selection; a per-test duration cap; one BLAS thread per process | [#380](https://github.com/michaelJwilson/snakes_and_ladders/pull/380) (#372) |
| §0 Development loop | A generator rather than a seed for the relaxed benchmark, and full history for the committed-PDF rule, which every pull request had been failing on a shallow checkout | [#378](https://github.com/michaelJwilson/snakes_and_ladders/pull/378) |
| §0 Development loop, §1.3 | Every problem statement carries the same five labelled parts, the applicability tables gain a third reading by method family, and five hand-rolled implementations are refereed against the frameworks that duplicate them | this release ([#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376)) |
| Milestone 1.3 | The tree's two data-driven starts --- neighbor joining on the pairwise Jukes--Cantor and log-det distances, and the closest tree of the Hadamard conjugation --- with Atteson's radius and the four-point condition as the guarantees they carry, measured against the objective's own start at equal evaluations | [#373](https://github.com/michaelJwilson/snakes_and_ladders/pull/373) (#364) |
| Milestone 2.3 | A sixth budget-matched comparison, the tree's starts at 2,000 evaluations over twenty seeds, recorded as experiment 006 | [#373](https://github.com/michaelJwilson/snakes_and_ladders/pull/373) (#364) |

The 0.4.0 cut itself did not move: no tag exists, so `0.4.0` and `0.5.0` are
both unreleased and the version in `Cargo.toml` is still `0.3.0`.

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
do by inspection: the documents job rebuilds only the QA figures
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

`0.3.0` was built into `CHANGELOG.md` on 2026-09-03 from seven fragments and
never tagged — the repository carried no tag at all — so the `0.4.0` release
([#358](https://github.com/michaelJwilson/snakes_and_ladders/issues/358)) tags
both at its merge commit; 81 fragments accumulated between them and every
milestone moved, which the summary above carries row by row.

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

**External frameworks arrive as referees and adapters, not replacements**
([#322](https://github.com/michaelJwilson/snakes_and_ladders/issues/322), closing
[#242](https://github.com/michaelJwilson/snakes_and_ladders/issues/242) into it). A
`frameworks` extra carries `gymnasium` 1.3.0, `rustworkx` 0.18.1, `torchrl`
0.13.3 and `torch_geometric` 2.8.0, and `snakes_and_ladders.sandbox` is the
home whichever side of a measurement is not on the hot path moves to, with a
guard that only tests and QA import it. No framework has beaten the
implementation it would replace, so nothing has moved on that rule; four
measured declines are conserved there instead, each with the tests that
referee it — the three framework fronts below, and `sandbox.tropical` (#408).
`search.gym.GymnasiumEnvironment` wraps
any `learn.Environment` unchanged — the protocol stays stateless and scores a
neighbourhood at once, which `learn.exact` rests on — and passes Farama's
`check_env` on the Potts chain and the 5-taxon tree environment while an
episode round-tripped through both interfaces from one seed has the same
states, rewards, termination and candidate count. `PottsGraph` converts to and
from a `rustworkx` multigraph exactly, doubled bonds included; the open
lattices are `rustworkx.generators`' grid and path graphs as edge sets of the
same integers, and `G(n, p)` agrees with `undirected_gnp_random_graph` at both
ends of `p` and on the mean edge count of 400 draws. The installed `rustworkx`
has a global Stoer–Wagner cut and no s–t flow, asserted so the oracle moves
when that changes; `ising_ground_state` is pinned instead against `networkx`'s
minimum cut at extents 8 to 16. TorchRL's `GAE` and `ClipPPOLoss` reproduce
`learn.ppo`'s advantages, objective and gradient to 1e-10 once their float32
buffers are handed float64, and PyTorch Geometric's `GINConv` reproduces
`GraphSurrogate` to 1.33e-15 on tied first-layer weights — 0 to 8.88e-16 over
ten model seeds at #390's own size — and on general weights GIN sums node and
neighbours before its network where ours concatenates them, so the two are
different architectures and the test says so. Conversion cost,
timed apart from any call on 4 cores at a 1-minute load of 0.23: `to_rustworkx`
12.5 µs and `from_rustworkx` 46.4 µs at extent 8, 50.9 and 196.9 µs at 16,
747.2 µs and 3.78 ms at 64, against `rx.connected_components` at 4.6, 15.7 and
212.3 µs and the Python `ising_ground_state` at 1.40, 5.85 and 186.5 ms. The
conversion is below the cost of the cheapest call it would front at every
size, and no hot path moves until a measured adoption says so.

**The three candidates were measured, and none of them moved**
([#388](https://github.com/michaelJwilson/snakes_and_ladders/issues/388),
[#389](https://github.com/michaelJwilson/snakes_and_ladders/issues/389),
[#390](https://github.com/michaelJwilson/snakes_and_ladders/issues/390);
`docs/experiments/008-frameworks-on-three-hot-paths.md` carries the tables).
PyTorch Geometric fails the profile bar: the `index_add` it would front is
7.0% of a `GraphSurrogate` fit, under #341's 10%, and it is 1.04x slower at
60 examples and 1.08x faster at 240. `scipy.sparse.csgraph.maximum_flow`
clears the bar — Dinic is 61.1%, 66.2% and 73.5% of `ising_ground_state` at
extents 16, 32 and 64 — and is 3.5x to 8.8x the Python reference at 2.15,
7.26 and 33.02 ms, but 2.3x to 4.2x slower than the Rust Dinic already
fronting that call at 0.51, 2.37 and 14.12 ms, all three reporting the same
energy. The three energies are −317.489949509, −1359.092598119 and
−5480.514606575. scipy takes `int32` capacities, so the real-valued reduction
is scaled and rounded: over 20 fields at extent 16 the energy is exact from
1e3 to 1e8 and wrong on 3 of them at 1e2, by 7.18e-4, 1.92e-3 and 4.31e-3,
which narrows the single-field range #388 recorded and is why no rounding
bound is claimed. `rustworkx.connected_components` clears the bar at extent 16
and above, where `_find` and `_union` are 14.8% and 15.6% of a Swendsen–Wang
sweep; the labelling alone is 28.5, 101.9, 262.9 and 1158.4 µs at extents 8,
16, 24 and 48 against the union-find walk's 54.6, 172.7, 431.1 and 1699.8, and
the whole sweep 0.229, 0.725, 1.568 and 5.846 ms against `potts_mcmc`'s own
pointer doubling and sorted grouping at 0.195, 0.650, 1.386 and 5.220 — 1.12x
to 1.17x slower, and renumbering the clusters where ours leaves the chain
equal entry for entry. All three fronts are conserved in `sandbox/`
(`pyg_surrogate`, `rustworkx_clusters`, `scipy_mincut`) with the tests that
pin each against what beat it, so a library release that moves any of these
numbers fails a test rather than dating a paragraph. `alpha_expansion` —
49.9% Dinic at extents 16 and 32, with no compiled path because
`maxflow_rust.max_flow` returns the value and not the cut — is the one call
site where a compiled flow would still pay, tracked under
[#405](https://github.com/michaelJwilson/snakes_and_ladders/issues/405).

**Two of the replacements are measured, and both are declined
([#391](https://github.com/michaelJwilson/snakes_and_ladders/issues/391),
[#392](https://github.com/michaelJwilson/snakes_and_ladders/issues/392),
`docs/experiments/009-batched-rollout-and-torchrl-ppo.md`).**
`sandbox.gym_vector.rollout_batch` collects a batch of episodes through
`gymnasium.vector.SyncVectorEnv`, with `TimeLimit` carrying the decision budget
and `RecordEpisodeStatistics` the episode boundary, and at one copy equals
`learn.rollout.rollout` under the same generator draw for draw. It costs 1.381,
1.018 and 0.938 ms per episode at batch 1, 4 and 16 on the Potts chain against
the sequential rollout's 0.795, 1.570, 1.499 and 1.639 ms against 0.929 on the
5-taxon tree, and 6.314, 6.261 and 6.654 ms against 4.139 on the 7-taxon one: `SyncVectorEnv` is a serial loop in one process, so it has no
parallelism to amortize the vector API's observation stacking and autoreset
bookkeeping against, and no batch size makes it pay. TorchRL's `GAE` is 220x
slower than the recursion it would front, 6.384 ms against 0.029 ms per
32-episode batch. Its `ClipPPOLoss` cannot front `ppo_loss` at that function's
signature at all --- it takes the actor, not the log-probabilities --- and a
whole PPO loop rebuilt on it ran the 1,920-episode Potts-chain budget in 6.31 s
against 8.06 s, at the identical result: 96.30% of the 81 starts, exact expected
return 2.2779, and a sampled learning curve agreeing iteration for iteration to
0.0. Of that 1.28x, 1.13x is the neighbourhood scoring hoisted out of PPO's
epoch loop and the rest is clipping the concatenated batch instead of looping
over episodes; both are changes to `learn.ppo`, which now runs the same budget
in 5.83 s, and TorchRL stays what it already was here, the second implementation
refereeing ours at 1e-10.

All three declined fronts are kept as code in
`python/snakes_and_ladders/sandbox/` --- `gym_vector`, `torchrl_advantage` and
`torchrl_clip` --- rather than as numbers in a merged pull request, because the
sandbox is the home of whichever side of a measurement is not on the hot path
(`sandbox/CLAUDE.md`). Each is still pinned against ours ---
`tests/regression/learn/test_learn_ppo_torchrl.py` at 1e-10 on both value and
gradient, `tests/regression/search/test_search_gym_vector.py` at draw-for-draw
equality --- and still timed beside ours by
`tests/benchmarks/test_learn_ppo_bench.py` and
`tests/benchmarks/test_search_gym_bench.py`, so a library release that changes
an answer fails and one that moves a ratio is re-measured. The
`GymnasiumEnvironment` adapter the batched rollout is built from was not
declined and stays in `search.gym`.

**CPU parallelism has one seam and, at the mid-size tier on a 4-core host,
three negative results
([#344](https://github.com/michaelJwilson/snakes_and_ladders/issues/344)).**
`snakes_and_ladders.parallel.map_tasks` runs a loop of independent tasks on a
thread or process pool with results in input order and one generator per
task, spawned in item order, so a run at four workers is bitwise the run at
one; `opt.fit.fit_from`, `opt.budget.compare` and
`search.support.bootstrap_support` go through it with an explicit `workers=`,
and each pins `workers=1` against `workers=4` with `==` or `torch.equal`. The
inventory below is the serial baseline each site was measured from, the
matrix the result at 1, 2 and 4 process workers with the intra-op thread
count at 1 and at the default (4). Measured once per cell on the 4-core
development host at a 1-minute load of 2.2 rising to 3.8 during the run —
shared with other jobs, so the wall clocks carry contention and the ratios
are what is read. The pool's spawn and import cost 1.81 s for 4 workers with
trivial tasks, more than the multi-start fit or the comparison does in total
and a third of the bootstrap; no site reaches the 2× at 4 workers the plan
set, so all three stay serial by recommendation (callers pass `1`) and keep
the argument, which is the contract a bigger tier or a bigger machine is
measured against. The one positive number is inside a task, not across
them: the serial multi-start fit at the default thread count is 2.87× the
fit at one thread, because torch's intra-op parallelism over 1000 sites
already uses the cores, which is why the sites leave the count at the
default rather than pinning one thread per worker as the plan assumed.

| site | per-task (s) | count | serial wall (s) | loop fraction | Amdahl bound at 4 |
| --- | --- | --- | --- | --- | --- |
| `opt.fit.fit_from` (8 taxa, 1000 sites, 8 starts) | 0.17 | 8 | 1.40 | 0.999 | 3.99× |
| `opt.budget.compare` (Rastrigin, 4 instances × 4 seeds, 4 fits each) | 0.03 | 16 | 0.53 | 1.000 | 4.00× |
| `search.support.bootstrap_support` (5 taxa, 300 sites, 8 replicates, NNI) | 0.66 | 8 | 5.81 | 0.910 | 3.15× |

| site | intra-op threads | workers | wall (s) | speedup | equal to serial |
| --- | --- | --- | --- | --- | --- |
| `opt.fit.fit_from` | 1 | 1 | 1.40 | 1.00× | yes |
| `opt.fit.fit_from` | 1 | 2 | 3.24 | 0.43× | yes |
| `opt.fit.fit_from` | 1 | 4 | 3.05 | 0.46× | yes |
| `opt.fit.fit_from` | default | 1 | 0.49 | 2.87× | yes |
| `opt.fit.fit_from` | default | 2 | 3.14 | 0.44× | yes |
| `opt.fit.fit_from` | default | 4 | 3.21 | 0.44× | yes |
| `opt.budget.compare` | 1 | 1 | 0.53 | 1.00× | yes |
| `opt.budget.compare` | 1 | 2 | 3.19 | 0.17× | yes |
| `opt.budget.compare` | 1 | 4 | 2.95 | 0.18× | yes |
| `opt.budget.compare` | default | 1 | 0.49 | 1.09× | yes |
| `opt.budget.compare` | default | 2 | 3.13 | 0.17× | yes |
| `opt.budget.compare` | default | 4 | 3.13 | 0.17× | yes |
| `search.support.bootstrap_support` | 1 | 1 | 5.28 | 1.00× | yes |
| `search.support.bootstrap_support` | 1 | 2 | 5.41 | 0.98× | yes |
| `search.support.bootstrap_support` | 1 | 4 | 4.79 | 1.10× | yes |
| `search.support.bootstrap_support` | default | 1 | 4.89 | 1.08× | yes |
| `search.support.bootstrap_support` | default | 2 | 4.70 | 1.12× | yes |
| `search.support.bootstrap_support` | default | 4 | 5.65 | 0.93× | yes |

Speedup is against the site's serial run at one intra-op thread; the serial
wall clock in the inventory includes the setup the loop does not carry (the
alignment and the starting search for the bootstrap). The sites the plan
lists after these — the candidate fits of `search.infer`, `learn.rollout`
batches, tempering replicas, `qa.build`, `check_notebooks` and `pytest-xdist`
— are measured on the same matrix before any is switched on (`TICKETS.md`).

## Milestone 1.1 — Simulation & Ground Truth Engine

**Phylogenetics: landed.** A `k`-state Jukes-Cantor simulator generates an
alignment and the ancestral tree in Newick from a typed
tree fixture, retaining the parameters that produced them
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

**The external field is per site, and three instances need one.** `h` was one
row every site shared, so the spatial half of the coupled model could not be
studied with anything site-specific driving it and nothing declared a lattice
at the transition. `h` is now `(n_states,)` or `(n_nodes, n_states)`, widened
once by `snakes_and_ladders.sim.potts.site_field` at each entry point, and the
exact open-chain recursion, the Gibbs sweep, `enumerate_potts` and
`strip_log_partition` index one shape. A per-site field whose rows are equal
reproduces the shared-field `log Z` and marginals to 0.0, and the strip
matches enumeration to 0.0 on a per-site field, which is what the widening had
to satisfy.

`potts_spots` declares the spatial half with the chains and the gated
emissions removed and the field `h[n, m] = alpha[m] * log(size[n] / size_bar)`
added: 3x3 at `ci`, where enumeration over 19,683 configurations gives the
exact marginals; a 12x6 strip at `stress`, where 3\*\*72 configurations are
past enumeration and the column transfer matrix is not; and 71x71 triangular
at `release`, the geometry `spatio_sequential_counts/stress` declares. At `ci`
Gibbs matches enumeration to 0.0153 against the declared 0.04, the covariate
moves the exact marginals by 0.2555 against its own site average, and `alpha`
is recovered inside its 95% intervals with the flat class covering zero
(0.0186 against a half-width of 0.0411). At `stress` the exact mean field
energy, `d log Z(t h) / dt` at `t = 1` from two more transfer-matrix
evaluations, is 15.121 against the sampler's 15.126, inside the declared 0.25
and 0.06 standard errors. At `release` the mean `alpha` of a label rises
monotonely across size quartiles (-0.2335, -0.0549, 0.0862, 0.2602) and the
tilt between the outer two is 0.4935 +/- 0.0101 over eight chains.

**A finding: at the coupled fixture's coupling the field is swamped.** The
71x71 triangular instance was proposed at the coupled fixture's `J = 1.0`.
The `q`-state Potts model on a triangular lattice orders above the coupling
solving `v**3 + 3 v**2 = q` for `v = exp(J) - 1` (Baxter, ch. 12), which at
`q = 10` is `J_c = 0.913`, so `J = 1.0` is on the ordered side: a seeded draw
put 3,991 of 5,041 sites into the two extreme classes after 200 sweeps (44.9 s)
and the counts were still moving, which makes the draw a coarsening front
rather than an equilibrium sample. The instance is declared at `J = 0.7`
instead, where the ten classes hold 415 to 593 sites each and the tilt above is
stable; the swamped configuration is recorded here rather than declared.

**A lattice at the transition, declared rather than built twice.**
`potts_lattice/stress` is the 12x12 open square at the exact 3-state
transition in zero field. The coupling is resolved by
`snakes_and_ladders.sim.potts.critical_coupling` from the file's own state
count rather than stored as a rounded float, so the instance cannot drift off
`J_c`. `tests/regression/search/test_potts_mcmc.py` and section 9 of
`docs/nb/potts_chain.ipynb` each built that lattice for themselves; both now
read the file, and the notebook re-executes to the same energy autocorrelation
times it printed before — 6.70, 4.17 and 2.34 site updates for single-site,
Swendsen-Wang and Wolff — which is the evidence the two copies were one
instance. The values are now pinned beside the ordering, at the file's 10%
relative tolerance. A duplication guard keeps `ln(1 + sqrt(q))` in one place
and found two further copies on its first run, in
`test_belief_propagation.py` and `test_potts_mcmc_bench.py`
([#413](https://github.com/michaelJwilson/snakes_and_ladders/issues/413)).

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

**A turbo code, the second member of the fourth problem class**
([#233](https://github.com/michaelJwilson/snakes_and_ladders/issues/233)).
`sim.convolutional` builds a recursive systematic convolutional encoder from
two octal generator polynomials and holds its trellis as `(state, input)`
arrays: `2 ** m` states, two edges leaving each, the parity bit per edge and
the tail input that empties the register. The default register is the
memory-2 `(7, 5)`, and LTE's `(13, 15)` is carried as the second. Two of
them fed the same message in two orders through a seeded random interleaver
make the rate-1/3 unpunctured turbo code: both encoders terminated, so a
`K`-bit message transmits `3 K + 4 m` bits and both ends of both trellises
are pinned. Pins, none sharing code with the arrays: the parity stream
equals an explicit `b(D)/a(D)` long division over GF(2) on 40 random inputs
for both registers; encoding is linear (`c(u + v) = c(u) + c(v)` on 10 pairs
at `K = 12` and `K = 40`) and systematic; the tail returns the register to
the zero state from every state it reaches and is not a string of zeros; the
interleaver's inverse composes to the identity at `K = 12`, `256` and
`1,024`; and `parity_check` turns the code into a `ParityCheck` by GF(2)
nullspace at `n <= 512`, whose `2 ** 10` codewords under #340's
`enumerate_codewords` are exactly the `2 ** 10` words the shift register
produces at `K = 10`. Three instances are declared: `K = 12` (4,096
messages, enumerable), `K = 256` (the figure's) and `K = 1,024` (the
waterfall's).

**A low-density parity-check code, the fourth problem class**
([#340](https://github.com/michaelJwilson/snakes_and_ladders/issues/340), part 1).
`sim.ldpc` draws a member of Gallager's regular ensemble by column
permutation — column weight 3, row weight 6, held as offsets into one edge
array in both orientations, no dense matrix past the sizes an oracle reaches
— and puts three channels behind one interface returning log-likelihood
ratios: binary symmetric, binary erasure and binary-input Gaussian. Gallager's
bands require `n` to be a multiple of the row weight, so the ticket's
`10,000 x 20,000` is realized as `9,999 x 19,998` (59,994 nonzeros) and the
per-PR mid-size code is 996 bits. A GF(2) encoder by Gauss–Jordan elimination
runs at `n <= 512` and asserts `H c = 0` on every word it returns; past it
the all-zero codeword is sent, on the symmetry argument the textbook states,
and the argument is itself pinned: on the symmetric and erasure channels a
codeword's ratios are the zero word's negated where `c_i = 1` on every
realization, and negating the ratios at a codeword negates every decoder
posterior exactly and moves every decided bit with it, for both check
updates and all three channels. The channels are held to their closed forms —
`+-log((1 - p) / p)`, zero or `+-30`, `2 y / sigma^2` with mean `2 / sigma^2`
and variance `4 / sigma^2` — and to binomial counts at 19,998 bits.

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

**A fourth backend, and two routes to the gradient that were not taken**
([#449](https://github.com/michaelJwilson/snakes_and_ladders/issues/449)).
`pruning_analytic` computes the same log-likelihood behind one
`torch.autograd.Function` whose backward is the closed form of
`alg:pruning-backward`, so the autograd graph is 2 nodes at every tree size
against the taped path's 59, 115 and 227 at 4, 8 and 16 taxa. One gradient at
8 taxa by 20,000 sites: **16.36 ms (IQR 1.37) against the taped 33.06 (3.56)**,
and one L-BFGS fit **454.2 ms (38.7) against 694.5 (52.3)**, both converging to
148940.229159. Half the saving is in the forward pass, which builds no graph:
12.38 ms taped against 8.10 under `no_grad`. The gradient agrees with the taped
one to 2.1e-13 relative, passes `gradcheck` in `float64`, and leaves
`search.infer`'s topology, trace, evaluations and fits unchanged.

A `burn` `Autodiff<NdArray<f64>>` port of the same recursion was measured
beside it and **declined**: 46.80 ms (3.86) per gradient and 1179.1 (144.1) per
fit, with a tape of 66, 134 and 270 nodes at 4, 8 and 16 taxa — larger than the
tape it was meant to replace. The boundary is not the reason; the kernel alone
is 48.01 ms [46.94, 49.27] by Criterion against 47.06 (6.09) for the same call
from Python, so the crossing is inside the spread and the kernel by itself
already costs 1.45x PyTorch's whole evaluation. Its `f64` path was sound —
5.9e-13 against the taped `float64` gradient — so precision is not why it lost.
`docs/experiments/010-pruning-gradient-routes.md` carries every number and the
prediction they were taken to test. The route is conserved rather than deleted,
as `snakes_and_ladders.sandbox.pruning_burn` over `src/pruning_burn.rs` behind
the `sandbox` Cargo feature, so the comparison can be re-run; the default
build, the wheel and every per-pull-request job link no `burn`, and
`infra/release.sh` is what compiles the feature.

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

**Sankoff's weighted step matrix, of which Fitch is the unit case**
([#335](https://github.com/michaelJwilson/snakes_and_ladders/issues/335)).
`sankoff_score` runs the same post-order pass with a `(k, n_sites)` cost
array in place of a bitmask — the pruning recursion in the min-plus
semiring — and is pinned twice: under the unit matrix it equals `fitch_score`
exactly on all 15 topologies of the five-taxon fixture at 1,200 sites, and
under a planted asymmetric matrix it equals a brute force over the 64
internal labellings of the four-taxon tree, reading every edge parent to
child. On the eight-taxon fixture it costs 0.66 ms at 2,000 sites and 11.5 ms
at 20,000 against Fitch's 0.09 ms and 0.87 ms — 7 to 13 times, the price of
a `(k, k, n_sites)` temporary per child. The score is a rooted tree's, so an
asymmetric matrix scores each rooting differently; that is the definition,
and the search is what refuses it.

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

**A problem no baseline solves, and the bar it is read against**
([#406](https://github.com/michaelJwilson/snakes_and_ladders/issues/406)). A baseline **solves** an
instance when it reaches the optimum from **0.4 or more** of starts; a
candidate qualifies below that, with the optimum from enumeration and not
from the search under test. Twenty-seven candidates were measured at 50
seeded starts over 16 seeds, each with its 95% interval. Fourteen came in
under the bar; the lowest is declared as `planted_glass/ci`, a planted
Viana-Bray glass at 18 sites and mean degree 4 with frustration 0.30, where
single-site descent reaches the enumerated ground state of -14.0 on
**0.079** of starts, interval (0.056, 0.101). Its planted state scores -7.0,
so the oracle is the enumeration of all 262,144 configurations and not the
planted bound. `tests/regression/search/test_search_hard_glass.py` pins it
and `infra/baselines.py` recomputes it at the release gate.

What was rejected, and why:

| candidate | baseline | rate over 16 seeds | verdict |
| --- | --- | --- | --- |
| `tree_search/ci` (5 taxa), `tree_search/stress` (6 taxa) | NNI hill climbing | 1.000 | solved |
| `tree_search/release` (7 taxa, #177) | NNI hill climbing | 0.476 (0.438, 0.515) | over the bar |
| 7-taxon Felsenstein-zone caterpillar, pendant 0.5/0.02, internal 0.02 | NNI hill climbing | **0.096 (0.076, 0.117)** | qualifies; not the lowest |
| the same at internal 0.05, and at pendant 0.75 and 1.0 | NNI hill climbing | 0.415 to 1.000 | over the bar |
| `frustrated_lattice/ci`, 3x3 triangular | single-site descent | 1.000 | solved |
| 4x4 triangular torus | single-site descent | 0.423 (0.379, 0.466) | over the bar |
| planted glass, 18 sites, frustration 0.10 and 0.20 | single-site descent | 0.853, 0.669 | over the bar |
| planted glass, 16 to 20 sites, frustration 0.30 to 0.50, two graph seeds | single-site descent | 0.079 to 0.661 | 13 of 18 qualify; **0.079** is the lowest |

Three things the table does not say on its own. The instance matters more
than the knobs: at 18 sites and frustration 0.30 one graph seed gives 0.079
and the other 0.539, so a fixture is a *declared instance* and never a
recipe. The headroom a policy could demonstrate is 1 - 0.079 = 0.921, and at
the per-seed standard deviation of 0.014 the tree comparison measured, an
exact two-sided sign test needs **6 paired seeds** to call a difference that
size --- which is the floor at which such a test can reach p <= 0.05 at all,
so the fixture is not what would limit the comparison. And random-restart
descent reaches the ground state on **every** seed at the declared 50
restarts, exactly as #198 found on the tree: the headroom is against a single
run, and the restart baseline stays unbeaten. The Felsenstein-zone tree is
the only candidate measured where restarts also fail (0.938 of seeds), which
is why it is recorded here rather than discarded.

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
for the model none of them can express. The audit of
[#341](https://github.com/michaelJwilson/snakes_and_ladders/issues/341) below
brought those ratios to 1.8x and 4.7x with the arithmetic unchanged.

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

**The repository was audited a second time, module by module, and three of
the ranked loops moved**
([#341](https://github.com/michaelJwilson/snakes_and_ladders/issues/341)).
`tests/benchmarks/profile_hotpaths.py` now covers one workload per module at
the enumerable tier the suite pins and the mid-size tier this file reports,
and prints the top five functions by `cProfile` self time with the fraction
of the run each carries. Measured on one development machine, 4 cores shared
with one other process, 1-minute load 0.2–1.9 throughout; the fractions do
not depend on load, and the wall clocks below were taken under 1.0. Loops
#287 ranked were not re-measured. The `ROADMAP.md` tier (1,000 taxa by
10,000 sites) and the `qa`/`infra` row (document build, notebook execution,
the test budget) were not run on this host and stay recorded as not measured.

| module | loop (mid tier; enumerable where it differs) | fraction | opportunity | expected gain | oracle |
| --- | --- | --- | --- | --- | --- |
| `likelihood` | `message_passing` flooding, 8x8: `logsumexp` per message 17.6%, ufunc reduce 14.7%, `graph.neighbours` scan 9.3%, `_run` 9.3%, `_normalize` 7.6% (3x3: 19.1 / 15.8 / 10.2%) | 58% Python per message | layout, allocation, vectorization | to within 3x of `belief_propagation` | the dictionary implementation, bitwise |
| `likelihood` | `message_passing` tree schedule, chain 200: `graph.neighbours` scan 15.7%, `graph.degree` scan 10.6% (400,000 calls), `logsumexp` 9.7% | 36% | layout, call overhead (two quadratic scans) | to within 2x of the forward recursion | the same, bitwise |
| `learn` | `PottsLandscape.features`, 60 x 32 episodes on the length-8 chain: 23.5%, with 520,968 ufunc reductions from its loop over sites; `policy.sample` 4.0% | 23.5% | vectorization | under 20% of the run | `_deltas`, exact |
| `search` | `maxflow.energy`, 32x32 x 64 configurations: the per-edge Python loop 91.6% (16x16: 93.0%) | 92% | vectorization | 5–10x on one configuration | `potts.log_weights`, 1e-12 relative |
| `search` | `spr_neighbours` at 20 taxa: 128.7 ms for 1,122 candidates, `build` and its generator 44%, `visit` 28% | 10% of one `infer` step (1.28 s); 67% of one candidate fit (193 ms) | allocation (a `Node` tree per new key) | at most 2x on the neighbourhood, under 5% of a step | not ported: under the 10% rule per step |
| `search` | `gibbs.sample_factor_graph`, 32x32: `conditional` 42.0%, `gibbs_sweep` 17.0%, `log_density` 9.7% (16x16: 40.8 / 18.6 / 9.2%) | 69% | compiled backend over the #341 edge layout | ~10x, from the Potts `numba` sweep's 7x | draw for draw on the same uniforms; not acted on |
| `search` | `alpha_expansion`, 32x32: Python Dinic `_augment` 28.8%, `_levels` 18.8%, `expand` 24.0% | 72% | FFI: `maxflow_rust` as the inner solver | 3x or more on the expansion | its energies, exact; not acted on |
| `search` | `potts_mcmc` single-site, 32x32: `_single_site_sweep` 45.8% | 46% | none: the Rust backend exists and is opt-in (#287) | — | not changed by default |
| `opt` | `hmc.sample`, 1,000 draws on the length-64 chain: `torch.logsumexp` 32.0% (512,000 calls, one per position per evaluation), `log_partition` 9.6%; the fit at length 64: 26.6% | 42% | call overhead: reassociate the homogeneous transfer-matrix product by repeated squaring, 6 products for 64 positions | 1.5–2x on the run | the sequential recursion at 1e-12 relative and central differences for the gradient; not acted on |
| `opt` | `fit` on the tree, 20 taxa x 500 sites: `run_backward` 45.2%, `_post_order` 22.9%; no Python-level call per site | — | none: the objective is one torch pass over sites | — | met by construction |
| `sim`, `likelihood` pruning, Fitch, `budget.compare`, `fit_surrogate`, Rust/FFI | every remaining loop under 10% of a run that is itself milliseconds: `sample_rows` kernel 74.4% of a 0.4 ms call with 19% in its wrapper, `pruning_rust` kernel 86.5%, `FactorGraph.__init__` 22.4% of 31 ms building the 32x32 graph | — | none | — | recorded, not ported |

Three loops moved, each pinned before it was timed and none changing what it
computes:

| loop | pin | before | after |
| --- | --- | --- | --- |
| `message_passing` flooding on the 8x8 lattice: messages as rows of two preallocated `(n_edges, width)` arrays, factors grouped by table shape and axis with their tables stacked once, one vectorized pass per group per sweep | bitwise against `message_passing_reference` (the dictionary implementation, kept) on every marginal and every schedule, 51 iterations both; `log Z` to 1e-12 relative | 514.6 ms, **75x** `belief_propagation` (6.88 ms) | **12.38 ms, 1.8x** |
| `message_passing` tree schedule on the 200-step chain: the same kernels grouped by height and depth, breadth-first, so a 2,000-step chain no longer exceeds the recursion limit | bitwise as above; the 2,000-step chain against the forward recursion to 1e-12 | 29.75 ms, **9.5x** the forward recursion (3.14 ms) | **14.77 ms, 4.7x** — the 2x target is not met: 800 levels at 10–15 µs of NumPy dispatch each is the floor of this layout |
| `maxflow.energy`: one gather over the edges and a `dot` | `log_weights` to 1e-12 relative (realized 7.7e-14; no longer bitwise, the edge terms sum in a different order) | 0.59 / 2.36 / 8.37 ms on one configuration at extents 16 / 32 / 64; 2.70 ms on 64 configurations at 32x32 | **0.08 / 0.28 / 1.00 ms (7.7–8.6x)**, below the Rust cut kernel at every extent; **1.14 ms (2.4x)** |
| `PottsLandscape.features`: one gather from a padded neighbour table, and `is_terminal` reading it instead of `_deltas` per action | `array_equal` to `_deltas`, unchanged | REINFORCE 60 x 32 episodes on the length-8 chain 2.43 s; one gradient update on the length-4 chain 27.31 ms | **1.85 s (1.32x)**; **23.99 ms**; `features` 23.5% of the run to 11.5%, `policy.sample` 5.4% — Python per action sits at the 20% line rather than under it |

No `numba` or Rust port was reached: on every loop acted on the vectorized
pass carried the gain, and the profile of what remains is dispatch per level
(the chain) rather than a call-bound inner loop.

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

**Site patterns, and a bound where they saturate**
([#408](https://github.com/michaelJwilson/snakes_and_ladders/issues/408)).
The log-likelihood reads one alignment column per term, so identical columns
collapse to distinct patterns carrying integer weights and the sum over
columns is reassociated, not approximated. `likelihood.patterns.compress`
builds the table once and `pruning`, `pruning_torch` and `pruning_rust` take
the weights, so no two backends can disagree about what a pattern is.
Compression at each fixture's declared size, as columns to distinct patterns:
`tree_search/ci` 5 taxa, 1,200 to 321 (**3.7x**); `tree_jc/ci` 4 taxa, 20,000
to 256 (**78.1x**); `tree_search/stress` 6 taxa, 1,500 to 526 (**2.9x**);
`tree_jc/stress` 4 taxa, 200,000 to 256 (**781.2x**); `tree_search/release`
7 taxa, 2,000 to 1,230 (**1.6x**); `tree_jc/release` 8 taxa, 200,000 to
19,646 (**10.2x**). The ratio is set by the taxon count against the site
count, not by either alone, which is why the widest alignment at the fewest
taxa compresses hardest and the seven-taxon fixture at 2,000 sites barely at
all.
Agreement with the uncompressed value is 1.8e-16 to 5.0e-13 relative over the
six backend-fixture pairs, inside `likelihood/CLAUDE.md`'s `1e-11` `float64`
bound, and the weights are checked against a `Counter` over the columns, which
knows nothing of `compress`.

The saving saturates at `k ** n` — 256 distinct columns at four taxa however
long the alignment — so blocks of `N > 1` consecutive sites buy a bound rather
than a saving. `likelihood.blocks` partitions the alignment into blocks of `N`
sites, evaluates exactly every block occurring at least `min_count` times
through the pattern compression of those blocks' columns, and brackets the
rest by the extremes of a single site's log-likelihood over the whole column
alphabet — one pass over the tree, the pruning recursion with each leaf
message replaced by the extremes of its transition row, with no data read.
The interval contains the exact log-likelihood on the `tree_search` and
`tree_jc` `ci` and `stress` fixtures over block sizes 1, 2, 3 and 5 against
cutoffs 1, 2, 4, 8 and 32, on the simulated alignment and on four
uniform-random ones each; the per-column extremes are checked against all 256
columns of the four-taxon fixture, one evaluation apiece.
`BlockFrequencyBound` is the tenth `Surrogate`, and its lower end is certified
over the 15 five-taxon topologies at block sizes 1 and 2 and cutoffs 1, 4 and
16 with no violation. What the cutoff buys is paid in width, exactly
`(bounded sites) x (per-site extreme range)`: at 2,000 sites on
`tree_search/ci` a cutoff of 8 at `N = 1` leaves 49 columns to evaluate and a
width of **1.006** `|LL|`, and at `N >= 3` nothing repeats, the whole
alignment falls in the tail and the width is **2.815** `|LL|` — so neither end
orders topologies, and the measurement is reported rather than asserted away.
The `POINT` claim drops the tail and ranks by the frequent blocks alone, the
same sites for every candidate; under it a lazy `search.infer` reaches the
same topology and the same fitted log-likelihood as the search that fits every
candidate, from three starts under both move sets at 1,200 sites and cutoff 4,
at **2 to 4** fits against **5 to 15** and **68 to 163** forward passes
against **209 to 731**. The cutoff is a count, so it scales with the
alignment: 8 at 1,200 sites loses the optimum from 6 of 12 starts and 4 does
not, while 8 at 2,000 sites keeps all 12.

**The interval is not a cheaper forward pass, and the measurement says so.**
One thread, mean over 50 calls: at five taxa by 2,000 sites the uncompressed
evaluation is **0.908 ms**, the pattern-compressed **0.660 ms**, the per-tree
extremes **0.384 ms**, and the interval **3.650**, **3.314** and **3.047 ms**
at cutoffs 1, 8 and 32; at four taxa by 20,000 sites, **3.974**, **0.511**,
**0.257**, and **23.183**, **22.699** and **22.588 ms**. So the interval costs
**4.0x** and **5.8x** the evaluation it stands in for, and raising the cutoff
from 1 to 32 buys **17%** and **2.5%** — because the two terms that scale with
the cutoff are bounded by 0.66 ms and 0.26 ms and everything else is the block
partition's `np.unique`, which sorts every block whatever the cutoff. The
bound's saving is in *fits*, not in passes: a fit is 254 ms at this fixture
against a 3.65 ms interval, which is what the ranked search's 2 fits against
13 buys. Making the interval itself cheap is a separate decision against a
profile and is not taken here.

**BCJR, Viterbi and the turbo iteration, held to enumeration and to the
general sum-product**
([#233](https://github.com/michaelJwilson/snakes_and_ladders/issues/233)).
`likelihood.convolutional.bcjr` runs the log-MAP forward and backward
recursions with the observation on the *edge* rather than the state, since
the bits transmitted at a step are a function of the transition taken, and
returns the extrinsic ratio a turbo iteration exchanges;
`viterbi` is max-product over the same branch metrics.
`sim.factor_graph.from_trellis` presents the chain to the general
implementation, extending #340's seam rather than adding a second.
Pins at the enumerable size, on the `(7, 5)` register: the BCJR bit
posteriors equal the sum over all `2 ** K` messages to **2.8e-14** in
log-odds and the evidence to **1.4e-14** at `K = 10`, `sigma = 0.9`; the
tree schedule of `message_passing.sum_product` on `from_trellis` gives the
same posteriors to **2.7e-15** and the same `log Z` to **3.6e-15** at
`K = 8`; `max_product` returns the Viterbi path on five draws; and a
received word at `sigma = 1.6` separates the bitwise from the blockwise MAP,
so the two decodings are pinned as different answers rather than assumed
alike. An a priori ratio is pinned by enumerating with it folded into the
systematic stream, which is the same posterior by definition.

`likelihood.turbo.decode_turbo` runs the serial extrinsic exchange and
returns #340's `Decoding` unchanged, with `decoded` reading the two
constituent decoders' agreement in place of a syndrome. The joint graph has
cycles, so equality against the exact bitwise MAP is not asserted and the
gap is measured on the declared instance: at `K = 12`, 100 frames and 1,200
message bits per point, the bit error rate at 8 iterations is 0.115, 0.061,
0.033 and 0.013 at 0, 1, 2 and 3 dB against the exact MAP's 0.092, 0.037,
0.019 and 0.004 — a factor of 1.26, 1.66, 1.74 and 3.20, which a `K = 12`
interleaver does not close. Per point the iteration is asserted only *no
worse* than its first iteration, because at 2 dB the two tie at 40 errors
and 1,200 bits cannot separate them; the strict improvement is asserted over
the four points together, 267 errors against 306. At `K = 1,024`, 200 frames
per point over six points (182 s), the waterfall turns between 0.4 and 1.2
dB.

*The one optimization, ranked before it was written.* `cProfile` over five
`K = 1024` decodings at 8 iterations put **21.9%** of self time in
`numpy.ufunc.at` and 7.3% in the `numpy.full` allocations feeding it --- the
scatter the forward recursion used to carry `alpha[t, s] + gamma` into
`next_state[s, u]`. But `next_state[:, u]` is a permutation, which a test
already pinned, so the scatter is a *gather* through its inverse; `Trellis`
now carries that inverse and the recursion reads it. Measured on
`pytest-benchmark`, single-threaded, exclusive host: one BCJR pass at
`K = 1024` **12.42 ms to 9.83 ms (20.9% faster)** and at `K = 256`
**3.29 ms to 2.56 ms (22.2%)**; eight turbo iterations **221.7 ms to
151.1 ms (31.8%)** and **53.9 ms to 38.7 ms (28.3%)**. The release waterfall
fell from 239 s to 182 s with it. Agreement with the enumeration oracle is
unchanged at 2.8e-14 in log-odds and 1.4e-14 on the evidence, which is what
says the reassociation moved nothing. After the change 95.1% of self time is
`bcjr`'s own bytecode --- the Python loop over `K + m` steps --- which is the
shape a compiled backend would take next; no port is proposed here, because
none has been measured through its binding.

*The departure, reported rather than asserted.* The ensemble bit error rate
is **not** monotone in the iteration count. Over the six declared points at
`K = 256` it rises between consecutive iterations at four of them, the
largest rise 2.3e-3 at 0 dB between iterations 7 and 8, and every rise is
inside one binomial interval of the 12,800 message bits the point rests on.
The suite therefore asserts only that the last iteration beats the first and
that no rise exceeds twice that interval. At `K = 12` with a shorter
interleaver the same non-monotonicity is larger relative to the rate. It is
a property of loopy sum-product on a serial schedule, not a defect here.

*A second negative result.* At `K = 12` the code buys nothing at 0 dB: the
exact bitwise MAP's bit error rate, 0.092, is the uncoded antipodal closed
form's 0.079 to within the sample, and the iteration's 0.115 is above it. Short-block turbo codes are worse than
uncoded below the turn, which is why the closed-form pin is stated at
`K = 256` and above, where the coded curve is under it at every declared
point.

**A cheaper climb, and a kernel the compiler can vectorize**
([#408](https://github.com/michaelJwilson/snakes_and_ladders/issues/408)).
Each change was ranked before it was written. `cProfile` over an eight-taxon
SPR search at 1,000 sites and 300 candidates — 80.9 s, 301 fits, 18,955
forward passes, 4 accepted moves, budget exhausted — puts **40.5%** of self
time in the autograd backward pass, **19.7%** in the Torch pruning post-order
and **6.7%** in the L-BFGS step, with neighbourhood generation nowhere in the
top twenty. The search's cost is therefore candidates fitted times passes per
fit, and each change attacks one of those two factors: a parsimony start and a
bounded regraft reduce candidates, a partial refit reduces passes per
candidate.

**What each change bought, alone and together.** SPR from a random start,
budget 500 candidates, medians over 8 seeds, at the two sizes enumeration
referees and at one past them. Every configuration reached the enumerated
maximum **8 of 8** at five and at six taxa, and at eight taxa every one
returned the unbounded search's own log-likelihood to the last digit printed,
so the saving below is not paid for in optima on these fixtures. Six taxa,
1,500 sites, as fits / forward passes / seconds: unbounded **49 / 2,848 /
10.09**; parsimony start **31 / 1,902 / 6.61**; radius 1 **15 / 755 / 2.58**;
radius 2 **43 / 2,482 / 8.63**; radius 3 **49 / 2,848 / 10.04**, which is the
unbounded search, since no edge of a six-taxon remainder is further than 3;
partial refit **50 / 1,810 / 6.06**; all three **31 / 1,009 / 3.37**. At eight
taxa by 1,000 sites the unbounded search spends 16,600 and 24,493 forward
passes over two seeds against all three's **1,930** and **3,359**, and 69.0
and 101.7 s against **7.6** and **13.8** — **8.6x** and **7.3x** the passes,
**9.1x** and **7.4x** the wall clock.

The three do different work and it shows in the counts. A parsimony start cuts
*moves* (0 and 1 accepted at eight taxa against 2 and 4) and leaves the
neighbourhood alone. A radius cuts *candidates per move* and pays for it in
moves (6 and 9 at radius 1 against 2 and 4), which is why it still wins by
**5.6x** in passes: a candidate costs a fit and a move costs nothing. A
partial refit changes neither, cutting *passes per candidate* — its fit count
rises slightly, from the full refit of each accepted move, while its passes
fall by a third. The smallest radius that keeps the optimum on these fixtures
is **1**, which is the NNI neighbourhood exactly; the combination is reported
at radius 2, the smallest radius that is still an SPR search.


**The Rust pruning kernel, laid out for the vector unit.** The kernel is
**93.2%** of the Python-visible call at 20 taxa by 11,000 sites and **96.1%**
at 200, so the marshalling of issue #232 is no longer the term to chase;
inside it the rescaling pass is **25.9%** and **32.5%** of the call, measured
against `rescale=False`, and the message pass is the rest. A partial changed
from `(site, state)` to `(state, site)`, making every inner loop a contiguous
run over sites with no early exit and no data-dependent reduction. **That
alone bought 3.9% at four taxa by 200,000 sites and nothing at eight**,
because the sites-contiguous message reads each child row `k` times and those
rows are not in cache: the traffic ate the vectorization. Tiling the site loop
so one tile's `k` child rows and `k` parent rows are live together fixed it.
Criterion against the committed baseline, tile swept: 128 sites per tile
**-28.2%** and **-35.5%** at the two cells, 256 **-29.1%** and **-28.1%**, 512
**-33.0%** and **-27.5%**, 1024 **-19.2%** and **-23.2%**, 4096 **-19.5%** and
**-21.5%**. 128 wins on the pair and is what ships: 47.59 to **34.17 ms** at
four taxa by 200,000 sites, 101.07 to **65.18 ms** at eight.

**Through the binding it buys nothing at the declared scale, and that is the
result.** The same two builds, measured from Python at the sizes `ROADMAP.md`
declares: 20 taxa by 11,000 sites, 10.28 to **10.46 ms**; 200 taxa by 11,000,
114.94 to **112.42 ms** — 1.8% the wrong way and 2.2% the right way, both
inside the run-to-run spread. Against the NumPy oracle in the same runs, 1.91x
to **2.06x** and 1.94x to **2.01x**. So the layout pays where an alignment is
long enough for one node's rows to leave cache and not at 11,000 sites, where
199 internal nodes and a `log` per site per node are the call and the message
pass is not. The remaining term at many taxa is not identified and no further
port is taken against a guess. Agreement with the NumPy oracle is unchanged in
kind and measured at **3.4e-15** and **1.5e-15** relative, inside
`likelihood/CLAUDE.md`'s `1e-11` `float64` bound; the one reassociation is the
rescale divide becoming a reciprocal and a multiply, which is why the pin is a
relative tolerance and not bitwise.

**What one fit costs, and which of three proposed ports the numbers license**
([#443](https://github.com/michaelJwilson/snakes_and_ladders/issues/443) PR 1;
[experiment 007](docs/experiments/007-per-fit-cost-of-the-tree-likelihood.md)).
Measurement, and one declined route conserved rather than implemented.
Caterpillar topologies at 4 to 20 taxa by
1,000 sites, under `with_lock measure` on the 4-core reference host at one
BLAS thread, medians with the range; the 8-taxon SPR profile reproduces #436's
to within 0.4 points, which is what licenses reading the 20-taxon one beside
it.

| question | measurement | verdict |
| --- | --- | --- |
| `torch.compile` on the pruning forward | forward and backward **1.21x** eager at 8 taxa, **1.09x** at 20; 0 graph breaks, 1.8e-16 relative agreement; the graph falls 105 to 94 nodes; the compile is 13.6 s on a fresh inductor cache and 3.7 s on a warm one, and one artifact serves 12 distinct random topologies without recompiling | does not close the gap; amortization is not the obstacle, there is no win to amortize. Kept as `sandbox.compiled_pruning` and pinned by `tests/regression/likelihood/test_pruning_torch_compile.py`, so the decline is re-checkable against the next torch |
| autograd graph nodes per fit | **7 x (tree nodes) + 9** exactly at every size, four per branch and five per internal node; 1,020 at 4 taxa, 12,925 at 20 | the mechanism #443 predicts, at 7x the constant it assumed |
| the L-BFGS step against #341's 10% bar | **6.63%** of an 8-taxon SPR search, **3.57%** of a 20-taxon one; 3.18% and 2.08% of a bare fit | under the bar at both sizes and falling with size; **closed** |
| the FFI boundary, decomposed | argument marshalling **1.1--4.9%** of a through-binding call, the binding call **90--100%**, the return **under 0.03%**; the residual is the wrapper's own validation at 0.018--0.097 ms, constant in sites. Per-crossing floor 0.0497 ms at 8 taxa, 0.0855 ms at 20 | the boundary is not most of the call; #436's kernel win did not go there |
| crossing once per fit instead of once per pass (calculated from the two rows above, not benchmarked) | 27 crossings x 0.0497 ms = 1.34 ms of a 99.8 ms fit; 47 x 0.0855 ms = 4.02 ms of 511.9 ms | **1.34%** and **0.79%**; an order under the bar |

What the candidates have to work with instead, at 1,000 sites: the backward
costs **1.32x** the taped forward at 8 taxa and **1.36x** at 20 (2.01 against
1.52 ms, 6.04 against 4.44 ms), and a level-synchronous schedule collapses the
recursion by the nodes-to-levels ratio, **2.0x** on the caterpillar profiled
and **2.8--3.8x** on a random topology — not to `O(log n)`, since a
caterpillar's depth is half its node count. So level-synchronous batching is
recommended and carries 10--18% of a search's self time as a floor and 30--53%
as an upper bound; the analytic gradient's ceiling is `0.34 / 2.34` of a
forward-and-backward pair, 12% of a search, and it shrinks further once
batching lands, so it is not built against these numbers; and no Rust port is
licensed, because the boundary it would amortize is worth 1.34%. The Rust
forward is nevertheless **3.4x and 3.6x** the torch taped forward, so a port
that paid would have to carry the gradient too, which neither #436 nor #443
proposed.

Of the three declines, one had code to conserve. The compiled front is
`snakes_and_ladders.sandbox.compiled_pruning` (`sandbox/CLAUDE.md`), and
`tests/regression/likelihood/test_pruning_torch_compile.py` holds the three
facts the ratio was read against: the compiled value and gradient are the
eager ones inside `likelihood/CLAUDE.md`'s `1e-11` `float64` bound, Dynamo
compiles the recursion with **no graph break**, and one artifact serves six
further random topologies at 8, 12, 16 and 20 taxa without recompiling. The
L-BFGS close is a profile and built nothing, so it has nothing to conserve;
the refused Rust port was never written, and what keeps its 1.34% re-measurable
is `tests/benchmarks/test_pruning_rust_bench.py`, which this pull request
already carries.

**The perimeter is not the cost either, and that closes the hypothesis**
([#444](https://github.com/michaelJwilson/snakes_and_ladders/issues/444), PR 1).
#436 left the FFI boundary as where a further port should look. It was
measured two ways and it is not where the time is. Decomposing one
through-binding call ([#443](https://github.com/michaelJwilson/snakes_and_ladders/issues/443))
puts **marshalling at 1-5%** of it and the kernel at 90-100%: 0.0170 ms of
0.3443 at 8 taxa by 1,000 sites, 0.0433 of 3.8828 at 8 by 10,000, 0.0397 of
0.9609 at 20 by 1,000, 0.1674 of 11.6924 at 20 by 10,000. Marshalling grows
at 1.6e-5 ms per site against the kernel's 1.1e-3, a factor of 68.
Amortizing **every** crossing a fit would make is worth **1.34%** at 8 taxa
and **0.79%** at 20 — a calculation from that decomposition, under the 10%
bar the table above records ports against.

The independent measurement agrees on the term. A `#[pyclass]`
(`sandbox.pruning_problem.PruningProblem`) holding the alignment, `k` and
`pi` and taking the branch lengths, a flat parent-index array and `leaf_row`
as three borrowed buffers cuts the bytes a caller materialises per pass from
**64,352 to 360** at 8 taxa by 1,000 sites and from **1,760,928 to 936** at
20 by 11,000 — 179x and 1,881x — and cuts the argument construction it was
built to remove from **0.173 ms to 0.023 ms** of a 5.606 ms call. The two
measurements agree on the term and disagree on the denominator: the
marshalling times are 0.173 ms here and **0.1674 ms** there at nearly the
same size, while the whole call is 5.606 ms here against **11.6924 ms**
there and 10.28 ms in #436, on fixtures built differently. So the fraction
reads 3.1% against 1.43% through the denominator and not the numerator, and
the conclusion is the same either way: marshalling is single digits.

Per pass, medians over 81 interleaved repeats with the interquartile range, one thread under the exclusive lock: 0.157 to
0.145 ms (**-7.9%**) at 8 by 1,000, 1.881 to 1.925 (**+2.3%**, the
interquartile ranges overlapping, so **no change**) at 8 by 11,000, 0.434 to
0.397 (**-8.4%**) at 20 by 1,000, and 5.606 to 5.271 (**-6.0%**) at 20 by
11,000. **The bytes fell by three orders of magnitude and the time did not
follow**, which is the result.

One measurement does not fit that and is reported rather than dropped: 20
candidates against one 20-taxon, 11,000-site alignment, each arm in its own
process, reads **124.09** and **127.06 ms** through the function against
**96.27** and **94.26 ms** through the handle (**-22.4%**, **-25.8%**), with
a handle rebuilt per candidate at **142.54** and **129.15 ms** — a control
that gives the held alignment back and lands at or above the function.
`pytest-benchmark` on the same sweep under the same lock reads **91.02**
against **88.06 ms**, **-3.3%**, and the gap between the two instruments is
not explained by the allocator's `mmap` threshold: forcing either extreme
made the function arm slower (156.17 ms and 192.43 ms against 121.44 default).
So a 22-26% figure exists at one shape and is unattributed, while the
decomposition that *is* attributed says 1-5%. The decomposition decides,
because #436's lesson is that a figure without a mechanism does not survive
the boundary.

**And nothing crosses the boundary in the fit path today.** A search's fits
run `likelihood.pruning_torch`; `likelihood.pruning_rust` is reached only by
`qa.backend_agreement` and the tests, so the 18,955 forward passes above are
on the PyTorch path. The handle therefore has no caller and would earn 1.34%
if it had one. It is a declined implementation: it moves to
`snakes_and_ladders.sandbox.pruning_problem` with its regression tests, which
keep it pinned bitwise against `pruning_rust.log_likelihood`, and
`likelihood/` is unchanged. `src/pruning.rs` keeps one kernel behind both
bindings — `pruning_log_likelihood_core` over a topology in offsets form,
with the existing `#[pyfunction]` an adapter from the children-lists shape —
so nothing here is a second implementation to hold to the oracle.

**The LDPC decoder, specialised from the general sum-product and held to it**
([#340](https://github.com/michaelJwilson/snakes_and_ladders/issues/340), part 1).
`likelihood.ldpc.decode` runs the log-domain `tanh` rule or min-sum under a
flooding schedule, vectorized over every edge with two preallocated message
buffers, a syndrome stop that refuses to count an undecided (zero-ratio) bit
as decided, and messages clipped at `+-30` so an erasure's certainty stays a
number; codeword enumeration at `2 ** k <= 200,000` gives exact bit posteriors
and the ML codeword. Pins: on a 22-bit cycle-free code the decoder equals the
tree-schedule `sum_product` and the enumeration over 32,768 codewords to
1.9e-13 in log-odds on the symmetric and Gaussian channels and 9.4e-14 in
probability on the erasure channel, where the cap's bound is `exp(-30) =
9.4e-14`; min-sum equals the tree-schedule `max_product` to 6.7e-16 and
returns the enumerated ML codeword with a margin above 0.1 nats; on six loopy
(3,6) codes at 12 and 18 bits the decoder and the general damped flooding
reach the same Bethe fixed point within 4.6e-11, both run to a message
residual of 1e-12. Density evolution on the erasure channel reproduces the
published thresholds 0.4294, 0.3834 and 0.5176 for (3,6), (4,8) and (3,5) to
5e-4, and at the release gate the 19,998-bit code resolves every erasure at
`epsilon = 0.42` on three seeds and leaves 25–30% of its bits erased at
`0.44`. Per pull request, the 996-bit code over 20 shared seeds has zero bit
errors at `p = 0.05` on the symmetric channel and a bit error rate of 0.049
at `p = 0.09`, either side of the (3,6) threshold `p* = 0.084`, and no
residual erasure at `epsilon = 0.35` against 0.433 at `0.5`. One finding is
recorded rather than pinned: min-sum on the symmetric channel, where every
ratio has one magnitude and the leave-one-out minimum ties everywhere, is
not monotone in `p` at 996 bits (frame error rate 0.72 at `p = 0.05`, 0.08
at `0.06` over 50 seeds), because the cap at 30 truncates the integer
multiples of `log((1 - p) / p)` at a different multiple for each `p`; the
sum-product decoder shows no such effect and the Gaussian channel orders the
two as the textbook expects. Sizes, memory, the profile that decides a Rust
kernel, and the optimization framing against the samplers are parts 2 and 3
of the ticket.

**The coupled model at 5,041 vertices, and the Rust E step that makes it
affordable** ([#399](https://github.com/michaelJwilson/snakes_and_ladders/issues/399)).
`sim.count_pairs` declares the coupled spatio-sequential model with a
two-channel count emission — a negative-binomial total and a beta-binomial
success count per hidden state — on a 71x71 triangular lattice, `M = 10`
classes, `K = 10` states, `S = 20,000` positions: 1.008e8 count pairs,
**384.6 MiB** as `uint16`, drawn once from the seed in
`tests/regression/fixtures/spatio_sequential_counts/stress.yaml` and binned by
1, 5 and 10, so the three instances are one draw. Aggregation is exact for the
first channel and a declared misspecification for the second, both pinned
against the f-fold convolution of the base mass: `BetaBinomial(f n, a, b)`
keeps the mean, carries more than four times the variance and sits at total
variation above 0.3 from the truth.

The NumPy E step at that size was profiled before anything was ported
(`CLAUDE.md`, Profile first), at bin factor 10 on the 4-core reference host:
one `class_posteriors` is **20.1 s**, of which `torch.lgamma` is **10.4 s** of
self time (51.5%) and the two emission `log_density` bodies 18.9 s cumulative
(94.3%); one `external_field` is **191.3 s**, `torch.lgamma` 102.8 s (53.7%)
and the emission densities 188.1 s (98.3%). Forward–backward is 0.72 s (3.6%)
and the `einsum` 1.1 s (0.6%), so the port is the emission density and nothing
else. Both counts are integers in a range of a few thousand, so
`oxi_snakes_and_ladders::coupled` tabulates the log-density by count —
`[count, M, K]`, count-major, because the field wants every class and state at
one count — and reads two doubles where the oracle calls `lgamma` three times.
NumPy against Rust through the binding, one thread:

| bin factor | positions | `class_posteriors` NumPy | Rust | | `external_field` NumPy | Rust | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2,000 | 20.1 s | **0.55 s** | 36.6x | 191.3 s | **1.10 s** | 174x |
| 5 | 4,000 | 39.2 s | **0.82 s** | 47.8x | 378.1 s | **2.13 s** | 177x |
| 1 | 20,000 | 278.5 s | **4.89 s** | 57.0x | ~2,690 s (extrapolated) | **12.59 s** | ~214x |

The bin-1 NumPy field is the one number not measured: at the ratio the two
coarser factors set it is three quarters of an hour on a host four agents
share, past the 20-minute cap a single measurement is allowed, so it is
extrapolated from the bin-5 measurement and marked as such. Criterion times
the kernel without the tables or the boundary at `S = 200`: 21.32 ms and
84.04 ms, which scaled to bin 10's 2,000 positions is 213 ms and 840 ms
against the 0.55 s and 1.10 s above — the tables' construction and the
crossing are 0.34 s and 0.26 s of each call, the boundary term
`likelihood/CLAUDE.md` requires measuring separately.

The kernel is pinned to the NumPy oracle at the ci instance and on a
64-vertex slice of the 5K one: the log evidence and the field agree to
**3.1e-15** and **3.8e-15** relative, against a stated bound of 1e-12; the
state and pairwise posteriors agree to **1.3e-9** absolute, against 1e-8, and
that residual is the *oracle's* own departure from summing to one — the
kernel's rows sum to one exactly, because the scaled recursion normalizes at
every position and the log-domain one does not.

Recovery at the declared size, from a labelling with 30% of its vertices
redrawn: the label block returns the planted labelling **exactly** at bin
factors 10 and 5 --- 0.734 to 1.000 up to a permutation of the ten class
names --- in one round, 22.8 s and 27.2 s of which the alpha-expansion is
about 20. The full test, simulate through assertion, is **59 s at factor 10
and 62 s at factor 5**, so factor 5 is the key fixture and factor 1 stays
`release` as the plan assigns it. The plan expected factor 1 to be out of
reach; it is not. It recovers the labelling exactly in **98 s**, inside the
same 120 s budget, because the port took one E sweep there from 278.5 s to
4.89 s. That the whole declared instance now fits a per-test budget is the
finding, and moving the key fixture onto it is the maintainer's call rather
than this ticket's. Two further measurements
are recorded because they are findings rather than passing numbers. From a
*uniform* start at `M = 10` the first E step's class densities are ten
mixtures of the same data and the field is at chance, so the ascent does not
move: which start drives this block is issue #306's question, not this
fixture's, and the test corrects a labelling rather than searching for one.
And the fixture's first draft declared a beta-binomial trial count per hidden
state, which puts a count drawn under one state outside another's support,
where the log-density is `lgamma` of a negative argument: two of ten classes
lost their evidence to `NaN` and the field argmin fell to 0.113, the fraction
of vertices in the first class. The loader now refuses a trials ladder that
varies, the trial count being a property of the observation and not of the
state, and the fixture's digest is what made the correction visible.

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

**The tree's data-driven start, and what it buys at equal evaluations.** The
HMM's spectral method is a method of moments with a consistency guarantee;
Mossel and Roch carry the argument to a phylogeny, and what it becomes in
practice is a distance per pair and a tree from the distances
([#364](https://github.com/michaelJwilson/snakes_and_ladders/issues/364)).
`likelihood.distance` inverts the Jukes–Cantor closed form and reads Steel's
log-det distance, each with its delta-method variance, pinned to the
transition probabilities they invert and to interval coverage of **0.940**
and **0.9425** at a nominal 0.95 over 400 seeds; `search.neighbor_joining`
returns every fixture and random trees at 20 and 50 taxa from their path
lengths with every branch to `1e-12`, and on the six-taxon fixture recovers the
topology on every replicate inside Atteson's radius of 0.030 — recovery
**0.72, 0.96, 1.00 and 1.00** over 50 seeds at 100, 300, 1,500 and 10,000 sites, with
**0, 0, 2 and 46** replicates inside the radius. The guarantee is narrower than
the recovery: against the largest distance standard error and the largest
realized error at each fixture's declared sites and seed, the radius is
**0.035** against **0.040** and **0.046** at five taxa, **0.030** against
**0.039** and **0.056** at six, and **0.025** against **0.0035** and
**0.0065** at eight, so neighbor joining recovers all three topologies and
only the eight-taxon fixture sits inside the theorem. `likelihood.hadamard` is the
two-state Hadamard conjugation at up to 12 taxa, exact to `1e-12` against the
pruning likelihood evaluated on every pattern, with the four-state alignment
reduced to it at `2/3` of every branch; the Kimura three-parameter conjugation
is not built. `FromDistances` and `FromHadamard` in `search.initialize` are the
fourth and fifth `Initializer`, in `search/` and not `opt/` because
`opt/` may import no application module, and each also names the topology a
search begins from.

Measured through `opt.budget.compare` over 20, 10 and 5 datasets at five, six and twenty taxa
([`docs/experiments/005`](docs/experiments/005-tree-initializers-at-equal-evaluations.md)).
On the branch-length fit every start reaches the one optimum; the
neighbor-joining start does so in **22.3** evaluations against the
objective's own **23.0** at five taxa, **23.5** against **24.4** at six and
**25.2** against **34.0** at twenty, the Hadamard start **21.8** at five: no
start separates on the fit below twenty taxa, because the stop is the gradient
relative to the objective and the curvature pairs cost the same from anywhere
in the basin. On the NNI search against the
enumerated optimum the neighbor-joining start reaches it from **20 of 20**
datasets at five taxa and **10 of 10** at six, scoring **4.0** and **6.0**
candidates — one neighbourhood, the start already being the optimum —
against the random start's 20 of 20 and 10 of 10 at **7.3** and **16.3**
(McNemar p = 1.000, no discordant dataset); at twenty taxa, against the best
found in 60 candidates, **5 of 5** against **0 of 5** (p = 0.062, the smallest
five pairs can give), the random climb still 3,510 nats short on the first
dataset. **The estimator buys the topology, not the fit.** No default changes: every number above was
produced from the objective's own start and still is.

**Tempering against restarts on the mixture, at equal evaluations.** The
comparison [#284](https://github.com/michaelJwilson/snakes_and_ladders/pull/284)
and [#303](https://github.com/michaelJwilson/snakes_and_ladders/pull/303)
deferred, recorded whichever way it fell
([#332](https://github.com/michaelJwilson/snakes_and_ladders/issues/332),
[`docs/experiments/004`](docs/experiments/004-mixture-tempering-vs-restarts.md)).
Five components 1.5 standard deviations apart with unequal weights, 500
observations, built from `sim.mixture` under seed 20260908 since #262 committed
neither of the five-component fixtures it measured. Multi-start EM, simulated
annealing with Hamiltonian proposals and parallel tempering — the continuous
counterpart of the Potts one, now in `opt.hmc` beside `anneal` — each spend
3,000 likelihood evaluations per start through `opt.budget.compare`, every
method ending with the same charged L-BFGS polish because raw EM sits 4 to 6
nats above its basin's optimum 500 iterations in. Against the best-known optimum — 1111.596 nats, reached by 16 of 1,000
polished restarts and 8.3 nats below the polished simulated parameters, whose
basin is not the maximum on this sample — over 40 shared starts: **restarts
7/40**, tempering 4/40 (McNemar p = 0.549 against restarts), annealing 1/40
(p = 0.031); mean gaps 2.6, 4.2 and 8.0 nats. **Restarts are not beaten on
the mixture**, at 3,000 evaluations: tempering does not separate from them
and annealing loses to them, the opposite of the glass row above and the
same finding as Rastrigin. The 8-start tier of the same test runs per pull
request and pins the ordering.
The paired test `ROADMAP.md` §2.4 asks for is now in the utility:
`opt.budget.mcnemar` on the per-start hits, exact rather than chi-square,
because 40 starts cannot support the approximation.

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

**Adaptation is a warm-up, opted into and reported, and the standing decision
against it is reversed on measurements**
([#333](https://github.com/michaelJwilson/snakes_and_ladders/issues/333)).
`hmc.sample` takes an `Adaptation(warmup, target_acceptance, step_jitter)`,
every field required: the warm-up sets a diagonal mass matrix from the sample
variance of its first window and the step size by dual averaging (Hoffman &
Gelman 2014 §3.2, `eq:dual-averaging`), run once at unit mass and once on the
metric; the chain is then drawn at those fixed values, and `HmcChain.adapted`
reports them with the warm-up acceptance and `force_evaluations` what the chain
cost. The mass matrix is a change of coordinates on the objective rather than a
change to the integrator, held to a hand-written mass-matrix leapfrog to
1e-12, and the fixed-parameter path is untouched — the 31 HMC tests pass with
no number moved. Two measurements shaped it. On a locally quadratic target the
acceptance is a cliff in the step — on the analytic Gaussian 0.88 at a step of
1.2 and 0.01 at 1.4, the stiff direction's stability limit being 1.26 — so a
target of 0.65 sits on the cliff, and each proposal's step is drawn from a
uniform band around the adapted one (Neal 2011 §5.4.2.2), which the drawn
chain keeps. And the published gain of 0.05 was set for a trajectory-averaged
statistic: with a single Metropolis probability per proposal the drawn chain's
acceptance against a target of 0.65 was **0.815** at 0.05, 0.691 at 0.1,
**0.643** at 0.2 and 0.610 at 0.5, so 0.2 is the constant, stated as a
deviation. What it achieves: the drawn chain's acceptance pooled over 20 seeds
is **0.650** on the Gaussian (per-seed 0.578 to 0.758) and **0.673** on the
four-taxon tree posterior (per-seed 0.573 to 0.750; 0.678 over the 3 seeds CI
runs, the 20 in the stress tier), against the target 0.65. The adapted chain and the fixed-parameter chain agree on both posteriors
— means within 1.83 standard errors on the Gaussian and 1.77 on the tree,
spreads within 0.54 and 2.84, each standard error from the chain's own
effective sample size — and the #268 interval is reported beside both: the
adapted chain's spread is 1.009 and 1.008 of the exact Gaussian one, and 1.01
to 1.13 of the delta-method interval on the tree's branch lengths against the
fixed chain's 0.99 to 1.08. The effective sample size, by Geyer's initial
positive sequence and held to an AR(1) whose autocorrelation time is a closed
form (estimate over truth 0.83 to 1.12 at a coefficient of 0.9), prices a
draw: on the tree's slowest branch the adapted chain gives **0.038** effective
draws per gradient against the fixed chain's **0.019** at unit mass, whose
masses the warm-up measured as spanning 5 to 124; on the Gaussian 0.099 and
0.105 against 0.098 and 0.015.

**A tempering ladder is chosen from its own exchange acceptance.**
`schedule.adapt_ladder` takes the measurement as a callable and knows no
model: a pair below a stated band is bisected geometrically, a rung both of
whose pairs are above it is removed, and a pair above the band beside one
inside it has their shared rung moved halfway toward the far end;
`potts_mcmc.adapt_ladder_potts` supplies the measurement as a
`parallel_tempering` run. On the 9×9 periodic triangular antiferromagnet from
the endpoints (2.0, 0.4) alone, a band of (0.25, 0.75) at 50 sweeps per
measurement settled inside the band **20/20** seeds in 4.2 rounds on average,
on 5 to 8 rungs, and a fresh run on the returned ladder exchanged at 0.16 to
0.72 on every pair; the hand ladder (2.0, 1.2, 0.7, 0.4) exchanges at 0.28,
0.15 and 0.12. At equal sweeps with the warm-up charged — 2400 per seed, of
which the warm-up spent 895 on average and 500 to 1900 — the adapted ladder
reached the closed-form ground state **20/20** against the hand ladder's
20/20, and 19/20 against 20/20 at 1600: the instance does not separate them,
since the hand ladder hits 18/20 at 100 sweeps, and what the adapted ladder
buys is the band on an instance where the band did not matter. NUTS remains
out of scope; the tree comparison did not ask for it.

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
posterior.

**The same weights serve labellings and decodings, and a tempered ensemble
gives a marginal one**
([#331](https://github.com/michaelJwilson/snakes_and_ladders/issues/331)).
`neighbourhood_labelling_support` and `enumerated_labelling_support` take any
factor graph, the neighbourhood every single-site change, so a Potts
configuration and a hidden path are one case: the enumerated weight equals
`enumerate_potts`'s Boltzmann weight at `beta = 1` on every one of the 729
labellings of a three-state 3 x 2 lattice to 1e-12, and the path posterior of
the ambiguous chain's decodings, where the Viterbi path's margin is the
fixture's 0.3033 nats and the posterior-decoded path's is -0.6066. The
single-site neighbourhood is the whole space only where one site is free — on
a two-site chain it reaches 5 of 9 labellings — and there the two weights
agree to 1e-12. The bootstrap stays a tree quantity: it resamples sites, which
a chain's ordered sites do not license and a labelling does not have.
`search.tempered` runs replica exchange from the moves of #309 on the ladder
of #267 with `parallel_tempering`'s exchange ratio, over labellings and over
topologies; the fraction of the temperature-one replica's sweeps at a
structure is its tempered weight (`eq:tempered-weight`), held to enumeration
over 20 seeds on the ladder (1, 2, 4) at 1,000 sweeps after 100 of burn-in.
The largest single-seed deviation is 0.039 on the four-taxon fixture's three
topologies (30 sites, weights 0.66, 0.17, 0.17), 0.024 on the 2 x 2 lattice's
ground state and one-flip excitation (0.68, 0.05) and 0.031 on the ambiguous
chain's two decodings (0.14, 0.08); the mean over seeds is within 0.004 on
every instance; asserted at 0.06 per seed and 0.01 on the mean. It is named a
posterior weight only beside its diagnostics, which are asserted too:
exchange acceptance per adjacent pair 0.82-0.93 on the topologies, 0.53-0.82
on the lattice and 0.71-0.85 on the chain, and no indicator's autocorrelation
time above 0.74 recorded sweeps. Over topologies the tempered weight is a
marginal over topologies of the *fitted* likelihood, not over branch lengths.
Calibration is re-measured at seven and eight taxa behind the release gate,
for the quantities a search there can afford — the NNI neighbourhood weight
and the bootstrap's smallest split support, 8 replicates — over 16 NNI
searches per taxon count at 50 to 400 sites (the hard seven-taxon fixture and
the balanced eight-taxon one); the fraction of returned trees equal to the
generating one per support bin `(0, 0.5]`, `(0.5, 0.9]`, `(0.9, 1]`:

The per-bin table was left as a placeholder when #350 merged: the
release-gated test that produces it (about 40 minutes at seven and eight taxa)
was not run at the merge, and the audit below records the gap; the table is
transcribed here at the next release-gate run rather than typed from memory.

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
the planted energy. The five-component mixture comparison, the one problem
class where restarts are the standard answer, is under Milestone 1.3 and in
[`docs/experiments/004`](docs/experiments/004-mixture-tempering-vs-restarts.md)
([#332](https://github.com/michaelJwilson/snakes_and_ladders/issues/332)).

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

**Viterbi is built as max-product over the chain's factor graph**
([#296](https://github.com/michaelJwilson/snakes_and_ladders/pull/296)), returning
the enumerated Viterbi path with its joint on four chains, and posterior
decoding is forward–backward
([#307](https://github.com/michaelJwilson/snakes_and_ladders/pull/307)); the
canonical ambiguous chain reads both decodings off one enumeration and the
tempered ensemble of #331 weights them. **Not built:** iterated conditional
modes over HMM state paths (#176) — the block move of #309 is the exact
sampler on the same object, and `snakes_and_ladders.search.alpha_expansion`'s
lattice ICM is a different one. Single-flip local search over the Potts chain
exists as an RL environment, not as a classical baseline suite.

**Large parsimony: the same climb, one pass per candidate, and it finds the
wrong tree where it should**
([#335](https://github.com/michaelJwilson/snakes_and_ladders/issues/335)).
`parsimony_search` walks NNI or SPR on the Fitch score or a metric step
matrix, with `infer`'s accounting — a budget in candidates scored, each
topology scored once — and enumeration as its oracle. From every one of the
15 starts at five taxa and the 105 at six, under both neighbourhoods and both
scores, it reaches the enumerated minimum, which is unique and the
generating tree at both sizes (1163 changes at five taxa and 1,200 sites;
1792 at six taxa and 1,500 sites, 37 ahead of the runner-up); a median
search scores 17 candidates under NNI and 61 under SPR at six taxa, the
whole 105-start study running in 0.2 to 3.4 s where the likelihood study on
the same fixture is release-gated at 50 s. Eight taxa is where the
neighbourhoods separate: over its 10,395 enumerated topologies at 1,000 sites
NNI reached the minimum from 9 of 12 random starts and SPR from 12 of 12, at
medians of 60 and 289 candidates. On the Felsenstein-zone fixture at 2,000
sites the search returns, from every start, the tree that groups the two
long branches at 1869 changes, where the generating tree scores 1982; maximum
likelihood on the same alignment puts the generating tree first by 12.99 log
units, and transition/transversion weighting does not move parsimony's
answer (3013 against 3273). The search is right and the criterion is wrong,
which is what the fixture is for.

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

**The tree policy learns once it has something to learn**
([#328](https://github.com/michaelJwilson/snakes_and_ladders/issues/328)).
#178 trained a policy over one feature, the improvement a move buys, and
measured a tie with greedy hill climbing on the hard seven-taxon fixture,
because a softmax over one column is an inverse temperature. `FeatureSet.FULL`
gives each move seven columns without a fit — the improvement, the Fitch
parsimony change, the pattern support of the split broken and of the split
made, and the sizes of the two exchanged subtrees — standardized within the
neighbourhood, each pinned to an independent computation and each shown to
vary within a neighbourhood (a planted Robinson–Foulds column, constant across
NNI moves, is refused). At #178's budget of 640 episodes over 50 starts and 16
training seeds, the single feature reaches the enumerated maximum from 0.487
of episodes against greedy's 0.480 (sign test p = 0.79) and the full set from
0.796, ahead on 16 of 16 seeds (p = 3.05e-5;
`docs/experiments/005-tree-policy-features.md`). The known-parameter reward
now scores GTR from a given rate matrix through the pruning recursion. Not
measured: the full set against random-restart hill climbing, which #194 showed
reaches every start on this fixture, and any column's individual necessity.

**A critic, an actor–critic and PPO, each pinned to enumeration before it is
measured** ([#313](https://github.com/michaelJwilson/snakes_and_ladders/issues/313),
part 1). `learn.exact` now returns action values and the optimal value beside
the expected return, and the three agree where they must: Bellman's equation
to 1e-12 on every state checked, the optimal value above every policy's, and
over eight decisions equal to the gap to the enumerated minimum energy. A
critic reads state features derived from the action features the protocol
already supplies (their mean and maximum, and the log of the neighbourhood's
size), so no instance changed; fitted to the exact `V^pi` on all 81
configurations of the chain, the linear critic explains 0.87 of its variance
and a 16-unit MLP 0.996. The advantage-weighted score function with the exact
critic as baseline is unbiased for the enumerated gradient to 5e-3 over 4,000
episodes, and the unclipped PPO objective at the collecting policy has the
actor–critic's gradient exactly. At the budget #135 trained REINFORCE on (60
iterations of 32 episodes) the trained policy reaches the enumerated optimum
from 88.9% of the 81 starts under REINFORCE, 88.9% under the actor–critic and
**96.3% under PPO** (greedy: 80.2%), with mean exact expected return 2.21,
2.23 and 2.28; at a quarter of that budget REINFORCE reaches it from 32.1% and
PPO from 87.7%. An `MLPPolicy` trained by PPO reaches it from 97.5% with mean
return 2.55, the worsening move a chain needs being representable where two
linear features cannot.

**On the hard tree fixture every algorithm lands on greedy, and the feature
set is why.** At #178's budget (40 iterations of 16 episodes, horizon 30, 50
seeded starts, the fixed-length NNI reward) greedy reaches the enumerated
maximum from 0.48 of the starts, REINFORCE from 0.49, PPO from 0.47, and PPO
collecting under an epsilon-greedy behaviour policy on a linear schedule from
0.3 to 0.02 from 0.46 — all within the noise #178 measured (standard
deviation 0.014 over seeds). The environment exposes one feature, the
improvement a move buys, so the policy is an inverse temperature and no
algorithm can learn what that class cannot express; the off-policy variant
is correct by the ratio `pi / beta` and buys nothing here. The feature set of
Milestone 2.1's first bullet is the prerequisite for a tree result, not a
better optimizer.

**A planner reaches the optimum at a fraction of greedy's evaluations, once
its prior and critic are trained.** `learn.planning` runs PUCT search over
any environment with the policy as prior and the critic as leaf value, and
expert iteration fits the policy to the root visit distributions and the
critic to the achieved returns. Pinned by enumeration: with the exact
optimal value as leaf and one decision of depth the most visited move is an
argmax of `Q*` on every state checked, and at depth three the visit
distribution's one-step value under `Q^pi` is no less than the prior's on 13
of 14 states. Measured on the chain, counted in successor evaluations: an
untrained prior with a fresh critic reaches the enumerated optimum from
76.5% of the 81 starts at 57 evaluations per episode against greedy's 80.2%
at 48; after 10 iterations of 8 planned episodes (1,066 evaluations of
training) the planner reaches it from **92.6% at 8.3 evaluations per
episode**, and at six simulations, 6.3 evaluations, matches greedy's 80.2%
— the same answer at an eighth of the cost. The policy alone, without the
search, reaches 30.9%: the visit distributions at 20 simulations are flat
targets, and what expert iteration taught here is the critic. The
factor-graph environment and the surrogate reward model wait on #296 and
#308 landing on `dev`.

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
and this file now cites them. Six are recorded at 0.5.0: the mixture at 3,000
evaluations (004), the tree policy's feature set (005), and the tree's starts
at equal evaluations (006). The applicability tables of §1.3 cite a file by
its number, and a citation naming one that does not exist fails the
generation, so a retraction cannot leave the textbook pointing at nothing.

**The record is capped at one screen, so what it holds is chosen**
([#458](https://github.com/michaelJwilson/snakes_and_ladders/issues/458)).
Seven prose sections collapse to three — question, numbers, finding — and the
body is at most ten non-blank content lines after the front matter, neither the
title nor a section heading among them; the front matter is untouched, since it
is the reproducibility record. The six files fall from **70, 69, 69, 114, 82
and 141** lines to **34, 34, 33, 34, 34 and 34**, bodies of **7, 7, 6, 7, 7 and
7** content lines, every one of them a question, a table and a finding. No number
was deleted to fit: each one displaced was already evidence in this file or an
argument in the pull request that carried it, and the guard that validates the
files now fails an eleventh content line. The Aim run store (#75) is part 2, behind the
dependency's approval; until then the numbers are typed from the
measurement and the file names the command that produced them.

## Stage 3 — Research Extensions

**Both halves are built, and the second's block was the referee, not the
effort.** `ROADMAP.md`'s differentiable-search bullet names two relaxations.
Potts configurations and HMM state paths are enumerable, so the exact optimum,
the exact expected score and the exact gradient are all computable and "does
the relaxation find what discrete search finds" is falsifiable
([#211](https://github.com/michaelJwilson/snakes_and_ladders/issues/211)).
Tree topologies were recorded as having no such referee. They have two below
nine taxa: exhaustive enumeration, and the Hadamard conjugation of a two-state
spectrum, which is a closed form for the relaxation's own coordinates
(#408).

**A tree is a point of the tropical Grassmannian, and a quartet's resolution
is an argmin.** `Gr(2, n)` is the set of pairwise-distance vectors satisfying
the tropical Plücker relation --- the four-point condition --- and of a
quartet's three pairing sums the smallest is attained by its own resolution.
Softening that argmin at temperature `tau` and weighting a per-quartet score
table by it gives an objective differentiable in the distances and linear in
the weights, so at a tree metric it is the discrete quartet score exactly.
Measured over the metric of every one of the 15, 105 and 945 topologies of the
5-, 6- and 7-taxon fixtures, and over 201 of the 10,395 at eight:
`3.8e-16` relative at worst. The softmin's own leakage is
certified under `1e-11` by `sum_Q 2 exp(-g_Q / tau) R_Q`, inverted for the
temperature each corner needs, and what is left at that temperature is float64
rounding of a sum over `C(n, 4)` terms — which is why the agreement is pinned
relatively and not at the `1e-11` the Gumbel-softmax half uses at a hundredth
of the magnitude.

**The closed form lands on the variety.** The Hadamard conjugation of the
exact two-state spectrum returns a weight per split, zero on every split the
tree lacks; summing those over the splits separating two taxa is a route to
the metric that shares no algebra with a walk over the tree, and its four-point
violation is under `1e-12` at 5, 6 and 7 taxa. It resolves every quartet as
the generating tree does, and so does the metric estimated from a recoded
alignment.

**What it buys is nothing, and that is the result.** Annealed ascent from
8 random metrics reaches the enumerated maximum of the quartet surface 8 of 8
at five and six taxa and 7 of 8 at eight, the miss 15.8 below in 302,287.
Neighbor joining on the estimated distances reaches the same maximum on every
fixture measured, at no gradient steps, so no budget-matched claim is made
against the bounded-radius search of #408's PR 3. On the 7-taxon fixture built
so hill climbing fails ([#177](https://github.com/michaelJwilson/snakes_and_ladders/issues/177))
the top two quartet scores differ by `3.4e-8` relative — inside the
convergence of the fits that produced them, so the enumerated argmax is not a
target there — and both methods return the generating topology. Ascent leaves
the variety: the four-point violation where it stops is 0.24, 0.05 and 2.15 in
units where the metric has mean 1, and at seven taxa its relaxed value exceeds
every corner's by 0.96, which is the outer relaxation's gap measured rather
than assumed absent. The module is therefore
`snakes_and_ladders.sandbox.tropical` and not a member of `search/`: a
declined implementation is conserved with the tests that declined it, so the
measurement keeps its subject, and `snakes_and_ladders.qa.tropical_relaxation`
keeps rendering `fig:tropical-relaxation` from it.

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

**The comparison fixture is not the repository's own.** `potts_chain/ci.yaml` has
`J = 0.75 > 0`, so its optimum is `argmax(h)` repeated and every method finds
it. That is the third time a fixture has been too easy to separate methods —
after [#177](https://github.com/michaelJwilson/snakes_and_ladders/issues/177),
[#198](https://github.com/michaelJwilson/snakes_and_ladders/pull/198) and #209's planted
spin glass — and it is why the baseline is now run before any claim is made.

**Not built:** the relaxation on the Potts *lattice*, where the identity holds
but nothing has been measured; a fixture on which the tropical relaxation beats
a classical baseline, without which its budget-matched comparison has nothing
to compare; and any joint optimization of structure alongside continuous
parameters, which is what the roadmap bullet ultimately asks for.

## §1.2 Requirements Ledger

| Requirement | Status |
| --- | --- |
| Phylogenetic RF ≤0.05 against simulated truth | **Met**, from 125 sites upward ([#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148)) |
| Potts/HMM parameter recovery within 95% intervals | **Met** for the 1-D chain, the discrete HMM ([#116](https://github.com/michaelJwilson/snakes_and_ladders/pull/116)) and the 2-D lattice — realized 0.981 at 100 samples and 0.956 at 400 and 1600, over 40 replicates each |
| Precise state-sequence decoding | **Met** where enumeration referees it: Viterbi as max-product returns the enumerated path on four chains to 1e-13 ([#296](https://github.com/michaelJwilson/snakes_and_ladders/pull/296)) and forward–backward posteriors agree with the path enumeration to 1e-12 ([#307](https://github.com/michaelJwilson/snakes_and_ladders/pull/307)); decoding accuracy against a planted path past enumeration is not measured |
| Coupled model: planted labels and emission parameters past enumeration | **Met** for the labels — 0.97 accuracy up to permutation on a planted 10x10 lattice after the annealed start, against 0.66–0.78 from a cold start ([#307](https://github.com/michaelJwilson/snakes_and_ladders/pull/307)); recovery of the emission parameters past enumeration is not measured |
| Codes: exhaustive maximum-likelihood decoding where it reaches, the block error rate against the channel where it does not | **Met** at the enumerable size — the decoder equals enumeration to 1.9e-13 on a cycle-free 22-bit code and min-sum returns the enumerated ML codeword ([#356](https://github.com/michaelJwilson/snakes_and_ladders/pull/356)); at 19,998 bits the code brackets the erasure threshold 0.4294 between 0.42 and 0.44, and the error rates against `p`, `epsilon` and `sigma` are #340 part 2 |
| Parity with exact oracles on small `n` | **Met** for tree search against exhaustive enumeration ([#128](https://github.com/michaelJwilson/snakes_and_ladders/pull/128)), and for large parsimony from every start at five and six taxa ([#335](https://github.com/michaelJwilson/snakes_and_ladders/issues/335)) |
| Parity with IQ-TREE 2 / RAxML-NG on large `n` | **Deferred**, not withdrawn: `CLAUDE.md` admits no external solver today, so the comparison is not attempted and what referees large `n` in the meantime is this repository's own exact oracles where they reach and the simulated truth past them. The requirement returns if a tool is adopted; `docs/external_tools.md` surveys the candidates (issue [#126](https://github.com/michaelJwilson/snakes_and_ladders/issues/126)) |
| `O(n×L×k)` memory inside 16 GB / 24 GB | **Met**: 87.2 MB at 100 taxa by 11,000 sites on the worst-case (caterpillar) topology, 879 MB at the declared maximum — a factor of 20 inside 16 GB, with each figure pinned against the allocator to within 2.5% |
| CUDA, Metal/MPS and CPU dispatch | **CPU only**; selection logic landed ([#112](https://github.com/michaelJwilson/snakes_and_ladders/pull/112)), accelerator paths not implemented (issue #280) |
| Declared cross-device tolerance, not bitwise | **Met**: `1e-11` relative in `float64`, `1e-6` where either side is `float32` ([#112](https://github.com/michaelJwilson/snakes_and_ladders/pull/112)) |

## §1.3 The Documents

`docs/tex/` now spans every problem class rather than the phylogenetic
application alone: the abstract, methods and appendices state the Potts
Hamiltonian and the HMM decoding problem beside the substitution model, and the
Reference Taxonomy appendix routes the literature by concern. It was cut down
in `14d32d6` from the academic-letter structure of
[#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148) to an
eight-page specification, and has since grown back toward the shape §1.3 asks
for section by section, as recorded below.

Twenty-one QA scripts render the figures, each committing a figure with a caption
naming the seed, the sizes and the model that produced it, and `docs/CLAUDE.md`
states the rules that keep a CI-regenerated artifact true
([#140](https://github.com/michaelJwilson/snakes_and_ladders/pull/140)). The two
documents cite nineteen of them — the textbook the simulated tree, the
Jukes–Cantor curves, the four problem-statement figures of #358 and the two
rendered instances of
[#394](https://github.com/michaelJwilson/snakes_and_ladders/issues/394) (the
Tanner graph of the enumerable (3, 6) code, 2.7 s, and the coupled model's
planted and recovered labelling, 3.0 s), the paper
the worked simulation, the backend agreement, both parameter-recovery figures,
interval coverage, the model recovery, the trajectory and the topology search,
the reward surface, the tree policy and the footprint table — and the
per-pull-request build regenerates only those; the remaining two, the
problem-size table and the topology-accuracy figure, are checked at the release
gate ([#157](https://github.com/michaelJwilson/snakes_and_ladders/pull/157),
[#325](https://github.com/michaelJwilson/snakes_and_ladders/issues/325)).

Measured against §1.3's required contents: the model formulations are present
for every class, and since
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
the pruning and forward–backward recursions, derived since
[#347](https://github.com/michaelJwilson/snakes_and_ladders/pull/347) (#326); the bounds and their proofs are Appendix B since #317, and the
branch-and-bound search over them is #329. The paper's two framed placeholders
— the comparison against classical software and hardware scaling — gave way at
0.4.0 to the budgeted comparisons the ledger records and to #353's
thread-scaling result, with the external-tool comparison stated as ticketed
(#126) rather than drawn.

**The textbook is now one document at one standard**
([#298](https://github.com/michaelJwilson/snakes_and_ladders/issues/298), [#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376)). Every problem section carries
the same five parts — the model, the sizes it is supported at read from the
fixtures and the size tiers, the model as an instance of the factor graph of
`sec:factor-graph` with a hand-drawn sketch of that structure, the algorithms
as instances of `eq:sum-product` or of the optimization each is, and the
validation, one referee at a time with what it does and does not establish — and the notation table states the factor-graph symbols once
with the identification each section's classical symbol makes. The discrete
solvers that were one sentence each are sections with a labelled equation, a
citation, a regime and a pin: ground states as cuts (`eq:cut-energy`,
`eq:gw`), alpha expansion and its bound (`eq:alpha-expansion`), the heat bath
and the cluster moves with the field accept step (`eq:heat-bath`,
`eq:cluster-accept`), and temperature, annealing and tempering with the
exchange ratio and the tempered weight it licenses (`eq:exchange`,
`eq:tempered-weight`). The hidden Markov section states the backward
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

**Each of the nine problem statements is complete, and a guard says so**
([#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376)). Nine sketch files under `docs/tex/` — one per
problem, hand-drawn TikZ pulled in with `\input`, outside the figure cache
and the render cap, which govern rendered output only — carry the factor graph
of each model beside the coupled model's, which was the only one before. The
algorithms the sections cite are stated in one appendix (`app:algorithms`)
grouped as initializers, samplers, optimizers and surrogates, twelve
environments joining the four the body derives in place; the sampler, the
cluster move, parallel tempering, the chain block sweep, alpha expansion, hill
climbing, expectation--maximization, the sum-product decoder, $k$-means$++$,
the Hadamard start, multi-start and the surrogate ranking had none.
`tests/regression/test_problem_statements_complete.py` fails a section missing
any of `sec:<p>:model`, `par:<p>:sizes`, an `\input{<p>_figure}` defining
`fig:<p>:sketch`, `sec:<p>:validation`, or one `\ref{alg:...}`, and fails an
algorithm environment nothing cites; `docs/CLAUDE.md` states the convention.
Before it, one section of nine carried all five parts.

**Every row of `PROBLEMS.md` is a problem statement, and the applicability is
generated** (#358). Large parsimony (`sec:parsimony`), the frustrated lattices
(`sec:frustrated`), the Gaussian mixture (`sec:mixture`) and the continuous
test functions (`sec:testfunctions`) join the tree, Potts, HMM, coupled and
LDPC (`sec:ldpc`, #356) sections on the same four-part skeleton, each with a
QA figure rendered from the fixture it states. Two tables
(`tab:algorithms-problems`, `tab:oracles-problems`) are written by
`infra/problems_tables.py` from the catalogue and the suite's kind markers:
which algorithm family runs on each problem, which kind of oracle referees it,
and — per method and per size tier — whether the referee is an oracle or the
simulated truth alone, the latter marked as a cell where an oracle is wanted.
A guard holds what the generator writes to what the textbook needs, and a
catalogue symbol the generator cannot name fails it.

A third reading joins them at 0.5.0 ([#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376)):
`tab:methods-initializers` to `tab:methods-surrogates`, one part per method
family, pairing every problem with every family. Twenty-six of the
forty-four pairings carry a method; each states the tier it is validated at
and the kind of referee, both read from the suite, and a sentence on when the
family wins on that problem and when it does not, taken from
`docs/tex/method_notes.yaml` — the one hand-written input to any of
the three, and the reason it does not live under `generated/`. Fourteen of those sentences cite an experiment, and a citation to a
file that does not exist fails the generation. One pairing is marked
*untested* rather than omitted — the general time-reversible model's log-det
start, named by no test of either significant kind — because a dropped pairing
reads as a question nobody asked. The mixture's $k$-means$++$ seeding was the
second until [#420](https://github.com/michaelJwilson/snakes_and_ladders/pull/420)
gave it an oracle.

**The three exact evaluators are derived, not stated**
([#326](https://github.com/michaelJwilson/snakes_and_ladders/issues/326)).
The derivations appendix carries pruning as the marginalization over internal
states (`app:pruning`), forward–backward as the same marginalization on a
chain, with pruning on the caterpillar tree shown to be the backward pass
(`app:forward-backward`), and sum-product on a tree from the subtree
factorization, with max-product as the same argument under a maximum and the
Bethe free energy of a factor graph of any degree (`eq:bethe-factor`) as the
loopy stationary point the existing Bethe derivation identifies
(`app:sum-product`). Each is cited from the point of use in the main text and
from the module that implements it, so the guard of #274 resolves them; the
`TICKETS.md` bullet that named this work without an issue is closed by it.

## What Is Not Claimed

- That a learned policy beats hill climbing on trees. It has now been
  measured on a fixture where hill climbing demonstrably fails, and it does
  not: 0.485 of episodes reach the enumerated maximum against greedy's 0.480,
  a difference of +0.005 with a standard deviation of 0.014 over 16 training
  seeds, 8 of them ahead, at an exact two-sided sign test of p = 1.0. The
  policy does train — an untrained one reaches the maximum on 0.018 — so this
  is a tie rather than a failure to learn. The environment is what bounds it,
  in two ways stated under Milestone 2.1 above. With the seven-column feature
  set of #328 the policy reaches the maximum from 0.796 of episodes, ahead of
  greedy on 16 of 16 seeds, and is not yet measured against random-restart
  hill climbing, which reaches 1.000 on this fixture.
- That a learned policy beats any baseline on the instance
  [#406](https://github.com/michaelJwilson/snakes_and_ladders/issues/406) declared. `planted_glass/ci` is a
  problem a baseline does not solve — the first the repository carries — and
  no policy has been run on it. `snakes_and_ladders.learn.potts.PottsLandscape.on_graph`
  takes one scalar coupling across every edge, deliberately, so that the
  greedy searcher stays inside the policy class; the instance's difficulty is
  its per-edge signs, which that constructor cannot express. Until an
  environment exists that can, the fixture states a gap rather than closes
  one.
- Any comparison against established software. IQ-TREE 2 and RAxML-NG are not
  installed, and no statement anywhere in the repository compares against them.
  This is a stance rather than an omission: `CLAUDE.md` admits no external
  solver today, and `docs/external_tools.md` records what adopting one would
  start from.
- Runtime scaling. Benchmarks are not ranked on CI hardware, so timings live in
  the benchmark suite on fixed hardware rather than in a committed figure.
- Rate variation across sites, and GPU dispatch. Neither is built; both are
  ticketed (#323, #280).

## Consistency audit at 0.5.0

What the release audit ([#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376)) found stale, contradictory or
duplicated between the planning documents, the two documents, the templates
and the code, and fixed in the same pull request. Its baseline is
[#375](https://github.com/michaelJwilson/snakes_and_ladders/pull/375) as merged; what that audit fixed is not repeated here.

- **The version this file is read at.** The header said `0.4.0`, "the release
  #358 cuts". No tag exists --- the repository has never carried one --- and
  `Cargo.toml` still reads `0.3.0`, the version the last built `CHANGELOG.md`
  section carries. The header now states the deferred cut, and the release
  template's tag precondition says what an audit does when it does not hold.
- **Counts the summary had left behind.** Four experiments were recorded
  where six exist, and five budget-matched comparisons where six do; the
  Milestone 1.3 row named neither the tree's data-driven starts nor
  [#373](https://github.com/michaelJwilson/snakes_and_ladders/pull/373), which landed them for #364 between the two audits.
  `TICKETS.md` still listed that work as open.
- **`DEV.md`'s restated counts.** `tests/benchmarks/` was said to hold 36 flat
  modules and holds 39; the frameworks bullet named four test modules that
  `importorskip` a package and there are six, beside three more that
  `importorskip` `scipy`; and the critical tier's **88 tests in 12.6 s** had
  been overtaken by #372's and this release's guards, at **149 in 13.0 s**.
  All three are re-measured on this host and dated. Its branch-protection note
  still spoke of #377's job rename as unmerged; it merged as #379, and
  confirming the replacement is now a release precondition.
- **`docs/CLAUDE.md`'s build rule.** It said the script regenerates the
  figures the documents cite. Since [#380](https://github.com/michaelJwilson/snakes_and_ladders/pull/380) it regenerates the
  cited figures whose input stamp differs, which is `DEV.md`'s to state in
  full; the local rule now carries the principle --- a partial rebuild moves a
  check and never removes one --- and points there.
- **`scipy` is not imported anywhere.** The audit's own plan recorded it as
  imported by `opt.fit` and `search.statistics`. Both modules say the opposite
  in their first paragraph and write out the constants they would need,
  precisely so that no dependency is taken for them. The three referee tests
  this release adds `importorskip` it and skip everywhere until it is
  declared, which `DEV.md` and `INSTALL.md` now say and which stands as the
  open question.
- **The textbook's problem statements were not one shape.** One of the nine
  carried all five parts; the rest carried two to four, and nothing said so,
  because a section missing its sizes or its validation reads as complete.
  Eight sketches, twelve algorithm environments and nine validation
  subsections close it, and a guard now fails the shape rather than a reviewer
  noticing. The plan recorded eight algorithm environments as existing; four
  did.
- **Two symbols were one letter.** The parity-check matrix and the
  Sylvester--Hadamard matrix were both `H`, distinguished only by weight and
  by context. `notation.tex` now defines both, which is where the rule says a
  symbol lives; and the coupled model's figure was labelled outside the
  convention the other eight now follow, so it is renamed.
- **`ROADMAP.md` did not describe two milestones it had reached.** Milestone
  1.3 named no deliverable for a start read from the data, which #364
  supplies, and §1.3 did not say what makes a problem statement complete. Both
  are stated in the document's existing register, and nothing else in it
  changed.
- **The release template.** Its version was `0.4.0`; it carried no per-problem
  completeness checklist and no frameworks table, both of which the last two
  audits needed and wrote by hand; and its tag precondition had no instruction
  for the case both audits actually met. All four are fixed, and the two
  ledger regenerations the audits ran are preconditions rather than folklore.

**Duplicated machinery.** The two seams the 0.4.0 audit recorded --- one
annealing driver behind eight entry points, one weighted enumeration behind
eight enumerators --- are unchanged and are not refactored here. Each is a
ticket, with the oracle that would pin a merge stated in that audit's tables
above; folding either into a release audit would put an untested rewrite of
the sampler and the enumerator inside the pull request that is meant to check
everything else.

**What the audit checked and found true.** The ten required checks are the ten
jobs `ci.yml` defines; the manifest holds nineteen figures and the documents
cite seventeen, as `DEV.md` says; `PROBLEMS.md` resolves every symbol it
names; `CHECKS.md` is a regeneration; the experiment index is a regeneration
and every experiment validates against the template; and no document links to
a path that does not exist.

**The applicability table after the tier move.** Moving the nineteen tests over the per-PR cap to the `release` tier (the throughput pull request) changed three cells of the generated applicability table: the Potts chain, the Potts lattice and the frustrated lattices now read `oracle` rather than simulated truth at the release tier, because the oracle-refereed tests that moved there are the tier's referee. The table is regenerated here; the textbook inputs it, so the rebuilt document carries the three cells.

**Incompleteness at 0.5.0 (issue #400).** `TICKETS.md`, whose job is what remains, carried twelve bullets naming a closed issue. Five described work that had landed and are removed: the textbook's pruning, forward-backward and sum-product derivations (#326, 45 citations in the textbook), the refusal of an unidentifiable fit (#122, `opt.fit` raises on it), the required checks on every pull request (#273, the workflow runs on `pull_request`), the references and blue-sky directions (#360, `docs/blue_sky.md`), and the 0.5.0 audit with the `scipy` question it left standing (#376, both settled). Five name work that remains with the carrier closed and are re-pointed: the profiling at the declared sizes, the efficiency assessment, the audit's four unmet targets and the parallelism sites beyond the first three go to #405; the discrete fixture no baseline solves goes to #406. Two keep a closed number for a reason the bullet states. Two reductions were measured and declined: the six emission families share 270 lines over six methods, every line the family's own distribution rather than scaffolding, and the QA renderers' argument handling is already `qa.runner`'s, leaving one line each. The notebooks' Further Work sections cited nine closed issues. Three named features that have since landed and are corrected here: the hidden Markov notebook said Viterbi and the forward-backward posteriors were not built, and the coupled notebook that no path is decoded, when both landed with #175 and #173, so the gap is coverage and #407 carries the sections; the Potts notebook's four bullets cited #278, which closed without the lattice pass, and now cite #404. The corrections are prose, so no notebook re-executed: the input digest is over code cells.

**Seams (issue #400).** The package is 33,900 lines of Python across six modules (`search` 6,579, `qa` 6,707, `likelihood` 5,487, `opt` 5,417, `learn` 3,792, `sim` 3,483, top level 2,448) and 1,385 of Rust, against 35,780 of tests. `SEAMS.md`, regenerated by `infra/seams_survey.py`, lists 11 protocols and 4 shared contracts: 7 protocols and 3 contracts have three or more consuming modules (`Objective` has 15 implementers and 14 consumers and reaches 7 of the 11 catalogue problems; `Environment` 12 consumers; `FactorGraph` 7); `CountEmissionFamily`, `RelaxedObjective` and `Channel` have no consumer outside their module, `Policy` one, `SpatioSequentialParams` two, each kept for the reason the table prints. No `Problem` protocol: the fixture registry is the classes' contract. One merge proposed under this ticket was measured and declined: the HMM and mixture EM loops share 16 lines, and a driver would add more than it removed. `infra/duplication_survey.py` at this audit: enumerate-shaped functions 15 (8 at #230's filing; #387 owns them), energy-shaped 8 (5), private logsumexp 0 (4), open-coded edge zips 0 (6).

## Consistency audit at 0.4.0

What the release audit ([#358](https://github.com/michaelJwilson/snakes_and_ladders/issues/358))
found stale between the planning documents, the technical documents and the
code, and fixed in the same pull request:

- The summary table above said Viterbi, a trained tree policy and Milestones
  2.2 to 2.4 were not started; the Milestone 1.4 text said Viterbi was not
  built and the requirements ledger had no decoder, while max-product over the
  chain returns the enumerated Viterbi path since #296 and forward–backward
  the posteriors since #307. Corrected in all three places.
- `TICKETS.md` listed landed work as open — forward–backward as an evaluator
  (#173), the Rust Gibbs sweep (#246), Viterbi and posterior decoding (#175),
  schedules and tempering (#267), discrete support (#270, #331), the
  initializers (#251), the emission families (#228, #229), the fixture API
  (#132), the documentation vetting and split (#244, #249), the rename (#250),
  the labels (#274), the kinds (#237), the root-detection fix (#168) and the
  hard tree fixture and trained policy (#177, #178) — and named as unfiled
  what has since been ticketed: rate variation (#323), multi-SPR and
  branch-and-bound (#329), device dispatch (#280),
  the milestone re-keying (#324), the uncited figures (#325).
- `DEV.md`'s repository layout put the temperature schedules in `search/`
  (they are `opt.schedule` since #272), described `opt/` and `learn/` by their
  first two instances, and said no memory helper existed (#232 added the
  footprint table); its measured counts were re-taken on this host and are
  restated with the date. `INSTALL.md` counted eleven QA scripts and put
  `mypy` over two directories where `pyproject.toml` lists three. `README.md`
  counted four objective instances.
- `PROBLEMS.md` had no row for the coupled model and named neither Sankoff,
  large parsimony, forward–backward, max-product, the factor-graph Gibbs
  sweeps nor the surrogates. `ROADMAP.md` §0.4 implied an oracle is always
  required; it now states the two-way standard, and §1.1 names the fourth
  class and the two derived instances.
- The textbook's coupled section listed as remaining the fixture, the E step,
  the block ascent and the initializer that #302 and #307 landed; its
  density-evolution appendix named a repository file, which the textbook may
  not. This section's own §1.3 text said the bounds were absent and three
  placeholders stood. And the Milestone 1.4 text above carried a literal
  `CALIBRATION_TABLE` token where #350's release-gated table belongs.

**Duplicated machinery.** Not refactored here; each group is stated with the
oracle that would pin a merge, and the seam is a follow-up ticket.

| entry point | what it is | classification | oracle for a merge |
| --- | --- | --- | --- |
| `search.potts_mcmc.anneal_potts` | heat-bath sweep per `Schedule` step, best state kept | one algorithm, Potts type | draw-for-draw equality with `anneal_factor_graph` on the Potts adapter (already measured, 2,000 of 2,000 sweeps) |
| `search.gibbs.anneal_factor_graph` | the same loop over any `FactorGraph` | one algorithm, general type | as above, plus the triangular ground state on 6 of 6 seeds |
| `search.gibbs.anneal_topology` | Metropolis over topologies per schedule step, best kept | a distinct move inside a copied driver loop | the flat-prior weight at `T = 1` (#270) and the enumerated best at `T -> 0` |
| `opt.hmc.anneal` | one Hamiltonian transition per schedule step, best kept | the same driver loop over a continuous transition | a constant schedule reproduces `hmc.sample` draw for draw |
| `learn.relaxed.anneal` | a geometric temperature evaluated at one step | a copy of `opt.schedule.Exponential` | equality at every step with both endpoints exact |
| `search.potts_mcmc.parallel_tempering` | replicas on spawned generators, Metropolis exchange | one algorithm, Potts type; `_swap_log_ratio` copied | per-replica chi-square against the unscaled enumeration, and identical exchange acceptances once the driver is shared |
| `opt.hmc.parallel_tempering` | the same replica and exchange loop over `torch` generators | one algorithm, continuous type | the analytic Gaussian's spread per replica, and the mixture comparison of experiment 004 unchanged |
| `search.tempered` (#350) | replica exchange from the Gibbs moves, weights from the cold replica | a third copy of the replica driver | the tempered weights against enumeration on the three instances #331 pins |

The fix is one driver — a schedule, a transition and a generator in, the
best state and the trajectory out — and one exchange step over
`(state, energy, beta)` triples, with the eight entry points as adapters.

| enumerator | what it enumerates | classification | oracle for a merge |
| --- | --- | --- | --- |
| `search.topology.enumerate_topologies` | every unrooted topology by stepwise insertion | distinct, and stays | the `(2n-5)!!` count |
| `search.max_cut.enumerate_max_cut` | every two-state assignment, keeping the best cut | product enumeration with a score, one side fixed | the lowest-energy configuration of `enumerate_potts` at two states |
| `learn.potts.enumerate_configurations`, `learn.hmm.enumerate_paths` | `itertools.product` over states and sites | two identical copies, neither under the enumeration cap of #230 | equality of the sequences |
| `learn.potts.optimum`, `learn.hmm.optimum`, `learn.relaxed.enumerate_optimum` | argmax over the product with a lexicographic tie rule | three copies of one kernel | each other's result on shared instances; `RelaxedPotts.discrete` equals `PottsLandscape.energy` |
| `likelihood.potts.enumerate_potts` | log weights over the product, then marginals | one algorithm, graph type | its own pin: the transfer matrix on a chain to machine precision |
| `likelihood.hmm_paths.enumerate_hidden_paths` | log joints over the product, then the evidence, posteriors and both decodings | one algorithm, chain type | its own pin: the forward recursion to 1e-12 |
| `likelihood.spatio_sequential.enumerate_spatio_sequential` | log joints over labellings times paths, then three posteriors | one algorithm, coupled type | its own pin: the per-class forward recursion at a relative gap of 0.0 |
| `likelihood.mixture_assignments.enumerate_mixture_assignments` (#393) | log joints over every component assignment, then the evidence, the responsibilities and the argmax | one algorithm, independent-observation type | its own pin: the factorized E step at 1.2e-16 relative |
| `likelihood.ldpc.enumerate_codewords` (#356) | every codeword from the generator matrix | product enumeration over the information bits | `H c = 0` on every word and the `2^k` count |

The fix is one weighted enumeration — cardinalities and a log-weight
function in, assignments and log weights out, the cap of #230 applied once —
with marginalization and argmax as two helpers over it; the four `learn` and
`search` copies become calls, and the three `likelihood` enumerators keep
their result types over the shared kernel.

**Methods refereed by the simulated truth alone, or by neither kind.** Read
from the suite by `infra/problems_tables.py` and typeset in the textbook's
applicability tables; each is a cell where an oracle is wanted:

Five of the six closed at 0.5.0 ([#393](https://github.com/michaelJwilson/snakes_and_ladders/issues/393)),
each at the CI size and each with the agreement it realized:

| problem | method | what now referees it | agreement | tolerance |
| --- | --- | --- | --- | --- |
| hidden Markov model | sampling (`chain_block_sweep`) | the enumerated path posterior over `3**8` paths, by chi-square | 0.0146 largest per-site deviation; smallest per-site p-value 0.062; joint p-value 0.251 over 36 lumped cells | p > 0.01 |
| hidden Markov model | expectation–maximization (`baum_welch_family`) | the enumerated path evidence at the start, at three iterates, and at the fixed point | 0.0 relative at the start; 8.5e-13 at convergence; 2.2e-12 on the initial distribution against the enumerated first-site posterior | 1e-11 and 1e-10 |
| coupled model | the annealed initializer (`graph_burn_in`) | the enumerated maximum-posterior labelling under the parameters it fitted | the mode on 9 of 12 instances, within 1.05 nats on the other 3, against a uniform start's 2 of 12 and a mean 1,020.8 nats | 8 of 12, 1.5 nats |
| Gaussian mixture | the evaluator, the gradient fit and k-means++ | `enumerate_mixture_assignments` over `2**16` assignments | evidence 1.2e-16 relative, responsibilities 4.4e-16; the seeded start in the maximum-posterior assignment on 20 of 20 seeds against uniform seeding's 10 | 1e-12; 20 of 20 |
| continuous test functions | the initializers (`RandomRestart`) | the published Himmelblau minimizers, since a continuous surface has nothing to enumerate | every fit within 6.2e-07 of a published minimizer, value 7.9e-31 | 1e-5, and 1e-12 on the value |

Two remain, and neither is an enumeration that was not attempted:

| problem | method | referee today | why it is not #393's to close |
| --- | --- | --- | --- |
| phylogenetic tree, general time-reversible | the distance start (`FromDistances` on `log_det_distance`) | no test of either significant kind names it as a start | a missing test rather than a missing oracle: `enumerate_topologies` is already the oracle the row would use, and what is absent is a test that starts a GTR fit from the log-det distance at all (#364) |
| phylogenetic tree, Jukes–Cantor | the learned surrogate (`fit_surrogate`) | simulated truth: the maximized likelihood of each enumerated topology | the target a surrogate is trained and scored against is itself a fit, so there is no exact answer to hold it to; the enumeration supplies the topology set, not the number |
| any problem at the release tier | the HMM, the mixture, the test functions, the lattice and the code | simulated truth only at that tier | out of #393's scope by its own statement: a branch-and-bound bound for trees past eight taxa (#329); a boundary contraction for lattices past the transfer-matrix width; density evolution beyond the erasure channel for the code (#340 part 2) |
