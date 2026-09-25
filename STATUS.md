# STATUS

What has landed against `ROADMAP.md`, how it was established, and the pull
request that carries it. Read at version `0.5.0`, the release
[#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376) audits.
The cut of `0.4.0` is deferred and the repository carries no tag, so the
release template's tag precondition is recorded here as not met; `Cargo.toml`
reads `0.3.0`, the version the last built `CHANGELOG.md` section carries.

This file is a ledger against the roadmap, not a project board. Open work lives
in GitHub issues, which is where it is read from (#804). A milestone is recorded
**landed** only where an independent oracle pins the claim; a capability that
runs but is checked against nothing is recorded **not started**, on the terms
§0.4 sets.

## Summary

`CHECKS.md` lists the checks each row rests on, per test, generated from the
suite by `infra/ledgers.sh` and not committed (issue #425); `PROBLEMS.md` names
the code behind each problem class.

| Roadmap item | Status | Evidence | Key PRs |
| --- | --- | --- | --- |
| §0 Development loop | Landed, its record widened since 0.3.0: the problem catalogue and checks ledger, the experiment ledger, the run logger, kind markers on every test, one fixture API, citation-driven figure builds with label guards, notebook re-execution, the frameworks extra as referees, and the parallelism seam as a negative result | Ten required checks; committed PDFs byte-compared and every notebook re-executed on each PR; `0.3.0` was built into `CHANGELOG.md` on 2026-09-03 and never tagged, so `0.4.0` tags both (#358) | [#49](https://github.com/michaelJwilson/snakes_and_ladders/pull/49), [#57](https://github.com/michaelJwilson/snakes_and_ladders/pull/57), [#72](https://github.com/michaelJwilson/snakes_and_ladders/pull/72), [#92](https://github.com/michaelJwilson/snakes_and_ladders/pull/92), [#102](https://github.com/michaelJwilson/snakes_and_ladders/pull/102), [#151](https://github.com/michaelJwilson/snakes_and_ladders/pull/151), [#205](https://github.com/michaelJwilson/snakes_and_ladders/pull/205), [#243](https://github.com/michaelJwilson/snakes_and_ladders/pull/243), [#248](https://github.com/michaelJwilson/snakes_and_ladders/pull/248), [#285](https://github.com/michaelJwilson/snakes_and_ladders/pull/285), [#294](https://github.com/michaelJwilson/snakes_and_ladders/pull/294), [#316](https://github.com/michaelJwilson/snakes_and_ladders/pull/316), [#318](https://github.com/michaelJwilson/snakes_and_ladders/pull/318), [#353](https://github.com/michaelJwilson/snakes_and_ladders/pull/353), [#355](https://github.com/michaelJwilson/snakes_and_ladders/pull/355), [#375](https://github.com/michaelJwilson/snakes_and_ladders/pull/375), [#379](https://github.com/michaelJwilson/snakes_and_ladders/pull/379), [#380](https://github.com/michaelJwilson/snakes_and_ladders/pull/380) |
| 1.1 Simulation & ground truth | Trees, the HMM under six emission families, Potts (1-D chain plus general N-D lattice/MRF and `G(n, p)`), the Gaussian mixture, the coupled spatio-sequential model, three canonical fixtures with outside answers, and two code constructions --- Gallager's ensemble and the bicycle code --- landed as first-class simulators | Simulated substitution frequencies against the closed-form JC probabilities; GTR reproduces JC to machine precision; HMM state and emission marginals against brute-force path enumeration; the count families reach their Poisson and binomial limits at the `O(1/x)` rate; Potts single-site and pair marginals against exhaustive enumeration at 3-state 3x3 and 2-state 4x4; the coupled simulator held to what it composes by chi-square at 0.001 over 400 draws; every encoded codeword satisfies `H c = 0` at `n = 24`, `96`, `510`, and a codeword's channel ratios are the zero word's up to sign on every realization; the bicycle construction's degrees, rank and self-orthogonality `H H^T = 0` exact over GF(2), and its 12-bit fixture at minimum distance 4 over 64 enumerated codewords | [#58](https://github.com/michaelJwilson/snakes_and_ladders/pull/58), [#64](https://github.com/michaelJwilson/snakes_and_ladders/pull/64), [#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115), [#120](https://github.com/michaelJwilson/snakes_and_ladders/pull/120), [#182](https://github.com/michaelJwilson/snakes_and_ladders/pull/182), [#190](https://github.com/michaelJwilson/snakes_and_ladders/pull/190), [#223](https://github.com/michaelJwilson/snakes_and_ladders/pull/223), [#224](https://github.com/michaelJwilson/snakes_and_ladders/pull/224), [#259](https://github.com/michaelJwilson/snakes_and_ladders/pull/259), [#261](https://github.com/michaelJwilson/snakes_and_ladders/pull/261), [#263](https://github.com/michaelJwilson/snakes_and_ladders/pull/263), [#302](https://github.com/michaelJwilson/snakes_and_ladders/pull/302), [#356](https://github.com/michaelJwilson/snakes_and_ladders/pull/356), [#546](https://github.com/michaelJwilson/snakes_and_ladders/pull/546) |
| 1.2 Likelihood & energy engine | CPU landed (NumPy, PyTorch, Rust at 2.5x the oracle at 200 taxa by 11,000 sites); belief propagation with two exact oracles; Fitch and Sankoff parsimony; one factor graph with sum-product and max-product over it, Viterbi included; forward–backward as an evaluator; certified bounds and learned surrogates; the LDPC decoder pinned to the general sum-product and to enumeration; two runtime audits; GPU dispatch not started (#280) | Worst relative deviation 4.0e-14 against brute-force marginalization across three backends and four site counts spanning a factor of 30; max-product returns the enumerated Viterbi path on four chains; flooding on the 8x8 lattice within 1.8x of belief propagation after the audit; decoder posteriors within 4.6e-11 of the general flooding on six loopy codes and 1.9e-13 of enumeration on a cycle-free one; the 19,998-bit (3,6) code brackets the erasure threshold 0.4294 between 0.42 and 0.44 | [#66](https://github.com/michaelJwilson/snakes_and_ladders/pull/66), [#74](https://github.com/michaelJwilson/snakes_and_ladders/pull/74), [#81](https://github.com/michaelJwilson/snakes_and_ladders/pull/81), [#112](https://github.com/michaelJwilson/snakes_and_ladders/pull/112), [#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148), [#219](https://github.com/michaelJwilson/snakes_and_ladders/pull/219), [#247](https://github.com/michaelJwilson/snakes_and_ladders/pull/247), [#296](https://github.com/michaelJwilson/snakes_and_ladders/pull/296), [#307](https://github.com/michaelJwilson/snakes_and_ladders/pull/307), [#317](https://github.com/michaelJwilson/snakes_and_ladders/pull/317), [#343](https://github.com/michaelJwilson/snakes_and_ladders/pull/343), [#346](https://github.com/michaelJwilson/snakes_and_ladders/pull/346), [#354](https://github.com/michaelJwilson/snakes_and_ladders/pull/354), [#356](https://github.com/michaelJwilson/snakes_and_ladders/pull/356) |
| 1.3 Continuous optimization | Landed for trees, the HMM, the Potts chain and lattice, and the mixture; an interval at any fit, whatever produced it; posterior sampling by leapfrog and Yoshida, tempered and adapted; initializers, multi-start, k-means++ and, since #373, the tree's two data-driven starts — neighbor joining on pairwise distances and the closest tree of the Hadamard conjugation (#364); since #541 three chain starts on a surrogate surface — `FromChain`, `FromAnnealing`, `FromTempering` — and eight seedings of the coupled model's emission parameters compared through them; closed-form test functions; the mixture at equal evaluations through the budget utility | Gradients against central differences; 95% intervals cover truth at the nominal rate over 60 replicates; the lattice fitted against an enumerated normalizer, coverage 157/160 at 100 samples and 153/160 at 400 and 1600; integrator orders realized at 4.000 and 16.001; the adapted chain's acceptance 0.650 pooled over 20 seeds at a target of 0.65; restarts reach the mixture's best-known optimum from 7/40 starts against tempering's 4/40 (`p = 0.549`) and annealing's 1/40 (`p = 0.031`); neighbor joining returns every branch of a known tree to 1e-12 at 20 and 50 taxa, and no start separates on the fit at five and six taxa, every one converging in 22 to 27 evaluations (experiment 006); the coupled model's emission seedings split the two orderings at 100 components on 4,000 observations — the prior draw reaches the best projected value on 6 of 6 paired instances while `tempering` and `data` recover more of the generating component on 6 of 6, both at `p = 0.031`, so no default moves (experiment 009) | [#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115), [#116](https://github.com/michaelJwilson/snakes_and_ladders/pull/116), [#119](https://github.com/michaelJwilson/snakes_and_ladders/pull/119), [#120](https://github.com/michaelJwilson/snakes_and_ladders/pull/120), [#256](https://github.com/michaelJwilson/snakes_and_ladders/pull/256), [#263](https://github.com/michaelJwilson/snakes_and_ladders/pull/263), [#269](https://github.com/michaelJwilson/snakes_and_ladders/pull/269), [#271](https://github.com/michaelJwilson/snakes_and_ladders/pull/271), [#272](https://github.com/michaelJwilson/snakes_and_ladders/pull/272), [#303](https://github.com/michaelJwilson/snakes_and_ladders/pull/303), [#345](https://github.com/michaelJwilson/snakes_and_ladders/pull/345), [#348](https://github.com/michaelJwilson/snakes_and_ladders/pull/348), [#352](https://github.com/michaelJwilson/snakes_and_ladders/pull/352), [#373](https://github.com/michaelJwilson/snakes_and_ladders/pull/373), [#553](https://github.com/michaelJwilson/snakes_and_ladders/pull/553) |
| 1.4 Move sets & classical baselines | Trees landed, with warm starts, lazy scoring and support; large parsimony; Potts cluster updates; the exact-baseline family — minimum cut, alpha expansion with its proved bound, Max-Cut with a certificate read against a Burer-Monteiro solve of the relaxation rather than an optimal SDP, which makes the ratio optimistic and is #334; schedules, annealing and parallel tempering on the Rust sweep by default; one Gibbs sampler and annealer over any factor graph, and a tempered ensemble over labellings, decodings and topologies; the coupled model fitted; Viterbi and posterior decoding landed; iterated conditional modes over HMM paths not started (#176) | NNI and SPR neighbour counts exhaustively verified at `n = 5..8`; hill climbing reaches the enumerated optimum from 12 of 12 starts and large parsimony from every start at five and six taxa; the two-state ground state exact against enumeration over 36 shape-coupling-field combinations; on the planted glass at equal sweeps tempering 12/12 against annealing 10/12 and restarts 4/12; the tempered weight within 0.039 of enumeration on every seed of 20 | [#82](https://github.com/michaelJwilson/snakes_and_ladders/pull/82), [#127](https://github.com/michaelJwilson/snakes_and_ladders/pull/127), [#128](https://github.com/michaelJwilson/snakes_and_ladders/pull/128), [#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148), [#212](https://github.com/michaelJwilson/snakes_and_ladders/pull/212), [#220](https://github.com/michaelJwilson/snakes_and_ladders/pull/220), [#221](https://github.com/michaelJwilson/snakes_and_ladders/pull/221), [#222](https://github.com/michaelJwilson/snakes_and_ladders/pull/222), [#255](https://github.com/michaelJwilson/snakes_and_ladders/pull/255), [#272](https://github.com/michaelJwilson/snakes_and_ladders/pull/272), [#284](https://github.com/michaelJwilson/snakes_and_ladders/pull/284), [#289](https://github.com/michaelJwilson/snakes_and_ladders/pull/289), [#304](https://github.com/michaelJwilson/snakes_and_ladders/pull/304), [#307](https://github.com/michaelJwilson/snakes_and_ladders/pull/307), [#310](https://github.com/michaelJwilson/snakes_and_ladders/pull/310), [#346](https://github.com/michaelJwilson/snakes_and_ladders/pull/346), [#350](https://github.com/michaelJwilson/snakes_and_ladders/pull/350) |
| 1.5 Continuous samplers, HMC & tempering | Landed: leapfrog and Yoshida integrators with step-size adaptation, temperature schedules, simulated annealing, and a parallel-tempered ensemble over any supported problem | A chain's spread against the analytic Gaussian; each replica's marginals against the unscaled model's enumeration with exchanges on; Yoshida measured fourth-order and **declined** on cost, 91 evaluations per trajectory against leapfrog's fewer | [#266](https://github.com/michaelJwilson/snakes_and_ladders/issues/266), [#267](https://github.com/michaelJwilson/snakes_and_ladders/issues/267), [#268](https://github.com/michaelJwilson/snakes_and_ladders/issues/268) |
| 2.0 RL definition | Landed: `app:rl` in the textbook is a self-contained account of the algorithms a learned proposal is drawn from, organised on the state-space and update axes, with the suitability of each for the supported problems argued rather than asserted | The section is cited from `sec:policy-gradient` at the point of use, and every algorithm it names is either implemented or stated as not built | [#313](https://github.com/michaelJwilson/snakes_and_ladders/issues/313) |
| 2.1 RL formulation & deployment | The estimator, the Potts, hidden-path and tree environments, a critic, an actor–critic, PPO and a PUCT planner landed, each pinned to enumeration; a tree policy trained on the fixture hill climbing fails, a tie over one feature and ahead of greedy over the seven-column set (#349); not yet measured against restarts | Enumerated gradient against finite differences at 1.5e-11 relative; on the Potts chain REINFORCE 86.6%, PPO 96.3% and the planner 92.6% at 8.3 evaluations per episode against greedy's 80.2% at 48; on the 7-taxon fixture the single feature reaches 0.487 against greedy's 0.480 (sign test `p = 0.79`) and the full set 0.796, ahead on 16 of 16 seeds (`p = 3.05e-5`), while restarts reach 1.000 at the same budget | [#135](https://github.com/michaelJwilson/snakes_and_ladders/pull/135), [#137](https://github.com/michaelJwilson/snakes_and_ladders/pull/137), [#139](https://github.com/michaelJwilson/snakes_and_ladders/pull/139), [#192](https://github.com/michaelJwilson/snakes_and_ladders/pull/192), [#193](https://github.com/michaelJwilson/snakes_and_ladders/pull/193), [#198](https://github.com/michaelJwilson/snakes_and_ladders/pull/198), [#320](https://github.com/michaelJwilson/snakes_and_ladders/pull/320), [#349](https://github.com/michaelJwilson/snakes_and_ladders/pull/349), [#355](https://github.com/michaelJwilson/snakes_and_ladders/pull/355) |
| 2.2 Curriculum learning | Started: the surrogate curriculum from 5 to 6 taxa, from 3x3 to 4x6 lattices, and across `spatio_only`'s 9 to 5,041 sites; weight transfer for a policy and batched rollout not started | Zero-shot at six taxa the set surrogate falls to `R^2` 0.68 and recovers to 0.94 after transfer, the MLP holds 0.92 and reaches 0.95; lattice surrogates transfer zero-shot at 0.99 at a shared field and collapse to -338.6 at a per-site one, recovering to 0.722 | [#317](https://github.com/michaelJwilson/snakes_and_ladders/pull/317), [#547](https://github.com/michaelJwilson/snakes_and_ladders/pull/547) |
| 2.3 Empirical validation | The budget utility and the exact paired test landed, and six budget-matched comparisons are recorded; no empirical alignment and no external tool ([#126](https://github.com/michaelJwilson/snakes_and_ladders/issues/126)) | Every comparison at one budget over shared seeds with McNemar's exact test: the glass, Rastrigin, the mixture, the relaxation against greedy, the cluster updates at the transition, and the tree's starts at equal evaluations | [#303](https://github.com/michaelJwilson/snakes_and_ladders/pull/303), [#348](https://github.com/michaelJwilson/snakes_and_ladders/pull/348) |
| 3.1 Surrogates & bounds | Landed as certified analytic bounds plus learned predictors on the gap above them, ranking a neighbourhood so only the top-`K` are re-scored exactly; the filter's cost ratio at large `n` is **unmeasured** and is what remains | Each bound proved in Appendix B and asserted against the exact value it bounds; the learned gap predictor refereed by the enumeration the bound is checked on | [#317](https://github.com/michaelJwilson/snakes_and_ladders/issues/317) |
| 4.1 Tracking, ablations & leaderboard | The experiment ledger, its generated index and the run logger landed; six experiments recorded, each capped at a ten-line body (#458); the Aim run store landed as the optional `track` extra, `Run` written from `aim.Run` ([#75](https://github.com/michaelJwilson/snakes_and_ladders/issues/75), #782) | Every file under `docs/experiments/` validated against the template and the cap per pull request, and this file cites the files rather than restating them | [#316](https://github.com/michaelJwilson/snakes_and_ladders/pull/316), [#318](https://github.com/michaelJwilson/snakes_and_ladders/pull/318) |
| Stage 5 Research extensions | Gumbel-softmax relaxation of Potts and HMM states landed; the tropical Grassmannian half landed, refereed by enumeration and the Hadamard closed form, and not shown to beat a classical baseline, so it is conserved in `sandbox/`; learned surrogates rank a neighbourhood with exact re-scoring of the top candidates; stochastic escape by epsilon-greedy landed | Gumbel-softmax exact at every corner to 1e-11 and deterministic ascent 18/40 against greedy's 5/40, McNemar `p = 0.00098`; the tropical relaxation exact at every corner to 3.8e-16 relative, four-point violation of the Hadamard metric under 1e-12, ascent 8/8 at five and six taxa and 7/8 at eight against the enumerated maximum, and neighbor joining reaching it at no gradient steps; a surrogate-ranked SPR search reaches its optimum from 4/4 starts at 5 fits against 312; escape from a local optimum rises from 0.111 at `epsilon = 0` to 0.883 at 0.4 | [#198](https://github.com/michaelJwilson/snakes_and_ladders/pull/198), [#225](https://github.com/michaelJwilson/snakes_and_ladders/pull/225), [#317](https://github.com/michaelJwilson/snakes_and_ladders/pull/317) |

## Progress Since the 0.4.0 Audit

What moved between [#375](https://github.com/michaelJwilson/snakes_and_ladders/pull/375), the 0.4.0 audit this one starts from,
and this release. A milestone not named here did not move.

| Roadmap item | What moved | Pull request |
| --- | --- | --- |
| Milestone 2.1 | Canonical control fixtures, the control arm of #596: a chain, a gridworld, a cliff walk and Towers of Hanoi, each with an optimum known from outside, and `value_iteration` as a second computation of it --- a sweep over an enumerated state set against `learn.exact`'s recursion over trajectories, agreeing to 1e-12 on all four. Hanoi's `2^d - 1` is asserted as an integer at d = 1..4 over `3^d` states. Both learners measured reach the optimum from every start of the 4x4 grid: REINFORCE at 300 iterations of 16 episodes, PPO at 60 of 16 --- 5x fewer iterations at 5.9x the cost each, 3.1 s against 3.6 s. The k-armed bandit and slippery Frozen Lake are **not** here: `Environment.step` is deterministic by contract and `learn.exact`'s enumeration depends on it | [#597](https://github.com/michaelJwilson/snakes_and_ladders/issues/597) |
| Milestone 2.1 | Tabular Q-learning and SARSA, whose only reason to exist is a claim neither can check alone: on the cliff walk they **disagree in the documented direction**. Q-learning's greedy route runs beside the cliff and returns the optimal -7; SARSA gives up exactly the two moves that buy a row of clearance, -9; and under the epsilon-greedy policy each was learned with the order reverses, -12.90 for SARSA against -25.54 over 200 episodes. Their values separate by four orders of magnitude on the 4x4 grid --- Q-learning 2.14e-05 from `V*`, SARSA 1.8889 from it, because its fixed point is `q_pi` and not `q*` --- while both reach the optimum greedily from all 15 starts. With the default pessimistic initialisation neither finds the chain's far prize: reaching it needs seven consecutive exploratory steps, 7.8e-10 per episode, and the learned value of the start is the consolation | [#597](https://github.com/michaelJwilson/snakes_and_ladders/issues/597) |
| Milestone 2.1 | The sizing harness the gate needs before any learning: `learn.failure` runs a baseline from seeded starts at one budget and sweeps sizes for the first at which it falls below a threshold. Refereed by reproducing #194 --- on the 7-taxon tree at 60 decisions a single greedy descent reaches the enumerated maximum from 0.560 of 50 starts and random restarts from 50 of 50, so the baseline to beat there is 1.000. A baseline that never fails reports no failure, which is the reportable outcome and not a gap | [#596](https://github.com/michaelJwilson/snakes_and_ladders/issues/596) |
| Milestone 2.1 | The Potts lattice sized at its critical coupling, the first per-problem arm of #596, and it **renames the baseline**. Against alpha-expansion's energy at 144 and 576 sites, best of eight starts: single-descent ICM is 5.31% and 10.59% above it and **unchanged at 25x the budget**, so its failure is structural rather than starved; random restarts close half of it, to 3.69% at 576 sites; Wolff degrades with size in a field, 3.38% to 34.35%, being blind to the unary term it moves sites against; and simulated annealing reaches the expansion's energy at every size and budget measured. So the gate is argued against annealing, and two instances cannot host it at all --- in zero field the optimum is the closed form `-J|E|`, agreeing with the exact two-label cut to 2.7e-16 and reached by restart-ICM and Wolff from every start to 1,024 sites, and at two labels in a field alpha-expansion equals that cut bitwise. The exact-hit fraction is also the wrong instrument for a continuous energy: it reads 0.00 for annealing at 144 sites where a 1% allowance reads 0.38 | [#596](https://github.com/michaelJwilson/snakes_and_ladders/issues/596) |
| Milestone 1.1 | Polar codes, conserved in `sandbox/` as a declined route rather than catalogued as a problem class: the Kronecker transform, the frozen set from Arikan's exact erasure recursion or the Gaussian approximation, Reed--Muller as the same transform under the weight rule, and successive-cancellation and list decoding. Capacity is conserved exactly under the transform (1e-12 at `n = 3, 5, 8, 11`); at `N = 16` enumeration gives the maximum-likelihood floor and the gap is measured --- 98 block errors for SC against 62 for ML over 200 shared draws at `sigma = 1.0`, a list of 4 recovering 34 of the 36; `SCL(1) == SC` bitwise and `SCL(2^k) == ML`; the naive recursion pins the list layout bitwise over 900 draws | [#605](https://github.com/michaelJwilson/snakes_and_ladders/pull/605) (#593) |
| Milestone 1.1 | The algebraic codes and the capacity they are read against: repetition, single parity check, Hamming, Golay and Reed--Solomon over `GF(2^m)`, each decoded in closed form at every length, and `sim.capacity` for the three channels. Perfection holds as an integer equality; RS(7,3) meets Singleton with equality at `d = 5` over all 512 enumerated codewords; the algebraic decoder is exact on 900 of 900 draws to `t` and, past it, refused 260 of 300 and was wrong on 40, recovering none by luck. Read against capacity, the (3,6) ensemble leaves 13.8% of the erasure channel unused at its realized rate 0.5020, and the four algebraic pairs give up factors of 2.7 to 5.6 at a block error rate of 1e-2 | [#700](https://github.com/michaelJwilson/snakes_and_ladders/pull/700) (#594) |
| §0 Development loop | A typeset map of the package's Python surface, generated from the docstrings by `ast` and edited by no one: `docs/api_map.pdf`. The tree carries 1,743 entries --- 167 modules, 284 classes, 656 functions, 636 methods, read 2026-09-18 on `main` 266cb65. The entry count is checked against a second walk of the tree, and a public function without a summary line is refused rather than typeset as a blank | [#578](https://github.com/michaelJwilson/snakes_and_ladders/pull/578) (#576) |
| §0 Development loop | The two documents are named the paper and the textbook, `infra/build_documents.sh` replaces the script named after the retired artifact, and a guard fails any live file naming it | [#379](https://github.com/michaelJwilson/snakes_and_ladders/pull/379) (#377) |
| §0 Development loop | Validation is one command under five minutes: an inputs stamp beside every committed figure and notebook (the stamps were measured wrong on 476 of 476 decisions and removed by #490); guard-only selection; a per-test duration cap; one BLAS thread per process | [#380](https://github.com/michaelJwilson/snakes_and_ladders/pull/380) (#372) |
| §0 Development loop | A generator rather than a seed for the relaxed benchmark, and full history for the committed-PDF rule, which every pull request had been failing on a shallow checkout | [#378](https://github.com/michaelJwilson/snakes_and_ladders/pull/378) |
| §0 Development loop, §1.3 | Every problem statement carries the same five labelled parts, the applicability tables gain a third reading by method family, and five hand-rolled implementations are refereed against the frameworks that duplicate them | this release ([#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376)) |
| Milestone 1.3 | The tree's two data-driven starts --- neighbor joining on the pairwise Jukes--Cantor and log-det distances, and the closest tree of the Hadamard conjugation --- with Atteson's radius and the four-point condition as their guarantees, measured against the objective's own start at equal evaluations | [#373](https://github.com/michaelJwilson/snakes_and_ladders/pull/373) (#364) |
| Milestone 2.3 | A sixth budget-matched comparison, the tree's starts at 2,000 evaluations over twenty seeds, recorded as experiment 006 | [#373](https://github.com/michaelJwilson/snakes_and_ladders/pull/373) (#364) |
| Milestone 1.1 | The Calderbank–Shor–Steane code the bicycle matrices define, and decoding it under degeneracy. `k = n - 2 rank(H)` is 0 at both classical fixtures, so two instances are declared at `m < n / 2`: [[16, 2]] with 30 four-cycles and [[96, 16]] with 160. Success is the residual lying in the row space, not equalling the error, and it is pinned from both sides before a decoder runs. Exact degenerate maximum likelihood over all 65,536 errors gives a logical error rate of 0.070297, against 0.072735 for the decoder maximizing one error's probability; sum-product on the same matrix gives 0.1705, failing 341 of 2,000 as 73 logical cosets and 268 non-convergences | [#568](https://github.com/michaelJwilson/snakes_and_ladders/pull/568) (#362) |
| Milestone 1.1 | The bicycle construction as a second code family: `H = [A | A^T]` from a sparse circulant, rows deleted to the target rate, on the channels and the decoder already carried. It decodes worse than a Gallager draw of the same length, degrees and rate --- more blocks failed at all six 96-bit settings and at eleven of twelve 996-bit ones --- and is carried for the CSS condition `H H^T = 0` the quantum half rests on | [#546](https://github.com/michaelJwilson/snakes_and_ladders/pull/546) (#362) |

The 0.4.0 cut itself did not move: no tag exists, so `0.4.0` and `0.5.0` are
both unreleased and `Cargo.toml` still reads `0.3.0`.

## §0 The Development Loop

The loop described in `ROADMAP.md` §0 is in force. Blank issues are disabled
and filings route through the task template
([#57](https://github.com/michaelJwilson/snakes_and_ladders/pull/57)); the pull-request
template carries the Definition of Done, the benchmark table, the realized
tolerance table, and the deferred-work section
([#49](https://github.com/michaelJwilson/snakes_and_ladders/pull/49),
[#86](https://github.com/michaelJwilson/snakes_and_ladders/pull/86),
[#89](https://github.com/michaelJwilson/snakes_and_ladders/pull/89)); labels are generated
from `.github/labels.yml` by a workflow, so the taxonomy cannot drift from the
documents that describe it.

Nine required checks gate a merge and a tenth, `documents`, runs on the push
to `main` instead ([#488](https://github.com/michaelJwilson/snakes_and_ladders/issues/488));
three do work no reviewer can do by inspection. The documents job renders every
figure in `sal.qa.manifest` — the two documents cite all 23
since [#492](https://github.com/michaelJwilson/snakes_and_ladders/issues/492), and
the stamps that once narrowed the set are deleted
([#490](https://github.com/michaelJwilson/snakes_and_ladders/issues/490)); the `lint`
job fails a pull request that changes `docs/paper.pdf` or `docs/textbook.pdf`
without being the rebuild a **Rebuild the documents** ticket asked for
([#72](https://github.com/michaelJwilson/snakes_and_ladders/pull/72),
[#483](https://github.com/michaelJwilson/snakes_and_ladders/issues/483)); and the notebooks job
re-executes every notebook under `docs/nb/`, failing one whose Further work
section is missing or names no ticket
([#278](https://github.com/michaelJwilson/snakes_and_ladders/issues/278)) and one whose
printed output has moved. The coverage floor cannot be lowered to pass a
change. Benchmarks run only when the diff touches code they measure, and the
release-gated suite is excluded per pull request — 138 s over 540 tests against
989 s for the full suite
([#159](https://github.com/michaelJwilson/snakes_and_ladders/pull/159)).

**Coverage counts a test only when it judges the science** (issue #729, PR
[#731](https://github.com/michaelJwilson/snakes_and_ladders/pull/731)). One run of
the regression tier on that branch with a context per test read **94.33%** of
16,840 statements under `--cov-fail-under`, **34.48%** of them reached by
importing the package with no test at all; recut to the tests carrying
`end2end` or `oracle` the figure is **77.93%**, and **82.68%** with `qa`
exempt, which `infra/coverage_recut.py` now holds on the push to `main` and at
the release gate, `search` at its own **86.18%**. The contexts cost the tier
33 s of 934 s. `structural` and `edge_case` are retired into `infra` and
`smoke`, `simulated_truth` is `end2end`, `mathematical` is `analytic`, and
`patch`, `backend`, `bug`, `warning` and `snapshot` are registered as the
finding axis, none of which counts; root `CLAUDE.md` states the rule and the
audit the ticket plans raises the judged figure directory by directory,
`search` first. The complement --- every kind and finding but the two that
judge --- is the third guard (issue #732, PR
[#733](https://github.com/michaelJwilson/snakes_and_ladders/pull/733)): **89.25%** with `qa` exempt,
`search` **88.66%**, floored at 89.2 and 88.6, and `--functions` lists the
330 public callables no judging test enters, 157 of them entered by no test.
The `search` audit is the first directory to raise them (step 2): judged
**83.75%** with `qa` exempt against 83.08% and `search` **89.99%** against
87.07%, floored at 83.7 and 89.9; the complement **89.29%** against 89.31%
and `search` **88.96%** against 88.66%, so the whole's 89.2 stands and
`search`'s rises to 88.9. `search/kernels.py` goes 12.68% to **97.18%**,
`search/surrogate.py` 79.82% to **97.37%** and `search/projection.py` 92.27%
to **94.48%**, and the package's statements no test reaches fall 141 to 62.
The codes are the second (step 3): judged **83.89%** with `qa` exempt and the
complement **89.28%**, floored at **83.8** and 89.2, with `sim` **77.44% to
78.64%** as convolutional, turbo, CSS, the elementary codes and Reed--Solomon
each gain the planted run they lacked. `search` reads **89.86%** against its
89.9 floor on that tree and the floor does not follow it down: the merges of
#744 to #749 deleted the `oracle` over the Rust max-flow kernel and `opt`'s
four Himmelblau minima, which is `search` 89.99% to 89.86% and `opt` 86.42% to
85.93% and is not this branch's.
`learn` is the third (step 4): judged **84.93%** with `qa` exempt against
83.89% and the complement **89.26%** against 89.28%, floored at **84.9** and
89.2, with `learn` **79.61% to 86.62%** and its judged deficit 342 to 191
statements. Six referees carry it --- a planted ground state recovered at both
grains on `potts_nd`, a hand-computed table on `arena`, a forward expansion
over 512 trajectories on `exact`, the enumerated score and gradient at four
interior points of an ascent on `relaxed`, the `gamma = 1` telescoped return
on `critic`, and the enumerated binomial on `failure` --- and `canonical.py`
stays at 77.98%, no learner being run on the chain, the cliff or Hanoi.
`likelihood` is the fourth (step 4): judged **85.35%** with `qa` exempt
against 84.93% and the complement **89.25%** against 89.26%, floored at
**85.3** and 89.2, with `likelihood` **89.17% to 91.24%**, its judged deficit
259 to 198 statements and its public callables no judging test enters 41 to
32, 27 of them declarations whose bodies run at import. Three referees and
thirteen re-marks carry it --- the device policy's every route against the
NumPy pruning oracle at the tolerance it returns, density evolution against
its scalar recursion and the closed-form threshold, and
`prune_with_matrices` against direct marginalization --- and
`belief_propagation.ConvergenceError` is left, its non-convergence asserted
and refereed by nothing.
`opt` is the fifth (step 4): judged **85.49%** with `qa` exempt against
85.35% and the complement **89.25%**, unmoved, so the judged floor rises to
**85.4** and 89.2 stands, with `opt` **85.93% to 87.28%**, its judged deficit
176 to 154 statements and its public callables no judging test enters 20 to
16, every one of the 16 a declaration whose body runs at import. Four
referees and four re-marks carry it --- `mcnemar` against
`scipy.stats.binomtest` over 168 contingency tables at 1.84e-16, the fixture's
six declared minimizers against the published values and each function's own
value, gradient and curvature, and a warm-up's 1,200 draws against the exact
Gaussian marginals within 2.17 standard errors of a bound of 3.0 --- and
`opt/budget.py` goes 64.35% to **76.52%** and `opt/schedule.py` 76.03% to
**81.51%**. `opt/objective.py` stays at 78.95%, its four unreached statements
a `Protocol`'s `...` bodies.
`sim` is the sixth (step 4): judged **86.11%** with `qa` exempt against
85.49% and the complement **89.27%** against 89.25%, so the judged floor
rises to **86.1** and 89.2 stands, with `sim` **78.64% to 82.05%**, its
judged deficit 400 to 311 statements and its public callables no judging
test enters 35 to 23, 18 of them declarations whose bodies run at import.
Fourteen referees and ten re-marks carry it --- the three families'
published rate-1/2 limits and `brentq` on the entropy equation, the
sphere-packing counts and the published CRC long division, the published
GF(16) table and Fermat over 501 elements, `rustworkx`'s own grid, and the
two trees the Newick documentation carries --- and `sim/capacity.py` goes
23.29% to **73.97%**, `sim/galois.py` 71.13% to **87.63%**, `sim/graph.py`
76.22% to **86.01%** and `sim/elementary_codes.py` 64.52% to **82.80%**.
`sim/ldpc.py` stays at 82.43%, the published (7,4) parity check reaching
what its oracles already reached.
The root modules are the last (step 4), and they close the ticket: judged
**86.34%** with `qa` exempt against 86.11% and the complement **89.47%**
against 89.27%, so the judged floor rises to **86.3** and 89.2 stands, with
the root **75.58% to 78.23%** and its judged deficit 250 to 217 statements.
Its 46 public callables stay 29 judged, 5 by a self-check and 12 by nothing,
11 of the 12 declarations whose bodies run at import and the twelfth
`inputs.module_closure`, which is infrastructure. Two modules move:
`parallel.py` **51.95% to 80.52%**, 41 seeded tasks mapped on threads and on
processes equal to the serial map bitwise beside the closed form
`n (n - 1) (2n - 1) / 6 = 22,140`; and `emissions.py` **82.30% to 83.82%**,
the four count families and issue #658's two covariate branches against
`scipy.stats`, 1.07e-14 relative at the widest of a declared 1e-12.
`numerics.py` stays at 84.00% and `incidence.py` at 77.03%, `scipy.special`
and `scipy.sparse` reading what their oracles already read; `log.py` and
`inputs.py` are the repository's own machinery and are `infra`, not judged.
`qa` is exempt, so its three renderers no test reached gain `snapshot` pins
beside the gate: the tropical relaxation's `0.0193186` temperature and
`7.276e-12` residual, the turbo waterfall's 121, 87, 32 and 8 errors of
1,200 at 8 iterations, and the tree policy's learned `0.486875` against
greedy's `0.480000` --- `docs/experiments/005-tree-policy-features.md`
records 0.487 against 0.480 --- which takes `qa`'s statements no test
reaches 225 to 156. Over the ticket the judged figure goes **82.68% to
86.34%** and its floor 83.7 to **86.3**, by referees added and never by a
widening of the set that counts.
The ladder the `oracle` tests form --- 520 of them in 148 files on this tree,
416 in 141 when issue #734 surveyed it --- is declared once in
`infra/ladder.py` and generated into the textbook as `tab:ladder`: 89 rungs
over the five problems, 89 pinned by a named `oracle` test and 0 carrying the
ticket instead of one --- the Potts ladder pinned to the foot, each of its five
open rungs against the rung below it (issue #734, step 2), the tree ladder
with it, its four against the closed-form mixture, enumeration, pruning and the
analytic surrogate (step 3), the HMM ladder, its four against the path
enumeration, the coupled enumeration, the re-estimation equations and the
coupled E step (step 4), the codes ladder, its five against the enumerated
maximum likelihood, belief propagation on the code's own parity check and one
constituent's BCJR (step 5), and the mixture ladder, its four against the
enumerated assignment posterior, the enumerated optimal partition, the
divergence each seeding declares and each chain run's own record (step 6).

Two releases have been cut under the procedure, each from a Release ticket
gated on `infra/release.sh`: `0.1.0`
([#102](https://github.com/michaelJwilson/snakes_and_ladders/pull/102)) and `0.2.0`
([#151](https://github.com/michaelJwilson/snakes_and_ladders/pull/151)). Each ran the
consolidation audit the template drives, and `0.2.0`'s found a categorical
sampler duplicated three times, two copies missing the clamp the third had, so
a probability row summing to `1 - 4e-16` could return a category past the end
of the alphabet.

`0.3.0` was built into `CHANGELOG.md` on 2026-09-03 from seven fragments and
never tagged, so the `0.4.0` release
([#358](https://github.com/michaelJwilson/snakes_and_ladders/issues/358)) tags
both at its merge commit; 81 fragments accumulated between them and every
milestone moved.

**Between `0.2.0` and `0.3.0`, six pull requests refined the loop and its
record; no roadmap milestone moved.** `ROADMAP.md` was restructured around the
development loop and the three problem classes, and `STATUS.md` and
`TICKETS.md` (deleted by #804) were introduced as the ledger and backlog
([#152](https://github.com/michaelJwilson/snakes_and_ladders/pull/152),
[#153](https://github.com/michaelJwilson/snakes_and_ladders/pull/153)); the thirteen QA
scripts were routed through one `sal.qa.runner`
([#156](https://github.com/michaelJwilson/snakes_and_ladders/pull/156)) and
`sal.qa.manifest` given the figure that renders each output
([#157](https://github.com/michaelJwilson/snakes_and_ladders/pull/157)); the regression
suite was split by submodule and its documented budget corrected after being
found stale
([#159](https://github.com/michaelJwilson/snakes_and_ladders/pull/159)); every module
`CLAUDE.md` was pointed at the Writing Style section instead of restating it
([#158](https://github.com/michaelJwilson/snakes_and_ladders/pull/158)); and a plan's
required shape — 2–5 validated steps ending in an Open Questions section — was
stated in `ROADMAP.md` §0.2, `DEV.md` and `infra/CLAUDE.md` alike
([#164](https://github.com/michaelJwilson/snakes_and_ladders/pull/164)).

**External frameworks arrive as referees and adapters, not replacements**
([#322](https://github.com/michaelJwilson/snakes_and_ladders/issues/322), closing
[#242](https://github.com/michaelJwilson/snakes_and_ladders/issues/242) into it). A
`frameworks` extra carries `gymnasium` 1.3.0, `rustworkx` 0.18.1, `torchrl`
0.13.3 and `torch_geometric` 2.8.0, and `sal.sandbox` is the
home an implementation moves to once a framework replaces it on a hot path,
with a guard that only tests and QA import it. No framework has yet beaten
the implementation it would replace; `sandbox.tropical` moved on the other
rule, a measured decline conserved rather than deleted (#408).
`search.gym.GymnasiumEnvironment` wraps any `learn.environment.Environment` unchanged and
passes Farama's `check_env` on the Potts chain and the 5-taxon tree
environment, while an episode round-tripped through both interfaces from one
seed has the same states, rewards, termination and candidate count.
`PottsGraph` converts to and from a `rustworkx` multigraph exactly, doubled
bonds included; the open lattices are `rustworkx.generators`' grid and path
graphs as edge sets of the same integers, and `G(n, p)` agrees with
`undirected_gnp_random_graph` at both ends of `p` and on the mean edge count of
400 draws. The installed `rustworkx` has a global Stoer–Wagner cut and no s–t
flow, asserted so the oracle moves when that changes; `ising_ground_state` is
pinned instead against `networkx`'s minimum cut at extents 8 to 16. TorchRL's
`GAE` and `ClipPPOLoss` reproduce `learn.ppo`'s advantages, objective and
gradient to 1e-10 once their float32 buffers are handed float64, and PyTorch
Geometric's `GINConv` reproduces `GraphSurrogate` to 1e-12 on tied first-layer
weights — on general weights GIN sums node and neighbours before its network
where ours concatenates them, so the two are different architectures and the
test says so. Conversion cost, timed apart from any call on 4 cores at a
1-minute load of 0.23: `to_rustworkx` 12.5 µs and `from_rustworkx` 46.4 µs at
extent 8, 50.9 and 196.9 µs at 16, 747.2 µs and 3.78 ms at 64, against
`rx.connected_components` at 4.6, 15.7 and 212.3 µs and the Python
`ising_ground_state` at 1.40, 5.85 and 186.5 ms. The conversion is below the
cost of the cheapest call it would front at every size, and no hot path moves
until a measured adoption says so; #388, #389 and #390 carry the three candidates.

**CPU parallelism has one seam and, at the mid-size tier on a 4-core host,
three negative results
([#344](https://github.com/michaelJwilson/snakes_and_ladders/issues/344)).**
`sal.parallel.map_tasks` runs a loop of independent tasks on a
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
the argument. The one positive number is inside a task, not across them: the
serial multi-start fit at the default thread count is 2.87× the fit at one
thread, because torch's intra-op parallelism over 1000 sites already uses the
cores, which is why the sites leave the count at the default rather than
pinning one thread per worker as the plan assumed.

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

**The leaf-count ladder is a cost axis, and it is now a fixture
([#582](https://github.com/michaelJwilson/snakes_and_ladders/issues/582)).**
The ticket asked where a ladder over leaf count stops; it does not. Neighbour
joining recovers the generating topology **exactly at 8, 20, 50, 100 and 200
leaves** --- normalized RF **0.0000**, zero variance over four seeds, 0.58 s at
200 --- so no rung fails and the terminating rule never fires. The axes that do
bite are sites (RF 0.2824 at 50 sites, 0.1529 at 100, 0.0235 at 200, 0.0000
from 500, all at 20 leaves) and tree shape, where the Felsenstein zone fails at
**four** leaves once pendant branches reach 0.8 (RF 0.4000 +/- 0.4899 over five
seeds --- bimodal, two failures, not noise).
`tests/regression/fixtures/tree_scale/` keeps the ladder for what it does
measure: 20 leaves per pull request, 50 under `-m stress`, 200 at release, each
declaring its topology as `{balanced: n, height: h}` rather than as yaml. On
its default rung the branch's post-order optimizations are **8.389 ms to 7.032
ms** per gradient and **4.035 s to 3.457 s** per search, at an identical
log-likelihood on identical evaluation and fit counts;
`docs/experiments/014-the-leaf-count-fixture.md` carries it.

**A fourth parallelism site is positive: one neighbourhood's candidate fits
([#405](https://github.com/michaelJwilson/snakes_and_ladders/issues/405)).**
`search.infer` takes `workers`, `backend` and `intra_op_threads` and fans the
candidates of a neighbourhood through the same seam. Unlike the three sites
above, a candidate fit is seconds rather than tenths, so the pool's spawn is
amortized: 58.87 s to **21.67 s at 50 taxa on 4 processes (2.72x)** and 14.94 s
to 9.92 s at 20 taxa (1.51x), both at 2,000 sites, each returning the serial
run's `log_likelihood` bitwise and its topology.
Which backend pays inverts with the regime, so neither argument gets a
parallel default: threads win 1.20x at 8 taxa by 20,000 sites, where the work
sits inside GIL-releasing `torch` kernels, and lose 1.99x at 20 taxa by 2,000
sites, where it is Python in the post-order; processes do the reverse, losing
1.63x in the first and winning in the second.
`docs/experiments/013-parallel-candidate-fits.md` carries the comparison.

Speedup is against the site's serial run at one intra-op thread; the serial
wall clock in the inventory includes the setup the loop does not carry. The
sites the plan lists after these — the candidate fits of `search.infer`,
`learn.rollout` batches, tempering replicas, `qa.build`, `check_notebooks` and
`pytest-xdist` — are measured on the same matrix before any is switched on
(#525).

## Milestone 1.1 — Simulation & Ground Truth Engine

**Modules.** The generators and the registry this milestone's ground truth comes from: `sim.jc`, `sim.gtr`, `sim.simulate`, `sim.simulator` (the one way to simulate, #829), `sim.tree`, `sim.newick`, `sim.params`, `sim.css`, `sim.emission_mixture`, `sim.count_pairs.rust` and `sim.fixtures`, which declares every instance the suite is checked on. `sim.galois`, `sim.reed_solomon`, `sim.elementary_codes` and `sim.capacity` are the algebraic codes and the capacity they are read against (#700); `sim.polar` is the polar construction, promoted from `sandbox` with its row (#826).

**Phylogenetics: landed.** A `k`-state Jukes-Cantor simulator generates an
alignment and the ancestral tree in Newick from a typed tree fixture, retaining
the parameters that produced them
([#58](https://github.com/michaelJwilson/snakes_and_ladders/pull/58)). Simulated
substitution frequencies are validated against the closed-form JC transition
probabilities within a yaml-declared Monte Carlo tolerance across several site
and taxon counts. Newick counting, validation and state-labelled serialization
are the package's single source of that functionality
([#64](https://github.com/michaelJwilson/snakes_and_ladders/pull/64)).

The general time-reversible model landed with the fitting work that needed it
([#120](https://github.com/michaelJwilson/snakes_and_ladders/pull/120)): Jukes-Cantor has no
free parameters. It is validated by reduction — equal exchangeabilities with a
uniform `π` reproduce the Jukes-Cantor rate matrix and its closed-form
transition probabilities to machine precision.

**Potts: 1-D chain plus a general N-D lattice/MRF simulator.** The 1-D chain
in an external field exists as an `opt` reference instance with an exact
transfer-matrix oracle ([#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115)),
and again as a `learn` environment. `sal.sim.graph.PottsGraph`
generalizes it to an arbitrary undirected graph with a per-edge coupling, and
`sal.sim.potts.simulate_potts` samples on it — exactly, by the
same backward-message recursion, when the graph is a 1-D open chain, and by
single-site Gibbs (heat-bath) MCMC otherwise — with an N-D lattice a
constructed case of the general graph rather than a second code path
([#190](https://github.com/michaelJwilson/snakes_and_ladders/pull/190), closing #170,
superseding the
sampling half of #149). `sal.opt.potts.simulate_chains` cannot import
`sal.sim` under `opt/CLAUDE.md`'s "no application imports" rule, so it
keeps its own copy of the exact recursion — a duplication
[#186](https://github.com/michaelJwilson/snakes_and_ladders/issues/186) tracks
resolving by moving `PottsParams` into `sal.sim.potts`. No
fitting, cluster updates, or evaluator on the general graph yet (issues #172,
#174).

**The external field is per site, and three instances need one.** `h` was one
row every site shared. It is now `(n_states,)` or `(n_nodes, n_states)`,
widened once by `sal.sim.potts.site_field` at each entry point,
and the exact open-chain recursion, the Gibbs sweep, `enumerate_potts` and
`strip_log_partition` index one shape. A per-site field whose rows are equal
reproduces the shared-field `log Z` and marginals to 0.0, and the strip matches
enumeration to 0.0 on a per-site field.

`spatio_only` declares the spatial half with the chains and the gated
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
and the counts were still moving, making the draw a coarsening front rather
than an equilibrium sample. The instance is declared at `J = 0.7` instead,
where the ten classes hold 415 to 593 sites each and the tilt above is stable.

**A lattice at the transition, declared rather than built twice.**
`potts_lattice/stress` is the 12x12 open square at the exact 3-state transition
in zero field. The coupling is resolved by
`sal.sim.potts.critical_coupling` from the file's own state
count rather than stored as a rounded float, so the instance cannot drift off
`J_c`. `tests/regression/search/test_potts_mcmc.py` and section 9 of
`docs/nb/potts_chain.ipynb` each built that lattice for themselves; both now
read the file, and the notebook re-executes to the same energy autocorrelation
times it printed before — 6.70, 4.17 and 2.34 site updates for single-site,
Swendsen-Wang and Wolff. The values are pinned beside the ordering, at the
file's 10% relative tolerance. A duplication guard keeps `ln(1 + sqrt(q))` in
one place and found two further copies on its first run
([#413](https://github.com/michaelJwilson/snakes_and_ladders/issues/413)).

**HMMs: a first-class simulator.** `sal.sim.hmm` draws a hidden state path
and an observation sequence jointly from a declared `(pi, A, B)`, retaining
the path alongside the data on the footing the tree simulator already has
([#182](https://github.com/michaelJwilson/snakes_and_ladders/pull/182), closing
[#171](https://github.com/michaelJwilson/snakes_and_ladders/issues/171)). The generator
embedded in `sal.opt.hmm`
([#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115)) is deleted; `opt`
now imports the truth type from `sim` and draws no data itself. Validated
against brute-force enumeration for the per-position state and emission
marginals, self-normalized importance sampling against the exact path
posterior for one realized observation, and the transition matrix's own
stationary distribution for long-run occupancy.

**Emission families: what a state emits, separated from how it is fitted.**
`sal.emissions` holds the interface — draw, score, re-estimate —
with the categorical matrix one implementation of it and a univariate Gaussian
the second; the simulator, the forward recursion, Baum-Welch, path enumeration
and the state aligner all go through it
([#228](https://github.com/michaelJwilson/snakes_and_ladders/issues/228)). Every
categorical test in the suite passes against the family without its assertions
being rewritten.

The Gaussian case is where the discrete assumption stops holding, and both
consequences are pinned. The evidence is a *density*, so `log P(observations)
<= 0` fails on correct code and no test asserts it. And the likelihood has **no
maximum**: a state's mean on one observation with its variance going to zero
diverges, at exactly `log 10` nats per tenfold narrowing. The variance floor is
derived from the data — `s**2 / n**2`, the nearest-neighbour spacing below
which a state has collapsed onto a point — and reaching it is a **refusal**,
since a clamped fit returns normally and reports an interval around a
degenerate optimum
([#122](https://github.com/michaelJwilson/snakes_and_ladders/issues/122)).

**The identifiable regime is measured, not assumed.** Coverage of the 95% Wald
intervals over 24 replicates of 240 observations, against the separation of the
emitting means:

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
too ill-conditioned to invert. Those replicates are counted rather than
dropped, since excluding them unannounced would select for the well-behaved
samples.

**Count emissions condition on a per-observation covariate.** The negative
binomial takes an exposure and the beta-binomial a trial count, conditioned on
and never fitted
([#631](https://github.com/michaelJwilson/snakes_and_ladders/issues/631)). The
rate is `e_i mu_k`, so `mu_k` stays the state's own association and the moments
keep a value under a varying covariate. The pre-covariate families are
conserved in `sandbox.count_emissions`, hash-frozen over 27 definitions, and
referee the new ones by an identity rather than a tolerance: at a constant
covariate the two are the same model, absorbing it as `log mu - log(c)`, and
scoring agrees **bitwise**. Recovery carries the regime the conserved families
cannot express --- planted `mu = 2.5, r = 3.0` fits 2.52, 2.99 over an
eightfold exposure spread, and `a = 3.0, b = 7.0` fits 2.96, 7.03 over a
sixteenfold trial-count spread --- and withholding the exposure recovers the
*average* rate instead, reading the spread it would have explained as
overdispersion at `r = 1.73`. Two measurements corrected the ticket's own
runtime analysis: the rate hoisted out of the dispersion bisection is 1.2 ms
against `digamma`'s 97.8 ms in a 270 ms solve, under half a percent and left
alone per the profile-first rule; and the profiled mean written as a
matrix-vector product is **13.72x** at the declared scale, 54.9 ms to 4.0 ms,
dropping an 80.7 MB intermediate for a relative 2.7e-16 --- five orders inside
the declared 1e-11, which is the first use of `CLAUDE.md`'s bitwise-to-tolerance
rule ([#649](https://github.com/michaelJwilson/snakes_and_ladders/issues/649)).

**Count emissions: the dispersion axis, bracketed.** Four count families join
the seam — binomial below equidispersion, Poisson exactly at it, negative
binomial and beta-binomial above
([#229](https://github.com/michaelJwilson/snakes_and_ladders/issues/229),
[#260](https://github.com/michaelJwilson/snakes_and_ladders/issues/260)). An
interface exercised only by overdispersed families has never been asked whether
it assumes overdispersion somewhere. Each referees the neighbour it is a limit
of: `BetaBinomial(n, 1, 1)` is the discrete uniform and `Binomial(1, p)` is
Bernoulli, both to 1e-14 or better; the negative binomial approaches Poisson as
`r -> inf` and the beta-binomial approaches the binomial as `a + b -> inf`,
both at the `O(1/x)` rate the truncation predicts, with `r` times the deviation
measured at 18.75, 18.88, 18.94, 18.97 and 18.98 across `r` from 500 to 8000 —
converging rather than drifting, which is what makes the extrapolated tolerance
legitimate.

**Two of the four have an M step that is an optimization**, and `reestimate`
returns what its M step did, not only what it produced. **Both solves bracket
rather than step**, by measurement. Newton on the negative binomial's
weighted score converges for moderate `r` and, on a near-Poisson sample,
overshoots in `log r` and underflows to zero; safeguarded bisection reaches
`|score| / weight` of 1e-14 to 1e-16 in 45 evaluations and agrees with a
40001-point grid search to zero relative difference. Minka's fixed point for
the beta-binomial is monotone but linearly convergent: at a true concentration
of 120 it was still moving in the third decimal after 500 iterations.
Alternating bisection in `(p, a + b)` settles in 3 to 9 iterations at a
residual of exactly zero. An M step that does not settle is refused, per
`likelihood/CLAUDE.md`.

**The beta-binomial solve runs every component at once, on distinct values**
(#892, PR #917). Within one solve the weights are fixed, so each score's sum
over the observations is a sum over each channel's distinct values weighted by
the responsibility summed there, and one alternating bisection runs over every
component in lockstep. One EM iteration on the `spatio_sequential_counts/release`
projection (K = 100, 4,000 pairs) drops from 5.9 s to 0.65 s and on
`emission_mixture/stress` from 0.61 s to 0.124 s, one thread; the fitted
parameters are bitwise the per-component solve's, which stays as the oracle.
Batching over the observations alone bought 1.18x: the cost was `digamma`
evaluations, not calls.

**The negative-binomial dispersion solve takes the same cut** (#918). One
lockstep bisection on `log r` over every state, on the distinct counts: the
dispersion solve at the release projection's size goes from 324 ms to 24.8 ms
(13.1x, `test_negative_binomial_batched_bench.py`, one thread), the joint M
step from 543 ms to 235 ms (2.3x), and one `emission_mixture/stress` EM
iteration from 112 ms to 96 ms, where the beta-binomial solve is now 84 ms of
it (#925). Against the per-state solve, kept as the oracle: bitwise on
`emission_mixture/ci`, and on `/stress` 7 of 10 states bitwise and the rest
within a relative 5.9e-13 of #648's 2e-06 floor.

**Both count solves are compiled** (#922). `src/count_mstep.rs` runs each
state's bisection in Rust and evaluates every `sum_u w_u (digamma(u + x) -
digamma(x))` as `sum_j T_j / (x + j)` over the tails of the weights, an
identity for integer counts that needs no special function. One thread, the
kernel against the batched torch solves: the beta-binomial solve 13.5x on
`emission_mixture/stress` (87.3 to 6.5 ms) and 19.5x on the release
projection (209.9 to 10.8 ms), the dispersion solve 9.5x and 3.9x; the joint
M step at the release projection 235 to 19.7 ms, and one stress EM iteration
96 to 12.7 ms. The beta-binomial results are bitwise the oracle's, since a
bisection's answer is set by its scores' signs; the dispersion within 1.2e-12
relative. `emission_mixture_starts` executes in 56 s, from 277 to 302 s, its
compared text unchanged; the E step is now 41% of an iteration (#924).

**The flat-likelihood hazard is the mirror of the Gaussian's.** Where a
Gaussian likelihood is *unbounded* as a variance falls, a count likelihood goes
*flat* as the dispersion rises toward its Poisson or binomial limit. The bound
is derived on the same construction for both count families — the dispersion at
which the overdispersion the model is for falls below the sampling noise on
measuring it, `mu sqrt(W/2)` for the negative binomial and `(n-1) sqrt(W/2)`
for the beta-binomial. Coverage of the 95% intervals over 16 replicates of 480
counts, against the true dispersion:

| true `r` | intervals covering | rate | replicates with no interval at all |
| --- | --- | --- | --- |
| 0.5 | 44/48 | 0.917 | 4/16 |
| 1.0 | 60/64 | 0.938 | 0/16 |
| 2.0 | 58/64 | 0.906 | 0/16 |
| 5.0 | 59/64 | 0.922 | 0/16 |
| 20.0 | 58/64 | 0.906 | 0/16 |
| 100.0 | 32/36 | 0.889 | 7/16 |

**The finding is not the coverage column.** Coverage sits near nominal at every
dispersion; what degrades is *whether an interval exists*, at **both** ends — a
heavy tail at `r = 0.5`, a flat likelihood at `r = 100`. A non-identified count
model announces itself as a singular information matrix, not as an interval in
the wrong place — the opposite of what the Gaussian case showed, and why both
were measured.

**A Gaussian mixture: the emission seam with the Markov chain removed.**
`sal.sim.mixture` draws component labels and observations
jointly, retaining the label so a clustering has something to be checked
against; `sal.opt.mixture` fits
([#262](https://github.com/michaelJwilson/snakes_and_ladders/issues/262)). Its
component M step **is** `GaussianEmission.reestimate`, called with
responsibilities where an HMM passes state posteriors, and a test asserts the
two produce identical numbers on the same responsibilities. The unbounded
likelihood transfers unchanged: a component collapsed onto a single
observation is refused, not clamped.

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

**The quantum code the bicycle matrices open, decoded on the coset**
([#362](https://github.com/michaelJwilson/snakes_and_ladders/issues/362)).
`Hx = Hz = H` is a Calderbank--Shor--Steane code whenever `H H^T = 0`, which
the bicycle construction satisfies because `A` and `A^T` are polynomials in the
same cyclic shift. The classical half measured that condition without using it;
this half builds the code, and with it the criterion degeneracy forces: two
errors differing by a stabilizer act identically on the codespace, so a decode
succeeds when `(e + e_hat)` is in `rowspace(H)` and not when `e_hat == e`.

**Degeneracy is worth 3.4%, and belief propagation gives back far more than
that.** At `n = 16`, `k = 2`, `p = 0.05`: exact degenerate maximum likelihood
**0.070297** against the likeliest single error's 0.072735, differing at **24**
of 128 syndromes with no tie broken. Sum-product reaches 0.178750 +- 0.005788
over three seeds of 4,000 trials --- **2.54x the floor** --- and the code's
defining property is why. `H H^T = 0` forces even row overlaps; an overlap of
exactly two is a four-cycle; the graph carries **30** of them at `n = 16` and
**160** at `n = 96`. A decoder that assumes a tree is being run on a graph the
CSS condition fills with short cycles.

**The two failures are counted apart because they invert with size.** A single
block-error number would hide it: of 2,145 failures at `n = 16`, 1,742 did not
converge and 403 converged on a logical coset; at `n = 96` it is 1,323 against
2,128. The first returns no usable correction, the second returns one that
damages the encoded state. The quotient is pinned by its own partition --- every
reachable syndrome carries exactly `2^k` cosets ---
and `docs/experiments/015-css-decoding-under-degeneracy.md` carries the
comparison.

## Milestone 1.2 — Differentiable Likelihood & Energy Engine

**Modules.** The evaluators and the oracle they are pinned to: `likelihood.brute_force`, `likelihood.parsimony`, `likelihood.algebraic`, `likelihood.polar` (successive cancellation and the list, #826), `likelihood.css`, `likelihood.mixture_assignments`, `likelihood.schedule`, `likelihood.spatio_sequential.rust`, and `likelihood.device`, which owns the cross-device tolerance this milestone's claims are stated against. `likelihood.ragged` is the gateway of the ragged path and its oracle, and `likelihood.ragged.rust` the compiled kernel behind it, conserved beside `sandbox.rectangular_hmm` (#666).

**Polar codes are a row, and the CRC-aided list is built and measured**
([#826](https://github.com/michaelJwilson/snakes_and_ladders/issues/826)).
`sandbox.polar` and `sandbox.polar_decoding` --- 711 lines conserved since #593
--- are `sim.polar` and `likelihood.polar`, moved bitwise with their 22 tests,
and the `polar` fixture joins the registry at three tiers with its row in
`PROBLEMS.md`, so the applicability tables carry it by generation;
`sandbox.polar_reference` stays as the oracle that pins successive
cancellation bitwise. `decode_scl(..., crc=)` returns the best survivor whose
check passes and the best metric when none does (Tal & Vardy 2015): at
`N = 16` with a three-bit check and the exhaustive list it is the
maximum-likelihood codeword of the enumerated 32-word outer code on every one
of 12 draws, and at `N = 64`, rate 1/2, CRC-8 and `L = 8` it decodes more of
120 shared blocks than the plain list. **At `N = 256` it buys nothing yet**:
over 200 shared blocks at `sigma` from 1.0 to 0.6 (0 to 4.4 dB) the CRC-aided
and plain block error rates are equal at `L = 8` (1.000, 0.995, 0.925, 0.570,
0.090) and at `L = 32` (0.805 and 0.395 at 0.8 and 0.7), because every list
failure is the sent path pruned --- in 1,000 decodes the sent word was never in
the final list below the top --- and the check can only re-rank what survives.
The gain the literature reports at `N = 2048`, `L = 32` waits on a list the
pruning does not reach first, which is a longer list or a better construction,
and the measurement says which is missing here.

**A covariate reaches the two-channel family, every seam above it, and the
compiled backend** ([#660](https://github.com/michaelJwilson/snakes_and_ladders/pull/660)).
The pair families take one covariate per channel, `(..., 2)` splitting where
the observation splits; the neutral covariate is ones for one channel and the
declared trials for the other, since they condition on different kinds of
quantity. `external_field` and `marginal_log_likelihood_torch` scored without
one — the first a defect, since the fit handed it a posterior computed *with*
the covariate. `baum_welch_family` unpacked the whole observation shape and so
refused every family carrying axes of its own, which a downstream consumer
measured on this branch and reported.

**The compiled backend conditions, and the first implementation of it was
slower than its own oracle.** The kernel indexes a table by a row, not a
count, so the covariate is caller-side. Tabulating the *distinct*
`(count, covariate)` pairs meant one `np.unique` over an `(S·V, 2)` array per
call: **135.6 ms of a 141.4 ms E step**, putting the backend at **0.6x** the
NumPy oracle it exists to beat. Factorizing the covariate alone and addressing
the outer product arithmetically — a larger table, built vectorized and never
sorted — gives **12.4 ms, 6.5x** the oracle, against **37.2x** uncovaried.
Fewer rows was the wrong thing to optimize. Agreement with the oracle under a
covariate: 2.7e-15 relative on the evidence, 2.9e-15 on the field, one thread.

**`FlowNetwork` keeps the contiguous form it built**
([#659](https://github.com/michaelJwilson/snakes_and_ladders/pull/659)).
`from_arcs` assembled the compiled consumer's arrays on its way to the list
store and discarded them, so a network built from arcs and solved in Rust made
the round trip NumPy → list → NumPy for nothing. `as_arrays` is **289 ns**
against **2.87 ms** at 20,000 edges; one alpha-expansion sweep of four labels
is **1.240x** at 16x16 and **1.112x** at 32x32 through the Rust backend, with
the Python backend — which never calls `as_arrays` — the control at 1.000x.
The arrays are bit-identical kept or derived, and both writers drop the form.
The list store stays on #586's measurement (1.95 ms as lists against 3.93 ms
with offsets). The survey's per-call finding no longer fires against a
constructor, which runs once per instance and so has no second payment to
remove; a paired control pins that an accessor over a non-compressed store is
still reported.

**A chain's transition kernel may be a function of position**
([#654](https://github.com/michaelJwilson/snakes_and_ladders/pull/654)).
`forward_backward`, `sample_path` and `forward_log_likelihood_from_density`
take either one `(K, K)` matrix for the whole chain or `(T - 1, K, K)`, one per
transition, so a spacing between sites or a rate that varies along the sequence
has a signature to arrive through. The constant form is a stride-zero view and
the choice is hoisted out of the recursion, so it sums the same terms in the
same order: the single-matrix result is reproduced **bitwise** in evidence,
posterior, pairwise and sampled path, and the constant path costs 0.990x of the
recursion that preceded it at `T = 100,000`, `K = 8`. The varying form costs
1.044x there and 512 B against 48.83 MiB in the kernel argument, which is why
both shapes stay. The new path is pinned against a path enumeration that shares
no recursion with it, and a planted two-regime kernel is recovered per regime
from the pairwise posteriors where the pooled estimate matches neither.
**A count pair can be drawn under a covariate, and fitted under one**
([#670](https://github.com/michaelJwilson/snakes_and_ladders/issues/670),
[#671](https://github.com/michaelJwilson/snakes_and_ladders/issues/671),
[#672](https://github.com/michaelJwilson/snakes_and_ladders/issues/672)).
Three seams carried one defect: a covariate that already holds the family's
channel axis had the broadcast singleton appended anyway, so its last axis read
1, named no channel, and the pair family refused it. `m_step` re-derived
`covariate_block`'s slice rather than calling it; `_drawing_covariate` did the
same and the coupled simulator reshaped every draw to `(S, V)`, so a pair could
not be drawn at all; and both count-pair simulators passed no covariate, so the
one problem class whose emission *is* a pair had no covaried planted instance.
The referee is recovery. Over an exposure spanning a factor of 16, a coupled
fit **told** it recovers planted rates of 24.0 and 72.0 as **23.3 and 71.3**,
within 3%; the same fit on the same data **not told** it reaches **47.9 and
151.7**, factors of 1.99 and 2.11 against an `E[U(0.25, 4)]` of 2.125 --- the
rate averaged over the exposures. In Rust the exposure scales the drawn gamma
rather than the distribution, `c Gamma(r, mu/r)` being `Gamma(r, c mu/r)`
exactly, so a covariate costs one multiply per draw and the per-(class, state)
objects stay. A fixture declaring no covariate draws what it drew before,
asserted rather than assumed.

**A covariate reaches the families from a fit, not only from a test**
([#657](https://github.com/michaelJwilson/snakes_and_ladders/pull/657)).
#631's exposure and trial count were reachable only by a test that built a
family directly. Three seams now thread one: the spatial params carry a
`(S, n_nodes)` covariate that `gated_log_density`, `class_log_density` and the
spatial M step each slice the way they slice the observations;
`baum_welch_family` takes a `(n_sequences, length)` one and reaches both the E
step's scoring and the emission M step; and the HMM objectives store one beside
their observations. The end-to-end referee is #631's recovery raised from the
family to the fit: over a `U(0.25, 4)` exposure whose draw spans 15.8x, a
negative-binomial Baum-Welch **told** the exposure recovers a planted 2.0 and
9.0 as **1.9688 and 8.8088**, and the same fit on the same data **not told** it
converges to **4.1505 and 18.5904** — the rate averaged over the exposures,
factors of 2.075 and 2.066 against an `E[U(0.25, 4)]` of 2.125. A constant
exposure of ones scores bitwise and moves a 200-iteration fit 1.5e-13 at the
single thread `tests/conftest.py` pins, two orders inside the declared 1e-11;
the same fit reads 5.1e-14 at two threads and 9.9e-14 at four, so the figure is
stated with the thread count it was taken at.

**Felsenstein pruning: three CPU backends, one oracle.** Vectorized NumPy is
the reference, with per-node rescaling accumulated in log space
([#66](https://github.com/michaelJwilson/snakes_and_ladders/pull/66)); differentiable
PyTorch takes branch lengths as a tensor separate from the topology
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
`search.infer`'s topology, trace, evaluations and fits unchanged. The ordering
is the same at 4 taxa — 5.68 ms (0.74) against the taped 9.82 (1.11) — and
against central differences swept over steps 1e-4 to 1e-7 all three routes
deviate by the same amount to four significant figures, worst 9.5e-4 at
h = 1e-4 and 1.007e-6 at h = 1e-6, so the deviation is the difference
quotient's and not any gradient's.

A `burn` `Autodiff<NdArray<f64>>` port of the same recursion was measured
beside it and **declined**: 46.80 ms (3.86) per gradient and 1179.1 (144.1) per
fit, with a tape of 66, 134 and 270 nodes at 4, 8 and 16 taxa — larger than the
tape it was meant to replace. The boundary is not the reason: the kernel alone
is 48.01 ms [46.94, 49.27] by Criterion against 47.06 (6.09) for the same call
from Python, so the crossing is inside the spread and the kernel by itself
already costs 1.45x PyTorch's whole evaluation. Its `f64` path was sound —
5.9e-13 against the taped `float64` gradient.
`docs/experiments/007-pruning-gradient-routes.md` carries the question, the
three routes' numbers and the prediction they were taken to test.
**Switching the fitted path to the closed form, which 007 left as
[#443](https://github.com/michaelJwilson/snakes_and_ladders/issues/443), is
not taken.** `likelihood.objective.BranchLengthObjective` now takes
`gradient=`, so the two routes are comparable through a fit and not only at
`log_likelihood`, and they agree at **5.4e-16** relative there and reach the
same optimum. The default stays the tape, because the ratio between them is
not a function of the problem's size: at one tree, one alignment and 20,000
sites, replacing the fixture's branch lengths with 0.1 moves it from
**2.11x** to **0.57x** — the routes swap places. Flushing denormals to zero
leaves the swing intact (2.11 to 1.70, 0.57 to 0.74), so that is not the
mechanism and the cause is unidentified; `docs/experiments/012-pruning-gradient-routes-after-the-post-order.md`
carries it, and it is what #443 now asks. The route is
conserved as `sal.sandbox.pruning_burn` over
`src/pruning_burn.rs` behind the `sandbox` Cargo feature; the default build,
the wheel and every per-pull-request job link no `burn`, and
`infra/release.sh` compiles the feature.

**Nine PyTorch optimization patterns were itemized and all nine declined**
([#544](https://github.com/michaelJwilson/snakes_and_ladders/issues/544)).
#528 found no second portable candidate and left the remaining cost inside
`torch`; these are the patterns that cost names, measured against the paths in
the tree. `cProfile` self time over one warm run each, 8 taxa by 2,000 sites,
one BLAS thread, `float64`: a JC fit spends **53.7%** in
`pruning_torch._post_order` and **27.0%** in `run_backward`; a GTR fit **43.6%**
in `run_backward`, **20.9%** in `_post_order`, **7.1%** in `LBFGS.step` and
**2.6%** in `matrix_exp`; a 30-candidate NNI climb **44.1%** and **20.5%** in
the same two, and with `lazy_top=3` **43.9%** and **20.3%** — the fits, not the
gradient-free scoring, which does not reach the top ten. Both terms of each
ratio come from one profiled run, so a shared load moves neither.

1. **`inference_mode` on evaluation-only paths.** Every such path already runs
   under `no_grad` — **20 blocks** outside the sandbox, across
   `likelihood.pruning.torch`, `learn.policy`, `planning`, `critic`, `ppo`,
   `actor_critic`, `surrogate`, `relaxed`, `search.rl` and `search.max_cut`.
   The residue `inference_mode` would remove is the version counter and view
   tracking. One gradient-free evaluation at 8
   taxa by 20,000 sites: **12.11, 19.04 and 18.99 ms under `no_grad` against
   12.26, 12.23 and 18.57 under `inference_mode`**, three readings, and the
   spread inside either arm is larger than the gap between them. Several of the
   14 could not take it in any case: PPO's stored log-probabilities re-enter an
   autograd graph, which an inference tensor may not.
2. **Batching independent fits, and `torch.vmap`.** `vmap` does reach the
   objective at a fixed topology and reproduces the sequential values, but
   nothing consumes a batched one: L-BFGS' strong-Wolfe line search branches on
   each start's own values, so `B` starts need `B` line searches. Across
   topologies it does not apply at all — the post-order is a Python recursion
   over a different tree per candidate. Four starts at 8 taxa by 20,000 sites
   measured 195.1 to 236.5 ms sequential against 163.9 to 366.7 batched, three
   readings, which resolves nothing and does not need to.
3. **Host–device and Python syncs.** There is no host–device boundary: CUDA and
   Metal are unimplemented (#280) and `device.available_device()` returns `cpu`,
   where a scalar read is a read and not a synchronisation. The package calls
   `.item()` **zero** times. What remains is 2 `float()` per L-BFGS outer
   iteration in `opt.fit` — one extra forward and backward for the relative
   convergence test, against 20 inner iterations — and one `float()` per child
   per node inside `log_likelihood_cached`'s cache key. Neither appears in any
   profile above.
4. **Preallocation and `out=`.** Forbidden rather than unused on the path that
   pays: `pruning_torch.log_likelihood` runs under a tape, so every intermediate
   is held for the backward and an in-place write would corrupt it. What is left
   is hoistable allocation — `torch.arange` per leaf, `torch.ones_like` per
   node for the rescaling `where` — which the profiles put at 1.4 to 1.9% and
   1.2 to 1.7%. Hoisted, the recursion returns the value and the gradient
   **bitwise**, at **1.02x** at 2,000 sites and **0.94x** at 20,000: no gain,
   and the sign turns with the size.
5. **Contiguity and stride order.** `partial @ transitions[i].T` passes BLAS a
   transposed view, which `gemm` takes as a flag. Pre-transposing to a
   contiguous copy is **slower** — 65.49, 68.30 and 65.60 µs against the view's
   64.25, 67.15 and 64.48, three readings — and the product is bitwise
   unchanged, so the copy buys nothing and costs itself.
6. **`torch.matrix_exp` against the closed form.** The Jukes–Cantor path already
   takes the closed form, and the measurement says to keep it: `matrix_exp` is
   **2.57 to 2.73x** the closed form over the same branch vector. For GTR, one
   eigendecomposition of the reversible `Q` with a vector exponential per branch
   is **0.84 to 0.88x** `matrix_exp` and agrees with it to **3.3e-16**, which is
   15% of the **2.6%** a GTR fit spends there — **0.4%** of the fit, below the
   bar #341 sets.
7. **Fused and `foreach` optimizer variants.** The continuous fits are
   `torch.optim.LBFGS`, which has neither parameter; only `Adam` does, and
   `learn/`'s eight `Adam` sites each hold one tensor or a small module, where
   `foreach` has nothing to fuse over. The ceiling was already #457's 6.63% and
   3.57%, and this profile's 1.2 to 7.1%.
8. **`float32` is a correctness decision, not a performance one.** The
   cross-device tolerance is relative and keyed on the lowest precision in the
   comparison, so adopting `float32` moves `CROSS_DEVICE_RTOL` from 1e-11 to
   1e-6 for every comparison it touches and costs five digits of every pinned
   value. It is not taken as a speed-up, and is not measured as one.
9. **CUDA and MPS** belong to
   [#280](https://github.com/michaelJwilson/snakes_and_ladders/issues/280) and
   are blocked on a device: this host has neither.

`tests/regression/likelihood/test_torch_patterns.py` pins what each decline
rests on — that the alternative computes the same thing — and
`tests/benchmarks/test_torch_patterns_bench.py` times the arms.
**The host was not quiet**, carrying three other agents' work throughout, so
every absolute above is an upper bound and none is comparable to the timings
recorded elsewhere in this file. What the readings support is the three ratios
that repeat across all three of them, all under a millisecond per arm, and the
profile shares, which are internal to one run.

**`pruning_analytic` is measured and not wired in**
([#453](https://github.com/michaelJwilson/snakes_and_ladders/issues/453)).
`likelihood.objective` calls `pruning_torch.log_likelihood` on both objectives,
and nothing under `python/` imports `pruning_analytic`, so the 694.5 → 454.2 ms
recorded above is available and unclaimed on `BranchLengthObjective`.
`SubstitutionModelObjective` cannot take it — it differentiates the rate matrix,
which that backward refuses by design. Adopting it is its own change with its
own measurement, filed rather than taken here.

**The Rust backend returned nothing at the declared scale, and now returns
2.5x.** Measured against the NumPy oracle end to end, it was **1.8x** at 10
taxa by 1,000 sites and **1.00x** at 200 by 11,000 — the top of the range
`ROADMAP.md` §1.2 declares. Two causes, both invisible in the benchmark cells
that existed at 4 and 8 taxa. The binding took nested Python lists, so PyO3
built one integer object per observed state: 2.2 million of them at the top of
the range, a cost growing with `n x L` while the kernel's advantage does not.
And the recursion held every node's partial likelihood for the whole
computation, 140 MB where the oracle needs 21. The alignment now crosses as one
borrowed array and a partial is released when its parent has consumed it,
giving **1.7x** at 20 taxa by 11,000 sites and **2.5x** at 200, with peak
memory down to 22.6 MB. The backend is still under the 3x bar a CPU port is
held to: the remaining cost is the kernel, which loops over states scalar-wise
where the oracle reaches BLAS, and the site-parallel recursion is the
data-parallel case the roadmap sends to the GPU.

**The memory requirement is settled.** The simulator retains every node's
states, `(2n - 1) x L x 8` bytes, and pruning retains one partial likelihood
per open node, `(2n - 2) x L x k x 8` on a caterpillar — the deepest tree on
its leaves, so the worst case, and the topology at which `O(n x L x k)` is
tight rather than loose. At 100 taxa by 11,000 sites that is **87.2 MB**, and
at the declared maximum of 1,000 by 11,000 it is **879 MB**: a factor of **20**
inside the 16 GB requirement. A balanced tree of the same size costs strictly
less, which makes the figure a bound.

Both terms are published as arithmetic over the arrays' shapes rather than as a
sampled peak: a `tracemalloc` figure does not survive a change of machine, and
an earlier draft published one that CI regenerated differently. The regression
suite pins every published cell against the allocator's own count: realized
agreement **2.5%** at the smallest cell and **0.3%** at the two larger ones,
against a 5% band.

**Device dispatch: declared, CPU-only.** Selection prefers CUDA, then
Metal/MPS, then CPU, and the cross-device tolerance is stated where the
roadmap promised it — relative, and keyed on the lowest precision in the
comparison: `1e-11` with `float64` on both sides, `1e-6` where either side is
`float32`, since Metal cannot do `float64`
([#112](https://github.com/michaelJwilson/snakes_and_ladders/pull/112)). Both figures are
derived from measured agreement, and the `float32` bound is exercised on CPU so
runners without an accelerator still check it. The CUDA and Metal paths are not
implemented.

**Parsimony landed, and it is here to be wrong.** Fitch's algorithm scores a
topology beside the likelihood, pinned against exhaustive enumeration over
internal-node labellings — equality, not a tolerance, since the score is an
integer. In the Felsenstein zone, four taxa with two long branches placed non-adjacently
make parsimony *statistically inconsistent*: over 12 replicates parsimony
recovered the true topology **0 of 12
times at 200, 1000 and 5000 sites** while likelihood went 10/12, 12/12, 12/12.

The Farris zone is the control that makes that interpretable: move the same
two long branches to be adjacent and parsimony is right **12 of 12 at every
site count**, while likelihood needs more data — 4/12, 6/12, 10/12. An implementation simply
broken would fail both zones.

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
asymmetric matrix scores each rooting differently.

**Belief propagation is now measured over an ensemble, not three fixtures.**
`sal.sim.graph.erdos_renyi_graph` draws `G(n, p)` beside
`lattice_graph`, and BP is checked per draw against enumeration. Over 60 sparse
draws, 106 across two ensembles were acyclic and BP was exact on every one —
worst relative deviation 3.7e-15 in `log Z` and 4.9e-13 in the marginals,
inside the `1e-11` float64 bound. The ensemble also reaches what no committed
fixture did: **104 of 120 draws carried an isolated vertex**, the boundary
between the general message-passing loop and the edgeless special case.

**A correction to what was expected.** #214 proposed asserting that the
deviation on a cyclic draw sits well below the lattice's, on the
locally-tree-like argument. It does not, at this scale: measured over 14
cyclic draws the relative deviation ran 2.7e-04 to 7.4e-03, median 3.6e-03,
against the lattice's peak of 5.2e-03 — comparable, not better. At `n <= 10` a
single cycle is a large fraction of the graph, and the locally-tree-like
argument is asymptotic. Nothing claims BP is more accurate on a random graph
than on a lattice at these sizes.

**Three canonical fixtures, each consumed by more than one module.**
`sal.sim.canonical` holds instances whose answer comes from outside this
repository, admitted on two clauses stated in `sim/CLAUDE.md`: the answer must
be independently known, and more than one module must consume it
([#209](https://github.com/michaelJwilson/snakes_and_ladders/issues/209)).

| Fixture | Known from | Consumed by |
| --- | --- | --- |
| Triangular Ising antiferromagnet | a double count that closes exactly: `N` of `3N` edges agree in any ground state, at every size | `sim` builds, `likelihood.potts` enumerates, `search.max_cut` optimizes, `sample.potts_mcmc` refuses |
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
degeneracies (42, 68, 42) are asserted instead, the discipline
[#214](https://github.com/michaelJwilson/snakes_and_ladders/pull/214) arrived at for graph
threshold results.

**The planted spin glass does not replace #177, and this is measured rather
than assumed.** It was proposed as the instance whose difficulty scales, after
[#198](https://github.com/michaelJwilson/snakes_and_ladders/pull/198) measured
random-restart greedy solving #177's tree at 1.000. Against 20-restart iterated
conditional modes at `n = 100` and mean degree 4, descent lands on the planted
energy below frustration 0.2 (mean gap +0.30 at 0.00, -0.12 at 0.05, +0.12 at
0.10, -0.25 at 0.15) and beats it above (-8.2 at 0.20, -25.7 at 0.30), where
the planted state is no longer near-optimal. Raising connectivity does not open
a window: at mean degree 12 and frustration 0.05 descent matches the planted
energy exactly on every instance. The fixture supplies a **known-energy
reference past the size enumeration reaches**. The search for an instance no
baseline solves stays open (#406).

**A problem no baseline solves, and the bar it is read against**
([#406](https://github.com/michaelJwilson/snakes_and_ladders/issues/406)). A
baseline **solves** an instance when it reaches the optimum from **0.4 or
more** of starts; a candidate qualifies below that, with the optimum from
enumeration and not from the search under test. Twenty-seven candidates were
measured at 50 seeded starts over 16 seeds, each with its 95% interval.
Fourteen came in under the bar; the lowest is declared as `planted_glass/ci`, a
planted Viana-Bray glass at 18 sites and mean degree 4 with frustration 0.30,
where single-site descent reaches the enumerated ground state of -14.0 on
**0.079** of starts, interval (0.056, 0.101). Its planted state scores -7.0, so
the oracle is the enumeration of all 262,144 configurations.
`tests/regression/search/test_search_hard_glass.py` pins it and
`infra/baselines.py` recomputes it at the release gate.

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

Two things the table does not say on its own. The instance matters more than
the knobs: at 18 sites and frustration 0.30 one graph seed gives 0.079
and the other 0.539, so a fixture is a *declared instance* and never a
recipe. The headroom a policy could demonstrate is 1 - 0.079 = 0.921, and at
the per-seed standard deviation of 0.014 the tree comparison measured, an
exact two-sided sign test needs **6 paired seeds** to call a difference that
size, so the fixture is not what would limit the comparison. And random-restart
descent reaches the ground state on **every** seed at the declared 50
restarts, exactly as #198 found on the tree: the headroom is against a single
run, and the restart baseline stays unbeaten. The Felsenstein-zone tree is the
only candidate measured where restarts also fail (0.938 of seeds).

**Viterbi and posterior decoding can now be told apart.** Neither decoder is
implemented, but the fixture that separates them is: on `ambiguous_hmm` the
Viterbi path is `(0,0,0,0,0)` — unique, 0.3033 nats clear of the runner-up —
while posterior decoding returns `(0,1,0,1,0)`, the observations themselves,
with every marginal above 0.6256. That posterior sequence is the **5th** most
likely path of 32, 0.6066 nats behind the Viterbi path. 

**Potts and HMM evaluators: partial.** The 1-D transfer matrix and the HMM
forward recursion exist, each with its exact oracle. Sum-product belief
propagation over a general `PottsGraph` joins them, with the 2-D strip transfer
matrix as the oracle past enumeration
([#206](https://github.com/michaelJwilson/snakes_and_ladders/pull/206)).

**What belief propagation is claimed to do, and what it is not.** On a tree it
is exact, and that is where the correctness claim sits: `log Z` agrees
with exhaustive enumeration to 2.0e-15 relative and the single-site marginals
to 2.8e-13, inside `likelihood/CLAUDE.md`'s `1e-11` `float64` bound. On a
loopy lattice it is approximate, so nothing asserts agreement — the deviation
from the exact strip transfer matrix is reported as a measurement, and it is
1.7e-15 at zero coupling, 1.1e-03 at `J = 0.5`, and peaks at 5.2e-03 at
`J = 0.875` on a 6x4 open strip in three states. The exact `q`-state
transition on a square lattice is at `J_c = ln(1 + sqrt(q)) = 1.005` for
`q = 3`, so the Bethe approximation is worst where the correlations it neglects
are longest-ranged, and it recovers on both sides. Messages that do not settle raise.

**The three evaluators are one algorithm on one structure**
([#290](https://github.com/michaelJwilson/snakes_and_ladders/issues/290),
part 1). `sim.factor_graph` holds variables, factors as log tables, adapters
from a tree, a Potts graph, a hidden Markov chain and the coupled
spatio-sequential model, and the Forney normal form; `likelihood.message_passing`
runs sum-product and max-product over it with a tree schedule that is exact
and refused off a tree, and a damped flooding schedule that is the Bethe
approximation. Each shape is pinned to the evaluator that predates it: on the
six-node Potts tree `log Z` and every marginal
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
The specialised evaluators stay; the factor graph is the structure for the
model none of them can express. The audit of
[#341](https://github.com/michaelJwilson/snakes_and_ladders/issues/341) below
brought those ratios to 1.8x and 4.7x with the arithmetic unchanged, and
`likelihood.message_passing.rust` --- the tree schedule's two passes in Rust,
the default since
[#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754) ---
turned the second around: the general algorithm is now 2.11x *ahead* of the
forward recursion on that chain. The section below carries the measurement.

**Forward–backward is an evaluator**
([#306](https://github.com/michaelJwilson/snakes_and_ladders/issues/306),
closing #173). `likelihood.forward_backward` returns the evidence, the position
posteriors and the pairwise posteriors of one chain in the log domain, and a
forward-filter backward-sample draw of the path; it is pinned against the path
enumeration on four chains to 1e-12 and the coupled model's E step against the
enumerated conditional posterior to 1e-11. Baum–Welch keeps its own recursion
for the gradient it needs. `likelihood.hmm` holds the scored evidence and the
Viterbi path at given parameters, compiled for the categorical, one-channel
Gaussian and count families (#997) and pinned to hmmlearn's `score` and
`decode` in `tests/validation/test_hmmlearn.py`; #1059 moved both out of
`opt.hmm`.

**The tree was audited against the runtime-optimization opportunities, and
five of the ten lines had a measurement behind them**
([#287](https://github.com/michaelJwilson/snakes_and_ladders/issues/287)). Profiling
first (`tests/benchmarks/profile_hotpaths.py` plus a Potts-side probe) ranked
the Python-level loops by self time: the SPR neighbourhood at 30 taxa spent 28
of 29 seconds building a `Node` tree and unioning `frozenset`s per candidate
*to deduplicate*; a 6-taxon hill climb charged 8,730 scalar transition-matrix
calls to 970 likelihood evaluations; a fifth of a REINFORCE run was Python
arithmetic per action. Each change is pinned to the code it replaces; no
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
p-values and exchange acceptances — which could not be relied on then: one
draw across a threshold moved by an ulp would part them, so the backend stayed
opt-in. [#599](https://github.com/michaelJwilson/snakes_and_ladders/issues/599), below, made
that agreement a bound rather than an observation, and the backend the
default. And the batched transition matrices leave the JC
log-likelihood bitwise unchanged while the gradient moves by 1.5e-11 absolute,
autograd summing the same terms in a different order; the finite-difference
check that pins the gradient is unaffected.

**The repository was audited a second time, module by module, and three of
the ranked loops moved**
([#341](https://github.com/michaelJwilson/snakes_and_ladders/issues/341)).
`tests/benchmarks/profile_hotpaths.py` covers one workload per module at the
enumerable and mid-size tiers, printing the top five functions by `cProfile`
self time with the fraction of the run each carries. Measured on one
development machine, 4 cores shared with one other process, 1-minute load
0.2–1.9 throughout; the wall clocks below were taken under 1.0. Loops #287
ranked were not re-measured, and the `ROADMAP.md` tier (1,000 taxa by 10,000
sites) and the `qa`/`infra` row are recorded as not measured.

| module | loop (mid tier; enumerable where it differs) | fraction | opportunity | expected gain | oracle |
| --- | --- | --- | --- | --- | --- |
| `likelihood` | `message_passing` flooding, 8x8: `logsumexp` per message 17.6%, ufunc reduce 14.7%, `graph.neighbours` scan 9.3%, `_run` 9.3%, `_normalize` 7.6% (3x3: 19.1 / 15.8 / 10.2%) | 58% Python per message | layout, allocation, vectorization | to within 3x of `belief_propagation` | the dictionary implementation, bitwise |
| `likelihood` | `message_passing` tree schedule, chain 200: `graph.neighbours` scan 15.7%, `graph.degree` scan 10.6% (400,000 calls), `logsumexp` 9.7% | 36% | layout, call overhead (two quadratic scans) | to within 2x of the forward recursion | the same, bitwise |
| `learn` | `PottsEnvironment.features`, 60 x 32 episodes on the length-8 chain: 23.5%, with 520,968 ufunc reductions from its loop over sites; `policy.sample` 4.0% | 23.5% | vectorization | under 20% of the run | `_deltas`, exact |
| `search` | `maxflow.energy`, 32x32 x 64 configurations: the per-edge Python loop 91.6% (16x16: 93.0%) | 92% | vectorization | 5–10x on one configuration | `potts.log_weights`, 1e-12 relative |
| `search` | `spr_neighbours` at 20 taxa: 128.7 ms for 1,122 candidates, `build` and its generator 44%, `visit` 28% | 10% of one `infer` step (1.28 s); 67% of one candidate fit (193 ms) | allocation (a `Node` tree per new key) | at most 2x on the neighbourhood, under 5% of a step | not ported: under the 10% rule per step |
| `search` | `gibbs.sample_factor_graph`, 32x32: `conditional` 42.0%, `gibbs_sweep` 17.0%, `log_density` 9.7% (16x16: 40.8 / 18.6 / 9.2%) | 69% | compiled backend over the #341 edge layout | ~10x, from the Potts `numba` sweep's 7x | draw for draw on the same uniforms; **acted on by [#561](https://github.com/michaelJwilson/snakes_and_ladders/issues/561) and [#563](https://github.com/michaelJwilson/snakes_and_ladders/issues/563), below: 122x on the sweep, 147x on the density, 18.5x on the run** |
| `search` | `alpha_expansion`, 32x32: Python Dinic `_augment` 28.8%, `_levels` 18.8%, `expand` 24.0% | 72% | FFI: `maxflow_rust` as the inner solver | 3x or more on the expansion | its energies, exact; **acted on by [#528](https://github.com/michaelJwilson/snakes_and_ladders/issues/528), below** |
| `search` | `potts_mcmc` single-site, 32x32: `_single_site_sweep` 45.8% | 46% | FFI: the Rust backend exists and was opt-in (#287) | — | **acted on by [#599](https://github.com/michaelJwilson/snakes_and_ladders/issues/599), below: the default, on an exact pin** |
| `opt` | `hmc.sample`, 1,000 draws on the length-64 chain: `torch.logsumexp` 32.0% (512,000 calls, one per position per evaluation), `log_partition` 9.6%; the fit at length 64: 26.6% | 42% | call overhead: reassociate the homogeneous transfer-matrix product by repeated squaring, 6 products for 64 positions | 1.5–2x on the run | the sequential recursion at 1e-12 relative and central differences for the gradient; **acted on by [#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754), below: 3.57x on the run, above the expected range, because the cut shortens the autograd tape as well as the forward pass** |
| `opt` | `fit` on the tree, 20 taxa x 500 sites: `run_backward` 45.2%, `_post_order` 22.9%; no Python-level call per site | — | none: the objective is one torch pass over sites | — | met by construction |
| `sim`, `likelihood` pruning, Fitch, `budget.compare`, `fit_surrogate`, Rust/FFI | every remaining loop under 10% of a run that is itself milliseconds: `sample_rows` kernel 74.4% of a 0.4 ms call with 19% in its wrapper, `pruning_rust` kernel 86.5%, `FactorGraph.__init__` 22.4% of 31 ms building the 32x32 graph | — | none | — | recorded, not ported |

Three loops moved, each pinned before it was timed and none changing what it
computes:

| loop | pin | before | after |
| --- | --- | --- | --- |
| `message_passing` flooding on the 8x8 lattice: messages as rows of two preallocated `(n_edges, width)` arrays, factors grouped by table shape and axis with their tables stacked once, one vectorized pass per group per sweep | bitwise against `message_passing_reference` (the dictionary implementation, kept) on every marginal and every schedule, 51 iterations both; `log Z` to 1e-12 relative | 514.6 ms, **75x** `belief_propagation` (6.88 ms) | **12.38 ms, 1.8x** |
| `message_passing` tree schedule on the 200-step chain: the same kernels grouped by height and depth, breadth-first, so a 2,000-step chain no longer exceeds the recursion limit | bitwise as above; the 2,000-step chain against the forward recursion to 1e-12 | 29.75 ms, **9.5x** the forward recursion (3.14 ms) | **14.77 ms, 4.7x** — the 2x target is not met: 800 levels at 10–15 µs of NumPy dispatch each is the floor of this layout |
| `maxflow.energy`: one gather over the edges and a `dot` | `log_weights` to 1e-12 relative (realized 7.7e-14; no longer bitwise, the edge terms sum in a different order) | 0.59 / 2.36 / 8.37 ms on one configuration at extents 16 / 32 / 64; 2.70 ms on 64 configurations at 32x32 | **0.08 / 0.28 / 1.00 ms (7.7–8.6x)**, below the Rust cut kernel at every extent; **1.14 ms (2.4x)** |
| `PottsEnvironment.features`: one gather from a padded neighbour table, and `is_terminal` reading it instead of `_deltas` per action | `array_equal` to `_deltas`, unchanged | REINFORCE 60 x 32 episodes on the length-8 chain 2.43 s; one gradient update on the length-4 chain 27.31 ms | **1.85 s (1.32x)**; **23.99 ms**; `features` 23.5% of the run to 11.5%, `policy.sample` 5.4% — Python per action sits at the 20% line rather than under it |

No `numba` or Rust port was reached: on every loop acted on the vectorized
pass carried the gain, and the profile of what remains is dispatch per level
(the chain) rather than a call-bound inner loop.

**One Potts model in code, and the consolidation was also the optimization**
([#277](https://github.com/michaelJwilson/snakes_and_ladders/issues/277)).
Seven builders returned one adjacency and four loops scored one energy.
`PottsGraph.compressed_adjacency` is now the only adjacency and
`sim.potts.energies` the only scorer, with `sim.potts.heat_bath_log_weights`
the conditional the vectorized simulator, the sequential sampler and the Rust
kernel share --- the three loops around it stay separate, being separate.
Which implementation survived was decided by measurement rather than by
location: the surviving energy is the vectorized gather and dot product, and
the surviving neighbour walk indexes the compressed rows. `likelihood.potts.log_weights`
is deliberately not merged in, being what the rest is refereed by.

Every chain, configuration and labelling is **bitwise** unchanged at the same
seeds on `potts_chain`, `potts_lattice`, `planted_glass` and `spatio_only`
(46 quantities). What moves is a reported *energy*, by at most **3.7e-16**
relative --- two units in the last place --- where the edge sum is a dot
product rather than a left-to-right loop. Timed on 4 cores with `ps` showing
nothing but the job; this host resolves nothing below about 20%, its first
process of a pair running 15--20% faster than its second, so every cell below
is the minimum over interleaved runs in both orders.

| path | before | after |
| --- | --- | --- |
| energy, 32x32 at 64 configurations | 2.89 ms (the sampler's, through `log_weights`); **1.34 ms the fastest replaced** (`maxflow.energy`) | **0.82 ms** |
| `maxflow.energy`, the two-state entry point | 1.34 ms | 1.31 ms, unchanged |
| heat-bath sweep, 32x32 at 2 / 10 states | 5.75 / 5.31 ms | 5.29 / 5.30 ms, unchanged |
| Gibbs simulator, 6x6 x 50 chains; 12x12 x 200 | 46.5 / 148.5 ms | 45.1 / 132.7 ms, unchanged |
| single-site descent, 16x16 at 3 / 5 labels | 2.59 / 2.62 ms | 2.67 / 2.47 ms, unchanged |

`compressed_adjacency` is built by a stable sort rather than a Python loop
over the edges, which `iterated_conditional_modes` pays per call: at 16x16 the
loop was 1.3 ms of a 2.1 ms descent, and the rows are identical on 47 graphs
including doubled bonds, self-loops and an empty edge set.

**The minimum cut `alpha_expansion` could not reach, and the claim that
looked like a boundary cost**
([#528](https://github.com/michaelJwilson/snakes_and_ladders/issues/528),
[`docs/experiments/008`](docs/experiments/008-ffi-profile-and-alpha-expansion.md)).
Re-run on this host, the audit's ranking holds: `maxflow._augment` 28.4% and
`_levels` 20.6%, **49.0%** of `alpha_expansion` at 32x32 in a Python Dinic
solver beside a Rust one. The reason it was never taken was read as the FFI
boundary and was not. `max_flow_impl` returns `(value, source_side)` and its
docstring already says that side *is* a minimum cut; the PyO3 wrapper bound
`(value, _)` and returned the value alone, so `expand`, which needs the cut
and not the flow, had nothing to call. Returning the side — one `Vec<bool>`
per call, no second traversal and no extra crossing — and hoisting a `set`
construction out of a per-edge loop that made it quadratic in the edge count:

| open lattice, labels | Python cut | + hoisted `set` | + Rust cut | overall |
| --- | --- | --- | --- | --- |
| 8x8, 3 | 17.10 ms | 16.54 ms | **3.54 ms** | **4.82x** |
| 8x8, 5 | 19.54 ms | 18.71 ms | **4.16 ms** | **4.70x** |
| 16x16, 3 | 94.24 ms | 90.07 ms | **14.15 ms** | **6.66x** |
| 16x16, 5 | 194.37 ms | 178.77 ms | **25.52 ms** | **7.62x** |
| 32x32, 3 | 648.47 ms | 583.02 ms | **60.29 ms** | **10.76x** |

Medians over 21, 21, 11, 11 and 5 rounds, one thread, load 0.75–0.81 either
side. The 3x the audit expected is met at every size and exceeded past 8x8,
because the port removes the whole solver rather than one loop inside it.
Criterion times the kernel alone at **15.2, 68.4 and 298.9 µs** per cut at
extents 8, 16 and 32, against 3.5, 14.2 and 60.3 ms for a whole expansion
through the binding, so the crossing is not the term here either — which is
the third measurement behind root `CLAUDE.md`'s corrected FFI rule.

The backend is **opt-in**, `Backend.PYTHON` by default. A minimum cut is a
combinatorial minimum whose value both solvers must report exactly, but the
cut attaining it need not be unique, and a degenerate network could hand back
a different labelling of the same energy and send the two routes down
different expansion sequences. On the seeded fixtures they do not: twelve
cells at 8x8 with two and four labels agree on the labelling, the energy, the
cycle count and the move count, and the pure implementation stays as the
oracle. Nothing else in the mid-tier ranking is a candidate — every other
module's first entry is a `torch` call, a NumPy ufunc or a Rust kernel, and
`gibbs.conditional` at 44.8% was #341's open item, unchanged there and
carried by [#561](https://github.com/michaelJwilson/snakes_and_ladders/issues/561),
below.

**The Gibbs conditional compiled, and the exponential that could not cross
with it** ([#561](https://github.com/michaelJwilson/snakes_and_ladders/issues/561)).
`gibbs.conditional` re-profiled on this host at **43.4%** of a 32x32 run,
`gibbs_sweep` at 18.0% and `log_density` at 9.0%, which holds #341's ranking.
The sweep now runs through a `numba` kernel over an **edge layout** — every
factor's log table flattened into one array, and offsets into it per
variable — so the list of lists, the `np.zeros` per site and the tuple key
built with a `slice` per factor are all gone.

| 8x8, 3 states, 20 sweeps | NumPy | compiled | ratio |
| --- | --- | --- | --- |
| `gibbs_sweep` (min / median) | 23.50 / 24.53 ms | **0.193 / 0.202 ms** | **121.8x / 121.7x** |

| `sample_factor_graph`, 20 sweeps | NumPy | compiled | ratio |
| --- | --- | --- | --- |
| 16x16 (min / median) | 101.7 / 103.3 ms | **14.4 / 14.6 ms** | **7.1x / 7.1x** |
| 32x32 (min / median) | 423.6 / 430.0 ms | **59.9 / 60.5 ms** | **7.1x / 7.1x** |

`pytest-benchmark` over 3,273 and 31 rounds for the sweep, five repeats for
the run, one thread, on a host `ps` showed carrying nothing but the job
(1-minute load 0.13 before the first reading). A second reading of the sweep
gave 129.6x and 128.8x, so the sweep is 122–130x over two runs; the run's
7.1x repeated to the decimal. The **7.1x** the run realizes is the audit's
~10x missed and the Potts `numba` sweep's 7x met, and the sweep's 122x is
what Amdahl's law then leaves: what the ticket priced at 69% of the run is
now 2.1% of it. Peak traced memory at 32x32 rises from 1.12 MiB to 2.40 MiB,
of which the layout is 351 KiB, built once per graph in 14.1 ms.

**The pin is exact, and a bound is what makes it exact.** A heat-bath draw
exponentiates, and NumPy's `exp` and `libm`'s disagree in the last place on
**4.6%** of `float64` inputs (2e6 draws, three ranges), so the arithmetic
alone would make this a distributional port like the Rust Potts sweep. The
kernel instead decides a site only where the draw clears every cumulative
boundary by more than the two exponentials can move it, and hands any other
site back to NumPy: 16 units of the last place per state, against the 4 the
error analysis needs. So the compiled sweep is the *same chain*, not a chain
of the same law — 8 of 8 runs at 16x16 and 32x32 over four seeds agree on
every state and every log-density, 76,800 draws, and the handing back is
pinned by driving it with a guard wide enough to take every site. That is
what lets the kernel be the default, and
[#599](https://github.com/michaelJwilson/snakes_and_ladders/issues/599), below, is the same
construction in `src/potts.rs`.

**Re-profiled after, the ranking inverts.** At 32x32 `log_density` carries
**47.8%** of the compiled run and its generator expression a further 12.0%,
`layout` 13.2% — paid once per graph — and `gibbs_sweep` **2.1%**, down from
18.0%. `conditional` does not appear: no call reaches it. `log_density` is
the loop that clears the 10% bar now, a Python sum over factors with a tuple
key built per factor; it is carried by [#563](https://github.com/michaelJwilson/snakes_and_ladders/issues/563), below, rather than here.

**The density compiled, and the layout left on top**
([#563](https://github.com/michaelJwilson/snakes_and_ladders/issues/563)).
`log_density` runs through a second `numba` kernel over the edge layout the
sweep already walks: each factor's element is the offset its axes fix, and the
sum runs left to right over factors in graph order. That order is the pin.
Floating-point addition is not associative, so a pairwise or vectorized sum
would move the last place of every recorded density; this one takes no
exponential either, so nothing here needs the bound #561's sweep needs, and
`FactorGraph.log_density` — unchanged, and still the definition every adapter
is enumerated against — is reproduced **bitwise**.

| 32x32, one call, 3,008 factors | dict | compiled | ratio |
| --- | --- | --- | --- |
| `log_density` (min / median) | 1,680.6 / 1,861.4 µs | **11.5 / 11.7 µs** | **146.7x / 159.5x** |

| `sample_factor_graph`, 20 sweeps at 32x32 | min / median |
| --- | --- |
| NumPy | 445.6 / 451.7 ms |
| compiled sweep, dictionary density (#561) | 65.2 / 70.4 ms |
| both compiled (#563) | **24.1 / 25.0 ms** |

The run realizes **18.5x** over NumPy against #561's 7.1x, and 2.7x over #561
alone. `pytest-benchmark` over 494 and 36,977 rounds for the density, five
repeats of `perf_counter` for the run, one thread, each reading on a host `ps`
showed carrying nothing but the job (1-minute load 1.36 to 1.39). Peak traced
memory over the run is 2.67 MiB against 1.05 MiB for the NumPy path; the
layout grows to 475 KiB, of which the four factor arrays are 125 KiB, and its
build from 13.7 to 16.1 ms — the allocation the density no longer repeats
3,008 times per recorded sweep.

**Re-profiled again, the layout is what is left.** At 32x32 over 20 sweeps
`_Indexed.layout` carries **37.0%** of the run in self time and **80.2%**
cumulatively, `_Indexed.__init__` 6.7% and 11.3%, `gibbs_sweep` 4.1% in self
time, and `log_density` does not reach the top eight: 20 calls at 11.5 µs is
0.2 ms of a 24.1 ms run. The layout is paid once per graph, and a compiled
sweep costs 0.125 ms at this size, so the layout is worth 130 sweeps: at 100
sweeps it is still 66.6% cumulatively, and it clears the 10% bar for any run
under about 1,300 sweeps of one graph. `sample_factor_graph` builds one per
call, so a caller that samples many graphs pays it every time and a long chain
amortizes it. It is the next candidate on this path and is not carried here.
Four readings on this host agree to within one point, the last two of them
with `ps` showing nothing but the job.

**The Rust sampling sweep is the default, on the same construction**
([#599](https://github.com/michaelJwilson/snakes_and_ladders/issues/599)). `src/potts.rs`
decides a site only where the draw clears every cumulative boundary by 16
units of the last place per state — four times the bound the error analysis
needs — returns the flat position of the first site it declines, and the
oracle's own site update, factored out of `_single_site_sweep`, decides that
one before the kernel resumes. `sample_potts` gains the `backend` argument it
lacked; it, `anneal_potts`, `parallel_tempering` and `adapt_ladder_potts`
default to `Backend.RUST`.

| 100 sweeps / steps, median of five | oracle | Rust | ratio |
| --- | --- | --- | --- |
| `sample_potts`, 16x16 | 292.2 ms | **2.7 ms** | **109.8x** |
| `sample_potts`, 32x32 | 1,201.9 ms | **8.5 ms** | **140.9x** |
| `anneal_potts`, 16x16 | 344.5 ms | 25.0 ms | 13.8x |
| `anneal_potts`, 32x32 | 1,381.6 ms | 79.9 ms | 17.3x |
| `parallel_tempering`, 4 x 20, 16x16 | 260.5 ms | 7.3 ms | 35.8x |
| `parallel_tempering`, 4 x 20, 32x32 | 1,066.0 ms | 23.9 ms | 44.7x |
| `adapt_ladder_potts`, 10 sweeps, 16x16 | 324.6 ms | 8.8 ms | 36.9x |
| `adapt_ladder_potts`, 10 sweeps, 32x32 | 1,288.9 ms | 26.8 ms | 48.1x |

One reading, one thread per process, 1-minute load 1.0 to 2.0 on the 4-core
host, which carried one other job. Two further readings of `sample_potts`
alone, medians of nine, gave **121.2x / 152.0x** at a load of 0.97 and
**110.7x / 140.1x** at 1.9, so the two extents are 110–121x and 140–152x
over three runs. Annealing trails because the per-sweep energy the schedule needs is
NumPy either way. The cluster moves are not ported and are unchanged; the
comparison between move sets is therefore run on `Backend.PYTHON`, in
`test_potts_mcmc_bench.py`, `test_ground_state_bench.py` and
`profile_hotpaths.py`, so it stays a statement about move sets.

**The chain is the oracle's, state for state.** 48 runs — two extents, four
seeds, two alphabets, with a field and without, 100 sweeps each — agree on
every recorded configuration, and the kernel handed **0 of 2,457,600** sites
back. So `_GUARD` is free at 16 and narrowing it toward 4 buys nothing, which
is what [#599](https://github.com/michaelJwilson/snakes_and_ladders/issues/599) asked to be
measured. On the same 48 runs the *unguarded* kernel also matched, so on this
host the two `exp` implementations never moved a decision: the guard is what
makes the agreement a bound rather than a property of this `libm` and this
NumPy build. Every autocorrelation time, goodness-of-fit p-value and notebook
output above is therefore unmoved, which the search regression suite (481
passed) is the check on.

**Every kernel releases the GIL**
([#604](https://github.com/michaelJwilson/snakes_and_ladders/issues/604)). Nine of the ten
`#[pyfunction]`s wrap their inner call in `Python::detach`; `double` is a
placeholder integer multiply and does not. Each already extracted its slices
and then touched no Python object, so `Ungil` makes a Python object inside
the closure a compile error rather than a crash.

| `oxisal.single_site_sweeps`, 32x32, 200 sweeps | held | released | control |
| --- | --- | --- | --- |
| 1 thread | 13.3 ms, 1.00x | 13.1 ms, 1.00x | 1.00x |
| 2 threads | 25.8 ms, 1.03x | 13.4 ms, **1.96x** | 1.87x |
| 4 threads | 50.8 ms, 1.05x | 14.2 ms, **3.70x** | 3.19x |

Throughput against one thread, medians of five, `OMP_NUM_THREADS=1` so one
thread is one core; the control is a 700x700 NumPy matmul, which releases the
GIL already. Held, four times the work cost **3.82x** the wall — serialization
exactly. No serial timing moves, and none should: this is throughput under
`backend="threads"`, not latency. What it buys is that `DEV.md`'s documented
`"threads"` case has eligible sites for the first time; which callers clear
the 2x bar through `sal.parallel` is not measured here and is
what remains of #604.

**Bounds with proofs, certified rather than trusted**
([#308](https://github.com/michaelJwilson/snakes_and_ladders/issues/308)). A
surrogate carries its claim — lower bound, upper bound or point prediction —
and `certify` holds it to that claim on every structure an oracle can score.
Two one-pass bounds bracket a topology's maximized log-likelihood: one pruning
evaluation at non-negative least-squares lengths from below, and
`sum_s log pi(x_1s) - F log k` from above, where `F` is the Fitch score — a
vertex argument on the multilinear Jukes–Cantor site likelihood, checked by
enumerating all 256 vertices at five taxa and finding the bound attained on
every site. Over the 15 topologies of the five-taxon fixture at 1,200 sites
both hold with no violation; the plug-in bound's worst gap is 6.7 nats and its
mean 2.9, at 2.4 ms against 254 ms for the fit; the parsimony bound costs 0.1
ms and its gap is 2.2 nats per site, so it is a bound and not an estimate. The
entrywise-limit bound the plan proposed was derived and dropped: it sits above
zero on every fixture, and the parsimony bound dominates it. For a lattice, the
naive mean-field bound and the
tree-reweighted spanning-tree bound (one tree per edge, uniform) sandwich
`log Z` within 0.049 and 0.028 nats per node on average over 40 random
open lattices (worst 0.078 and 0.052), both differentiable in the field and
couplings and both agreeing with central differences to 1e-6 relative; from
them a bracket on the ground-state energy that on the same lattices at
`beta = 3` sits 0.028 nats per node below the enumerated minimum. The
proofs are Appendix B of the textbook.

**Site patterns, and a bound where they saturate**
([#408](https://github.com/michaelJwilson/snakes_and_ladders/issues/408)). The
log-likelihood reads one alignment column per term, so identical columns
collapse to distinct patterns carrying integer weights and the sum over columns
is reassociated, not approximated. `likelihood.patterns.compress` builds the
table once and `pruning`, `pruning_torch` and `pruning_rust` take the weights.
Compression at each fixture's declared size, as columns to distinct patterns:
`tree_search/ci` 5 taxa, 1,200 to 321 (**3.7x**); `tree_jc/ci` 4 taxa, 20,000
to 256 (**78.1x**); `tree_search/stress` 6 taxa, 1,500 to 526 (**2.9x**);
`tree_jc/stress` 4 taxa, 200,000 to 256 (**781.2x**); `tree_search/release` 7
taxa, 2,000 to 1,230 (**1.6x**); `tree_jc/release` 8 taxa, 200,000 to 19,646
(**10.2x**). The ratio is set by the taxon count against the site count, not by
either alone, so the widest alignment at the fewest taxa compresses hardest.
Agreement with the uncompressed value is 1.8e-16 to 5.0e-13 relative over the
six backend-fixture pairs, inside `likelihood/CLAUDE.md`'s `1e-11` `float64`
bound, and the weights are checked against a `Counter` over the columns.

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
orders topologies. The `POINT` claim drops the tail and ranks by the frequent
blocks alone, the same sites for every candidate; under it a lazy
`search.infer` reaches the
same topology and the same fitted log-likelihood as the search that fits every
candidate, from three starts under both move sets at 1,200 sites and cutoff 4,
at **2 to 4** fits against **5 to 15** and **68 to 163** forward passes
against **209 to 731**. The cutoff is a count, so it scales with the
alignment: 8 at 1,200 sites loses the optimum from 6 of 12 starts and 4 does
not, while 8 at 2,000 sites keeps all 12.

**The interval is not a cheaper forward pass.** One thread, mean over 50 calls:
at five taxa by 2,000 sites the uncompressed evaluation is **0.908 ms**, the
pattern-compressed **0.660 ms**, the per-tree extremes **0.384 ms**, and the
interval **3.650**, **3.314** and **3.047 ms** at cutoffs 1, 8 and 32; at four
taxa by 20,000 sites, **3.974**, **0.511**, **0.257**, and **23.183**,
**22.699** and **22.588 ms**. The interval costs **4.0x** and **5.8x** the
evaluation it stands in for, and raising the cutoff from 1 to 32 buys **17%**
and **2.5%**: the two terms that scale with the cutoff are bounded by 0.66 ms
and 0.26 ms and everything else is the block partition's `np.unique`, which
sorts every block whatever the cutoff. The bound's saving is in *fits*, not in
passes: a fit is 254 ms at this fixture against a 3.65 ms interval, which is
what the ranked search's 2 fits against 13 buys.

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
0.019 and 0.004 — a factor of 1.26, 1.66, 1.74 and 3.20. Per point the
iteration is asserted only *no worse* than its first iteration, because at 2 dB
the two tie at 40 errors; the strict improvement is asserted over the four
points together, 267 errors against 306. At `K = 1,024`, 200 frames
per point over six points (182 s, **6.0 s** after #754's Rust trellis), the
waterfall turns between 0.4 and 1.2 dB.

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
`bcjr`'s own bytecode --- the Python loop over `K + m` steps --- the shape a
compiled backend would take next. Issue #754 took it: `src/bcjr.rs`, behind
`likelihood.convolutional.rust`, is 26.2x the decode at the declared
`K = 256` and is now `bcjr`'s default backend, with the `(7, 5)` register's
outputs bitwise unchanged. *The BCJR trellis in Rust*
below carries the table.

*The departure, reported rather than asserted.* The ensemble bit error rate
is **not** monotone in the iteration count. Over the six declared points at
`K = 256` it rises between consecutive iterations at four of them, the
largest rise 2.3e-3 at 0 dB between iterations 7 and 8, and every rise is
inside one binomial interval of the 12,800 message bits the point rests on.
The suite therefore asserts only that the last iteration beats the first and
that no rise exceeds twice that interval. At `K = 12` with a shorter
interleaver it is larger relative to the rate — a property of loopy sum-product
on a serial schedule.

*A second negative result.* At `K = 12` the code buys nothing at 0 dB: the
exact bitwise MAP's bit error rate, 0.092, is the uncoded antipodal closed
form's 0.079 to within the sample, and the iteration's 0.115 is above it, so
the closed-form pin is stated at `K = 256` and above.

**A cheaper climb, and a kernel the compiler can vectorize**
([#408](https://github.com/michaelJwilson/snakes_and_ladders/issues/408)).
`cProfile` over an eight-taxon
SPR search at 1,000 sites and 300 candidates — 80.9 s, 301 fits, 18,955
forward passes, 4 accepted moves, budget exhausted — puts **40.5%** of self
time in the autograd backward pass, **19.7%** in the Torch pruning post-order
and **6.7%** in the L-BFGS step, with neighbourhood generation nowhere in the
top twenty. The search's cost is candidates fitted times passes per fit: a parsimony start
and a bounded regraft reduce candidates, a partial refit reduces passes per
candidate.

**What each change bought, alone and together.** SPR from a random start,
budget 500 candidates, medians over 8 seeds. Every configuration reached the
enumerated maximum **8 of 8** at five and at six taxa, and at eight taxa every
one returned the unbounded search's own log-likelihood to the last digit
printed, so the saving below is not paid for in optima. Six taxa,
1,500 sites, as fits / forward passes / seconds: unbounded **49 / 2,848 /
10.09**; parsimony start **31 / 1,902 / 6.61**; radius 1 **15 / 755 / 2.58**;
radius 2 **43 / 2,482 / 8.63**; radius 3 **49 / 2,848 / 10.04**, which is the
unbounded search, since no edge of a six-taxon remainder is further than 3;
partial refit **50 / 1,810 / 6.06**; all three **31 / 1,009 / 3.37**. At eight
taxa by 1,000 sites the unbounded search spends 16,600 and 24,493 forward
passes over two seeds against all three's **1,930** and **3,359**, and 69.0
and 101.7 s against **7.6** and **13.8** — **8.6x** and **7.3x** the passes,
**9.1x** and **7.4x** the wall clock.

The three do different work. A parsimony start cuts *moves* (0 and 1 accepted
at eight taxa against 2 and 4). A radius cuts *candidates per move* and pays
for it in moves (6 and 9 at radius 1 against 2 and 4), still winning by
**5.6x** in passes: a candidate costs a fit and a move costs nothing. A
partial refit cuts *passes per candidate* — its fit count rises slightly from
the full refit of each accepted move while its passes fall by a third. The
smallest radius that keeps the optimum on these fixtures is **1**, the NNI
neighbourhood exactly; the combination is reported at radius 2, the smallest
radius that is still an SPR search.


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

**Through the binding it buys nothing at the declared scale.** The same two
builds, measured from Python at the sizes `ROADMAP.md` declares: 20 taxa by
11,000 sites, 10.28 to **10.46 ms**; 200 taxa by 11,000, 114.94 to
**112.42 ms** — 1.8% the wrong way and 2.2% the right way, both inside the
run-to-run spread. Against the NumPy oracle in the same runs, 1.91x to
**2.06x** and 1.94x to **2.01x**. The layout pays where an alignment is long
enough for one node's rows to leave cache and not at 11,000 sites, where 199
internal nodes and a `log` per site per node are the call. The remaining term
at many taxa is not identified and no further port is taken. Agreement with the
NumPy oracle is measured at **3.4e-15** and **1.5e-15** relative, inside
`likelihood/CLAUDE.md`'s `1e-11` `float64` bound; the one reassociation is the
rescale divide becoming a reciprocal and a multiply, which is why the pin is
relative and not bitwise.

**The LDPC decoder, specialised from the general sum-product and held to it**
([#340](https://github.com/michaelJwilson/snakes_and_ladders/issues/340), part
1). `likelihood.ldpc.decode` runs the log-domain `tanh` rule or min-sum under a
flooding schedule, vectorized over every edge with two preallocated message
buffers, a syndrome stop that refuses to count an undecided (zero-ratio) bit as
decided, and messages clipped at `+-30` so an erasure's certainty stays a
number; codeword enumeration at `2 ** k <= 200,000` gives exact bit posteriors
and the ML codeword. Pins: on a 22-bit cycle-free code the decoder equals the
tree-schedule `sum_product` and the enumeration over 32,768 codewords to
1.9e-13 in log-odds on the symmetric and Gaussian channels and 9.4e-14 in
probability on the erasure channel, where the cap's bound is `exp(-30) =
9.4e-14`; min-sum equals the tree-schedule `max_product` to 6.7e-16 and returns
the enumerated ML codeword with a margin above 0.1 nats; on six loopy (3,6)
codes at 12 and 18 bits the decoder and the general damped flooding reach the
same Bethe fixed point within 4.6e-11, both run to a message residual of 1e-12.
Density evolution on the erasure channel reproduces the published thresholds
0.4294, 0.3834 and 0.5176 for (3,6), (4,8) and (3,5) to 5e-4, and at the
release gate the 19,998-bit code resolves every erasure at `epsilon = 0.42` on
three seeds and leaves 25–30% of its bits erased at `0.44`. Per pull request,
the 996-bit code over 20 shared seeds has zero bit errors at `p = 0.05` on the
symmetric channel and a bit error rate of 0.049 at `p = 0.09`, either side of
the (3,6) threshold `p* = 0.084`, and no residual erasure at `epsilon = 0.35`
against 0.433 at `0.5`. One finding is recorded rather than pinned: min-sum on
the symmetric channel, where every ratio has one magnitude and the
leave-one-out minimum ties everywhere, is not monotone in `p` at 996 bits
(frame error rate 0.72 at `p = 0.05`, 0.08 at `0.06` over 50 seeds), because
the cap at 30 truncates the integer multiples of `log((1 - p) / p)` at a
different multiple for each `p`; the sum-product decoder shows no such effect.
Sizes, memory, the profile that decides a Rust kernel, and the optimization
framing against the samplers are parts 2 and 3 of the ticket.

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
`oxisal::coupled` tabulates the log-density by count — `[count,
M, K]`, count-major — and reads two doubles where the oracle calls `lgamma`
three times. NumPy against Rust through the binding, one thread:

| bin factor | positions | `class_posteriors` NumPy | Rust | | `external_field` NumPy | Rust | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2,000 | 20.1 s | **0.55 s** | 36.6x | 191.3 s | **1.10 s** | 174x |
| 5 | 4,000 | 39.2 s | **0.82 s** | 47.8x | 378.1 s | **2.13 s** | 177x |
| 1 | 20,000 | 278.5 s | **4.89 s** | 57.0x | ~2,690 s (extrapolated) | **12.59 s** | ~214x |

The bin-1 NumPy field is the one number not measured: at the ratio the two
coarser factors set it is three quarters of an hour on a host four agents
share, past the 20-minute cap, so it is extrapolated from the bin-5
measurement and marked as such. Criterion times
the kernel without the tables or the boundary at `S = 200`: 21.32 ms and
84.04 ms, which scaled to bin 10's 2,000 positions is 213 ms and 840 ms
against the 0.55 s and 1.10 s above — the tables' construction and the
crossing are 0.34 s and 0.26 s of each call, the boundary term
`likelihood/CLAUDE.md` requires measuring separately.

The kernel is pinned to the NumPy oracle at the ci instance and on a
64-vertex slice of the 5K one: the log evidence and the field agree to
**3.1e-15** and **3.8e-15** relative, against a stated bound of 1e-12; the
state and pairwise posteriors agree to **1.3e-9** absolute, against 1e-8, that
residual being the *oracle's* own departure from summing to one — the kernel's
rows sum to one exactly, the scaled recursion normalizing at every position.

Recovery at the declared size, from a labelling with 30% of its vertices
redrawn: the label block returns the planted labelling **exactly** at bin
factors 10 and 5 --- 0.734 to 1.000 up to a permutation of the ten class names
--- in one round, 22.8 s and 27.2 s of which the alpha-expansion is about 20.
The full test, simulate through assertion, is **59 s at factor 10 and 62 s at
factor 5**, so factor 5 is the key fixture and factor 1 stays `release`. The
plan expected factor 1 to be out of reach; it recovers the labelling exactly in
**98 s**, inside the same 120 s budget, because the port took one E sweep there
from 278.5 s to 4.89 s. That the whole declared instance now fits a per-test
budget is the finding; moving the key fixture onto it is the maintainer's call.
Two further findings. From a *uniform* start at `M = 10` the first E step's
class densities are ten mixtures of the same data and the field is at chance,
so the ascent does not move: which start drives this block is issue #306's
question, and the test corrects a labelling rather than searching for one. And
the fixture's first draft declared a beta-binomial trial count per hidden
state, which puts a count drawn under one state outside another's support,
where the log-density is `lgamma` of a negative argument: two of ten classes
lost their evidence to `NaN` and the field argmin fell to 0.113, the fraction
of vertices in the first class. The loader now refuses a trials ladder that
varies, the trial count being a property of the observation and not of the
state.

**The phylogenetic objectives' value and gradient under JAX**
([#1005](https://github.com/michaelJwilson/snakes_and_ladders/issues/1005)).
`likelihood.pruning.jax` traces `pruning_torch`'s post-order once per
topology under `jit`, the map from `theta` inside the same program;
`BranchLengthObjective` and `SubstitutionModelObjective` take
`backend=Backend.JAX`. Against the taped route on `tree_jc/release.yaml` at
2,000 sites the value agrees exactly and the gradient to 8.3e-16 (JC) and
9.8e-13 (GTR) relative; brute force pins the value at 1e-12. The default
stays `Backend.TORCH`: `infra/jax_decision.py pruning` on balanced trees of
4--32 taxa at 10^5 sites puts JAX at 0.95--1.59x the taped route's runtime and
1.23--1.56x its peak memory, and all 16 cells miss #1000's rule (at most 0.5x
runtime at 1.25x memory, or the converse). JAX stays the opt-in route.

**The pruning routes share their plumbing; the oracle shares nothing**
([#858](https://github.com/michaelJwilson/snakes_and_ladders/issues/858)). The
post-order, the leaf indicator, the rescaling step and the `pi`-shape,
missing-leaf, branch-length and branch-order validations were written five
times over. `likelihood.pruning_common` holds one of each and
`likelihood.pruning.rust`, `pruning_torch`, `pruning_analytic`,
`surrogate.prune_with_matrices`, `blocks` and `sandbox.pruning_burn` call it:
130 lines out of the routes. `likelihood.pruning` is byte for byte unchanged,
being the oracle each route is pinned against, and `brute_force`, the referee
that pins *it*, keeps its own checks for the same reason. No arithmetic is
unified: the vanished-scale fallback and the scatter index's device differed
between copies and are parameters, so no call site's bits or device move.
Evidence is 38 blake2b digests of the float64 bytes --- every route, rescaled,
unrescaled, weighted, cached, with the analytic gradient --- on `tree_jc/ci`
and `tree_search/ci`, identical before and after.

## Milestone 1.3 — Continuous Optimization via Autodiff

**Modules.** The optimization interface and what is fitted through it: `opt.objective`, `opt.constrain`, and `opt.testfunctions`, whose functions are the problem a fit is checked on before any model is. `sample.langevin` and `sample.slice`: the two samplers an HMC number is read against, one module each, both over the same `Objective` (#756; under `opt` until #777, with `sample.hmc` and `sample.schedule`). `opt.em`: the E step, M step alternation and the relative stopping rule the three expectation-maximization entry points ran a copy of each, which no oracle is pinned against (#859). `cost`: the unit a method spends, declared once for the oracle ladder and for `opt.budget.Budget`, which took a bare string until #860. `opt.termination`: whether a loop finished and why, one answer on fourteen results that carried five encodings between them (#860). `opt.starts`: one loop that seeds every start, polishes it at a held budget and records its curve, which four consumers wrote by hand (#894). `opt.split_merge`: split-and-merge moves on a converged mixture fit, kept only where the log-likelihood rises; the planted fixed point on `emission_mixture/ci` gains 165.7 nats on its first move (#904). `opt.hmm.jax`: the HMM objectives' default value and gradient under JAX, the scaled forward recursion with the backward recursion as its reverse pass, at 0.07x--0.18x PyTorch autograd's runtime at 10^4--10^5 positions and pinned to autograd at 1e-10; `opt.objective.DeclaredGradient` is how `fit` and the samplers read it (#1000). `sample.metropolis`: gradient-free random-walk Metropolis over any `Objective`, a `Kernel` on the shared `run_chain` so its warm-up is `Adaptation`'s; on an energy declared in `sample.declared` (Gaussian, Rosenbrock) the chain, the warm-up and `Power` operators' Kalman statistics run in `oxisal.MetropolisWalk`, draw for draw with BlackJAX's `rmh` on shared randomness and at 0.03x--0.37x its runtime (#1006). `sample.hmc` on the same compiled loop (`src/chain.rs`): the declared Gaussian, Rosenbrock, mixture and Gaussian HMM run the whole chain, warm-up and `Power` filters in `oxisal.HmcWalk`, and an objective with a traceable JAX energy in `sample.hmc.jax.JaxWalk`; against BlackJAX 0.11x--0.39x on Rosenbrock and the Gaussian with warm-up, 0.34x--0.36x on the HMM, 0.56x--0.58x on the mixture after two attempts (#1008).

**The interface is model-agnostic, and that is measured rather than asserted.**
An `Objective` is an unconstrained parameter vector, a differentiable scalar,
and a map back to named constrained parameters
([#115](https://github.com/michaelJwilson/snakes_and_ladders/pull/115)). Four instances now
run against it unchanged — the Potts chain, the HMM, branch lengths on a fixed
topology, and the GTR substitution model — and none required a change to
`sal.opt`. A test asserts the module imports nothing from `sal.sim`,
`sal.likelihood` or `sal.search`.

**The optimizer is now pinned to minimizers known in closed form, not only to
likelihood surfaces.** Every earlier test of `fit` measured a statistical
property — the first-order condition, coverage at the nominal rate, agreement
with Baum-Welch — under which an optimizer that stops early and a parameter
that is weakly identified look identical. Three standard test functions
separate them: Rosenbrock is reached to `1e-11` of its analytic minimizer at
2, 3 and 5 dimensions, the autodiff gradient matches the hand-written closed
form exactly on all three functions, and all four of Himmelblau's equal minima
are reachable, each from its own basin.

On Rastrigin, over 200 starts drawn uniformly from the standard `+/-5.12`
domain, a single L-BFGS fit reached the global minimum **0 times**; restricted
to `+/-2` it reached it in 4%. Every one of those runs reported `converged`,
satisfying the first-order condition. `converged` says nothing about global
optimality, and a result resting on a single fit of a multimodal surface has to
say so.

**The lattice is fitted, against an exact normalizer.** `log Z` is
enumerated over all 19,683 configurations of a 3-state 3x3 lattice, so the
fitted optimum is checked against a brute-force scan of the likelihood rather
than against the optimizer's own convergence, and the enumerated normalizer
reduces to `sal.opt.potts.log_partition`'s transfer matrix on a
chain to machine precision. Interval coverage over 40 replicates is 157/160 at
100 samples, 153/160 at 400 and 153/160 at 1600.

That closes the requirements row, and leaves the hidden Markov model's half of
it less settled than the committed coverage figure reads: its 45/48 = 0.938 at
150 sequences and 91/96 = 0.948 at 2400 both sit within one binomial standard
error of 0.95 (0.032 and 0.022), so the under-coverage the caption describes is
not distinguishable from sampling noise at those replicate counts. The two
identified causes are real — an emission fitted near zero, and the
post-selection cost of aligning the hidden states — but stating a sample size
at which nominal coverage begins to hold would need more replicates than the
figure runs, and none is claimed.

**Intervals now have a second, non-asymptotic source.** Hamiltonian Monte
Carlo samples the posterior over any `Objective`, so an interval can be a
quantile rather than a curvature estimate at the mode. The integrator is
pinned where it is exact first: reversible to `1e-15`, with its energy error
second order in the step size, measured at a ratio of exactly 4.00 across four
halvings at fixed trajectory length. The chain is then checked against two
references that are not samplers -- an analytic Gaussian, and the Potts
chain's own two-dimensional
posterior integrated on a grid, which it matches to 0.005 in the mean and 10%
in the spread. On that fixture the Laplace standard error agrees with the
posterior's to 15%, the expected outcome for a well-identified two-parameter
model.

**A step size too large biases the spread while the acceptance rate looks
healthy**, which is why `HmcChain` reports the per-proposal energy error.
Measured against quadrature: at a step of 0.020 the acceptance rate was 0.982
and the posterior standard deviation 12% low, divergent trajectories being
rejected preferentially in the tails. Acceptance rate does not detect it;
`max |dH|` tracks it monotonically.

**A fourth-order integrator lands, and loses.** Yoshida's (1990) triple jump
joins leapfrog as a selectable symplectic integrator, both compositions of the
same kick-drift-kick sub-step
([#266](https://github.com/michaelJwilson/snakes_and_ladders/issues/266)). The
orders are measured as the ratio by which halving the step divides the energy
error: leapfrog realizes 3.999, 4.000, 4.000, 4.000, 4.000 against a predicted
4, and Yoshida 16.310, 16.077, 16.019, 16.005, 16.001 against a predicted 16 —
converging rather than drifting, which makes it an order and not a coincidence
at one step size.

**It is slower anyway, and the mechanism is worth recording.** A higher-order
method pays where the step is limited by *accuracy*; here it is limited by
*stability*. Yoshida's middle sub-step runs backwards with
`|w0| = 1.70` times the nominal step, so its stability limit in the step size
is about 0.59 of leapfrog's — measured at 0.0333 against 0.0500, a ratio of
1.50 against the 1.70 the coefficient predicts. With three force evaluations
per step on top, the order advantage is spent twice over. At equal
acceptance on the Potts posterior, leapfrog reaches 0.855 at **21** gradient
evaluations per trajectory while Yoshida needs **91** to reach 0.975 and
accepts *nothing* at 61; on the analytic Gaussian it is 3 against 7.

**Where a fit starts is now the caller's to choose, and multi-start is
measured rather than assumed.** `Objective.initial()` was already the seam;
what went through it was one fixed constant per objective.
`sal.opt.initialize` adds the objective's own start, a
deterministic perturbation, and random restarts from a passed-in generator, and
`fit_from` reports every fit and their spread rather than only the best.

Multi-start is a tool for a particular shape of surface and not a general
improvement. On Himmelblau, four equal minima, a single fixed
start reaches exactly **one** basin however often it is run and four random
restarts reach all **four**. On Rastrigin, roughly `10**n` local minima each
satisfying the first-order condition, sixteen restarts reach the global minimum
**2 times in 30** against **0 in 30** from one start -- sixteen times the cost
for a success rate still near zero, and widening the draw does not help (the
same 2 in 30 at scale 4.0 as at 2.0): the obstacle is the density of the minima
and not the reach of the proposal. Held to eight fits by
`opt.budget.compare` ([#281](https://github.com/michaelJwilson/snakes_and_ladders/issues/281))
it is 0 of 10 either way.

No default changes on that evidence. Every number below was produced from the
objective's own start and still is.

**Fitting and intervals.** L-BFGS with a strong-Wolfe line search, convergence
judged on the gradient relative to the objective's own magnitude, and
confidence intervals from the observed Fisher information pushed through the
constraint map by the delta method
([#116](https://github.com/michaelJwilson/snakes_and_ladders/pull/116)). Validation is
parameter recovery: the Potts chain's 95% intervals cover the truth at exactly
the nominal rate over 60 replicates, and the HMM's gradient fit is cross-checked
against Baum-Welch, which shares no optimizer, parameterization or constraint
map with it.

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
measured at **203 ms** at 1000 sites.

**k-means++ lands, and buys nothing the fit can use.** The first initializer
that reads its objective's data
([#262](https://github.com/michaelJwilson/snakes_and_ladders/issues/262)), and
the case [#251](https://github.com/michaelJwilson/snakes_and_ladders/issues/251)
built the `Initializer` protocol for. **The protocol needed no change.** A
data-dependent strategy is model-*specific* rather than protocol-incompatible:
it takes an `Objective` like every other initializer and refuses the ones whose
parameter vector it cannot interpret, which is why it lives beside the mixture.

Measured against its published guarantee — Arthur & Vassilvitskii (2007) bound
the expected seeding cost at `8 (ln k + 2)` times optimal, and in one dimension
the optimal clustering is computable exactly by dynamic programming over
contiguous runs. On three components six standard deviations apart, over 200
replicates: mean cost ratio
**2.91** against a bound of **24.79**, worst draw 14.41. Uniform seeding
realizes **11.03** mean and a worst draw of **58.08**, outside the k-means++
guarantee.

**And none of that reaches the likelihood, which is the finding.** EM reaches
the same optimum from either seeding on that mixture — 200/200 from k-means++,
195/200 from uniform — and from the objective's own quantile start. Harder
fixtures make both fail: at five components 1.5 standard deviations apart
neither seeding reached the best optimum found in 200 draws, and at five with
unequal weights uniform reached it 9 times in 150 against k-means++'s 3 —
noise, opposite to the direction a default change would need. **No default
moves**; k-means++ lands as a strategy a caller may choose, at a cost of one
objective evaluation (271 us of seeding against 260 us per evaluation at 4000
points).

**Eight seedings of the coupled model's emission parameters, through the seam
that already existed**
([#541](https://github.com/michaelJwilson/snakes_and_ladders/issues/541)). The `M x K`
parameters are seeded a prior draw, the data's own quantiles, `kmeans++`, the
same construction scored under the family's Bregman divergence, a short
burn-in fit, and three chains on a Gaussian surrogate --- Hamiltonian,
annealed and tempered. **The `Initializer` protocol needed no change** for the
last three, as it needed none for `kmeans++` at
[#262](https://github.com/michaelJwilson/snakes_and_ladders/issues/262); they land
as `FromChain`, `FromAnnealing` and `FromTempering`, three more consumers of
the one protocol. Cost is counted in passes over the data rather than seconds,
which is what makes a 144-pass seeding comparable to a 6-pass fit.

**At 100 components on 4,000 observations the likelihood prefers the seeding
scored under the family's own divergence, and the truth prefers the one that
cost most.** Over six paired instances at a six-pass budget, `emission++`
reaches the best projected value on **6 of 6** at a mean gap of **0.4** nats
and the prior draw on 5 at **3.2** --- an ordering #560 produced by correcting
the scoring from the negative log density to the Bregman divergence, and one
the paired test does not separate (**p = 1.00** against the prior draw, on one
discordant instance of six). `kmeans++` follows at 6.8, and the six candidates
whose fits #560 did not re-run at 11.5 to 16.1 against the best that move
produced, 3.18 nats below the old one.
The same fits reverse on the simulated truth --- recovery of
the generating component **0.089** for `tempering` and **0.082** for `data`
against `prior`'s **0.060**, each 6 of 6 at **p = 0.031**, against the
generating parameters' own 0.109, with mean relative error in a component's
negative-binomial mean 0.303 and 0.691 against 0.884. A seeding cannot be
chosen on the projected likelihood at this size, so **no default moves**; the
disagreement is [#559](https://github.com/michaelJwilson/snakes_and_ladders/issues/559),
and experiment 009 carries the table.

**What the chains charge, and what the free baseline is worth.** The three
chain seedings cost **144 passes against the fit's 6** and buy 0.007 of
recovery over `data`, which costs none. Their acceptance is 1.00 at both sizes
and the tempered ladder's lowest swap acceptance is 0.00 at the key model: at
4,000 observations the surrogate posterior is sharp, so a chain on it descends
rather than mixes and what the bill buys is an optimizer of a Gaussian
surrogate, not a draw from it. At the notebook's four-component size the two
orderings agree and the chains win both --- best value reached, recovery 0.75
to 0.76 against the generating parameters' 0.78 --- so the disagreement is a
property of 40 observations a component rather than of the candidates.

**The tree's data-driven start, and what it buys at equal evaluations.**
Mossel and Roch carry the HMM's spectral method of moments to a phylogeny,
where it becomes a distance per pair and a tree from the distances
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
is not built. `FromDistances` and `FromHadamard` in `search.initialize` are the fourth and
fifth `Initializer`, in `search/` and not `opt/` because `opt/` may import no
application module, and each names the topology a search begins from.

Measured through `opt.budget.compare` over 20, 10 and 5 datasets at five, six
and twenty taxa
([`docs/experiments/005`](docs/experiments/005-tree-initializers-at-equal-evaluations.md)).
On the branch-length fit every start reaches the one optimum; the
neighbor-joining start does so in **22.3** evaluations against the objective's
own **23.0** at five taxa, **23.5** against **24.4** at six and **25.2**
against **34.0** at twenty, the Hadamard start **21.8** at five: no start
separates on the fit below twenty taxa, because the stop is the gradient
relative to the objective and the curvature pairs cost the same from anywhere
in the basin. On the NNI search against the enumerated optimum the
neighbor-joining start reaches it from **20 of 20** datasets at five taxa and
**10 of 10** at six, scoring **4.0** and **6.0** candidates — one
neighbourhood, the start already being the optimum — against the random start's
20 of 20 and 10 of 10 at **7.3** and **16.3** (McNemar p = 1.000, no discordant
dataset); at twenty taxa, against the best found in 60 candidates, **5 of 5**
against **0 of 5** (p = 0.062, the smallest five pairs can give), the random
climb still 3,510 nats short on the first dataset. **The estimator buys the
topology, not the fit.** No default changes: every number above was produced
from the objective's own start and still is.

**Tempering against restarts on the mixture, at equal evaluations.** The
comparison
[#284](https://github.com/michaelJwilson/snakes_and_ladders/pull/284) and
[#303](https://github.com/michaelJwilson/snakes_and_ladders/pull/303) deferred,
recorded whichever way it fell
([#332](https://github.com/michaelJwilson/snakes_and_ladders/issues/332),
[`docs/experiments/004`](docs/experiments/004-mixture-tempering-vs-restarts.md)).
Five components 1.5 standard deviations apart with unequal weights, 500
observations, built from `sim.mixture` under seed 20260908 since #262 committed
neither of the five-component fixtures it measured. Multi-start EM, simulated
annealing with Hamiltonian proposals and parallel tempering — the continuous
counterpart of the Potts one, now in `sample.hmc` beside `anneal` — each spend
3,000 likelihood evaluations per start through `opt.budget.compare`, every
method ending with the same charged L-BFGS polish because raw EM sits 4 to 6
nats above its basin's optimum 500 iterations in. Against the best-known
optimum — 1111.596 nats, reached by 16 of 1,000 polished restarts and 8.3 nats
below the polished simulated parameters, whose basin is not the maximum on this
sample — over 40 shared starts: **restarts 7/40**, tempering 4/40 (McNemar p =
0.549 against restarts), annealing 1/40 (p = 0.031); mean gaps 2.6, 4.2 and 8.0
nats. **Restarts are not beaten on the mixture**, at 3,000 evaluations:
tempering does not separate from them and annealing loses to them, the opposite
of the glass row above and the same finding as Rastrigin. The 8-start tier of
the same test runs per pull request and pins the ordering. The paired test
`ROADMAP.md` §4.1 asks for is now in the utility: `opt.budget.mcnemar` on the
per-start hits, exact rather than chi-square, because 40 starts cannot support
the approximation.

**Eight ways to seed a Gaussian mixture, and the control that named the
difference.**
[#548](https://github.com/michaelJwilson/snakes_and_ladders/issues/548) ran
[#541](https://github.com/michaelJwilson/snakes_and_ladders/issues/541)'s
seeding comparison on a Gaussian instead of on counts, because a Gaussian is
where #541's candidate 3 predicts no gain: the Bregman divergence of an
isotropic Gaussian's log-partition *is* the squared Euclidean distance
([`docs/experiments/010`](docs/experiments/010-gaussian-mixture-seeding-control.md)).
**It ties on the one-channel rung and loses on the two-channel one, and #560
had to correct the scoring before either was a measurement of the divergence.**
Over 200 seedings of `mixture/ci.yaml`, scored against the exact optimal
k-means cost: the divergence and squared Euclidean produce **identical
seedings, draw for draw**, at **1.8496** times the optimum (worst 6.21), while
the rule `opt.emission_mixture.plus_plus_start` applied until #560 — the
family's **negative log density** — costs **3.9111** (worst 33.80) against
uniform seeding's **4.3470**, nine tenths of the way from the divergence to
uniform. The cause is arithmetic and not statistical: the negative log density
is the divergence plus the log normalizer, D-squared sampling normalizes its
scores rather than shifting them, and an additive constant therefore dilutes
the rule toward uniform. The identity is exact and is pinned as one:
`plus_plus_start` reproduces `opt.mixture.kmeans_plus_plus`'s seedings centre
for centre under a shared generator. It is a **one-scale** identity, and the
key rung has two channels of different pooled scale, where the divergence
divides each by its own and is therefore a different rule: **749** nats there
against squared Euclidean's 424, further than the 668 the negative log density
reached.
The two-channel rung this was sized against —
`tests/regression/fixtures/mixture/release.yaml`, ten components over
5,041 x 4,000 = 20,164,000 observations, mirroring
`spatio_sequential_counts/stress.yaml` at bin factor 5 — **is declared and not
run**: one expectation-maximization iteration over it holds four
(20,164,000 x 10) float64 arrays, 1.613 GB for the responsibilities alone, and
peak resident set measured 2,169 MiB at 5,041,000 observations against 682 MiB
at 500,000, so 328 MiB per million above a 518 MiB floor and 7.0 GiB at the
full size, against 8 GiB free on the 15 GiB host. One iteration takes 9.473 s
and 0.917 s at those two sizes, so 38 s at the full size and 25 min for one
40-evaluation fit. The comparison runs at bin factor 40, 504,100 observations,
the largest whose simulate-fit-assert run fits the 120 s key cap, at **36.7 s**.
Those four numbers were taken on the 4-core host at a 1-minute load of 0.90
with one BLAS thread and no other job; the comparison's own wall clock, 79 min
57 s and 60 min 29 s over two runs, was not, and is an upper bound.

**The nine seedings, at 40 evaluations of expectation-maximization each.**
Every candidate is #541's, unchanged, so the two studies are comparable. The
fit is budget-matched through `opt.budget.compare` and the seeding is not, so
each seeding's own cost is reported in the fit's unit and as a fraction of it.
Left, `mixture/ci.yaml` over 40 shared starts; right, the two-channel rung over
8, which is what nine fits over 504,100 observations allow — the run took
79 min 57 s and 60 min 29 s on a contended host, so 40 starts would be the
whole release tier and the paired test is correspondingly weak there.

| seeding | ci hits of 40 | ci mean gap | ci recovery | key hits of 8 | key mean gap | key recovery | seeding cost, evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random-restart | 0 | 5.30 | 0.6800 | 0 | 3,030 | 0.4116 | 0.00, 0% |
| kmeans++ | 0 | 0.723 | 0.6240 | 0 | 424 | 0.4632 | 0.90, 2.2% |
| emission-d2 | 0 | 0.723 | 0.6240 | 0 | 749 | 0.4538 | 1.00, 2.5% |
| burn-in | 0 | 18.6 | 0.6240 | 0 | 30,700 | 0.3126 | 7.00, 17.5% |
| family-sample | 0 | 1.87 | 0.5980 | 0 | 717 | 0.4093 | 0.00, 0% |
| spectral | 0 | 0.723 | 0.6240 | 0 | 424 | 0.4632 | 1.00, 2.5% |
| hmc | 6 | 0.785 | 0.5260 | 0 | 595 | 0.4370 | 48.00, 120% |
| tempering | 2 | 1.06 | 0.5620 | 0 | 595 | 0.4370 | 88.00, 220% |
| anneal | 0 | 1.39 | 0.6220 | 0 | 595 | 0.4370 | 44.00, 110% |

Gaps are nats against the fit started from the generating parameters; recovery
is the fraction of observations assigned their planted component, against a
Bayes ceiling of **0.688** on the ci rung and **0.5509** on the key rung, which
1.5 standard deviations of separation fixes. **Nothing but a chain reaches the
referee, and no chain reaches it cheaply**: `hmc` reaches it from 6 of 40 ci
starts (McNemar p = 0.031 against restarts) at 120% of the fit budget, and
parallel tempering from 2 at 220%. **And on the key rung the chains did not
mix at all** — acceptance 0.000 at the step that accepted every proposal on
the ci rung, so all three returned their own starting point and are one
uniform-seeded fit bought at 110 to 220% of the fit, which is what
`STATUS.md` reports rather than averages in. Spectral is k-means++ on both
rungs, and for the reason the route predicts: the leading principal subspace
of a one- or two-channel sample is the whole of it, so the projection is a
rotation. The burn-in initializer is last on both rungs by an order of
magnitude, at 17.5% of the fit budget.

**The two orderings, #541's and this one.** #541 scores on the projected count
mixture of `spatio_sequential_counts/ci` — four components, 4,000
observations, 6 passes — in nats below the best value any candidate reached;
this experiment scores in nats below the fit started from the generating
parameters. The units differ, the ranks are what is compared.

| candidate | #541, counts | here, Gaussian ci | here, Gaussian key |
| --- | --- | --- | --- |
| tempering | 0.0 | 1.06 | 595 |
| hmc | 2.2 | 0.785 | 595 |
| anneal | 2.2 | 1.39 | 595 |
| burn-in | 7.0 | 18.6 | 30,700 |
| random-restart (#541's `data`) | 12.0 | 5.30 | 3,030 |
| kmeans++ | 20.4 | 0.723 | 424 |
| emission-d2 (#541's `emission++`) | 200.0 | 0.723 | 749 |
| family-sample (#541's `prior`) | 114.2 | 1.87 | 717 |
| spectral | not run (#554) | 0.723 | 424 |

**Which of #548's three outcomes, once the rule is the divergence.** It is
outcome three on the Gaussian, and by more than #548 measured: candidate 3 ties
k-means++ exactly on the one-channel rung, because there it *is* k-means++, and
loses by 749 nats against 424 on the two-channel one, where the divergence
whitens the channels and k-means++ does not. On counts the answer is the size's
rather than the family's: at four components and 1,000 observations a component
it ends **200.0** nats short against squared Euclidean's 20.4, and at 100
components and 40 a component it reaches the best projected value on 6 of 6
instances and leads the nine (#541, above). The divergence is relative and a
squared distance is absolute, so which wins is a property of how far apart the
components sit in the family's own geometry, which is a measurement per
instance and not a default that moves. The two studies agree on the chains as well, from opposite
directions: #541 measured acceptance 1.00 at 4,000 observations with hmc and
anneal returning the same seeding, and this one measured acceptance 0.000 at
504,100, all three returning their starting points. One fixed step size does
not serve two sample sizes, which #756 carries.

**An interval at a fit, whatever produced the fit.** The observed information
is a property of an objective *at a point*, not of the route that reached it,
but until now only a gradient fit could ask for one: expectation-maximization
works in the model's own parameters and never builds an unconstrained vector
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
within 3.6e-9 relative. Their intervals agree to **0.31%**, the width of the
flat ridge EM approaches slowly. The EM-derived intervals cover truth at
**243/264 = 0.920** over 12 replicates, with 1 of 12 reaching the boundary and
contributing none.

**The refusals survive.** An interval from a Hessian is a statement about a
maximum, and a Gaussian component at its variance floor is not one: the
likelihood is unbounded there, the information is not positive definite, and
the new entry point refuses exactly as the old one does — checked against the
healthy point beside it, since a guard that refused everything would pass a
refusal-only test.

**And the comparison `hmc.py` promised is now complete, in both regimes.** The
missing halves were the delta-method interval on the parameters a person
names, against a chain, where the approximation is exact and where it is not.
On the analytic Gaussian the Laplace interval equals
`sqrt(diag(covariance))` to 1e-8 and the chain's spread matches it to **0.04%
and 0.47%** at 4000 draws, so agreement is asserted. On the Potts posterior the
sampled spread is **1.057, 1.031 and 1.036** times the Laplace one: slightly
optimistic, the expected direction for a mildly non-Gaussian posterior.

**Adaptation is a warm-up, opted into and reported, and the standing decision
against it is reversed on measurements**
([#333](https://github.com/michaelJwilson/snakes_and_ladders/issues/333)).
`hmc.sample` takes an `Adaptation(warmup, target_acceptance, step_jitter)`,
every field required: the warm-up sets a diagonal mass matrix from the sample
variance of its first window and the step size by dual averaging (Hoffman &
Gelman 2014 §3.2, `eq:dual-averaging`), run once at unit mass and once on the
metric; the chain is then drawn at those fixed values, and `HmcChain.adapted`
reports them with the warm-up acceptance and `force_evaluations` what the chain
cost. The mass matrix is a change of coordinates on the objective, held to a
hand-written mass-matrix leapfrog to 1e-12, and the fixed-parameter path is
untouched. Two measurements shaped it. On a locally quadratic target the
acceptance is a cliff in the step — on the analytic Gaussian 0.88 at a step of
1.2 and 0.01 at 1.4, the stiff direction's stability limit being 1.26 — so a
target of 0.65 sits on the cliff, and each proposal's step is drawn from a
uniform band around the adapted one (Neal 2011 §5.4.2.2). And the published
gain of 0.05 was set for a trajectory-averaged statistic: with a single
Metropolis probability per proposal the drawn chain's acceptance against a
target of 0.65 was **0.815** at 0.05, 0.691 at 0.1, **0.643** at 0.2 and 0.610
at 0.5, so 0.2 is the constant, stated as a deviation. The drawn chain's
acceptance pooled over 20 seeds is **0.650** on the Gaussian (per-seed 0.578 to
0.758) and **0.673** on the four-taxon tree posterior (per-seed 0.573 to 0.750;
0.678 over the 3 seeds CI runs, the 20 in the stress tier), against the target
0.65. The adapted chain and the fixed-parameter chain agree on both posteriors
— means within 1.83 standard errors on the Gaussian and 1.77 on the tree,
spreads within 0.54 and 2.84, each standard error from the chain's own
effective sample size — and the #268 interval is reported beside both: the
adapted chain's spread is 1.009 and 1.008 of the exact Gaussian one, and 1.01
to 1.13 of the delta-method interval on the tree's branch lengths against the
fixed chain's 0.99 to 1.08. The effective sample size, by Geyer's initial
positive sequence and held to an AR(1) whose autocorrelation time is a closed
form (estimate over truth 0.83 to 1.12 at a coefficient of 0.9), prices a draw:
on the tree's slowest branch the adapted chain gives **0.038** effective draws
per gradient against the fixed chain's **0.019** at unit mass, whose masses the
warm-up measured as spanning 5 to 124; on the Gaussian 0.099 and 0.105 against
0.098 and 0.015.

**A tempering ladder is chosen from its own exchange acceptance.**
`schedule.adapt_ladder` takes the measurement as a callable and knows no
model: a pair below a stated band is bisected geometrically, a rung both of
whose pairs are above it is removed, and a pair above the band beside one
inside it has their shared rung moved halfway toward the far end;
`potts_mcmc.adapt_ladder_potts` supplies it as a `parallel_tempering` run. On
the 9×9 periodic triangular antiferromagnet from
the endpoints (2.0, 0.4) alone, a band of (0.25, 0.75) at 50 sweeps per
measurement settled inside the band **20/20** seeds in 4.2 rounds on average,
on 5 to 8 rungs, and a fresh run on the returned ladder exchanged at 0.16 to
0.72 on every pair; the hand ladder (2.0, 1.2, 0.7, 0.4) exchanges at 0.28,
0.15 and 0.12. At equal sweeps with the warm-up charged — 2400 per seed, of
which the warm-up spent 895 on average and 500 to 1900 — the adapted ladder
reached the closed-form ground state **20/20** against the hand ladder's
20/20, and 19/20 against 20/20 at 1600: the instance does not separate them,
since the hand ladder hits 18/20 at 100 sweeps. NUTS remains out of scope.

## Milestone 1.4 — Discrete Move Sets & Classical Baselines

**Modules.** The discrete solvers and their compiled counterparts: `search.ground_state`, `search.projection`, `sim.topology` and `sample.gibbs.numba` (`sample.kernels`, then `sample.numba.{gibbs,factor_graph}`, until #1059, `search.topology` and `search.kernels` until #830, which also put the chain's params in `sim.potts_chain`, the sampler-built initializers in `sample.initialize` and the algebraic decoders in `likelihood.algebraic`) (`search.potts_mcmc_rust`, a one-line twin, folded by #717). `search.decoding`: two estimators of a labelling, and which loss each one minimizes (#696). `search.tightening`: a dual bound on a Potts ground state, and the plaquettes that tighten it (#696). `sample.potts_keyed`: the cluster moves, as something a deterministic ``step`` can call (#706). `sample.balanced`: the locally balanced proposal kernel the Potts lattice and the factor graph share (#756). Those two, `sample.potts_mcmc`, `sample.gibbs`, `sample.tempered`, `sample.annealed` and `sample.statistics` were under `search` until #777. `search.mixture_starts`: the joint count-pair mixture started every way the package can start it, each start polished by EM at one budget and timed through `track` (#891). `search.potts_starts`: every ground-state solver as a start of the `opt.starts` seam on a size-tilted lattice, the energy an `Objective` over labellings and ICM the polish; at `potts_lattice/release`, q = 3, the graph cuts hand over the q = 2 sibling's exact optimum (#906). `search.icm`: iterated conditional modes and its minimum-sites floor, moved out of `search.alpha_expansion` (#1055); its `numba` kernel is `search.icm.numba`, the `<subpackage>.<backend>` layout (#1059). `search.trws`: the local-polytope lower bound on a Potts ground state by sequential tree-reweighted message passing, its Python reference the oracle of the `numba` kernel `search.trws.numba` (#1060).

**NNI and SPR: landed and counted.** Both neighbourhoods sit behind one
`Topology -> Iterator[Topology]` interface and are verified exhaustively
against `2(n - 3)` and `2(n - 3)(2n - 7)` at `n = 5..8` over every distinct
topology, with neighbour validity, symmetry and NNI-in-SPR containment
cross-checked ([#82](https://github.com/michaelJwilson/snakes_and_ladders/pull/82)).

**Hill climbing, with an oracle that settles the question.** `infer` climbs
over either neighbourhood, fitting the continuous parameters of every candidate
([#127](https://github.com/michaelJwilson/snakes_and_ladders/pull/127)). Exhaustive
enumeration of unrooted topologies gives search quality an independent
reference below 8 taxa
([#128](https://github.com/michaelJwilson/snakes_and_ladders/pull/128)): on the 6-taxon
fixture both move sets reach the enumerated maximum and recover the generating
topology from all 12 starting points, at a median of 14 candidate fits for NNI
against 48 for SPR. Budgets are counted in candidate fits rather than seconds,
so a run reproduces from its seed, and a topology is scored at most once per
search, keyed on its leaf bipartitions. One candidate fit measures 213 ms
against 22 us to generate an entire NNI neighbourhood.

**A discrete result now states how sure it is**
([#270](https://github.com/michaelJwilson/snakes_and_ladders/issues/270)).
`search.support` reports three quantities and names which: the
*neighbourhood* weight of the returned tree among itself and its neighbours
under a move set, with the margin over the best neighbour; the *enumerated*
weight over every topology, the exact flat-prior posterior over maximized
likelihoods where `(2n-5)!!` fits; and Felsenstein's *bootstrap*, per internal
split, over site-resampled searches. The first is held to the second: equal to
1e-9 at four taxa under NNI, where the neighbourhood is the whole space, and at
five taxa never smaller than it on any of the 15 topologies, with the best
tree's NNI margin equal to its enumerated margin. The bootstrap is held to its
definition — a frequency over the returned topology's internal splits,
reproducible from its generator — and gives the generating splits support of at
least 0.8 at 1,000 sites. The exact weight is calibrated: over 24 NNI searches
on the five-taxon fixture at 30 to 300 sites, the fraction of returned trees
equal to the generating one does not fall from one support bin to the next.
Each weight is over *maximized* likelihoods under a flat prior over topologies
and is named so, not called a posterior.

**The same weights serve labellings and decodings, and a tempered ensemble
gives a marginal one**
([#331](https://github.com/michaelJwilson/snakes_and_ladders/issues/331)).
`neighbourhood_labelling_support` and `enumerated_labelling_support` take any
factor graph, the neighbourhood every single-site change, so a Potts
configuration and a hidden path are one case: the enumerated weight equals
`enumerate_potts`'s Boltzmann weight at `beta = 1` on all 729 labellings of a
three-state 3 x 2 lattice to 1e-12, and the path posterior of the ambiguous
chain's decodings, where the Viterbi path's margin is 0.3033 nats and the
posterior-decoded path's -0.6066. The single-site neighbourhood is the whole
space only where one site is free — on a two-site chain it reaches 5 of 9
labellings — and there the two weights agree to 1e-12. The bootstrap stays a
tree quantity: it resamples sites, which a chain's ordered sites do not license
and a labelling does not have. `sample.tempered` runs replica exchange from the
moves of #309 on the ladder of #267 with `parallel_tempering`'s exchange ratio,
over labellings and over topologies; the fraction of the temperature-one
replica's sweeps at a structure is its tempered weight (`eq:tempered-weight`),
held to enumeration over 20 seeds on the ladder (1, 2, 4) at 1,000 sweeps after
100 of burn-in. The largest single-seed deviation is 0.039 on the four-taxon
fixture's three topologies (30 sites, weights 0.66, 0.17, 0.17), 0.024 on the 2
x 2 lattice's ground state and one-flip excitation (0.68, 0.05) and 0.031 on
the ambiguous chain's two decodings (0.14, 0.08); the mean over seeds is within
0.004 on every instance; asserted at 0.06 per seed and 0.01 on the mean. Its
diagnostics are asserted too: exchange acceptance per adjacent pair 0.82-0.93
on the topologies, 0.53-0.82 on the lattice and 0.71-0.85 on the chain, and no
indicator's autocorrelation time above 0.74 recorded sweeps. Over topologies
the tempered weight is a marginal of the *fitted* likelihood, not over branch
lengths. Calibration is re-measured at seven and eight taxa behind the release
gate, for the quantities a search there can afford — the NNI neighbourhood
weight and the bootstrap's smallest split support, 8 replicates — over 16 NNI
searches per taxon count at 50 to 400 sites; the fraction of returned trees
equal to the generating one per support bin `(0, 0.5]`, `(0.5, 0.9]`, `(0.9,
1]`:

The per-bin table was left as a placeholder when #350 merged: the
release-gated test that produces it (about 40 minutes at seven and eight taxa)
was not run at the merge. The table is transcribed here at the next
release-gate run.

**The coupled model is fitted, and the finding is about the start, not the
move** ([#306](https://github.com/michaelJwilson/snakes_and_ladders/issues/306),
#290 parts 3 to 6). `search.spatio_sequential` runs block-coordinate ascent
on `log p(x, l | theta)` with the chains marginalized: an E step per class,
an M step through each family's `reestimate` with `Pi_m` and the shared `t`
in closed form, and a label block that is the ground state of a Potts model
in the external field the posterior defines — by alpha expansion, by
single-site descent, or by an annealed Wolff move whose best visited labelling
is taken only when it improves the joint. The M-step identity holds through
autograd on a Gaussian instance to 1e-10 relative; the joint is non-decreasing
across every block for every solver; and with the parameters at the truth the
label block reaches the enumerated MAP labelling from the planted labels on 5
of 6 draws of the canonical instance for every solver. Past enumeration, on a
planted 10x10 lattice with weak emissions, the label problem alone is easy —
0.98 accuracy up to permutation with the parameters known — and every cold
start freezes:

| start, then ten blocks | alpha expansion | single-site descent | annealed Wolff |
| --- | --- | --- | --- |
| uniform labels, true parameters | 0.66 | 0.78 | 0.68 |
| `Graph_BurnIn++` (thirty annealed steps), then alpha expansion | 0.97 | — | — |

The cluster move does not escape what descent freezes into; the trap is the
parameters, which a cold EM collapses before the labels can separate them.
The annealed start — labels nearly free while the emissions are fitted to
what the data alone supports, the prior tightening as the classes separate,
seeded by `Emission_Mixture++` (k-means++ under the family's own negative
log-density) — is what recovers the labels. **The escape claim is retracted
for this instance.**

**One Gibbs sampler and one annealer serve every problem, over the factor
graph** ([#309](https://github.com/michaelJwilson/snakes_and_ladders/issues/309)).
`sample.gibbs` runs a heat-bath sweep over any `FactorGraph` — a variable's
conditional is the product of the factors touching it — tempered by a
schedule for annealing, with an exact block move for a chain-shaped subset
by forward filter and backward sample, and a Metropolis move over tree
topologies on the fitted likelihood. Each instance is held to the
distribution it targets by chi-square at 0.001: the 2x2 Potts lattice in a
field against enumeration; the five-step chain against the enumerated path
posterior, for the single-site sweep and for the block move; the four-taxon
tree at one site against sum-product's exact marginals; the coupled model's
labels against the enumerated posterior. The sweep copies the Potts kernel's
arithmetic and agrees with it draw for draw on 2,000 of 2,000 sweeps;
annealing reaches the triangular antiferromagnet's closed-form ground state on
6 of 6 seeds, as `anneal_potts` does at the same schedule. At temperature one
the topology move's visits
over the 15 five-taxon topologies match the flat-prior weight over fitted
likelihoods, the quantity #270 enumerates, and annealed to 0.02 it reaches
the enumerated best from 6 of 6 random starts with every topology fitted
once. The price of generality was 1.3x against the Python Potts sweep and 3.2x
against the Rust one; since #561 compiled the sweep it is a discount.
Re-measured together on one quiet host, twenty sweeps of an 8x8 three-state
lattice take 0.202 ms over the factor graph's edge layout and 24.53 ms over
its factor tables in NumPy, against 20.52 ms and 7.90 ms for twenty steps of
the Python and Rust Potts annealers, which carry an energy evaluation per
step (medians, `tests/benchmarks/test_gibbs_bench.py`). The specialised
kernels stay the default for the Potts lattice; whether they should is a
measurement to take rather than a change made here.

**What carries between neighbours, and what does not.** A branch is identified
by the leaf split it induces rather than by a node name, so a neighbour sharing
all but a few branches with its parent starts its fit from the parent's lengths
([#289](https://github.com/michaelJwilson/snakes_and_ladders/issues/289)). The
warm fit reaches the cold optimum — worst relative gap 5.3e-12 in
log-likelihood over the 90 SPR neighbours of an eight-taxon tree — and the
search's answer does not move. What it saves is smaller than the ticket hoped,
and in one measurement negative: over those 90 neighbours the warm fits spent
6,053 likelihood evaluations against 5,145 cold, a single neighbour fit costs
the same 49 either way, and only a refit of the *same* topology from its own
lengths drops to 14 — L-BFGS spends its evaluations on the branches the move
changed. The larger saving is lazy scoring: one cached likelihood evaluation at
the warm lengths ranks a neighbourhood and only the top `K` candidates are
fitted, the accepted move always in full. Subtree partials are cached on the
subtree's structure and lengths, and a cached partial equals a recomputed one
bitwise. Over four
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
neighbourhoods. Warm starts are the default, `lazy_top` is opt-in with `K` chosen per move set
from this table, and `Inference` reports fits and likelihood evaluations beside
the budget. RAxML's three-branch local optimization is not built; its gap to
the full optimum is the measurement that would license it.

**The accuracy requirement's first half is met.** Normalized Robinson-Foulds
distance from the inferred to the generating topology is met at the 0.05 bound
from 125 sites upward, with 8 of 8 replicates recovering the topology exactly at
2000 sites against 5 of 8 at 60
([#148](https://github.com/michaelJwilson/snakes_and_ladders/pull/148)). The normalizer
counts internal splits only: including the trivial ones would shrink every
distance by a taxon-count-dependent factor and silently weaken the bound.

**Potts cluster updates landed, validated by the distribution they converge
to** ([#212](https://github.com/michaelJwilson/snakes_and_ladders/pull/212)). Swendsen-Wang
and Wolff run beside single-site heat bath behind one interface, checked by
chi-square goodness-of-fit against the exact Boltzmann distribution at an
enumerable size, at a significance of 0.001, for all three move sets with and
without an external field; the worst p-value over 36 runs spanning six seeds
was 0.0145. The test's power is itself pinned: replacing the field accept step
with an unconditional recolouring is rejected at p = 0.0.

Two errors a test asserting only that the chain ran would have missed. Wolff's
cluster construction alone does not preserve detailed balance in a field:
without the Metropolis correction on `|C| * (h_new - h_old)` the sampler
converges to the wrong distribution. And a Wolff sweep sized by running
clusters until their cumulative size reached the site count --- a
state-dependent stopping rule --- biased an aligned two-site chain to 0.384 per
aligned state against an exact 0.334.

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
These lattices are small and their boundary open, both of which soften the
transition. Recorded as
`docs/experiments/001-potts-cluster-autocorrelation.md`.

**Two gradient-informed proposals landed, and on this energy they are one
kernel** ([#756](https://github.com/michaelJwilson/snakes_and_ladders/issues/756)).
`PottsMove.LOCALLY_BALANCED` weights every single-flip change by
`sqrt(pi(s') / pi(s))` (Zanella 2020); `PottsMove.GIBBS_WITH_GRADIENTS`
weights it by the same function of the first-order Taylor estimate of that
ratio at the one-hot state (Grathwohl et al. 2021). The estimate **is** the
ratio here: the relaxed log weight is affine in each site's row, so a
single-flip change carries no second-order term. Pinned three ways at
`1e-12` — the heat bath's own `heat_bath_log_weights`, the tape's gradient
through `torch.autograd.grad`, and the enumerated energy of every flipped
configuration — so the coincidence is refereed rather than assumed. Both
leave the exact Boltzmann law invariant at the enumerable sizes: chi-square
against enumeration at a significance of 0.001, realized
0.0157 to 0.9926 over two seeds on the 2x2, the 3x3 and the frustrated
triangular instance, the two move sets agreeing to four figures on every one
of them. Dropping the Metropolis correction, which leaves the chain
stationary at `pi(s) Z(s)`, is rejected at p = 0.0.
The same two proposals run over the factor graph as
`GibbsMove.LOCALLY_BALANCED` and `GibbsMove.GIBBS_WITH_GRADIENTS`, where the
estimate is exact for the same reason and against the same referee
(p = 0.2208 and 0.0483 over two seeds).

**They halve the sweeps and pay eight times as much to take one.** Energy
autocorrelation time at the exact transition on a 16x16 open lattice, two
readings each, a sweep being `n_nodes` updates for all three, at a 1-minute
load of 0.96 to 1.09:

| 4,000 sweeps, 400 burn-in | single-site | locally balanced | Gibbs with gradients |
| --- | --- | --- | --- |
| tau in sweeps, seed 0 | 12.93 | 5.12 | 5.12 |
| tau in sweeps, seed 1 | 8.07 | 4.79 | 4.79 |
| wall, NumPy sweep | 9.3 s | 75.7 s | 109.3 s |

An independent energy sample costs 4.96 sweeps against the heat bath's 10.50,
**2.1x fewer**, at **8.1x** the wall per sweep against the NumPy heat bath and
250x against the Rust sweep that is the default --- so 3.9x behind at equal
wall clock, and 380x behind the shipped sampler. Recorded as
`docs/experiments/021-potts-gradient-proposals.md`;
[#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754) owns
the port that would close it.

**Two cluster moves landed for the frustrated lattice, and both percolate on
it** ([#756](https://github.com/michaelJwilson/snakes_and_ladders/issues/756)).
`PottsMove.NIEDERMAYER` activates a bond on its energy relative to a threshold
`E_0` (Niedermayer 1988) rather than on its endpoints agreeing, so it runs on a
coupling of either sign --- the instance `sample_potts` refuses Wolff and
Swendsen-Wang on. At `niedermayer_threshold`'s value it **is** Wolff on a
ferromagnet: the same bonds, the same field accept step, and the same labelling
to the last bit over 900 draws at three temperatures. Below that value the
boundary terms of its ratio survive and the move interpolates down to a
single-site Metropolis flip, which is pinned against `energies` on the flipped
configuration. `sample_potts_pair` and `tempered_potts_pair` carry Houdayer's
isoenergetic move (2001): two replicas at one temperature exchange labels on a
component of the region where they disagree, which leaves `E(s) + E(s')` where
it found it --- worst `|dE|` 3.6e-15 over 500 drawn pairs --- so the acceptance
is 1 by an identity. Both leave the exact Boltzmann law invariant at the
enumerable sizes: chi-square against enumeration at a significance of 0.001,
realized 0.0221 to 0.9945 over two seeds on the 3x3 open square and the 3x3
periodic triangular antiferromagnet, and 0.0039 to 0.9913 per replica for the
pair. Two ablations are rejected at `p = 0.0`: the accept step dropped, and
Houdayer's component replaced by the single site it was seeded from. Where a
chain cannot referee the move --- mixed couplings, where the cluster is 8.94
sites of 9 and a chain is a global spin reversal --- the kernel's own flow
`pi(s) K(s, s')` is read against its transpose instead, 0.0016 against a Monte
Carlo error of 0.0089.

**Neither buys a round trip on this lattice.** Houdayer's cluster is 43 to 59
sites of a 65 to 72-site defect region on the 12x12 periodic triangular
antiferromagnet, so the move is a near-global exchange of the two replicas:

| 8 seeds, 2,000 sweeps, 10 rungs | without Houdayer | with Houdayer |
| --- | --- | --- |
| round-trip time, sweeps (12x12 triangular) | 1,115.5 | 1,090.7 |
| round-trip time, sweeps (60-site planted glass) | 955.9 | 772.3 |
| wall, 8 seeds | 17.4 s | 67.8 s |

The paired sign test reads `p = 0.6875` on the lattice and `p = 0.2891` on the
glass, so no direction is established in either sense, at 3.90x and 4.01x the
wall over two readings. Recorded as
`docs/experiments/022-cluster-moves-for-frustrated-lattices.md`. The moves are
exact and cheap to have; what the measurement rejects is this instance, whose
overlap percolates, and not the construction. `MoveKind.NIEDERMAYER` is an arm
`PottsNDEnvironment` can select beside Wolff and Swendsen-Wang, through
`cluster_moves`; Houdayer's move is not one and cannot be, an arm's action
carrying one labelling to one labelling where his carries a pair.

**`log Z` landed for the lattice, estimated and with its error**
([#756](https://github.com/michaelJwilson/snakes_and_ladders/issues/756)).
`sample.annealed` carries three estimators on one ladder of inverse
temperatures, from `beta = 0` --- where `log Z_0 = n log q` is exact, and is
returned bitwise --- to the target, each stepping the shipped kernel once per
rung so every move set is admissible. `annealed_importance_sampling` (Neal
2001) weights independent annealing runs; `population_annealing` (Hukushima &
Iba 2003) resamples a population by the same weights and accumulates the
per-rung normalizers; with the resampling off the second **is** the first, the
two agreeing to 1.8e-15 on the same trajectories. Both recover
`strip_log_partition` at strip widths 4, 6 and 8 and the enumerated `log Z` on
the two 3x3 lattices, every deviation inside three of its own standard error
over two seeds --- worst 1.63 --- at an ESS of 59 to 110 of 128. Two ablations:
dropping the importance weight leaves `mean(log w)`, Jensen's lower bound,
**7.9 standard errors low** on a six-rung ladder, and multinomial resampling in
place of systematic leaves a family entropy of 0.19 to 0.80 nats against 4.01
to 4.17 of a possible 4.85 --- two effective families of 128, which is why the
default is systematic. `simulated_tempering` (Marinari & Parisi 1992) samples
the rung as a variable on the weights `g_k = -log Z_k` those runs estimate: its
configurations are the enumerated Boltzmann law at every rung by chi-square at
a significance of 0.001, realized 0.0138 to 0.7028, and its rung occupation is
the one the weights predict, 0.1907 to 0.7083 under exact weights and 0.2765 to
0.3747 under a pilot's.

**A tempering ladder can now be placed by its round trips, and that does not
beat placing it by its acceptance.** `sample.schedule.adapt_ladder_by_round_trips`
redistributes a ladder of fixed length so the local diffusivity is flat
(Katzgraber et al. 2006), from the fraction of walkers moving up at each rung;
`sample.tempered.up_fraction` measures it off the walker trace
`parallel_tempering` now records, and `adapt_ladder_round_trips` is the Potts
binding. The acceptance-placed ladder stays the default and nothing changes for
a caller that does not ask. Over 8 seeds paired by seed at 2,000 recorded
sweeps:

| round-trip time, sweeps | acceptance-placed | round-trip-placed |
| --- | --- | --- |
| 16x16 open square at `J_c`, 12 rungs | 1,194.4 | 1,053.7 |
| 12x12 periodic triangular, 9 rungs | 360.1 | 366.6 |

The paired sign test reads `p = 0.2891` and `p = 0.7266`, and a second reading
at another warm-up seed `p = 1.0` and `p = 0.7266`, so no direction is
established either way. What the readings do establish is the warm-up: both
placements beat the geometric ladder they start from, 1,279.7 against 1,690.3
on the 16x16, and a feedback placement read from 400 sweeps instead of 1,000
ranges from 1,469.5 to 5,357.1 --- 3.2x worse than geometric --- because `f` is
then read from its own noise. Recorded as
`docs/experiments/023-placing-the-tempering-ladder.md`.

**The two baselines every Hamiltonian number is read against landed, and they
beat the trajectory at equal evaluations**
([#756](https://github.com/michaelJwilson/snakes_and_ladders/issues/756)).
`sample.langevin.mala` is `sample.hmc.sample` at one leapfrog step (Roberts &
Tweedie 1996), written as the two Gaussian transition densities rather than as
a trajectory, so the identity is measured between two implementations: over
400 proposals at step sizes 0.3, 0.7 and 1.0 the draws agree to 2.7e-15, the
energy error to 1.4e-14 and every accept/reject decision exactly, against a
declared 1e-12. It adapts through the same two warm-up windows toward Roberts
& Rosenthal's 0.574 rather than HMC's 0.65 --- the drawn chain's acceptance is
0.537 pooled over four seeds --- and carries an opt-in uncorrected route whose
bias is reported and never argued away: on a unit Gaussian at a step of 1.0
its variance is 1.338 against the closed-form `s^2 / (1 - h^2 / (4 s^2))` of
1.333, 0.14 standard errors, and against the target's 1.000, 11.09 of them,
while the corrected chain lands at 1.45 and the acceptance rate reads 1.00 for
the biased chain and 0.927 for the correct one. `sample.slice.slice_sample` is
Neal's (2003) stepping-out and shrinkage, coordinate-wise and hit-and-run,
with no tuning beyond an initial width; it reports objective evaluations
because it spends no gradient, and its shrinkage is ablated on cost rather
than on bias --- the rejection form reaches the 100-candidate refusal at
widths 2.0, 10.0 and 40.0 where the shrinkage holds a sweep to 11.3, 12.6 and
16.0 evaluations. Both recover the analytic Gaussian inside three of their own
ESS-corrected standard errors over two seeds (MALA worst 0.87 on a mean and
0.47 on a variance; slice 0.64 and 1.66) and the enumerated assignment
posterior inside four (MALA 1.15, slice 1.62), which puts four new rungs on
the mixture ladder.

Effective samples per 1,000 evaluations, worst coordinate, two seeds, each
sampler at its own warm-up, at a 1-minute load of 3.90 to 4.32 on a 4-core
host shared with two other agents:

| per 1,000 evaluations, seeds 1 / 2 | HMC (gradients) | MALA (gradients) | slice (objective) |
| --- | --- | --- | --- |
| analytic Gaussian, 2,000 draws | 25.3 / 22.0 | 59.1 / 71.8 | 30.8 / 29.4 |
| analytic Gaussian, 8,000 draws | 27.8 / 30.5 | 53.0 / 54.7 | 31.4 / 30.3 |
| mixture posterior, 800 draws | 19.4 / 20.5 | 191.8 / 166.5 | 100.8 / 116.5 |
| mixture posterior, 3,200 draws | 38.8 / 30.2 | 196.2 / 235.0 | 125.2 / 115.1 |
| wall, 8,000 / 3,200 draws | 17.5 s / 15.1 s | 3.8 s / 3.8 s | 2.6 s / 2.5 s |

The trajectory is not paid for at these dimensions: MALA takes 1.7x to 2.6x
HMC's samples per gradient on the two-coordinate Gaussian and 5.1x to 7.8x on
the one-coordinate mixture posterior, where ten steps retrace a line already
crossed. A gradient is a forward evaluation and a backward pass, so slice
sampling's column is not HMC's column and the ranking is settled on the wall
instead, where it is the same. Recorded as
`docs/experiments/026-what-a-draw-costs-in-gradients.md`.

**An exact ground state landed, and it is the repository's first optimum that
is proved rather than enumerated.** For two states with every coupling
non-negative the Ising energy is submodular, so a minimum cut finds its global
minimum in polynomial time. Every other discrete claim here rests on
exhaustive enumeration and stops at about twenty sites; this does not.

Validated three ways: against enumeration where it fits, at **exact equality**
over 36 shape-coupling-field combinations; against two analytic corners far
past it — zero field gives an aligned state at `-J |E|`, zero coupling gives
`argmax` per site; and by the max-flow min-cut theorem as a self-check, the
flow value equalling the capacity of the cut residual reachability induces.

A Rust kernel (`src/maxflow.rs`) runs **26-32x** faster than the NumPy
reference measured on its own, and **6.3-10.7x** as a caller sees it. #220
attributed the difference to the Python lists crossing the FFI boundary by copy
and deferred the fix;
[#336](https://github.com/michaelJwilson/snakes_and_ladders/issues/336) applied
it, passing `float64` and `int64` buffers through `rust-numpy` and returning
the configuration as an array, and measured that the copy was not the term.
Minimum of 20 or more rounds, in ms, on square lattices with a random per-node
field:

| Extent | NumPy reference | Kernel, lists (#220) | Kernel, buffers (#336) | Caller (#220) | Caller (#336) | `energy()` |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 4.47 | 0.142 | 0.138 | 0.733 | 0.711 | 0.575 |
| 32 | 22.0 | 0.835 | 0.836 | 3.30 | 3.22 | 2.60 |
| 64 | 180 | 7.15 | 6.93 | 17.3 | 16.9 | 9.94 |

The boundary copy was 0.03-0.2 ms of a 0.7-17 ms call and removing it moved the
caller-visible number by under 3%. What a caller pays for is
`sal.search.maxflow.energy`, which scores the returned
configuration edge by edge in Python and is 59-81% of the wrapper's time; it is
the oracle's function and is left as it is. Output is unchanged: the
configuration is equal element by element to the reference's and to the
previous binding's at extents 16, 32 and 64, and the energy is bitwise equal.
The reference stays the oracle. The port also removes a fragility: the Python
blocking flow recurses to the depth of the level graph and needs
`setrecursionlimit` raised past a few thousand nodes, where the Rust one uses
an explicit stack.

The boundary is refused rather than approximated. A negative coupling is
NP-hard and raises; more than two states is alpha expansion (#207), which
takes this as its inner solver.

**Alpha expansion landed, with the repository's first proved approximation
bound.** `k`-state MAP is NP-hard, so the exact cut above stops at two labels;
alpha expansion recovers the general case as a sequence of binary cuts, each
handled by the exact solver unchanged. For a metric pairwise term its local
minimum is within `2 c_max / c_min` of the global one — exactly 2 for a uniform
Potts coupling, and a bound that holds at every size where belief propagation
reports a measured deviation, the samplers report a distribution, and
enumeration stops at nine sites. Measured at `3x3` with three labels over 40
runs, alpha expansion found the global optimum **39 times** and recovered
99.554% of the achievable improvement in the one miss — far inside the bound,
which is not tight.

The move set earns its complexity past enumeration. At `3x3` alpha expansion
and single-site descent are indistinguishable, both finding the optimum in 31
of 32 runs between them; at `8x8` with four labels, expansion beat the best of
eight single-site descents on every trial, by 1.8 to 11.0 in energy.

**Two construction errors, and what caught them.** The first draft swapped the
cut's terminal capacities and mis-costed the auxiliary nodes. Neither broke
loudly — both produce a labelling that is merely worse — and both passed every
enumeration test at `3x3`. The **reduction** caught them: at two labels one
expansion is exact, so it must reproduce the minimum cut energy for energy,
and it was failing by up to 2.55.

**Ground-state recovery at 5,041 sites ranks the cut-based minimizers first,
and refutes the fixture's simulated truth as a referee for a ground state**
([#552](https://github.com/michaelJwilson/snakes_and_ladders/pull/552),
[`docs/experiments/009`](docs/experiments/011-potts-ground-state-recovery.md)).
Ten entries on `spatio_only` over three rungs, budget-matched through
`opt.budget.compare` at **2,083,260 site visits** --- 60 heat-bath sweeps of
`n_nodes + 2 |E|` --- over 8 independent starts. The unit is site visits and
not sweeps: a Wolff step flips one cluster while a heat-bath sweep touches
every site, and equal sweeps would hand the cluster moves a free lattice per
move.

Rung 2 is 5,041 sites at q = 2, where the ferromagnet in an arbitrary
per-site field is submodular and `search.maxflow.ising_ground_state` is
**exact**, so every gap below is a measured gap:

| method | best of 8 | median | gap to exact | tilt | spend |
| --- | --- | --- | --- | --- | --- |
| alpha-expansion | **-10454.1563** | -10454.1563 | **0** | 0.2151 | 138,884 |
| alpha-beta-swap | **-10454.1563** | -10454.1563 | **0** | 0.2151 | 69,442 |
| anneal | -10235.13 | -10165.77 | 219.03 | 0.5334 | 2,083,260 |
| tempering | -10012.21 | -9916.33 | 441.95 | 0.5848 | 2,083,260 |
| icm / gibbs-T0 | -9794.98 / -9681.92 | -9723.81 / -9592.21 | 659.17 / 772.24 | 0.5619 / 0.6290 | 2,083,260 |
| swendsen-wang | -9532.56 | -9465.97 | 921.59 | 0.7361 | 2,083,260 |
| wolff | -7641.53 | -6898.83 | 2812.63 | 0.0082 | 317,358 |
| greedy | -7418.10 | -7418.10 | 3036.06 | 1.8000 | 5,041 |
| max-product | no convergence in 60 flooding iterations | --- | --- | --- | 2,083,260 |

Both cut-based minimizers reach the exact optimum on every one of the 8
starts, at 3.3% and 6.7% of the budget; no sampler comes within 219 of it at
30x the spend. ICM and Gibbs at `T = 0` are reported on one axis because at
`T -> 0` the heat bath is the argmin over each site's conditional, which is
ICM's update: they differ in sweep order and the 112.9 between them is that
order's whole effect.

Rung 3 is the same lattice at q = 10, where there is no exact energy. The
expansion's factor-2 bound is stated on the **non-negative** form of the
energy, since this repository's is negative and the bound is false as
written on a negative quantity: shifted by
`sum_n max_m h[n, m] + J |E|` both terms are non-negative and the bound is
carried back through the same shift. The bracket is
**[-11521.58, -10454.16], width 1067.42**, against a winner-to-second gap of
540.60 --- so **the bracket is wider than the differences being ranked and
the energy ranks nothing at this rung**, which is the outcome, not a caveat.

The structural referee does not rescue it; it refutes its own premise. The
exact q = 2 ground state's size tilt is **0.2151**, *below* the fixture's
thermal 0.4935 rather than above it, and its occupancy is **3286/1755**
against the recorded 415-593. Both differences have one cause: a ferromagnet
orders as `T` falls, domain walls stop being affordable, and the majority
class takes sites whose own field points the other way. A temperature sweep
of the same instance measures the tilt rising to 0.6255 at `T = 1` and then
falling --- 0.5009 at 0.7, 0.4695 at 0.5, 0.2151 at zero --- so the tilt is
not monotone in temperature and a ground state cannot be checked against a
thermal constant at all. The cut that scores 0.2151 reproduces the
enumerated ground state exactly at nine sites, so the referee is wrong here
and the cut is not.

At q = 10 the best labelling any method found uses **2 of the 10 classes**
and is element-for-element the exact q = 2 ground state: only the ladder's
extremes are ever field-optimal, so the middle eight classes buy field at no
site and cost agreement at every one. The method that best matches the
fixture's per-class occupancy is Wolff, at 484-567 against the recorded
415-593 --- and Wolff is **last** on energy at -1141.56, because it barely
moved. All three of the fixture's structural statistics are
finite-temperature statistics of a sampler, and this experiment is the
record that they do not transfer to a minimizer.

**The cluster moves lose, and the instrumentation says exactly why**
(issue #551, Step 3). Per schedule step over 60 steps from `T = 2.0` to
`T = 0.05`, the Fortuin-Kasteleyn construction being exact only at zero
field and this fixture having one:

| `T` | SW mean / max cluster | SW accept | SW spanning | Wolff mean cluster | Wolff accept | Wolff spanning |
| --- | --- | --- | --- | --- | --- | --- |
| 2.000 | 2.30 / 111 | 0.7266 | 0.000 | 3.67 | 1.0000 | 0.000 |
| 0.944 | 12.42 / 1459 | 0.4171 | 0.001 | 11.67 | 0.7500 | 0.000 |
| 0.446 | 46.60 / 1733 | 0.0878 | 0.009 | 58.50 | 0.2500 | 0.000 |
| 0.211 | 62.23 / 1739 | 0.0043 | 0.012 | 770.17 | 0.2000 | 0.500 |
| 0.068 | 64.63 / 1739 | **0.0000** | 0.013 | 1193.50 | **0.0000** | 0.833 |

At q = 2 the prediction holds in full and compounds to a standstill: the mean
cluster grows 28x for Swendsen-Wang and 325x for Wolff while both accept
rates reach **exactly zero** below `T = 0.145`, so the cold sweeps that
should be doing the optimizing propose global moves that are never taken.
Wolff's clusters span the lattice in 83.3% of steps at the coldest
temperature and none of them is accepted. At q = 10 the same shape is milder
--- Swendsen-Wang's mean cluster grows 1.15 to 27.47 and its accept rate
falls 0.893 to 0.057 --- and Wolff degenerates instead: its cluster stays at
1.2 to 2.0 sites at every temperature, so a Wolff step is a single-site move
with a bond construction's overhead and it spends **618 of 2,083,260** site
visits. A Wolff run matched on visits rather than steps would need about
29,000 steps at this rung, which the per-PR budget does not buy; the entry is
reported at equal steps with its underspend stated, because stopping a sweep
when a budget is exhausted is a stop on the state and `search/CLAUDE.md`
refuses it.

Rung 1, `spatio_only/ci` at 9 sites and q = 3, is a **correctness pin and not
a comparison**: its enumerated ground state is 8 sites in one class and 1 in
another, class 2 unused, and all ten entries reach it. `search/CLAUDE.md`'s
"a fixture whose answer is trivial measures nothing" applies, and it is kept
for what it does referee --- that the graph cut, both its implementations,
alpha expansion and the new alpha-beta swap agree with exhaustive
enumeration exactly.

No wall clock is claimed. Every run was made with eight agents on a 4-core
host, verified busy with `ps` rather than quiet, so the seconds column of
`docs/experiments/009` is recorded and not used; the ranking rests on
energies, spends and counts, none of which move with load.


**Max-Cut landed as the other side of the same model.** Maximizing the weight
of separated edges *is* minimizing the energy with every coupling negative,
the NP-hard side of the boundary the exact cut refuses to cross. A
Goemans-Williamson relaxation gives a run a certificate where
enumeration cannot reach: measured on random graphs with triangles at 12, 16
and 18 nodes the rounded cut reached the enumerated optimum **every time**,
and the computable certificate `value / relaxation` came out 0.95 to 0.98
against a guarantee of 0.87856.

**The certificate is weaker than the theorem, and the repository says so.**
Goemans-Williamson assumes the semidefinite program is solved to optimality;
there is no SDP solver here, so it is solved approximately by Burer-Monteiro
gradient ascent in `torch`. The value returned can therefore sit *below* the
relaxation's optimum, making a ratio measured against it optimistic. That is
asserted rather than glossed: on a complete bipartite graph, whose maximum cut
is exactly `|E|`, the ratio comes out slightly **above 1** — impossible for an
exact solve.

**Simulated bifurcation is the first kernel the GPU rule can be read against, and on CPU it reads 1.9x** ([#823](https://github.com/michaelJwilson/snakes_and_ladders/issues/823)). `search.bifurcation.simulated_bifurcation` relaxes each site and label to an oscillator, integrates every one at once, and reads the labelling off the arg max; the same arithmetic runs on NumPy and on `torch` (`Backend.TORCH`, new), the two labellings pinned equal. Where couplings decide it wins: one replica reaches the declared 18-node glass's enumerated ground state from **0.562** of 32 seeds against a single descent's recorded 0.079, and the frustrated 3x3 antiferromagnet's from 0.938 of 16 at two states and 0.438 at three. Where a per-site field decides it loses: on a 4x4 lattice with `N(0, 1)` fields it reaches the cut's optimum on 1 of 6 fields at a mean gap of 1.09 against ICM's 0.089, which is the regime the textbook states rather than a number tuned away. The force scale is read from the instance --- one half over the largest drive a site can see --- because Goto's dense-matrix scale left the field as the whole drive and returned the field-only labelling (mean gap 3.24). At 5,041 sites and three states the torch path is **0.29 us per site-step against NumPy's 0.58**, 1.9x on the 4-core CPU host; no device is attached to this host, so the >=10x rule is not yet read and the kernel is admitted as the candidate, not the result.

**Temperature is one object, and it lives where all three consumers can reach
it.** `sal.sample.schedule` carries the schedules — constant, linear,
geometric, cosine, each mirroring its `torch.optim.lr_scheduler` counterpart
and checked against it to 1e-12 (1e-10 for the cosine, whose torch form is a
recursion) — with both endpoints reached *exactly* at the declared steps, and
a step past the end refused rather than clamped
([#267](https://github.com/michaelJwilson/snakes_and_ladders/issues/267)). The
Potts sampler takes a temperature as model scaling: the tempered energies equal
the energies over `T` with a deviation of **0.0**, and every move set's chain at
`T = 2` and `T = 0.5` in a field passes the chi-square against `exp(-E/T)`
enumerated from the unscaled model (p-values 0.016 to 0.89 at the 0.001
significance). The Hamiltonian sampler takes it as the momentum's variance —
the tempered dynamics are the untempered ones in rescaled time — and on the
analytic Gaussian a chain at `T` is the chain at 1 with its deviations scaled
by `sqrt(T)` **draw for draw to 1e-10**. At `T = 1` every operation is the
identity bitwise, and the 31 existing HMC and 13 Potts tests pass untouched.

**Annealing is the sampler on a schedule, and the first instance is a wash.**
`anneal_potts` and `hmc.anneal` run one sweep or one proposal per schedule step
and return the best state seen. On the 9×9 periodic triangular antiferromagnet,
whose ground-state energy is a closed form, geometric annealing from `T = 2` to
`0.05` over 200 sweeps reaches it **20/20** against single-site descent's
**2/20** and a constant `T = 1` control's **7/20** — the schedule, not the
wandering. Descent converges in 2.6 sweeps, so the same 200 sweeps buy 78
restarts, and the best of 78 also reaches it 20/20. On Rastrigin, measured at
equal *objective evaluations* with a counting wrapper: at 14,400 evaluations
annealed Hamiltonian proposals plus a polishing fit reach the global basin
**6/20**, and 101 random-restart fits on the same budget **10/20**; at 2,900
evaluations it is 0/20 against 1/20. Restarts win on the continuous surface;
neither is a default.

**Parallel tempering, and the instance where restarts lose.** Replicas at fixed
temperatures exchange configurations on `(β_i − β_j)(E_i − E_j)`, each replica
on its own spawned generator from one seed. With exchanges on, every replica
passes the chi-square against `exp(-E/T_r)` enumerated from the unscaled model
(p 0.024 to 0.70, exchange acceptance 0.78 and 0.57), and the paired negative
case — an exchange that omits the energy term — is caught at p = 0.0 on every
replica. The comparison, at **400 sweeps per method** on the planted Viana–Bray
spin glass, against the best energy any method found over 12 instances:

| instance | restarts of descent (100 × ≤4 sweeps) | annealing (1 × 400) | tempering (4 × 100) |
| --- | --- | --- | --- |
| 60 sites, degree 4, frustration 0.2 | 5/12, mean gap 0.75 | **12/12** | **12/12** |
| 60 sites, degree 4, frustration 0.35 | 5/12, gap 1.00 | 9/12, gap 0.50 | 9/12, gap 0.25 |
| 100 sites, degree 6, frustration 0.3 | 2/12, gap 2.58 | 7/12, gap 1.08 | **8/12**, gap 0.50 |

Every method beats the planted energy on every instance, as frustration
predicts. **The plan's prediction that tempering would be hard to justify at
these sizes is retracted**: on the one class of instance the roadmap needs —
frustrated, past enumeration — the tempered methods beat restarts at equal
budget and tempering carries the smallest gap. The first row is now held equal
by `opt.budget.compare`
([#281](https://github.com/michaelJwilson/snakes_and_ladders/issues/281)):
with the utility's own streams and the best any method found as the
reference, tempering **12/12**, annealing 10/12 with a mean gap of 0.17, and
restarts of descent 4/12 with a mean gap of 0.75, every method at or below
the planted energy. The five-component mixture comparison is under Milestone 1.3 and in
[`docs/experiments/004`](docs/experiments/004-mixture-tempering-vs-restarts.md)
([#332](https://github.com/michaelJwilson/snakes_and_ladders/issues/332)).

**The single-site sweep has a Rust backend, beside the oracle.** Issue #232
profiled it as the one place a Python-level loop dominates -- one interpreter
iteration per site per sweep, five NumPy calls to move one spin. The port runs
**77x** the Python sweep at 64 nodes and **108x** at 1,024, and unlike the
pruning backend the ratio *rises* with size: the adjacency crosses once in
compressed-row form and the uniforms cross as one array, so there is no
per-element marshalling to grow against it.

Nothing switches to it. `f64::exp` agrees with NumPy's to within a unit in the
last place rather than exactly, and `searchsorted` is a threshold, so one draw
across a boundary that moved by 1 ulp sends the two chains apart permanently;
replacing the oracle would move every autocorrelation figure above, every
committed notebook output that reads a chain, and the goodness-of-fit fixtures'
chain lengths. Agreement is therefore distributional: the Rust chain is scored
against the exact enumerated Boltzmann distribution at 2x2, with a field and
without, at the significance and thinning the Python sweep is held to -- and a
chain drawn under no field is rejected against the with-field truth.

**Viterbi is built as max-product over the chain's factor graph**
([#296](https://github.com/michaelJwilson/snakes_and_ladders/pull/296)), returning
the enumerated Viterbi path with its joint on four chains, and posterior
decoding is forward–backward
([#307](https://github.com/michaelJwilson/snakes_and_ladders/pull/307)); the
canonical ambiguous chain reads both decodings off one enumeration and the
tempered ensemble of #331 weights them. **Not built:** iterated conditional
modes over HMM state paths (#176). Single-flip local search over the Potts
chain exists as an RL environment, not as a classical baseline suite.

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
the same fixture is release-gated at 50 s. The neighbourhoods separate at
eight taxa: over its 10,395 enumerated topologies at 1,000 sites
NNI reached the minimum from 9 of 12 random starts and SPR from 12 of 12, at
medians of 60 and 289 candidates. On the Felsenstein-zone fixture at 2,000
sites the search returns, from every start, the tree that groups the two
long branches at 1869 changes, where the generating tree scores 1982; maximum
likelihood on the same alignment puts the generating tree first by 12.99 log
units, and transition/transversion weighting does not move parsimony's
answer (3013 against 3273).

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
gap above the plug-in bound, so a poor fit falls back to the bound. A linear
model, a deep MLP, a Deep Sets model over branch tokens, a one-block
attention model and a graph network over the tree all reach held-out R^2 at
or above 0.999 on 15-topology neighbourhoods of the five-taxon fixture
(16 alignments, split 10/3/3 by alignment) and 0.98 on the two held-out SPR
neighbourhoods at eight taxa, ranking the fitted best first on every held-out
neighbourhood; the three token models return the same value for a tree with
its children shuffled, to 1e-13.

**Four max-flow kernels measured, one kept** ([#715](https://github.com/michaelJwilson/snakes_and_ladders/issues/715)). Boykov-Kolmogorov replaces Dinic as the package kernel behind `search.maxflow.rust`, and Dinic, highest-label push-relabel and a synchronous parallel push-relabel move to `sandbox.maxflow_declined` behind the `sandbox` Cargo feature, each still pinned to the package kernel's cut: every kernel certifies the source-reachable set of the residual graph, the minimal minimum cut every maximum flow shares, so 40 seeded networks agree arc for arc and the expansion gives the same labelling, cycle count and bitwise energy under each. Measured 2026-09-18 on the 4-core host, min of 5 rounds, ms, a random per-node field on an open lattice:

| kernel | 16x16 | 32x32 | 64x64 | 128x128 | 256x256 | exponent in `n` | 64x64 x 10-label expansion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Boykov-Kolmogorov** | 0.20 | 0.75 | 2.73 | 12.8 | **61.4** | **1.03** | **352** |
| Dinic (the kernel before) | 0.41 | 1.70 | 11.8 | 44.9 | 234 | 1.15 | 637 |
| push-relabel, highest label | 0.31 | 1.30 | 8.74 | 108 | 1,680 | 1.56 | 437 |
| parallel push-relabel | 2.16 | 3.74 | 16.1 | 287 | 339 | 1.04 | 2,036 |

3.8x on the 256x256 cut and 1.8x on the expansion, both at stress size. Push-relabel has the best generic worst case and the worst constant on a grid: its gap and global-relabel scans are `O(n)` each, and its last doubling reads 1.98. The parallel kernel's rounds are barrier-bound, 269 / 255 / 297 ms at 1 / 2 / 4 threads on 256x256, so no thread buys anything inside a cut; threads pay across cuts, where the batch entry `ising_ground_states` releases the GIL over eight independent 256x256 cuts, 2,158 to 900 ms from one thread to four (2.4x). One defect found by the move: the parallel kernel's global relabel kept a source-side node's running label, which on a network with no path to the sink (the expansion's first swap network, 81 nodes, flow 0) pinned every label at `n`, so none reached `n + 1` and pushed its excess back, and the round never ended. It is two-sided now, as the sequential kernel's, and the case is a test.

**ICM takes a minimum-sites floor** ([#1055](https://github.com/michaelJwilson/snakes_and_ladders/issues/1055), [#1056](https://github.com/michaelJwilson/snakes_and_ladders/pull/1056)). After each sweep a state holding `0 < count < min_sites` sites is dissolved, each site to `surviving[floor(u * m)]` from uniforms drawn up front. `min_sites = 0` reproduces the descent before, bitwise: `ground_state` `icm` and `icm-random` on both backends, `descend`, and the ICM label step and fit, seeds 0 to 4 at `spatio_only` ci and release, as sha256 digests recorded before the change. The `numba` kernel equals the Python oracle bitwise at floors 0, 1, 3 and 10 in both orders over six seeds, and releases the GIL. At `spatio_only/release`, 5,041 sites and q = 10, floor 50: 8.4 ms against 377.7 ms in Python, medians of 5, 45x.

**Hybrid solvers are compositions** ([#1077](https://github.com/michaelJwilson/snakes_and_ladders/issues/1077)). `opt.compose.Then` runs stages in order: each starts from the one before's answer, gets the budget the stages before it left less its own reserve, and draws from one generator. It replaces three hand-written Potts loops: the warm chain (`search.ground_state.warm`) and #1041's two expansion hybrids. The composed arms reproduce the pre-#1052 hashes of all 5 arms on `spatio_only/{ci,release}` bitwise (`test_ground_state_start.py`). Their medians over 7 seeds at `spatio_only/release`, 1,000 sweeps, are 490–560 ms, against 500–570 ms for the hand-written loops. `ground_state` reads a chain from its name, e.g. `field_argmax>descent>alpha-expansion`, and each chain is pinned bitwise to its parts run by hand (`test_ground_state_compose.py`). `run_descent` is the part that charges only the sweeps ICM ran, since `run_icm` charges its whole budget and leaves a later part nothing.

## Milestone 1.5 — Continuous Samplers, HMC & Parallel Tempering

**Landed, and one of its two integrators was declined on measurement.** A
Hamiltonian chain with leapfrog and with Yoshida's (1990) triple jump, step-size
adaptation, temperature schedules, simulated annealing, and a parallel-tempered
ensemble that runs over any of the supported problems. The detail sits under
Milestone 1.3 above, where the chain first appeared as a second, non-asymptotic
source of intervals; this section is where the roadmap's own numbering puts it.

**Yoshida is fourth-order and loses anyway**
([#266](https://github.com/michaelJwilson/snakes_and_ladders/issues/266)). The
convergence order was confirmed against its prediction, and the integrator was
still declined: it needs **91** gradient evaluations per trajectory to reach an
acceptance of 0.975 where leapfrog needs fewer, because the middle sub-step runs
backwards and costs stability rather than buying it. The order is a property of
the method; the cost is a property of this problem, and only the second decides
adoption.

## Milestone 2.0 — RL Definition

**Landed as `app:rl` in the textbook**
([#313](https://github.com/michaelJwilson/snakes_and_ladders/issues/313)). A
self-contained account of the algorithms a learned proposal is drawn from,
organised on the two axes that separate them --- whether the state space is a
table over discrete states or a parameterized function over a continuous one,
and whether an estimate is updated from a full observed return or by
bootstrapping from the estimate at the next state. It runs from an exact
framework needing a model of the environment to the methods that do not.

It is a definition milestone, so what makes it done is coverage rather than a
measurement: the section is cited from `sec:policy-gradient` at the point of
use, and every algorithm it names is either implemented in `learn/` or stated
there as not built. A named algorithm with neither is the defect this section
is checkable against.

## Milestone 2.1 — RL Agent Formulation & Deployment

**One table per problem, from one call, and the chain's five rows come back through it** ([#705](https://github.com/michaelJwilson/snakes_and_ladders/issues/705), steps 1-3). The rows existed and each was written for its own experiment: the Potts chain had all five, the tree three and the HMM path none, so a new problem cost a new harness and the rows were not read side by side. `learn.arena` is that harness, and it is a *list* of the five calls rather than a reimplementation — each learner reached through the registry returns **bitwise** what calling its function directly returns on the same four streams, asserted per learner. Through it the chain reads **65 / 72 / 72 / 78 / 79 of 81 starts** — 80.2 / 88.9 / 88.9 / 96.3 / 97.5%, #313's numbers to the start, under both scorings.

**Two things the harness found in the numbers it reproduces.** The MLP row's 97.5% depends on a hyper-parameter the published row carried silently: at `ppo`'s default step of 0.05 it reads 78 of 81 rather than 79, which is the entire margin between the MLP and linear rows, so `MLP_LEARNING_RATE = 0.01` is declared in the registry rather than defaulted. And greedy's cost is **17.4** scored actions an episode on the chain, not the 48 the planner comparison quotes: 48 is `max_steps x |actions|`, the budget, which hill climbing does not spend because it stops at a local maximum after 2.2 steps. The planner's 8.3 is in its own unit — simulated evaluations inside a tree search, which a trajectory count does not see — so #135's "8.3 against greedy's 48" compares two units and is not reproduced here. One unit for it is step 5's.

**`TopologyEnvironment` is `TreeEnvironment`,** on #644's rule that an environment is named for its problem; the problem is `tree` and its fixtures are `tree_jc`, `tree_scale` and `tree_search`. No behaviour moved. All three retired names — `PottsLandscape`, `StatePathLandscape`, `TopologyEnvironment` — are now refused by a guard over the whole repository rather than the package alone, since the old names survived longest in the suite and in a notebook cell, and `Topology` was the third spelling of one seam.
**A Potts environment whose field is per site, and whose action carries a temperature** ([#706](https://github.com/michaelJwilson/snakes_and_ladders/issues/706), the single-site arm). Every learned policy before this scored a field of shape `(n_states,)` — one global tilt per label — while `search.ground_state` and `search.maxflow` score `(n_nodes, n_states)`, so no policy had been measured on the instance the classical ground-state methods are ranked on. `PottsNDEnvironment.score` is now the negation of `sim.potts.energies` **bitwise** over 600 seeded labellings at three instances, which is what says the two callers score one problem.

**The classical baseline is a point in the action space, and which point it is was measured rather than assumed.** A sweep action at temperature zero is one iterated-conditional-modes sweep — each site taking its conditional mode in index order — and repeating it reproduces `iterated_conditional_modes`'s labelling **label for label** at 16 and 36 sites from the start that method draws for itself. The best single flip at zero temperature is a *different* baseline, steepest ascent: on the 6x6 fixture it reaches 61.87 from the field-only start where ICM's own random start reaches 57.33, which is two methods and two starts rather than one beating the other. Mistaking the second for the first is the defect the test caught.

**Both cluster arms are `sample.potts_mcmc`'s own moves, and each is its classical run** ([#706](https://github.com/michaelJwilson/snakes_and_ladders/issues/706), the Wolff and Swendsen-Wang controls). Above zero temperature the move delegates to `_wolff_sweep` or `_swendsen_wang_sweep` and its labelling matches a direct call **bitwise**, so no second cluster kernel exists to drift from an oracle; what the wrapper adds is the root and colour the action names, and the `T = 0` limit, where the bond probability is one, a cluster is a like-coloured connected component and the accept step keeps only a recolouring that does not lower the score. The arm walking `search.ground_state`'s declared exponential ladder is indistinguishable from the classical run over 32 seeds at 144 sites: Swendsen-Wang reaches -263.5 ± 5.2 against `run_swendsen_wang`'s -262.3 ± 5.8 at an identical 134,400 site visits (Welch p = 0.38), and Wolff -203.6 ± 13.2 against -206.5 ± 17.8 at 25,393 ± 7,584 site visits against 26,328 ± 11,998 (p = 0.47, p = 0.72). A learned schedule now has a declared one to be measured against.

**A feature column an arm cannot vary is dropped rather than carried.** A Swendsen-Wang action names a temperature and nothing else, so a gain column would be constant across every action at a state and would sit in the direction a softmax over scored actions cancels — a weight nothing identifies. The width is decided from the declared move set and ladder, so it is fixed for a policy: three columns for the single-site arm, four for Wolff and the mixed set, two for Swendsen-Wang.

**The first draft of that feature map broke the ticket's own feasibility premise, and the profile said so.** #706 was argued on 0.30 µs per scored candidate; reading the gain through a per-action call measured **12.4 µs** — 70% of a decision, 0.442 s of 0.634 s over 300 — which is `learn/CLAUDE.md`'s inlining rule violated exactly. Vectorized into one gather, and with the greedy bound read only where an action needs it, a Wolff decision fell from 1,562 µs to **171 µs** over 126 candidates (**9.1×**) and the mixed arm's from 2,670 µs to 393 µs (6.8×); the vectorized gain is pinned **bitwise** against the scalar one. Against the move a decision buys — 2,068 µs for a Swendsen-Wang pass, 5,927 µs for a sweep — the overhead is now 1% and 8%. Folding the remaining tuple reads into one structured pass measured *slower* (293 µs against 279 µs) and was not kept.

**Reinforcement learning lives in one package (issue #779).** `search.rl`
(490 lines), `search.gym` (206) and `search.surrogate` (323) are `learn.tree`,
`learn.gym` and `learn.ranking`; `search.surrogate` and `learn.surrogate` share
no name, so it moved rather than folding. `MoveKind` went the other way, from
`learn.potts_nd` to `sample.potts_mcmc`, beside the moves it names. Excluding
the three deprecation shims, `search/` now imports nothing from `learn/`, where
it imported it on four lines. The move is pure: 25 import statements and 16
prose references rewritten across 29 files, and no reward, feature, policy,
default, RNG order or return value moved --- the ten test modules that moved to
`tests/regression/learn/` pin the same values from the new paths, markers
unchanged. `learn/`'s no-application-imports rule now names its three
exceptions --- `tree.py`, `ranking.py` and `potts_nd.py` --- rather than
admitting none.

**A Monte Carlo move keeps a deterministic `step`.** The realization is keyed on a `blake2b` digest of the labelling and the action rather than drawn from a stream (`learn/keyed.py`), so replaying an action replays its successor, a different state draws differently, and `learn.exact`'s enumeration stays valid — the obstacle #597 left open for the bandit and slippery Frozen Lake, settled here without widening the protocol. Prices are `search.ground_state`'s own: a sweep costs `n_nodes + 2 n_edges` and a flip its degree plus one, so a matched-budget comparison is against that module's unit.
**Modules.** The learning interface, the estimators and the episodes they run on: `learn.environment`, `learn.reinforce`, `learn.rollout`, `learn.potts`, `learn.hmm`, `learn.tree`, the phylogenetic environment, and `learn.gym`, the Gymnasium adapter over the same interface (both under `search` until #779). `learn.canonical`: canonical control problems, with an exact optimum written a second way (#597). `learn.failure`: where a classical baseline first fails, which is where a gate can be argued (#597; under `opt` until #717). `learn.tabular`: tabular Q-learning and SARSA, which differ in one expression (#597). `learn.keyed`: randomness inside a move, without giving up a deterministic ``step`` (#706). `learn.potts_nd`: a Potts lattice in N dimensions with a per-site field, searched by Monte Carlo moves (#706).

**The estimator is pinned to a closed form, not to a training curve**
([#135](https://github.com/michaelJwilson/snakes_and_ladders/pull/135)). With a finite
action set and horizon the expected return is exact by trajectory enumeration,
and its gradient follows by differentiating it. That oracle carries every
claim: the enumerated gradient agrees with central finite differences to
1.5e-11 relative, the sampled estimator with the enumerated gradient to
9.9e-03 over 6000 episodes, and a myopic variant crediting each action with
only its own reward is rejected at 71%. A score-function estimator with a sign
error is wrong by a factor and still trains, so the sampled return is a
diagnostic rather than a result.

**Learning is demonstrated where it can be refereed.** On the Potts environment
the reward decomposes exactly into the two features the policy scores, which
puts hill climbing *inside* the policy class as the weight vector proportional
to `(J, 1)`. The learned policy reaches the enumerated optimum from 86.6% of
the 81 starts against greedy's 80.2%, in 8 of 8 training seeds. Recorded as
`docs/experiments/003-potts-chain-reinforce-vs-greedy.md`.

**The phylogenetic environment exists, and the reward it can afford is
measured** ([#137](https://github.com/michaelJwilson/snakes_and_ladders/pull/137)). A state
is a topology, an action an NNI or SPR neighbour, the reward the improvement in
log-likelihood. Two reward models are implemented: fitting branch lengths per candidate costs
113.7 ms against 352 us at fixed known parameters, a factor of 323, and only
the second makes an episode affordable. The substitution is validated
([#139](https://github.com/michaelJwilson/snakes_and_ladders/pull/139)): the two surfaces
score the generating topology highest, agree on the best of all 105 topologies,
and correlate at 0.9568, agreeing on the best topology across a 50-fold range
of the fixed branch length with correlation never below 0.8719.

The fitted surface does not totally order topologies: many candidates share a
maximized log-likelihood to within the optimizer's convergence, the branch
distinguishing them fitting to zero and the tree collapsing to the same
polytomy — so a rank correlation moves by up to 0.04 under a perturbation of
one part in 1e9 and is not a measurement.

**The tree policy has now been trained, and the result is negative.** On the
7-taxon fixture where NNI hill climbing reaches the enumerated maximum from
only 24 of 50 starts, a trained policy reaches it on 0.485 of episodes against
greedy's 0.480 at a matched per-episode budget — +0.005, standard deviation
0.014 over 16 training seeds, 8 ahead, sign test p = 1.0. Two measured
properties of the environment bound it. An episode
terminates when no move improves, and all 945 topologies contain 10 such
states, so every run — 50 of 50 greedy, 800 of 800 learned — ends at one: the
agent selects which local optimum to enter, and cannot leave one. And the
policy ranks moves by a single feature, the improvement a move buys, which
places hill climbing inside the policy class as a temperature. A richer feature
set (#806) and accepted-worsening steps (Stage 3) would change the
answer.

**Escape is now built, and it moved the comparison's baseline rather than its
result.** An episode can run past a local optimum, and an epsilon-greedy policy
can take the worsening move that leaves one: escape from one of the 9
non-global optima rises from 0.111 at `epsilon = 0` to 0.883 at `0.4`, and
matched-budget success from a random start from 0.560 to 0.908 against the
0.480 a single greedy run reaches. But random-restart hill climbing
reaches the enumerated maximum from **every** start at the same 60-decision
budget: greedy stops after about four decisions, so 60 buys roughly fifteen
restarts, and with the global basin covering about 48% of starting topologies
`1 - 0.52**15` is indistinguishable from 1. The baseline Milestone 2.1 has to
beat on this fixture is therefore 1.000, not the 0.480 the tree-policy
comparison above was stated against, and nothing measured here beats it.

**Accepted worsening is a first-class action, and the measurement says what
it buys and what it does not**
([#820](https://github.com/michaelJwilson/snakes_and_ladders/issues/820)).
Every learner and the arena take `stop_at_local_optimum`; under `False` an
episode runs past a local optimum and the greedy row is hill climbing
restarted until the decision budget is spent, the baseline `learn/CLAUDE.md`
names for a wandering searcher. On the chain (81 starts, the published
budget of 60 x 32 x 6) the rule moves greedy and nothing else: restarted
greedy reaches **81 of 81** against 65 stopped (McNemar `p < 1e-4`, 16
discordant), while REINFORCE reads 71 against 72, actor--critic 70 against 72,
PPO 80 against 78 and the MLP 81 against 79 (`p` from 0.50 to 1.0), and
against restarted greedy REINFORCE and actor--critic **lose** (`p = 0.002`,
`0.001`) where PPO and the MLP tie. On the 7-taxon fixture (9 traps and 50
random starts, 40 x 16 x 60) the rule lifts every row --- escape 0 of 9 to 8
or 9 of 9, random-start success 14 to 29 of 50 up to 46 to 50 of 50, `p <
1e-4` on all five --- and no trained policy beats restarted greedy's 59 of 59:
PPO ties at 59, the MLP 57, REINFORCE and actor--critic 55 (`p = 0.125`). The
wandering rule buys the *baseline* its escape, and a policy that wanders has
to beat a searcher that restarts; on these two fixtures none does. Recorded in
the pull request's tables; the null default is bitwise the arena before it.

**All three problem classes are now MDPs.**
`sal.learn.environment.Environment` had one instance, a 1-D Potts chain. It
now carries the Potts environment over an arbitrary graph — the chain is the
one-dimensional case of the same class — and the hidden Markov state path,
whose objective is a decoding problem rather than an energy. Both are pinned
against the enumerated estimator oracle carried over unchanged from the chain,
and against exhaustive enumeration of their own state spaces: 19,683
configurations for a 3-state 3x3 lattice, 729 paths for a 3-state sequence of
six. Neither takes an application type, so `sal.learn` still
imports nothing from `sal.sim`, `sal.likelihood`
or `sal.search`, and a test asserts it.

**The tree policy learns once it has something to learn**
([#328](https://github.com/michaelJwilson/snakes_and_ladders/issues/328)).
#178 trained a policy over one feature, the improvement a move buys, and
measured a tie with greedy hill climbing on the hard seven-taxon fixture,
because a softmax over one column is an inverse temperature. `FeatureSet.FULL`
gives each move seven columns without a fit — the improvement, the Fitch
parsimony change, the pattern support of the split broken and of the split
made, and the sizes of the two exchanged subtrees — standardized within the
neighbourhood, each pinned to an independent computation and each shown to
vary within one (a planted Robinson–Foulds column, constant across NNI moves,
is refused). At #178's budget of 640 episodes over 50 starts and 16
training seeds, the single feature reaches the enumerated maximum from 0.487
of episodes against greedy's 0.480 (sign test p = 0.79) and the full set from
0.796, ahead on 16 of 16 seeds (p = 3.05e-5;
`docs/experiments/005-tree-policy-features.md`). The known-parameter reward
now scores GTR from a given rate matrix through the pruning recursion. Not
measured: the full set against random-restart hill climbing, which #194 showed
reaches every start on this fixture, and any column's individual necessity.

**A critic, an actor–critic and PPO, each pinned to enumeration before it is
measured** ([#313](https://github.com/michaelJwilson/snakes_and_ladders/issues/313),
part 1). `learn.exact` returns action values and the optimal value beside the
expected return, and the three agree where they must: Bellman's equation to
1e-12 on every state checked, the optimal value above every policy's, and over
eight decisions equal to the gap to the enumerated minimum energy. A critic
reads state features derived from the action features the protocol already
supplies (their mean and maximum, and the log of the neighbourhood's size), so
no instance changed; fitted to the exact `V^pi` on all 81 configurations of the
chain, the linear critic explains 0.87 of its variance and a 16-unit MLP
0.996. The advantage-weighted score function with the exact
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
is correct by the ratio `pi / beta` and buys nothing here.

**A planner reaches the optimum at a fraction of greedy's evaluations, once
its prior and critic are trained.** `learn.planning` runs PUCT search over any
environment with the policy as prior and the critic as leaf value, and expert
iteration fits the policy to the root visit distributions and the critic to the
achieved returns. Pinned by enumeration: with the exact optimal value as leaf
and one decision of depth the most visited move is an argmax of `Q*` on every
state checked, and at depth three the visit distribution's one-step value under
`Q^pi` is no less than the prior's on 13 of 14 states. Measured on the chain in
successor evaluations: an
untrained prior with a fresh critic reaches the enumerated optimum from
76.5% of the 81 starts at 57 evaluations per episode against greedy's 80.2%
at 48; after 10 iterations of 8 planned episodes (1,066 evaluations of
training) the planner reaches it from **92.6% at 8.3 evaluations per
episode**, and at six simulations, 6.3 evaluations, matches greedy's 80.2%
— the same answer at an eighth of the cost. The policy alone, without the
search, reaches 30.9%: the visit distributions at 20 simulations are flat
targets, and what expert iteration taught here is the critic. The
factor-graph environment and the surrogate reward model wait on #296 and #308.

## Milestone 2.2 — Curriculum Learning

**The regimen is measured rather than assumed, and it is the surrogates
that carry it: a policy does not transfer yet.** The evidence below sat
under Milestone 1.4, whose subject is discrete move sets, while
`ROADMAP.md` §2.2 pointed at a 2.2 row that did not exist (#683); it is
moved here unchanged.

The curriculum 5 → 6 taxa measures what
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

**Learned surrogates across the three `spatio_only` rungs**
([#365](https://github.com/michaelJwilson/snakes_and_ladders/issues/365)). The
declared instances span 9 sites at 3 classes to 5,041 at 10, and the referee
changes with the size: enumeration at `ci`, the column transfer matrix at
`stress`, and at `release` --- `oracle: none` --- the energy alpha expansion
reaches, which the discrete solvers do compute at 5,041 sites. Every fit
predicts a gap above an analytic offset and is bracketed before it is scored.
The spanning-tree upper bound does not reach the release lattice, being one
exact tree pass per edge over 14,840 edges, so a second bracket was proved for
it: the decoupled energy bound below and the per-site data optimum above, each
one pass over the sites and one over the edges, 1.03 per site wide against the
spanning-tree bracket's 0.081 per node at `ci`.

At `ci`, over 24 held-out instances of 96 split whole by draw, the models
explain the gap above the mean-field bound --- 0.29 to 0.49 nats over 9 sites
--- at `R^2` 0.975 for attention, 0.961 for the set model, 0.949 for the MLP,
0.906 for the graph model and 0.836 for the linear one. The bound alone
explains none of that gap and yet already ranks every held-out group's best
first, so the ranking there measures the instance and not the model; at
`stress` it ranks 0.667 of them and at `release` 0.333. From scratch at
`stress` over 48 instances, attention reaches 0.958 and the graph model 0.091.
Transfer from 9 to 72 sites collapses zero-shot and recovers: the graph model
-338.6 to 0.722 and attention -360.8 to 0.522, against the 0.99 zero-shot
recorded on `potts_lattice` at a shared field (#317), so the per-site field is
what the earlier number did not have to carry.

At `release`, the first fits run at this size, the deep MLP explains 0.515 of
the 2,066-to-2,200-nat gap and the linear model 0.332 over 12 held-out
instances of 48. **Both token models fail there, and the cause of each is
measured.** `GraphSurrogate` diverges --- `R^2` -47.5 over 150 epochs and
-976.7 over 30 --- because `_Batch.pool` sums over nodes, so the vector its
decoder reads is three orders of magnitude larger at 5,041 sites than at the 9
the architecture was fitted on. `AttentionSurrogate` does not run at all: one
attention matrix is 5,041^2 `float64` = 203.3 MB, 24 training examples over two
heads is 9.76 GB, and the kernel killed the process at 9.96 GB resident, twice.
What survives both is the offset --- every prediction, the diverged fit
included, stays inside the bracket, which is what predicting a gap above a
bound buys. Both failures are ticketed rather than worked around.

## Milestone 2.3 — Empirical Validation & Benchmarking

**The comparison machinery landed and the comparison the milestone asks for
has not been run.** What exists: `opt.budget.compare` puts two methods on one
budget over shared seeds, and `opt.budget.mcnemar` decides the pair by
McNemar's exact test rather than by two means read side by side. Six
budget-matched comparisons are recorded against it --- the planted glass,
Rastrigin, the mixture, the relaxation against greedy, the cluster updates at
the transition, and the tree's starts at equal evaluations, the last as
experiment 006 over twenty seeds at 2,000 evaluations (#364). The paired test
is exact rather than asymptotic, which is what lets a comparison over 40
shared starts state a result at all.

**What the milestone asks for and this file cannot evidence:** the *RL agents*
benchmarked past the size enumeration reaches, against the classical
heuristics this repository implements --- large parsimony under NNI and SPR,
and hill climbing. Every comparison above is between classical methods or
between a relaxation and greedy; none has a learner on one side at a size an
oracle cannot reach. The empirical alignment against an external solver stays
out of scope by rule (`docs/external_tools.md`), and #126 carries what would
replace it. Recorded **not started** on the milestone's own terms, with the
machinery it will use already built and refereed.

## Milestone 3.1 — Model Surrogates & Bounds for Supported Problems

**Modules.** The surrogates and the bounds they claim: `likelihood.surrogate`, `learn.surrogate` and `learn.ranking`, the examples and targets joining the two halves (`search.surrogate` until #779).

**Landed as bounds first and predictors second**
([#317](https://github.com/michaelJwilson/snakes_and_ladders/issues/317)). The
roadmap asked for a neural surrogate queried 10,000x faster than the exact
evaluation to filter proposal batches. What landed inverts the emphasis: a
*certified analytic bound* on the objective, with a learned predictor on the
gap above it, ranking a neighbourhood so that only the top-`K` candidates are
re-scored exactly. The bound is what makes the filter safe --- a learned score
alone can rank the true optimum out of the top-`K` and nothing would say so ---
and the proofs are Appendix B.

**What remains is the number the roadmap asked for.** The filter's cost ratio
at large `n` is unmeasured: the surrogate's speed against the exact evaluation's
is the claim "10,000x" was a target for, and this file cannot repeat it because
nothing here has measured it. #800's release audit is where it is next read for.

## Milestone 4.1 — Experiment Tracking, Ablations & Leaderboard

**The ledger has a record format before it has a run store**
([#314](https://github.com/michaelJwilson/snakes_and_ladders/issues/314)). An
experiment is a file under `docs/experiments/`, written from a template: the
commit, the feature under test, the fixture and its size tier, the methods
compared at one budget over shared seeds, the results, the finding, and the
tickets it filed. `infra/experiments.py` validates every file against the
template's fields, vocabularies and sections and generates the index that is
the leaderboard, and a guard runs it per pull request. The first entries are
three comparisons this file already stated — the cluster updates'
autocorrelation at the transition, the Gaussian-emission interval coverage
against separation, and REINFORCE against greedy on the Potts chain. Six are
recorded at 0.5.0: the mixture at 3,000 evaluations (004), the tree policy's
feature set (005), and the tree's starts at equal evaluations (006). The
applicability tables of §1.3 cite a file by its number, and a citation naming
one that does not exist fails the generation.

**The record is capped at one screen, so what it holds is chosen**
([#458](https://github.com/michaelJwilson/snakes_and_ladders/issues/458)).
Seven prose sections collapse to three — question, numbers, finding — and the
body is at most ten non-blank content lines after the front matter, neither the
title nor a section heading among them; the front matter is untouched, being
the reproducibility record. The six files fall from **70, 69, 69, 114, 82
and 141** lines to **34, 34, 33, 34, 34 and 34**, bodies of **7, 7, 6, 7, 7 and
7** content lines. No number was deleted to fit: each one displaced was already
evidence in this file or an argument in the pull request that carried it, and
the guard now fails an eleventh content line. The Aim run store (#75) is part 2,
behind the dependency's approval; until then the numbers are typed from the
measurement and the file names the command that produced them.

**A run reports as it goes through Aim's own interface, and the untracked run
is the run that was there**
([#778](https://github.com/michaelJwilson/snakes_and_ladders/issues/778)).
`sal.track.Run` is a `runtime_checkable` Protocol of the three
members a hook uses --- `track`, `__setitem__`, `close` --- written with
`aim.Run`'s signatures, so an Aim run is the store and there is no adapter to
keep in step; a `release` test asserts the `isinstance`. `track(run,
metrics=)` binds a `TrackedOptimization` to a `contextvars.ContextVar` for a
block, and `opt.fit.fit`, `sample.hmc`'s three drivers, `sample.potts_mcmc`'s
two, `sample.tempered`'s two ensembles and `qa.figure.write_qa_figure` call
its one `record` per iteration, sweep, round or figure. Every counter
recorded is one the result already returns --- the acceptance rate, the
gradients spent, the best energy, the swap acceptance --- so a series ends at
the field; `peak_rss_bytes` and the state's bytes are recorded once at the
end. **What the state means is the other half, and it is not a hook's to
define.** Six `Metrics` sets sit beside their problem's objective or energy
--- `PottsMetrics` (`sim/potts.py`), `HmmMetrics` (`opt/hmm.py`),
`TreeMetrics` (`likelihood/objective.py`), `MixtureMetrics`
(`opt/mixture.py`), `CodeMetrics` (`likelihood/ldpc.py`) and
`TestFunctionMetrics` (`opt/testfunctions.py`) --- and each metric is a call
to a function the package already had, pinned against it and against an
independent answer where the problem carries one: zero split distance on the
topology that generated the alignment, zero bit errors on the word that was
sent, zero distance at a known minimizer, and the likelihoods against
`enumerate_hidden_paths` and `enumerate_mixture_assignments`. Outside a block
the bound object is `NULL`, whose run is the shared `NULL_RUN`: `record`
returns on its first line and no metric is computed, which
`tests/regression/test_track.py` pins bitwise on `hmc.sample`, `anneal_potts`
and `potts_mcmc.parallel_tempering`. **The cost is one returned call a sweep,
and at the stress instance the walls cannot resolve it.** Timed directly on
the null run, `record` costs **0.246 us** at the annealer's four arguments
and **0.352 us** at the chain's six (minimum of five runs of 200,000 calls):
**0.50 ms over 2,000 sweeps** and **0.35 ms over 1,000 draws**. Both calls
were then read against `main` at 121a1c7, the tree without the seam, three
processes each and alternating so the two share the machine, at the minimum
of three repeats per process. `hmc.sample` at the instance #754's stress
ranking uses --- a Potts chain of length 64 over 200 chains, 1,000 draws,
five leapfrog steps --- reads **6.418 s** hooked against **6.389 s**,
**1.005x**, over spreads of 2.2% and 3.3% between processes of one variant.
`anneal_potts` on the 64x64 open lattice at the critical coupling, three
states and the Rust sweep over 2,000 sweeps, reads **0.7068 s** against
**0.7066 s**, **1.0003x**, over spreads of 1.0% and 0.3%. Both are inside
the ticket's 1% bar, and both differences are smaller than the spread of
either variant, which is what the direct timing predicts: 0.35 ms is 0.006%
of the chain and 0.50 ms is 0.07% of the anneal. Measured 2026-09-19 on the
4-core host at a 1-minute load of 0.04 rising to 0.97. **The tracker reaches every loop, and the five metrics nothing recorded are
recorded** ([#799](https://github.com/michaelJwilson/snakes_and_ladders/issues/799)).
`slice_sample` records its objective evaluations per draw, the unit it is
counted in; `annealed_importance_sampling` and `population_annealing` record
`log Z`, its standard error and its effective sample size per rung by the
closing formulas, so the last entry is the result's bitwise; `simulated_tempering`
records the rung, the acceptance, the sweep rate and, once, the occupation per
rung under the rung's context; the two tempered ensembles record the round
trips and the up fraction read from the trace so far, and the sweep rate;
`mala` was recorded already through `hmc._run_chain`. `record_cost` reaches
`fit`, `hmc.anneal`, `hmc.parallel_tempering` and both ensembles. The
per-rung reductions and the per-sweep round-trip read are assembled behind
`is_null`, so the null path pays one returned call: on the 12x12 lattice
(64 replicas over 12 rungs, 300 tempering sweeps, a 100-sweep pair ensemble)
and 2,000 slice draws, two alternating reads of the minimum of three runs
read 1.003x and 0.97x for AIS, 1.012x and 0.97x for population annealing,
1.003x and 0.94x for tempering, 0.98x and 1.00x for slice sampling and 0.99x
and 0.98x for the pair ensemble, each inside its own spread. Twenty-six tests
in `tests/regression/test_track.py` pin the six loops bitwise on the null run
and every new series' last value against the result's field.

**Aim is the store #75 asked
for, and it is the optional `track` extra**: nothing imports it at module
scope --- `track.as_aim` imports it where it is called --- so no CI job
installs it and the audit job, which syncs `dev`, stays clean. What the
extra carries is stated where it is declared: `pip-audit` reports
PYSEC-2026-1087 and PYSEC-2026-1088 against 3.29.1, its current release,
with no fixed version, both in the server `aim up` runs. Verified both ways
on 3.29.1: with the extra synced, `mypy --strict` is clean and 22 of
`test_track.py` pass; without it, `mypy --strict` is clean, 20 pass and the
two Aim tests skip, and `pip-audit` reports nothing. two `release` tests assert that an `aim.Run`
satisfies `Run` and that a run written to a temporary repository reads back
each sequence equal to the `MemoryRun` record of the same call; both
`importorskip` the package, so they skip wherever Aim is absent.

## Stage 5 — Research Extensions

**Both halves are built, and the second's block was the referee, not the
effort.** Potts configurations and HMM state paths are enumerable, so the exact
optimum, expected score and gradient are all computable and "does the
relaxation find what discrete search finds" is falsifiable
([#211](https://github.com/michaelJwilson/snakes_and_ladders/issues/211)).
Tree topologies were recorded as having no such referee; they have two below
nine taxa: exhaustive enumeration, and the Hadamard conjugation of a two-state
spectrum, a closed form for the relaxation's own coordinates (#408).

**A tree is a point of the tropical Grassmannian, and a quartet's resolution
is an argmin.** `Gr(2, n)` is the set of pairwise-distance vectors satisfying
the tropical Plücker relation --- the four-point condition --- and of a
quartet's three pairing sums the smallest is attained by its own resolution.
Softening that argmin at temperature `tau` and weighting a per-quartet score
table by it gives an objective differentiable in the distances and linear in
the weights, exactly the discrete quartet score at a tree metric. Measured over
the metric of every one of the 15, 105 and 945 topologies of the 5-, 6- and
7-taxon fixtures, and over 201 of the 10,395 at eight: `3.8e-16` relative at
worst. The softmin's own leakage is
certified under `1e-11` by `sum_Q 2 exp(-g_Q / tau) R_Q`, inverted for the
temperature each corner needs; what is left at that temperature is float64
rounding of a sum over `C(n, 4)` terms, which is why the agreement is pinned
relatively and not at the `1e-11` the Gumbel-softmax half uses.

**The closed form lands on the variety.** The Hadamard conjugation of the
exact two-state spectrum returns a weight per split, zero on every split the
tree lacks; summing those over the splits separating two taxa gives the metric
by a route sharing no algebra with a walk over the tree, and its four-point
violation is under `1e-12` at 5, 6 and 7 taxa. It resolves every quartet as the
generating tree does, and so does the metric estimated from a recoded
alignment.

**What it buys is nothing, and that is the result.** Annealed ascent from
8 random metrics reaches the enumerated maximum of the quartet surface 8 of 8
at five and six taxa and 7 of 8 at eight, the miss 15.8 below in 302,287.
Neighbor joining on the estimated distances reaches the same maximum on every
fixture measured, at no gradient steps, so no budget-matched claim is made
against #408's bounded-radius search. On the 7-taxon fixture built so hill
climbing fails ([#177](https://github.com/michaelJwilson/snakes_and_ladders/issues/177))
the top two quartet scores differ by `3.4e-8` relative — inside the convergence
of the fits that produced them — and both methods return the generating
topology. Ascent leaves
the variety: the four-point violation where it stops is 0.24, 0.05 and 2.15 in
units where the metric has mean 1, and at seven taxa its relaxed value exceeds
every corner's by 0.96, the outer relaxation's gap measured rather than assumed
absent. The module is therefore `sal.sandbox.tropical` and not a
member of `search/`: a declined implementation is conserved with the tests that
declined it, and `sal.qa.tropical_relaxation` keeps rendering
`fig:tropical-relaxation` from it.

**The relaxation is an extension, checked at every corner.** Over every
configuration of an enumerable instance the relaxed score equals the discrete
one to `1e-11` relative, for both spaces. The HMM check crosses a module
boundary — `sal.learn` may not import `sal.likelihood`, so
`RelaxedHmmPath.discrete` and `sal.likelihood.hmm_paths.path_log_probability`
are independent implementations — and the relaxed objective's enumerated
optimum is the Viterbi path.

**One identity carries the result, and its boundary is not what it looks
like.** For a multilinear objective under a factorized `q`,
`E_q[score] = score(q)` exactly. Two plausible statements of the limit are
refuted by tests — the model need not be a chain, and terms need not be
pairwise. What breaks it is a term using one site twice,
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

The deterministic ascent — which the identity licenses — beats the baseline;
adding Gumbel noise gives that up for a tie, and annealing does not recover it.
It is also 15% cheaper per run: 43.6 ms against 50.1 ms for 100 gradient steps.
Three of four variants tie, reported as a tie
([#193](https://github.com/michaelJwilson/snakes_and_ladders/pull/193)).

**The budgets are not the same unit and no claim is made that they are.**
Greedy stops at a local maximum after 3.5 decisions on average at 14 discrete
evaluations each; the relaxation takes gradient steps and evaluates no discrete
configuration until the end. What is matched is the restart count and the
seeds. The advantage is not bought with the larger budget: the relaxation wins
at 25 gradient steps (15/40, `p = 0.0064`).

**The comparison fixture is not the repository's own.** `potts_chain/ci.yaml` has
`J = 0.75 > 0`, so its optimum is `argmax(h)` repeated and every method finds
it — the third fixture too easy to separate methods, after
[#177](https://github.com/michaelJwilson/snakes_and_ladders/issues/177),
[#198](https://github.com/michaelJwilson/snakes_and_ladders/pull/198) and #209's
planted spin glass, and why the baseline is now run before any claim is made.

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
Reference Taxonomy appendix routes the literature by concern.

Twenty-three QA scripts render the figures, each committing a figure with a
caption naming the seed, the sizes and the model that produced it, and
`docs/CLAUDE.md` states the rules that keep a CI-regenerated artifact true
([#140](https://github.com/michaelJwilson/snakes_and_ladders/pull/140)). The two
documents cite twenty-three of them, every script the manifest carries, so none
is left to the release gate alone
([#157](https://github.com/michaelJwilson/snakes_and_ladders/pull/157),
[#325](https://github.com/michaelJwilson/snakes_and_ladders/issues/325)).

Measured against §1.3's required contents: the model formulations are present
for every class, and since
[#274](https://github.com/michaelJwilson/snakes_and_ladders/issues/274) every
equation and algorithm the code cites is stated in the textbook under a label,
with `tests/regression/test_document_labels.py` failing on a citation no
document resolves. The old document had labelled none of them, so nine
citations had never resolved and two named equation numbers from a numbering
that no longer existed. The bounds and their proofs are Appendix B since #317,
and the branch-and-bound search over them is #329. The paper's two framed
placeholders — the comparison against classical software and hardware scaling —
gave way at 0.4.0 to the budgeted comparisons the ledger records and to #353's
thread-scaling result, with the external-tool comparison stated as ticketed
(#126).

**The textbook is now one document at one standard**
([#298](https://github.com/michaelJwilson/snakes_and_ladders/issues/298),
[#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376)). Every
problem section carries the same five parts — the model, the sizes it is
supported at read from the fixtures and the size tiers, the model as an
instance of the factor graph of `sec:factor-graph` with a hand-drawn sketch,
the algorithms as instances of `eq:sum-product` or of the optimization each is,
and the validation, one referee at a time — and the notation table states the
factor-graph symbols once with the identification each section's classical
symbol makes. The discrete solvers that were one sentence each are sections
with a labelled equation, a citation, a regime and a pin: ground states as cuts
(`eq:cut-energy`, `eq:gw`), alpha expansion and its bound
(`eq:alpha-expansion`), the heat bath and the cluster moves with the field
accept step (`eq:heat-bath`, `eq:cluster-accept`), and temperature, annealing
and tempering with the exchange ratio and the tempered weight (`eq:exchange`,
`eq:tempered-weight`). The hidden Markov section states the backward pass, the
posterior and Viterbi as max-product (`eq:posterior`, `eq:viterbi`), and the
coupled spatio-sequential model of #290 has its own section (`sec:coupled`).
Four derivations the main text depends on — the Bethe fixed point, detailed
balance for a cluster move in a field, the exchange ratio, the delta method —
sit in an appendix cited from the point of use. The document is 21 pages;
`texlive-pictures` joins the CI TeX install for the figure.

**Each of the eleven problem statements is complete, and a guard says so**
([#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376)). Twelve
sketch files under `docs/tex/` — one per problem statement and one for the
polar code conserved in `sandbox/`, hand-drawn TikZ pulled in with `\input` —
carry the factor graph of each model beside the coupled model's, which was the
only one before. The algorithms the sections cite are stated in one appendix
(`app:algorithms`), twelve environments joining the four the body derives in
place.
`tests/regression/test_problem_statements_complete.py` fails a section missing
any of `sec:<p>:model`, `par:<p>:sizes`, an `\input{<p>_figure}` defining
`fig:<p>:sketch`, `sec:<p>:validation`, or one `\ref{alg:...}`, and fails an
algorithm environment nothing cites. Before it, one section of nine carried all
five parts.

**Every row of `PROBLEMS.md` is a problem statement, and the applicability is
generated** (#358). Large parsimony (`sec:parsimony`), the frustrated lattices
(`sec:frustrated`), the Gaussian mixture (`sec:mixture`) and the continuous
test functions (`sec:testfunctions`) join the tree, Potts, HMM, coupled and
LDPC (`sec:ldpc`, #356) sections on the same skeleton, each with a QA figure
rendered from the fixture it states. Two tables
(`tab:algorithms-problems`, `tab:oracles-problems`) are written by
`infra/problems_tables.py` from the catalogue and the suite's kind markers:
which algorithm family runs on each problem, which kind of oracle referees it,
and — per method and per size tier — whether the referee is an oracle or the
simulated truth alone. A guard holds what the generator writes to what the
textbook needs; a symbol it cannot name fails it.

A third reading joins them at 0.5.0
([#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376)):
`tab:methods-initializers` to `tab:methods-surrogates`, one part per method
family, pairing every problem with every family. Twenty-six of the forty-four
pairings carry a method; each states the tier it is validated at and the kind
of referee, both read from the suite, and a sentence on when the family wins,
from `docs/tex/method_notes.yaml`. Fourteen of those sentences cite an
experiment, and a citation to a file that does not exist fails the generation.
One pairing is marked *untested* rather than omitted — the general
time-reversible model's log-det start, named by no test of either significant
kind; the mixture's $k$-means$++$ seeding was the second until
[#420](https://github.com/michaelJwilson/snakes_and_ladders/pull/420) gave it
an oracle.

**The three exact evaluators are derived, not stated**
([#326](https://github.com/michaelJwilson/snakes_and_ladders/issues/326)).
The derivations appendix carries pruning as the marginalization over internal
states (`app:pruning`), forward–backward as the same marginalization on a
chain (`app:forward-backward`), and sum-product on a tree from the subtree
factorization, with max-product as the same argument under a maximum and the
Bethe free energy of a factor graph of any degree (`eq:bethe-factor`) as the
loopy stationary point (`app:sum-product`). Each is cited from the point of use
and from the module that implements it, so the guard of #274 resolves them.

## What Is Not Claimed

- That a learned policy beats hill climbing on trees. Measured on a fixture
  where hill climbing demonstrably fails, it does not: 0.485 of episodes reach
  the enumerated maximum against greedy's 0.480, a difference of +0.005 with a
  standard deviation of 0.014 over 16 training seeds, 8 of them ahead, at an
  exact two-sided sign test of p = 1.0. The policy does train — an untrained one
  reaches the maximum on 0.018 — so this is a tie rather than a failure to
  learn; the environment bounds it, in two ways stated under Milestone 2.1.
  With the seven-column feature set of #328 the policy reaches the maximum from
  0.796 of episodes, ahead of greedy on 16 of 16 seeds, and is not yet measured
  against random-restart hill climbing, which reaches 1.000 on this fixture.
- That a learned policy beats any baseline on the instance
  [#406](https://github.com/michaelJwilson/snakes_and_ladders/issues/406) declared. `planted_glass/ci` is a
  problem a baseline does not solve — the first the repository carries — and
  no policy has been run on it. `sal.learn.potts.PottsEnvironment.on_graph`
  takes one scalar coupling across every edge, deliberately, so that the
  greedy searcher stays inside the policy class; the instance's difficulty is
  its per-edge signs, which that constructor cannot express.
- Any comparison against established software. IQ-TREE 2 and RAxML-NG are not
  installed, and no statement anywhere in the repository compares against them:
  `CLAUDE.md` admits no external solver today, and `docs/external_tools.md`
  records what adopting one would start from.
- Runtime scaling. Benchmarks are not ranked on CI hardware, so timings live in
  the benchmark suite on fixed hardware rather than in a committed figure.
- Rate variation across sites, and GPU dispatch. Neither is built; both are
  ticketed (#323, #280).

## Throughput audit at #525

The first measurements taken on a host that was quiet rather than assumed
quiet. Issue #521 removed the readers-writer lock and #526 the last
generated-file class; every reading below was taken with `ps` showing no
process over 20% CPU but the run itself, the 1-minute load recorded before and
after, and each number repeated.

**The host.** Four cores. `pytest`'s own clock is quoted, not wall clock, so
`uv` startup is out of every figure.

### Variant A, the baseline every document now carries

| Step | Reading | Repeat | Was |
| --- | --- | --- | --- |
| `pytest -m critical` | **15.9 s**, 168 tests | 15.85, 15.94 s at load 0.54, 0.57 | 177 tests in 16.6 s ([#524](https://github.com/michaelJwilson/snakes_and_ladders/pull/524)) |
| `pytest -m "not release and not stress"` | **1,226 s**, 2,315 collected | 1,226.2, 1,234.2 s | 1,098 s over 2,121 |
| the same, `--benchmark-disable` | **994 s** | 993.7, 998.3 s | not previously measured |
| `infra/check_notebooks.py`, all six | **116 s** | 115.9, 115.2 s at load 0.94, 1.03 | 137 s at load 1.7 |
| `qa.build --all --check`, all 23 figures | **387 s** | one reading | never measured as a pass |
| `infra/baselines.py`, all five records | **28 s** | 28.4 s at one and at four threads | 31 s |
| `infra/review_gates.sh` | 26–27 s, eight rows | reused from [#524](https://github.com/michaelJwilson/snakes_and_ladders/pull/524) | — |

The figure pass is the first *measured* whole-manifest run: **387 s against the
431.8 s the manifest declares between its 23 entries**, so the declared sum —
each figure timed alone — is an upper bound on the pass rather than a floor, as
`DEV.md` had assumed in the other direction. It returned non-zero: one committed
figure, **`backend_agreement.pdf`, is stale on `main`**, reproducibly so over
three renders. Nothing here touched a renderer, so it is reported rather than
fixed.

The CI tier grew 12% and the critical tier shrank by nine tests, which went
with the host lock and the figure stamps.

### The grid: processes against BLAS threads

`n` is `pytest-xdist -n`; `t` is `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` together. On four cores `n·t > 4` oversubscribes. CI tier,
`pytest`'s clock, throughput against A:

| | t = 1 | t = 2 | t = 4 |
| --- | --- | --- | --- |
| **n = 1** | 1,226.2 / 1,234.2 s — 1.00x | not taken | 1,227.4 s — **1.00x** |
| **n = 2** | not taken | 541.1 s — 2.27x | — |
| **n = 3** | 361.9 / 361.9 s — 3.39x | — | — |
| **n = 4** | 330.4 s — 3.71x | — | — |

Critical tier, two repeats per cell, `pytest`'s clock:

| | t = 1 | t = 2 | t = 4 |
| --- | --- | --- | --- |
| **n = 1** | 15.94 / 16.79 | 15.91 / 16.53 | 15.65 / 15.49 |
| **n = 2** | 14.56 / 12.87 | 13.56 / 12.56 | 13.64 / 12.86 |
| **n = 3** | 12.48 / 11.75 | 11.91 / 11.62 | 12.03 / 12.20 |
| **n = 4** | 11.46 / 12.09 | 11.27 / 10.53 | 11.50 / 11.02 |

**The interaction is that there is none to trade.** Thread width contributes
nothing at any process count, so `n = 2, t = 2` — exactly the core count — is
2.27x where `n = 3, t = 1` is 3.39x: the second thread per worker is a worker
not taken.

**Three is not the peak on this tier, and the prior that said so measured
something else.** The 22.0 / 23.4 / 28.7 / 39.2 s series behind the ~2.30x
prior was *n concurrent whole suites* — the throughput of n agents each running
everything — on a tier of 16 s where process startup is most of the cost.
`-n` splits one suite instead, and on a tier of twenty minutes the startup
amortizes: the CI tier is still improving at four workers. On the critical tier,
where the prior's conditions hold, xdist gives 1.31x at three workers and 1.37x
at four, and startup is why.

### The correctness checks, which are what decide this

**1. Ordering and seeding under xdist: no change.** Every cell of the CI grid
returned **2,302 passed, 13 skipped** — the same result set at one, two, three
and four workers, and at one, two and four threads. Nothing in the suite
depends on collection order, a module-scoped fixture or a shared temporary
path.

**2. xdist silently stops benchmarking, and that is the real cost of C.**
`pytest-benchmark` prints `Benchmarks are automatically disabled because xdist
plugin is active` and the 212 benchmark tests then collect and pass while
measuring nothing. **The 3.39x is therefore not like-for-like.** With
benchmarks disabled on both sides the CI tier is **993.7 s** at `n = 1` against
361.9 s at `n = 3`, which is **2.75x** — and the 232.6 s difference says the
benchmarks are **19.0%** of the tier, not the 29.6% `DEV.md` carried from an
older and much smaller measurement. A tier run under `-n` is a correctness run
and no longer a benchmark run.

**3. Numerical agreement under wider BLAS: nothing moved, and the tier could
not have shown it.** The CI tier at `t = 4` returned the same 2,302 passed as
at `t = 1`. That is weaker evidence than it looks: the bitwise pins in
`tests/regression/` are almost all *within-run* comparisons — `workers=1`
against `workers=4`, a decoder against its own sign-flipped input — and a
global thread change moves both sides together. The pins that a reduction-order
change could actually break are the ones against committed constants, and there
are **59 float values across the five records under
`tests/regression/fixtures/`**, compared by `infra/baselines.py` with
`fresh.value != stored.value` — exact equality, no tolerance. Running that
recomputation at both widths settles it directly: **`infra/baselines.py`
reports all five records matching at `t = 1` and again at `t = 4`**, 28.4 s
each. Not one of the 59 moved. The reduction order these fits go through is
insensitive to the thread count at this problem size, so B was never blocked on
numerics — it is blocked on being no faster.

### What variant B would stale, which is the number that decides it

**221 recorded timings**, all taken at one BLAS thread: `DEV.md` 84,
`STATUS.md` 132, `INSTALL.md` 4, `TICKETS.md` 1 (deleted by #804), counting only numeric-workload
timings and excluding the compile, LaTeX and I/O readings that thread width
cannot touch. Plus the **59 committed baseline floats** above. The 23 declared
render times in `sal.qa.manifest` are *not* among them:
`qa.build` strips the three variables from a render's environment, so every
figure already renders at full width.

### The decisions

| | Verdict |
| --- | --- |
| **B, wider BLAS threads** | **Declined, on its own clock.** 1,227.4 s against A's 1,226.2 s is a 0.1% difference inside the repeat spread of A itself. It is numerically safe — the 59 committed floats are unmoved at four threads — and it is simply not faster, so the 221 timings it would have staled never had to be weighed. `tests/conftest.py` keeps pinning one thread per process |
| **C, `pytest-xdist`** | **Adopted for the correctness tiers, not for benchmarks.** 2.75x on the CI tier at `n = 3`, like-for-like; 1.31x on the critical tier, where startup dominates. Because xdist disables `pytest-benchmark`, a run that must produce benchmark numbers runs at `n = 1` |

The prediction recorded on the ticket was C at `-n 3` for ~2.3x and B declined
on cost rather than clock; C's like-for-like 2.75x is close to it. **B is
declined for the opposite reason to the predicted one**: the prediction
expected a real speed-up outweighed by the re-measurement it forced, and there
is no speed-up to outweigh.

**Dependency clearance.** `pytest-xdist` is **MIT**, an OSI-approved licence,
at **1.9k** GitHub stars — above the 1,000 the flag rule sets. Its one runtime
dependency, `execnet`, is MIT as well. Adopting it means declaring it in the
`test` extra and committing `uv.lock` in the same pull request, which this one
does not do: the measurement is the deliverable here and the adoption is
#405's.

## Consistency audit at 0.5.0

What the release audit
([#376](https://github.com/michaelJwilson/snakes_and_ladders/issues/376)) found
stale, contradictory or duplicated between the planning documents, the two
documents, the templates and the code, and fixed in the same pull request. Its
baseline is
[#375](https://github.com/michaelJwilson/snakes_and_ladders/pull/375) as
merged.

- **The version this file is read at.** The header said `0.4.0`, "the release
  #358 cuts". No tag exists --- the repository has never carried one --- and
  `Cargo.toml` still reads `0.3.0`. The header now states the deferred cut, and
  the release template's tag precondition says what an audit does when it does
  not hold.
- **Counts the summary had left behind.** Four experiments were recorded
  where six exist, and five budget-matched comparisons where six do; the
  Milestone 1.3 row named neither the tree's data-driven starts nor
  [#373](https://github.com/michaelJwilson/snakes_and_ladders/pull/373), which landed them for #364 between the two audits.
  `TICKETS.md`, deleted by #804, still listed that work as open.
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
  full; the local rule now carries the principle and points there.
- **`scipy` is not imported anywhere.** The audit's own plan recorded it as
  imported by `opt.fit` and `sample.statistics`. Both modules say the opposite
  and write out the constants they would need. The three referee tests this
  release adds `importorskip` it and skip everywhere until it is declared,
  which `DEV.md` and `INSTALL.md` now say and which stands as the open
  question.
- **The textbook's problem statements were not one shape.** One of the nine
  carried all five parts; the rest carried two to four, and a section missing
  its sizes or its validation reads as complete. Eight sketches, twelve
  algorithm environments and nine validation subsections close it, and a guard
  now fails the shape. The plan recorded eight algorithm environments as
  existing; four did.
- **Two symbols were one letter.** The parity-check matrix and the
  Sylvester--Hadamard matrix were both `H`. `notation.tex` now defines both;
  and the coupled model's figure was labelled outside the convention the other
  eight follow, so it is renamed.
- **`ROADMAP.md` did not describe two milestones it had reached.** Milestone
  1.3 named no deliverable for a start read from the data, which #364
  supplies, and §1.3 did not say what makes a problem statement complete. Both
  are stated in the document's existing register.
- **The release template.** Its version was `0.4.0`; it carried no per-problem
  completeness checklist and no frameworks table, both of which the last two
  audits wrote by hand; and its tag precondition had no instruction for the
  case both audits met. All four are fixed, and the two ledger regenerations
  the audits ran are preconditions rather than folklore.

**Duplicated machinery.** The two seams the 0.4.0 audit recorded --- one
annealing driver behind eight entry points, one weighted enumeration behind
eight enumerators --- are unchanged and are not refactored here. Each is a
ticket, with the oracle that would pin a merge stated in that audit's tables
above.

**What the audit checked and found true.** The ten required checks are the ten
jobs `ci.yml` defines; the manifest holds nineteen figures and the documents
cite seventeen, as `DEV.md` says; `PROBLEMS.md` resolves every symbol it names;
`CHECKS.md` and the experiment index are regenerations, every experiment
validates against the template, and no document links to a path that does not
exist.

**The applicability table after the tier move.** Moving the nineteen tests over the per-PR cap to the `release` tier changed three cells of the generated applicability table: the Potts chain, the Potts lattice and the frustrated lattices now read `oracle` rather than simulated truth at the release tier, the oracle-refereed tests that moved there being the tier's referee. The table is regenerated here, and the textbook inputs it.

**Incompleteness at 0.5.0 (issue #400).** `TICKETS.md`, since deleted by #804, carried twelve bullets naming a closed issue. Five described work that had landed and are removed: the textbook's pruning, forward-backward and sum-product derivations (#326, 45 citations in the textbook), the refusal of an unidentifiable fit (#122), the required checks on every pull request (#273), the references and blue-sky directions (#360), and the 0.5.0 audit with the `scipy` question it left standing (#376). Five name work that remains with the carrier closed and are re-pointed: the profiling at the declared sizes, the efficiency assessment, the audit's four unmet targets and the parallelism sites beyond the first three go to #405; the discrete fixture no baseline solves goes to #406. Two keep a closed number for a reason the bullet states. Two reductions were measured and declined: the six emission families share 270 lines over six methods, every line the family's own distribution rather than scaffolding, and the QA renderers' argument handling is already `qa.runner`'s, leaving one line each. The notebooks' Further Work sections cited nine closed issues, three of them features that have since landed and are corrected here: the hidden Markov notebook said Viterbi and the forward-backward posteriors were not built and the coupled notebook that no path is decoded, when both landed with #175 and #173, so the gap is coverage and #407 carries the sections; the Potts notebook's four bullets cited #278, which closed without the lattice pass, and now cite #404. The corrections are prose, so no notebook re-executed.

**Seams (issue #400).** The package is 33,900 lines of Python across six modules (`search` 6,579, `qa` 6,707, `likelihood` 5,487, `opt` 5,417, `learn` 3,792, `sim` 3,483, top level 2,448) and 1,385 of Rust, against 35,780 of tests. At that audit the package declared 11 protocols and 4 shared contracts (`SEAMS.md`, deleted by issue #586 in favour of the declarations themselves): 7 protocols and 3 contracts have three or more consuming modules (`Objective` has 15 implementers and 14 consumers and reaches 7 of the 11 catalogue problems; `Environment` 12 consumers; `FactorGraph` 7); `CountEmissionFamily`, `RelaxedObjective` and `Channel` have no consumer outside their module, `Policy` one, `SpatioSequentialParams` two, each kept for the reason the table prints. One merge proposed under this ticket was measured and declined: the HMM and mixture EM loops share 16 lines, and a driver would add more than it removed. `infra/duplication_survey.py` at this audit: enumerate-shaped functions 15 (8 at #230's filing; #387 owns them), energy-shaped 8 (5), private logsumexp 0 (4), open-coded edge zips 0 (6).

**Slimming baseline (issue #717).** The survey gained twelve rows on 2026-09-18 at `main` a5b6fa4, each pinned at its count by `test_duplication_guards.py` so nothing grows while the ticket's eight pull requests land, and the pull request that lowers a row lowers its pin: Potts energies of a labelling beyond `sim.potts.energies` 3, site-field broadcasts beyond `sim.potts.site_field` 2, annealers beyond `sample.gibbs` 4, `ground_state.run_*` wrappers 10, backend enums beyond `search.backend` 1, Python paths above a compiled kernel 8, surrogate modules 4, modules without a docstring 1, root exports 1 (`double`), regression test modules pinning a twin to its oracle beyond the first 12, flat modules 150, API-map entries 1,576. Two of the ticket's numbers were impressions the query corrected: it named 14 modules without a docstring and there is one (`scripts/__init__.py`), and nine compiled twins where eight sit beside an oracle (`ragged_rust` has no `ragged`). The sandbox is excluded from every row, since what sits there is declined.
**The survey retired (issue #813).** `docs/reviews/2026-09-20-calling.md` reads the same overlap by hand and finds the rows counted names: the four annealers share a six-line loop, the seven `run_*` wrappers are seven signature adapters, and one of the eight compiled paths (`pruning_torch`) is the differentiable method. The one copy the rows held --- `opt.potts.simulate_chains`, an exact chain simulator written twice --- is folded, bitwise at the declared instance with 5 of 5 baselines matching. `infra/duplication_survey.py`, `appraise_structures.py`, `appraise_seams.py` and their tests are removed (1,446 lines) and the counts are read by command: package **51,317** non-blank lines over 154 modules (sandbox, scripts and `__init__` excluded), tests **57,226** over 249 modules, `find <dir> -name '*.py' | xargs cat | grep -cve '^\s*$'`. The thirteen guards in `test_duplication_guards.py` stay.

Pull request 2 folded the energies and the site fields: `search.maxflow.energy`, `search.maxflow.site_field`, `search.alpha_expansion.energy` and `alpha_expansion._site_field` are gone, `sim.potts.energy` is the scalar entry and `sim.potts.site_field` takes the column check the two carried, so the rows read energies 1 (`maxflow.cut_energy`, a cut value's energy and not a labelling's), site fields 0, API-map entries 1,574; every numeric pin ran unchanged, and the one test that compared the two entry points to each other was deleted as a tautology, `log_weights` remaining the referee.

Pull request 3 (issue #387) put one enumeration behind the oracles: `enumeration.configurations` is the product space under the cap, `posterior`, `site_marginals` and `argmax` its reductions, and the five enumerators and three `optimum` helpers call it with every signature and value unchanged --- the row of enumerate-shaped names stays at 19 because #387 keeps every name; what went were five copies of the product space, two of them uncapped, and three of the argmax loop; the seam's four functions read API-map entries 1,580.

Pull request 5 folded the first compiled twin: `numerics_rust.sample_rows` is `numerics.sample_rows(backend=)`, the extension by default since the two agree bitwise on the same generator state, and `Backend` moved from `search.backend` to `sal.backend` so a module below `search` can name it; the rows read twins 7, flat modules 149, API-map entries 1,575. The remaining six twins follow one per pull request, `maxflow_rust` after #716 lands on it.
Pull request 7 made the root a surface: `sal` exports `Objective`, `Environment`, `FactorGraph`, `PottsGraph`, `fixture`, `Backend` and `parallel`, each resolved on first use so the bare import loads no submodule (asserted in a fresh process), and `double` is the extension's alone; `scripts` gained the docstring it lacked. The rows read root exports 7, modules without a docstring 0.
Pull request 8 folded the second twin, `search.potts_mcmc_rust`, which since #599 was one line of dispatch onto `potts_mcmc.sample_potts(backend=Backend.RUST)`, that function's default; the name issue #246 published lives on in the test and benchmark that cite it. The rows read twins 6, flat modules 148.

Pull request 6 judged `opt`'s three misplaced modules by their importers rather than by the ticket's list: `failure`, the sizing harness, is imported by nothing but the learning gate's tests and is `learn.failure` now, imports only; `budget` and `schedule` stay, because `sample.hmc` and `opt.initialize` import the schedule and `opt.failure` did the budget, and moving either under `search` would make `opt`, `likelihood` and `learn` import `search`, which `learn/CLAUDE.md` forbids. The four surrogate modules were judged on the same rule and stay four: `bound` is the seam and imports nothing, `likelihood.surrogate` the analytic bounds with their proofs, `learn.surrogate` the fit over tensors with no application import, and `search.surrogate` the examples and targets that join the two halves, which is the one place that may import both; dissolving it would move an import of `search` into `learn` or of `learn` into `likelihood`, and the row reads 4 with this as its reason.

Pull request 4 judged the annealers rather than folding them by name: `learn.relaxed.anneal` was `sample.schedule.ExponentialTempSchedule` in a second spelling (3.2e-16 relative apart, one ulp) and is that class now, and the three annealed `ground_state.run_*` wrappers are one run with a move set, so the rows read annealers 3, wrappers 7, API-map entries 1,572. The three that remain are three algorithms and not three spellings: `potts_mcmc.anneal_potts` runs a compiled heat-bath sweep with cluster moves and the counters `STATUS.md`'s seed-for-seed pins read, `gibbs.anneal_factor_graph` a factor-graph Gibbs sweep, and `hmc.anneal` with `projection.annealed_seeding` above it a Hamiltonian chain on a continuous surrogate; a driver over the three would carry a sweep callback across the compiled boundary to save the ten lines they share, and is declined.

**A dual bound on a ground state at #696.** `search.tightening` decomposes the
energy into subproblems whose shares sum to it, so
`max_x E(x) <= sum_s max_{x_s} E_s(x_s)` holds **at every iteration by
construction** and not at convergence --- which is what lets validity be
asserted after one sweep rather than after two hundred. Measured 2026-09-17
against exhaustive enumeration of all 19,683 labellings:

| instance | ground state | bound | gap | verdict |
| --- | --- | --- | --- | --- |
| square 3x3, `J = +0.6` | -0.961675 | **-0.961675** | 0 | certified optimal, 1 sweep |
| square 3x3, `J = -0.8` | 7.665024 | **7.665024** | 0 | certified optimal, 4 sweeps |
| triangular 3x3, `J = -0.8`, pairwise | 10.112322 | 7.843362 | 2.27 | |
| triangular 3x3, `J = -0.8`, **+ triangles** | 10.112322 | **9.668997** | 0.44 | **80%** of the gap closed |
| triangular 3x3, `J = -1.5`, pairwise | 10.920268 | 7.843362 | 3.08 | |
| triangular 3x3, `J = -1.5`, **+ triangles** | 10.920268 | **10.221722** | 0.70 | **77%** closed |

The square rows include `J = -0.8`, which is **non-submodular**: minimum cut
cannot take it and this certifies the labelling anyway, with no oracle
consulted --- the enumeration checks the claim rather than supplying it.

**The pairwise bound is the same number at both couplings**, 7.843362, and
that is the clearest statement of what the relaxation misses: with three
states every triangle is 3-colourable, so the relaxation satisfies every edge
at no cost whatever the coupling, and only a constraint *over* the triangle
can charge for the frustration. It is asserted as a prediction, not recorded
as an observation.

**The local-polytope bound at 5,041 sites, #1060.** `search.trws` maximizes
the pairwise relaxation `dual_bound` maximizes with no plaquettes, by TRW-S
(Kolmogorov 2006) over monotone chains, with an `O(q)` Potts message. Measured
2026-09-25 on the compiled kernel at its defaults (tolerance 1e-12 relative,
5,000 iterations):

| instance | TRW-S bound | best known labelling | gap | iterations | seconds |
| --- | --- | --- | --- | --- | --- |
| `spatio_only/release`, q = 10 | **-10,454.16** | -10,454.16, the graph cut (#1041) | 0.00 | 43 | 0.23 |
| `spatio_tiling/release`, q = 10 | **-17,022.98** | -17,022.18, annealing after ICM (#1050) | 0.81 | 1,825 | 8.7 |

The first row certifies the ten-state optimum with no reduction; the second
narrows the tiling bracket from 293.3, the factor-2 bound, to 0.81. At that
size `dual_bound` ran 200 iterations in 177.9 s to -10,455.25 on the first
row and in 176.3 s to -17,025.92 on the second, both unconverged. The compiled
kernel runs 10 iterations in 68.6 ms against the Python reference's 10,338 ms,
151x (`test_trws_bench.py`, 1-minute load 4). On two frustrated 3x3
triangular lattices TRW-S converges below `dual_bound`'s value of the same
relaxation, and both below the explicit LP (`test_trws.py`).

**The explicit LP, #1063.** `validation.highs` solves the local-polytope LP
with every node and edge marginal written out, by HiGHS through SciPy's
`linprog`, in a subprocess. Where TRW-S converges to it, it is the LP value to
1e-8 relative, and so is `dual_bound`'s: seven of the nine 3x3 and 2x4
lattices, `potts_lattice/{ci,stress,release}`, `spatio_only/{ci,stress}` and
`spatio_tiling/ci`, the largest 1.8 s for HiGHS against 0.07 s for TRW-S. On
the two stalled triangular lattices the LP is -1.14911 and -1.40024, TRW-S
0.0160 and 0.0271 below it, `dual_bound` 7.4e-5 and 0.0204. The primal's node
marginals are integral on every instance but the three frustrated ones, and
there the LP value is the minimum (`tests/validation/test_highs.py`). At
5,041 sites and ten states, 1,534,410 columns, one run each at a 1-minute load
of 5.3, 2.1 GB peak:

| instance | LP value | HiGHS seconds | node marginals | TRW-S bound | TRW-S below LP |
| --- | --- | --- | --- | --- | --- |
| `spatio_only/release` | -10,454.16 | 343.9 | integral | -10,454.16 | 8e-14 relative |
| `spatio_tiling/release` | **-17,022.93** | 267.6 | fractional at 95 sites | -17,022.98 | 0.0496 |

So of the tiling bracket's 0.81, TRW-S stopping short is 0.05 and the other
0.75 lies between the LP and the best labelling, -17,022.18. TRW-S takes 0.82 s
and 8.7 s on the two, the `highs` goals in `test_goals.py`.

Two implementation notes worth keeping. The block update is the exact
minimizer of its own block, checked against a numerical minimum over the
block's messages; an early version left the site's own share inside the
maximization, which adds half of it to every update and reads as a coordinate
descent that does not descend. And the tightest iterate is retained rather
than the last: every iterate is a valid bound, so this costs one comparison
and removes any need to rely on monotonicity (none of 480 updates raised the
dual once the block update was right).

**Label marginals, and which estimator the reported metric asks for (#696).**
`search.decoding` carries the maximum-posterior-marginal labelling and the
loss it minimizes. The distinction is not a preference: `label_accuracy`
scores **per-site** agreement, and the estimator minimizing per-site error is
the marginal one, while a maximum-a-posteriori labelling minimizes the chance
of getting the **whole field** wrong. Measured 2026-09-17 on the 3x3
triangular antiferromagnet at `J = -0.9`, against exhaustive enumeration of
all 19,683 labellings:

| labelling | energy | posterior | rank | expected wrong sites |
| --- | ---: | ---: | ---: | ---: |
| maximum a posteriori | 10.37330 | 1.668e-03 | **1** of 19,683 | 6.0051 |
| maximum posterior marginal | 21.58657 | 2.251e-08 | **19,555** of 19,683 | **5.4906** |

The two differ at **six of nine sites**, and each wins on its own loss and
loses on the other's. The marginal labelling sits in the worst one per cent of
configurations by posterior --- minimizing per-site error does not require the
answer to be jointly plausible, and on an antiferromagnet it puts every site
at its own field-preferred label, which no draw would produce. That is the
cost of the loss, and it is why a decoder reported without naming its loss
hides the question it answered. On a ferromagnet at the same field the two
agree exactly, which is recorded so the difference is not read as general.
**Generalized belief propagation at #689.** The plaquette regions see the
4-cycles the Bethe approximation cannot, and the measurement is what the ticket
was for. On the 3x3 lattice at three states, against exhaustive enumeration of
all 19,683 configurations (2026-09-16, 4-core host):

| `J` | exact `log Z` | Bethe error | Kikuchi error | factor | sweeps B/K | ms B/K |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 1.538947 | 8.53e-04 | **3.65e-08** | 23,378x | 49 / 372 | 58.7 / 277.3 |
| 0.6 | 3.483551 | 1.11e-02 | **6.66e-06** | 1,672x | 69 / 380 | 76.0 / 280.0 |
| 0.9 | 5.924288 | 2.99e-02 | **7.75e-05** | 386x | 87 / 394 | 95.5 / 291.4 |
| 1.2 | 8.845411 | 3.13e-02 | **2.41e-04** | 130x | 86 / 402 | 96.5 / 305.5 |

**Both halves of the case.** The ratio is three to four orders of magnitude and
the cost is **3.7x the wall clock** --- 5.5x the sweeps over tables of 81
entries rather than 9 --- so the plaquette buys its accuracy at a stated price
rather than at none. **The advantage decays with coupling**, 23,378x at
`J = 0.3` to 130x at `J = 1.2`: deep in the ordered phase both approximations
concentrate on the same configuration and what Bethe neglects stops mattering,
which is also why Kikuchi is not the tool for a ground state.

**At size, convergence is the binding constraint and not accuracy.** On the
6x4 open strip, refereed by `strip_log_partition` where enumeration cannot
reach (2026-09-17, this host):

| `J` | pairwise: sweeps, ms, error | plaquette: sweeps, ms, error | factor |
| --- | --- | --- | --- |
| 0.25 | 50, 182.7 ms, 2.147e-03 | 1,176, 5,196.6 ms, **1.510e-07** | 14,219x |
| 0.5 | 85, 296.8 ms, 3.778e-02 | **does not settle** | --- |
| 0.875 | 169, 610.5 ms, 2.285e-01 | **does not settle** | --- |

The refusals are not a cap chosen too low: at `J = 0.875` damping 0.7, 0.8,
0.9, 0.95 and 0.98 all reach 20,000 sweeps with the residual at 0.377, 0.140,
0.070 and 0.020 against a tolerance of 1e-12 --- rising damping buys a slower
approach, not a fixed point. So the plaquette regions pay 28x the wall clock
for four orders of magnitude at weak coupling, and at the couplings where
Bethe is worst they return nothing at all rather than a number. That is the
result the ticket asked for, and it is why nothing here is reported as a
replacement for `belief_propagation`: the module is conserved as a declined
route in `sandbox/region_graph.py`, imported by `tests/` alone.

The construction is refereed by the case it generalizes rather than by its own
claim: at the Bethe region graph `-F_K` is `log Z` to **8.9e-16** on a chain
and the value `likelihood.message_passing` reports to **1.8e-10** on a 3x3 and
a 4x4, and the parent-to-child updates find that module's fixed point to
**1.65e-10**. The singleton counting numbers come out `1 - d` --- -1, -2, -3 on
a 3x3 --- with the closed form written nowhere.

**Kikuchi is not a bound**, and nothing here is read as one: mean field bounds
`log Z`, Bethe and Kikuchi are stationary points of a non-convex functional and
may fall either side. Every claim above is accuracy against an exact referee.
**Compiled kernels at #678.** The crate is **9 modules and 4,048 lines**, and
**3 take the thread pool** --- `coupled`, `pruning`, `sampling` --- read from
each module's own parallel iterators by `infra/appraise_kernels.py` rather than
from a list. Every module exporting a `#[pyfunction]` is pinned against a
referee its own tests import; the one exclusion is `lib.double`, the
extension-loads probe, which arithmetic checks. The tool's first draft called
the ragged kernel unpinned and the tree said otherwise: `test_ragged_rust.py`
imports `posteriors` and `posteriors_oracle` from one module, so an oracle
beside the kernel is a third placement, and a survey reporting a false gap is
worse than one reporting none.

Four paths measured on the 4-core host, 2026-09-16, at a 1-minute load of 1.2
to 1.7 (a test run held one core; `DEV.md`'s own readings were taken at 1.19
to 1.30):

*The count-pair draw does not clear its own bar at the size that matters.* At
the **declared instance** --- 71x71 at `M = K = 10` over 20,000 positions,
1.008e8 pairs, 384.6 MiB of counts --- the Rust draw is **30.14 s and 30.32 s**
against the NumPy oracle's **37.32 s and 37.16 s**, two readings each: a ratio
of **1.23x**, where root `CLAUDE.md` sets **2x** for keeping a Rust backend and
says a backend that is never faster is a maintenance cost with no counterpart.
At the CI instance the same comparison reads **2.5x** (50.0 ms against 20.2
ms), which is the gate-size illusion the same rule names --- a ratio read at a
gate size decides nothing in either direction. The conclusion is not that the
kernel is wrong but that **parallelism is what would justify it**: the draw is
5,041 independent per-vertex streams by construction, `default_rng([seed, v])`,
which its own docstring states and no thread uses. That is #693.

*The ragged kernel's two recorded ratios measure two different things, and
neither is the third.* Against the **padded route** at honest padding it is
**2.8x** (93.14 to 33.41 ms at 0% padding, 121.63 to 43.37 ms at 47.8%) --- that
is the Rust claim. At **97.0% padding** it is **97.1x** (386.42 to 3.98 ms),
which is a statement about padding and not about Rust. Against the
**per-segment Python oracle** at 600 segments of 8 to 40 it is **65x** (519 to
8 ms), a statement about Python loop overhead. A survey that lists the three in
one column invites the wrong port next.

*Dinic is the shape a compiled kernel wins on.* `ising_ground_state` on a
40x40 lattice is **145.6 ms**, of which `_augment` is **42%** and `_levels`
**25%** of self time --- two thirds of the call in two pure-Python functions,
over 25,216 and 14 calls. #642 carries the layout half.

*The factor-graph Gibbs sweep is not a port; its first call is.* Warm it is
**0.20 ms a sweep** (10 ms for 50), already `njit`-compiled, so there is
nothing for Rust to take. The **cold call pays 738 ms of `llvmlite`
compilation** --- 74x the entire warm run of 50 sweeps --- which is a caching
question and not a language one. Its warm profile's top self-time entry is
`sample.gibbs._Indexed.layout` at **30%**, the same line the structure survey
surfaces once it stops dropping findings outside clusters (#690).

**Data structures (issue #586).** `infra/appraise_structures.py` walks the tree
rather than a hand list: **202 state-carrying classes, 7 clusters** at three or
more members. `role:incidence` is 12 members over 78 consuming references --- one
relation, three layouts --- and is the cluster the ticket acted on.
`sal.incidence.SparseIncidence` is now that one layout;
`ParityCheck` wrote it by hand, `PottsGraph` derived it per call and
`FactorGraph` had none. A Wolff cluster move on a 64x64 periodic lattice is
**3.69x** faster (3.451 to 0.935 ms, of which the per-call rebuild was 2.519),
`FactorGraph.degree` over 800 variables **15.1x** (42.350 to 2.801 ms) and
`.neighbours` **19.1x**, both linear now rather than quadratic. The parity check
pays a fixed 28 us of calls --- 25 per cent at 120 bits, nothing by 3,000, and
nothing at the 19,998-bit instance the ticket names --- against a decode paid
per iteration. `ParityCheck`'s five fields and `compressed_adjacency`'s three
arrays are bitwise unchanged, and the arrays are now shared and read-only.

**Data structures at #677.** The survey read **218** classes and **8** clusters
and did not name `Ragged` once, the largest addition since #586 wrote it. Two
blindnesses, both in the tool. The classifier knew `offsets` and `indptr` and
not the third spelling of one relation --- lengths beside one flat payload,
offsets derived --- so `ragged.Ragged` was filed as carrying no layout;
`role:incidence` is **14 members over 84 consuming references** with it and
`sim.hmm.SimulatedHmmDataset` in, against 12 over 81. The corroboration rule
of #586 is what keeps that from over-matching: a payload partner is required,
so `HmmParams.lengths` stays a declaration rather than a layout, and `sizes`
is refused as a length field because it is a histogram in
`sample.potts_mcmc.ClusterCounter` and a set of problem sizes in
`search.ground_state.Rung` and `sim.potts.SpatioOnlyParams` --- the
`restarts`-for-`starts` failure one spelling later. Second, a finding printed
only inside a cluster, so a cost on a class sharing its shape with nobody was
derived and dropped: **nine findings** were invisible, on
`likelihood.schedule.Layout`, `sample.gibbs._Indexed` (three between them),
`learn.surrogate.Examples`, `learn.surrogate._Batch`,
`emissions.NegativeBinomialEmission`, `learn.potts.PottsEnvironment` and
`sandbox.pruning_burn._Flattened`; they are #690's to price. One finding the
change raised was priced here and declined: `Ragged` rebuilds its offsets per
call in a Python scan, **39.3 us** on the 600-segment `hmm/ci` batch against
**335.11 ms** for one Baum-Welch iteration over it --- **0.012%**, and the
NumPy `cumsum` that would replace it saves 6 us of that (2026-09-16). A ratio
with no effect size, so it is recorded as measured rather than asked again.
Reading `self.offsets` is no longer counted as deriving them, which had
reported two costs where `Ragged` pays one. The covariate rule five seams
restated --- add the broadcast singleton only where the covariate carries no
axes of its own --- is stated once in `sal.emissions`, which is
what enforces it, and the seams point there; replacing them is #691. One
finding was considered and declined on the survey's own rule: `suffix:Params`
is 16 members over 38 references and
`sim.count_pairs.SpatioSequentialCountsParams` holds a `SpatioSequentialParams`
where no other row holds a params, but a `suffix:` key says the names agree and
nothing calls the sixteen polymorphically, so there is no seam to write ---
`DEV.md`'s three consumers are three consumers *through* a contract.

Three proposals under the ticket were measured and declined, which is the half
a survey exists to produce. `search.maxflow.FlowNetwork` keeps its list of
lists: over 16,384 rows of degree six a Dinic row walk is 1.95 ms as lists,
3.93 ms flat with offsets and **25.63 ms** as a NumPy slice, so the layout rule
in root `CLAUDE.md` is about a NumPy or compiled consumer and not about a
pure-Python inner loop. `likelihood.message_passing`'s plan keeps its lists of
lists: the whole layout is 0.6 per cent of a `sum_product` over a
2,000-variable chain, and what the profile ranks instead is the tree schedule
at 24.4 per cent, filed as #592. No `ExactSolution` contract and no `Params`
base class are written: `prefix:Exact` is five members at **zero** consuming
references and `suffix:Params` fifteen bundles sharing `seed` and `tolerance`,
each read by one simulator, both below the three-consumer rule. One merge was
taken: fourteen sites across seven modules reduced a score vector by
`logsumexp(values[None, :], axis=1)[0]`, which is `axis=0`.
`docs/experiments/016` and `017` carry the runs.

**Data structures at #755.** The survey reads **245** classes and **8** clusters,
with 12 `Protocol`s across 10 modules and 1,577 API-map entries over 148 flat
modules. Every cluster was decided against root `CLAUDE.md`'s rule and the
reason recorded per row (`docs/reviews/2026-09-19.md`): **one meets it and is
folded, one is already the seam and wants a guard, six are left as they are**
--- `suffix:Params` (17), `suffix:Decoding` (7), `prefix:Exact` (5),
`suffix:Fit` (6), `suffix:Result` (6) and `suffix:Dataset` (5) share a name and
no surface, and `SimulatedDataset` is the phylogenetic alignment rather than
their base. The fold is the composition #387 left: enumerate, score, take the
first maximizer, written in full by `learn.potts.optimum`, `learn.hmm.optimum`
and `learn.relaxed.enumerate_optimum` and now one function,
`enumeration.enumerated_optimum`. The three return bitwise what they returned,
asserted against the deleted body in `tests/regression/test_enumeration_seam.py`
at four sizes. `likelihood.hmm_paths` keeps its own argmax: it reads the score
vector again for the posterior. `tests/regression/test_duplication_guards.py`
pins the 245 classes, the eight clusters at their member counts and the one
named argmax consumer.

**Sampling is one directory (issue #777).** Eleven modules and 7,667 lines
moved to `python/sal/sample/`: four from `opt/` --- `hmc`,
`langevin`, `slice`, `schedule` --- and seven from `search/` --- `potts_mcmc`,
`gibbs`, `balanced`, `potts_keyed`, `tempered`, `annealed`, `statistics`. No
module carried a relative import, so each moved file is what it was; the
change is which directory names it and which `CLAUDE.md` states its rules.
`sample/CLAUDE.md` holds the sampler rules that were split between
`opt/CLAUDE.md` and `search/CLAUDE.md`, and each donor keeps what is its own:
an optimizer is judged by the optimum it reaches and a sampler by the
distribution it converges to. 83 files in the tree name `sample` and none
names a donor. Neither counting row moves: the same modules and the same
public names, under another directory. The twelve test modules moved with them
to `tests/regression/sample/`, markers unchanged.

**The `MessageSchedule` guard (issue #755).** The seam #592 wrote is now
asserted rather than remembered: `tests/regression/test_duplication_guards.py`
reads the class tree and the registry, and fails a schedule-shaped class that
does not inherit `likelihood.schedule.MessageSchedule`, a schedule no
`MessageScheduleName` reaches, or a consumer branching on a schedule's name.
**Five schedules, one base, five modules calling through it** --- the
`fields:name` cluster's seventh member, `search.ground_state.Entry`, shares the
field and is not a schedule, so the pin is five. 0.41 s on the `critical` tier,
and both halves are exercised on a violating source.

**The incidence seam is guarded (issue #755).** `test_duplication_guards.py`
reads every package module's syntax tree: no `offsets` or `indptr` array by
`cumsum` and no pair list sorted into row-major order outside `incidence.py`,
against **10** modules that build a store through `SparseIncidence.from_pairs`
or `compressed_adjacency` and **3** files excluded against a reason each ---
`ragged.py` and `search.maxflow` on the measurements above, and
`learn.surrogate`, which the guard found on `main`: `_Batch.__init__` lays a
batch's token blocks end to end and computes their starts with `np.cumsum`.

**Message schedules (issue #592).** The order messages go in is an interface,
`likelihood/schedule.py`, where it was two branches of an `if`. Five schedules
declare a `Guarantee` of three values rather than a boolean, because the
leaf-to-root pass alone is exact where it speaks and silent elsewhere and
neither "exact" nor "approximate" says that: **`upward` returns all of `log Z`
for 1.71x less work** than the two-pass schedule (85.8 ms against 146.4 ms on a
1,000-variable chain), agreeing with it to one unit in the last place and with
`likelihood.pruning.log_likelihood` --- an oracle sharing no code --- to 1e-13
relative over seven sites. A marginal a schedule does not compute is absent
rather than wrong, and `downward` reports `nan` for `log Z` rather than the
nearest available number. `sim.factor_graph`'s six adapters already reach every
graph class, so one seam covers the Potts lattice, the HMM chain, the tree, the
coupled model, the Tanner graph and the trellis with no seventh adapter.

The refactor is bitwise on the two schedules that predate it, asserted on a
200-leaf star and a 50-node caterpillar as well as the suite's chains. It also
uncovered a real defect: `sum_product(graph, schedule="tree")` ran **flooding**,
because a plain string compares equal to a `StrEnum` member without being it and
the branch used `is`.

The cost the ticket named was only half recovered, and the half is recorded
rather than rounded up. Carrying the axis from the breadth-first walk instead of
a `list.index` scan paid **nothing** --- a chain's factors have degree two, so
the scan it replaced was over two elements. Building the group arrays once per
shape instead of once per level paid **1.21x** on the plan (64.2 to 52.9 ms at
2,000 variables), taking its share of the run from 24.9 to 20.8 per cent. What
remains has no hotspot: it is Python bookkeeping proportional to the edges, and
removing it would mean vectorising the level assignment wholesale.
`docs/experiments/018` carries the run.

**A sixth schedule, and where it earns its heap**
([#825](https://github.com/michaelJwilson/snakes_and_ladders/issues/825)).
`residual` orders sends by the largest residual a factor's inputs last saw
(Elidan, McGraw & Koller 2006): a heap over the factors, the runner feeding each
applied step's residual back through `send`, so the seam gains `adaptive` and
`sweep_length` and no consumer names the schedule. It reaches the Bethe fixed
point flooding and sequential reach, pinned against the reference flooding on
the loopy lattice to 1e-9 in `log Z` and 1e-8 in the marginals. Sweeps to a
residual of 1e-10 at damping 0.5, one sweep being every factor sent once
whatever the order: on the 3x3 `ci` lattice 60 flooding, 53 sequential, 60
residual; on an 8x8 at `J = 0.5` in a random field 61, 53, 55; on a 12x12 at
the critical coupling in a random field **254, 196, 176**, where the residual
order is the only one under 200. On the (3,6) Gallager codes at 96 and 996
bits over a Gaussian channel at `sigma = 0.8` it is the worst of the three:
1,761, 1,133, **2,357** sweeps at 96 bits and 1,769, 1,139, 2,135 at 996,
because a parity factor's outgoing residual bumps every neighbour whether or
not its own inputs moved, and the hard constraints keep the residuals large
late into the run. Wall is not the comparison: flooding's one vectorised step
runs 128 ms where the two per-factor orders run 4.9 s and 5.2 s on the 12x12,
a cost of the Python step loop and not of the order, and the decoder keeps its
own layered schedule.

## Consistency audit at 0.4.0

What the release audit ([#358](https://github.com/michaelJwilson/snakes_and_ladders/issues/358))
found stale between the planning documents, the technical documents and the
code, and fixed in the same pull request:

- The summary table above said Viterbi, a trained tree policy and Milestones
  2.2 to 2.3 and 4.1 were not started, and the Milestone 1.4 text said Viterbi was not
  built, while max-product over the chain returns the enumerated Viterbi path
  since #296 and forward–backward the posteriors since #307. Corrected in all
  three places.
- `TICKETS.md`, deleted by #804, listed landed work as open — forward–backward as an evaluator
  (#173), the Rust Gibbs sweep (#246), Viterbi and posterior decoding (#175),
  schedules and tempering (#267), discrete support (#270, #331), the
  initializers (#251), the emission families (#228, #229), the fixture API
  (#132), the documentation vetting and split (#244, #249), the rename (#250),
  the labels (#274), the kinds (#237), the root-detection fix (#168) and the
  hard tree fixture and trained policy (#177, #178) — and named as unfiled what
  has since been ticketed: rate variation (#323), multi-SPR and
  branch-and-bound (#329), device dispatch (#280), the milestone re-keying
  (#324), the uncited figures (#325).
- `DEV.md`'s repository layout put the temperature schedules in `search/`
  (they are `sample.schedule` since #272), described `opt/` and `learn/` by their
  first two instances, and said no memory helper existed (#232 added the
  footprint table); its measured counts were re-taken on this host and dated.
  `INSTALL.md` counted eleven QA scripts and put `mypy` over two directories
  where `pyproject.toml` lists three. `README.md` counted four objective
  instances.
- `PROBLEMS.md` had no row for the coupled model and named neither Sankoff,
  large parsimony, forward–backward, max-product, the factor-graph Gibbs
  sweeps nor the surrogates. `ROADMAP.md` §0.4 implied an oracle is always
  required; it now states the two-way standard, and §1.1 names the fourth class
  and the two derived instances.
- The textbook's coupled section listed as remaining the fixture, the E step,
  the block ascent and the initializer that #302 and #307 landed; its
  density-evolution appendix named a repository file, which the textbook may
  not. §1.3 said the bounds were absent and three placeholders stood, and the
  Milestone 1.4 text carried a literal `CALIBRATION_TABLE` token where #350's
  release-gated table belongs.

**Duplicated machinery.** Not refactored here; each group is stated with the
oracle that would pin a merge.

| entry point | what it is | classification | oracle for a merge |
| --- | --- | --- | --- |
| `sample.potts_mcmc.anneal_potts` | heat-bath sweep per `Schedule` step, best kept | Potts type | draw-for-draw equality with `anneal_factor_graph` on the Potts adapter (measured, 2,000 of 2,000 sweeps) |
| `sample.gibbs.anneal_factor_graph` | the same loop over any `FactorGraph` | general type | as above, plus the triangular ground state on 6 of 6 seeds |
| `sample.gibbs.anneal_topology` | Metropolis over topologies per schedule step | a distinct move in a copied driver | the flat-prior weight at `T = 1` (#270) and the enumerated best at `T -> 0` |
| `sample.hmc.anneal` | one Hamiltonian transition per schedule step | the same driver over a continuous transition | a constant schedule reproduces `hmc.sample` draw for draw |
| `learn.relaxed.anneal` | a geometric temperature at one step | a copy of `sample.schedule.Exponential` | equality at every step with both endpoints exact |
| `sample.potts_mcmc.parallel_tempering` | replicas on spawned generators, Metropolis exchange | Potts type; `_swap_log_ratio` copied | per-replica chi-square against the unscaled enumeration, and identical exchange acceptances once the driver is shared |
| `sample.hmc.parallel_tempering` | the same replica and exchange loop over `torch` generators | continuous type | the analytic Gaussian's spread per replica, and experiment 004's mixture comparison unchanged |
| `sample.tempered` (#350) | replica exchange from the Gibbs moves, weights from the cold replica | a third copy of the replica driver | the tempered weights against enumeration on the three instances #331 pins |

The fix is one driver — a schedule, a transition and a generator in, the best
state and trajectory out — and one exchange step over `(state, energy, beta)`
triples, the eight entry points becoming adapters.

| enumerator | what it enumerates | classification | oracle for a merge |
| --- | --- | --- | --- |
| `search.topology.enumerate_topologies` | every unrooted topology by stepwise insertion | distinct, and stays | the `(2n-5)!!` count |
| `search.max_cut.enumerate_max_cut` | every two-state assignment, best cut kept | product enumeration with a score, one side fixed | the lowest-energy configuration of `enumerate_potts` at two states |
| `learn.potts.enumerate_configurations`, `learn.hmm.enumerate_paths` | `itertools.product` over states and sites | two identical copies, neither under #230's enumeration cap | equality of the sequences |
| `learn.potts.optimum`, `learn.hmm.optimum`, `learn.relaxed.enumerate_optimum` | argmax over the product with a lexicographic tie rule | three copies of one kernel | each other's result on shared instances; `RelaxedPotts.discrete` equals `PottsEnvironment.energy` |
| `likelihood.potts.enumerate_potts` | log weights over the product, then marginals | one algorithm, graph type | its own pin: the transfer matrix on a chain to machine precision |
| `likelihood.hmm_paths.enumerate_hidden_paths` | log joints over the product, then the evidence, posteriors and both decodings | one algorithm, chain type | its own pin: the forward recursion to 1e-12 |
| `likelihood.spatio_sequential.enumerate_spatio_sequential` | log joints over labellings times paths, then three posteriors | one algorithm, coupled type | its own pin: the per-class forward recursion at a relative gap of 0.0 |
| `likelihood.mixture_assignments.enumerate_mixture_assignments` (#393) | log joints over every component assignment, then the evidence, the responsibilities and the argmax | one algorithm, independent-observation type | its own pin: the factorized E step at 1.2e-16 relative |
| `likelihood.ldpc.enumerate_codewords` (#356) | every codeword from the generator matrix | product enumeration over the information bits | `H c = 0` on every word and the `2^k` count |

The fix is one weighted enumeration — cardinalities and a log-weight function
in, assignments and log weights out, #230's cap applied once — with
marginalization and argmax as helpers over it; the four `learn` and `search`
copies become calls, and the three `likelihood` enumerators keep their result
types over the shared kernel.

**Methods refereed by the simulated truth alone, or by neither kind.** Read
from the suite by `infra/problems_tables.py` and typeset in the textbook's
applicability tables. Five of the six closed at 0.5.0
([#393](https://github.com/michaelJwilson/snakes_and_ladders/issues/393)), each at the
CI size and with the agreement it realized:

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
| phylogenetic tree, general time-reversible | the distance start (`FromDistances` on `log_det_distance`) | no test of either significant kind names it as a start | a missing test rather than a missing oracle: `enumerate_topologies` is already the oracle the row would use (#364) |
| phylogenetic tree, Jukes–Cantor | the learned surrogate (`fit_surrogate`) | simulated truth: the maximized likelihood of each enumerated topology | the target a surrogate is trained and scored against is itself a fit, so there is no exact answer to hold it to |
| any problem at the release tier | the HMM, the mixture, the test functions, the lattice and the code | simulated truth only at that tier | out of #393's scope: a branch-and-bound bound for trees past eight taxa (#329); a boundary contraction for lattices past the transfer-matrix width; density evolution beyond the erasure channel for the code (#340 part 2) |

## Two Python paths above a Rust kernel ([#598](https://github.com/michaelJwilson/snakes_and_ladders/issues/598))

Every number below is a median of seven on an idle host — 1-minute load average
0.00, `OMP_NUM_THREADS=1`, nothing else running. The ticket's own profile was
taken with two test suites on 4 cores and is withdrawn; what it claimed and what
a quiet host says are recorded together, because the gap is the finding.

**`likelihood.potts.log_weights`** scored each edge with one NumPy call per
element. Below `GATHER_BELOW` = 128 configurations it gathers instead.

| path | before | after |
| --- | --- | --- |
| `log_weights`, one configuration, 16x16 | 1.405 ms | **0.019 ms (72x)** |
| `log_weights`, one configuration, 32x32 | 5.449 ms | **0.033 ms (166x)** |
| `anneal_potts`, 200 steps, 16x16, `Backend.RUST` | 47.8 ms | 46.4 ms, **unchanged** |
| `anneal_potts`, 200 steps, 32x32, `Backend.RUST` | 158.5 ms | 157.7 ms, **unchanged** |

The function is two orders of magnitude faster and the run that calls it is
not. The ticket put `energies` at 84% of `anneal_potts`; 200 steps at 5.449 ms
is 1,090 ms against a whole run of 158.5 ms, so the per-step term cannot have
been 84% of the run containing it. Where the 158 ms goes is
[#608](https://github.com/michaelJwilson/snakes_and_ladders/issues/608).

The crossover is measured rather than chosen: over lattices from 18 to 2,048
edges the gather wins 1.68–2.77x at 128 configurations and 2.4–5.4x below 96,
and runs 0.72–1.70x at 256, so the threshold sits below the mixed region.

**`search.maxflow.FlowNetwork.from_arcs`** builds a network with no Python call
per arc. It is the smaller half by an order of magnitude. The 8x8 row is a
validation size, not an optimization size, and is printed for completeness
rather than as a result:

| network build, `Backend.RUST` | `add_edge` loop | `from_arcs` |
| --- | --- | --- |
| 8x8, 3 / 5 states, 112 edges (a gate size) | 0.254 / 0.265 ms | 0.390 / 0.351 ms (0.7 / 0.8x) |
| 16x16, 3 / 5 states, 480 edges | 1.059 / 1.098 ms | 0.966 / 1.089 ms (1.1 / 1.0x) |
| 32x32, 3 / 5 states, 1,984 edges | 4.860 / 4.760 ms | 4.046 / 4.176 ms (1.2 / 1.1x) |

End to end, `alpha_expansion` runs 18.8 → 16.4 ms and 30.0 → 27.9 ms at 16x16
for 3 and 5 labels, and 74.0 → 62.4 ms and 172.0 → 142.9 ms at 32x32:
**1.08–1.20x**. The ticket's 53.0% / 12.1% split was measured under the same
load as the withdrawn profile and is withdrawn with it.

So the two halves are kept for different reasons. The gather is kept because it
is 72–166x on its own shape. The network build is kept for the layout — one
construction against 76,928 `add_edge` calls, pinned arc for arc on 320
networks — and **not** for a speed claim: 1.2x at the largest size measured,
slower below 16x16, and no threshold was added because `alpha_expansion` is not
run at 8x8 in anger.

## Three kernels take a thread pool, three proposals do not ([#627](https://github.com/michaelJwilson/snakes_and_ladders/issues/627))

Every number is a criterion median of 100 samples on an idle 4-core host ---
1-minute load 0.07, `OMP_NUM_THREADS=1`, nothing else running --- against the
serial medians the same command took before the port.

| kernel | serial | `rayon` | ratio | kept |
| --- | --- | --- | --- | --- |
| `pruning_log_likelihood/8taxa_200000sites` | 117.64 ms | **30.818 ms** | **3.82x** | yes |
| `external_field 200x5041 M=K=10` | 93.432 ms | **39.310 ms** | **2.38x** | yes |
| `sample_rows/2000000` | 26.694 ms | **13.091 ms** | **2.04x** | yes |
| `pruning_log_likelihood/4taxa_200000sites` | 58.119 ms | 15.259 ms | 3.81x | yes |
| `sample_rows/200000` | 2.2644 ms | 1.2126 ms | 1.87x | yes |
| `class_posteriors 200x5041 M=K=10` | 23.064 ms | 26.703 ms | **0.86x** | **no** |
| `external_field`, parallel over positions | 93.432 ms | 43.642 ms | 2.14x | **no** |
| `pruning_log_likelihood/8taxa`, reassociated | 117.64 ms | 30.903 ms | 3.81x | **no** |

**Three kept, and all three are bit-identical to the serial path.** Each writes
each result exactly once, and where there is a reduction it stays sequential,
so no oracle tolerance is spent and a run on a 4-core host equals a run on a
64-core one. `tests/` pins that for pruning at four window widths.

**`class_posteriors` is declined by measurement**, the way #591 declined three
proposals. `m` is a clean axis --- each class writes its own slice and nothing
is summed across them --- and it still runs at **0.86x**: ten items against
four cores, and `forward_backward` allocates per class, so the pool costs more
than it saves.

**The other two are declined although they were permitted.** The maintainer
relaxed the requirement from bit-identical to `likelihood/CLAUDE.md`'s relative
`1e-11` where a looser form ran faster, and a reassociated pruning sum measured
a relative deviation of **1.85e-13**, two orders inside it. Neither looser form
was faster. The reassociated reduction saves one `f64` per site and runs
**30.903 ms against 30.818 ms**; parallelising `external_field` over its 5,041
positions instead of its 200 vertices runs **43.642 ms against 39.310 ms**. So
the tolerance stays unspent, not on principle but because buying nothing with
it was the measured outcome.

**Two expectations the measurement overturned.** Inverting `external_field`'s
loops to keep the sum sequential was expected to cost locality, since
`totals[s * n_nodes + v]` walks contiguously in `s` and strides in `v`; it is
11% *faster*, because each vertex then owns a contiguous output slice and the
table reads are shared-read. And the per-site buffer pruning needs to keep its
sum ordered was expected to cost time; at 8 taxa the bit-identical form is
faster than the one without it.

The maximum-flow network and `double` are unchanged, as they must be: neither
is touched, and `max_flow_expansion_network/32x32` reads 396.93 us against
385.89 us, inside the spread.

## A graph owns its array form ([#623](https://github.com/michaelJwilson/snakes_and_ladders/issues/623))

`PottsGraph` stores its edges as a tuple of pairs, which is what a fixture
declares, and no consumer uses that: every one of them indexes an array by it.
Fifteen call sites across seven modules wrote `np.asarray(graph.edges)` or
`np.asarray(graph.coupling)` *inside* a function, so a 200-step run paid the
conversion 604 times ---
[#608](https://github.com/michaelJwilson/snakes_and_ladders/issues/608)
measured that at **81% of `anneal_potts` at 32x32**, against the Rust kernel's
8%. `edge_index` and `edge_coupling` derive it once, as `cached_property`.

Medians of seven, one BLAS thread, on an idle 4-core host, the same script run
against `main` and against this branch:

| run | `main` | this branch | ratio |
| --- | --- | --- | --- |
| `anneal_potts`, 200 steps, 32x32, `Backend.RUST` | 159.9 ms | **27.6 ms** | **5.79x** |
| `anneal_potts`, 200 steps, 16x16, `Backend.RUST` | 45.9 ms | **11.1 ms** | **4.14x** |
| `alpha_expansion`, 32x32, 5 labels | 70.1 ms | 55.1 ms | 1.27x |
| `alpha_expansion`, 32x32, 3 labels | 46.8 ms | 36.4 ms | 1.29x |
| `alpha_expansion`, 16x16, 5 labels | 15.7 ms | 12.7 ms | 1.24x |
| `alpha_expansion`, 16x16, 3 labels | 9.8 ms | 7.8 ms | 1.26x |

**The prediction and the measurement agree, which is the point of stating
both.** A term at 81% of a run predicts `1 / (1 - 0.81)` = 5.3x once it is
gone; the measurement is 5.79x. That is also what settles the #598 dispute
recorded above: #609 made `log_weights` 72--166x faster and moved
`anneal_potts` not at all, because the run's time was in the conversion rather
than in that function. This is the change that moves it.

**The cached arrays are handed out unwritable.** A `cached_property` returns
the same object to every caller and the graph is frozen, so a consumer writing
through one would corrupt every later call silently; `setflags(write=False)`
makes it a `ValueError` at the write. It decides a real case rather than a
hypothetical one: `torch.as_tensor` shares a writable NumPy buffer, so the two
surrogate call sites copy explicitly instead.

## The package has a map, and building it found the roadmap silent on 35 modules ([#664](https://github.com/michaelJwilson/snakes_and_ladders/issues/664))

`docs/mind_map.pdf` is one page: the two concerns, the eight packages, and
every one of the 139 modules, each labelled with the milestone this file claims
it under. Generated by `infra/mind_map.py`, never drawn.

**The join is the work; the figure is what it prints.** A module's role is not
its docstring restated --- it is the roadmap claim it carries, and this file is
where those are recorded. Reading it that way found the record incomplete:

| | modules |
| --- | ---: |
| claimed by a milestone, before | 67 |
| application modules claimed by none | **35** |
| of those, named nowhere in this file | **28** |
| claimed after | **102**, and 0 application modules unclaimed |

The 28 include `sim.jc`, `sim.simulate`, `sim.tree`, `sim.newick`,
`likelihood.brute_force`, `likelihood.device` and `opt.objective` --- the
foundations of the simulator, the oracle and the fitting interface. Their work
had landed years of tickets ago; no milestone section named the module, so no
reader could get from a roadmap claim to the code that implements it. Six
milestone sections now carry a **Modules** line, and a guard fails a PR that
adds an application module without one.

**A first measurement of this was wrong and is corrected here.** Bounding each
milestone's section at the next *milestone* heading swept the free-form
sections after Milestone 4.1 into it, crediting 4.1 with 11 modules it claims
nothing about. Sections end at the next `##` heading of any kind; the numbers
above are after that fix.

**`forest` was the intended typesetting and is not usable.** `forest.sty` ships
in this TeX Live, but `environ.sty`, `trimspaces.sty` and `elocalloc.sty` do
not, so it cannot load without installing TeX packages. TikZ `graphdrawing`
needs LuaLaTeX where the build runs pdflatex. The polar coordinates are
computed in the generator instead and emitted as plain TikZ, so no `.tex`
places a node and no dependency was added.

**The map is radial, and two numbers decide whether that is readable.** Each
branch takes a wedge proportional to the leaves it carries --- `application`
draws 95 against `infrastructure`'s 20, so equal halves would give one branch
four times the room per leaf --- which leaves every leaf the same **3.1
degrees**. At the 96 mm leaf radius that is 5.2 mm of arc against a name of
about 15 mm, so a horizontal label collides with its neighbour by a factor of
three. A label rotated to run radially outward is bounded by its *height*,
about 2 mm, and fits with room to spare. Past the top of the circle it is
turned through 180 degrees so it does not read upside down.

**Hovering a module shows its docstring**, as a transparent PDF annotation over
each label --- 115 of them, one per drawn leaf. `\pdfannot` is a pdfTeX
primitive so this needs no package: `pdfcomment` is the idiomatic route and is
absent here, with `soul`, `soulpos`, `zref-abspage` and `marginnote`. The text
crosses two escapes, LaTeX's before the PDF's, which is where a first attempt
failed: parentheses become brackets, the LaTeX specials are dropped and `_` is
written `\string_`. Support is the reader's --- Acrobat and most desktop
readers show it, Chrome's viewer and pdf.js do not --- so the page states every
role without it.
## The problem axis reads imports, not only fixture calls ([#622](https://github.com/michaelJwilson/snakes_and_ladders/issues/622))

#619 derived a per-problem selection from the registry call a test module
makes. Measured over the tree, it missed more than it found.

| | before | after |
| --- | ---: | ---: |
| modules carrying no problem marker | 138 of 258 | **70** |
| of those, application or benchmark code | 78 | **16** |

Both modules #614 names as its motivation --- `search/test_maxflow.py` and
`search/test_alpha_expansion.py`, "that problem's ground-state tests" --- were
among the 78, so a green `-m potts_lattice` run over a broken solver was the
failure the axis was opened to prevent and did not.

Neither loads a fixture, and neither can. They *sweep*: `test_alpha_expansion.py`
builds 15 lattices, each chosen for the property under test --- zero coupling,
a dominant one, a negative one, a periodic boundary. There is no single declared
instance to load, and declaring 15 fixtures to carry 15 deliberate variations
would make the registry a list of test arguments.

The second reading is over the module's **imports**, against `PROBLEMS.md`'s
**Defines** column. That file already stated the rule --- *a test module
importing any of it exercises the problem* --- and `tests/_problems.py` did not
implement it, so the catalogue's own sentence was untrue. It is held by
`test_problems_catalogue.py`, which resolves every symbol a row names, so the
reading is as current as the code rather than a second map to maintain. A
module-level `PROBLEM = "<name>"` constant was the alternative and is the map
this work removes, one level up: rewrite a module's model, forget the constant,
and the axis is confidently wrong rather than visibly empty.

One catalogue gap fell out. `opt.potts` is "a 1-D Potts chain in an external
field" in its own first line and no row named it; it now defines `potts_chain`.

The upper-bound guard changed with the premise: a module selected for a problem
must spell the problem's name **or** import code the catalogue says defines it.
It re-reads `PROBLEMS.md` by regex where the scan uses `ast`, so the two
readings still share no code.

## Chains of unequal length, and what padding costs ([#666](https://github.com/michaelJwilson/snakes_and_ladders/issues/666))

A batch of chains is not one long chain. The recursions restart at each
boundary, so the number of boundaries is part of the problem rather than of its
size, and `sal.ragged.Ragged` carries the segments end to end
with their lengths. A segment of one position is refused: it is all initial
distribution and no transition.

**There is one recursion, not two.** `baum_welch_family` takes either shape and
the rectangular form converts; the route it replaced is conserved in
`sandbox.rectangular_hmm` and referees the equal-length case **bit for bit**.
88 existing HMM tests pass unchanged through the new path.

**The compiled kernel pads nothing, and that is the whole finding.** The Python
path pads to the longest segment and masks, which buys one batched step per
position of the longest; `src/ragged.rs` walks the segments in place. Which
wins is a question about the *lengths*, not about the language. Every number is
a best of five on this host at `OMP_NUM_THREADS=1`, `torch` single-threaded,
four states:

| segments | total | padding waste | torch | Rust | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 x 500 | 32,000 | 0.0% | 93.14 ms | **33.41 ms** | 2.8x |
| 200 mixed, 5 to 400 | 41,546 | 47.8% | 121.63 ms | **43.37 ms** | 2.8x |
| 63 x 30 and 1 x 2000 | 3,890 | 97.0% | 386.42 ms | **3.98 ms** | **97.1x** |

**The third row is the one to read, and it is not a Rust result.** At 3,890
positions the padded path takes 386 ms, while at 41,546 positions --- ten times
the data --- it takes 122 ms. The cost follows the *longest* segment times the
segment count, which is 128,000 padded positions for 3,890 real ones. A single
long segment among short ones is therefore the shape where the Python path is
worst, and it is the shape the downstream consumer has.

The per-segment NumPy oracle runs 980 ms, 1,311 ms and 128 ms on the same
three, so it referees and does not compete. The compiled kernel is pinned to it
at a relative `1e-11` on the marginals, the transition counts and the evidence.

**Two keys, not two problems** (#666 step 4). `ragged_hmm` and
`spatio_sequential_ragged` join the HMM and coupled rows: each model is
unchanged and only its instance's segmentation differs, which is what
`PROBLEMS.md` means by a row with two keys. `ragged_hmm/ci` declares 2, 9, 9, 9
and 60 --- the shortest a segment may be, a run of equal lengths, and one long
enough that a padded block would be 70.3% padding --- so the batch takes 84
transitions rather than 88. `spatio_sequential_ragged/ci` splits the same S = 6
chain as 2 and 4, so enumeration still referees it.

`SpatioSequentialParams` gains `segments`, and the simulator draws the first
position of every segment from the initial distribution rather than from the
transition out of the position before it, which belongs to another chain. With
none declared the draws are the ones it has always made.

**The HMM fixture loader now takes `lengths` or the rectangular pair, and
exactly one.** A fixture that declared both could contradict itself.

**`lengths` is the only declaration of a batch's shape.** `hmm/ci.yaml` now
writes its 600 chains of 15 as the lengths themselves, and `HmmParams` derives
`n_sequences` and `sequence_length` rather than storing them --- a second field
for a derived fact is a field that can disagree, and on a ragged batch there is
no shared length to hold. Asking a ragged instance for `sequence_length`
**raises**; an earlier draft returned the longest, which is the quiet wrong
number the segmentation exists to prevent.

**The draws did not move.** The simulator groups segments by length and draws
each group as it always did, so the equal-length case is one group and the same
RNG stream: `hmm/ci` reproduces its states and observations byte for byte
against the old spelling, which is checked rather than assumed.

**What is left is the coupled scorer, and it is named rather than deferred
vaguely.** `sim.spatio_sequential` draws the declared segments; the enumeration
oracle and the message-passing fit in `likelihood.spatio_sequential` still
score the chain as one, at eight sites reading `n_positions`. Until they carry
the segmentation, a fit at `spatio_sequential_ragged` would optimize a
likelihood the instance does not have, so the three pairings that need it say
exactly that in `docs/tex/method_notes.yaml`, and the work is issue #669. `ragged_hmm`'s two say something
different: a path sampler over segments is the unsegmented sampler run S times,
and a bound over 89 positions costs more than the exact evaluation.

## Concurrency shape before a thread pool ([#612](https://github.com/michaelJwilson/snakes_and_ladders/issues/612))

Neither `rayon` nor `tokio` is a dependency. #610 has just put nine of the ten
`#[pyfunction]`s under `Python::detach`, taking threaded throughput from 1.05x
to **3.70x** at four threads, so a thread pool inside a kernel can overlap with
Python for the first time. What it should be is decided here by shape, not by
subsystem.

Counts are at the declared scale (`ROADMAP.md`: `n` to 1000, `L` to 11,000) on
a 4-core host.

| axis | independent items | against 4 cores | does an item ever wait? |
| --- | --- | --- | --- |
| `pruning_log_likelihood_impl`, sites | up to **11,000** | ~2,750x | no: one contiguous row, no early exit |
| `count_pairs`, vertex blocks | `n / VERTEX_BLOCK` = 1000/64 ~ **16** | 4x | no |
| tempering replicas | the ladder length, single digits | ~1x | no |
| heat-bath sweep | **1** | --- | a Markov chain: the next site reads the last |
| Dinic maximum flow | **1** | --- | each augmenting path reads the residual the last left |

**`tokio` is declined, and not for the reason first written down.** It is not
only an I/O runtime --- it carries M:N lightweight tasks over a work-stealing
scheduler, `spawn_blocking` and `block_in_place`. The argument is the table: a
lightweight task is cheap because it is a `Future` polled cooperatively, which
pays where concurrency greatly exceeds cores *and tasks spend their life
suspended*. Every item above is CPU-saturating with no await point, so it holds
its worker to completion and achieved parallelism is the worker count --- what
a plain pool gives with less machinery, and what `tokio`'s own guidance sends
to `rayon`. That no kernel in `src/` references `std::fs`, `std::net` or
`std::io` is corroboration, not the argument.

**The deciding property is an ordered reduction.** A parallel sum over sites
reassociates `log L` and moves its last bits, which the oracle tests pin.
`rayon` has `fold` and `reduce` over an *indexed* parallel iterator, so
per-site partials combine in index order and the arithmetic is unchanged.
`tokio` offers no such primitive; it would be hand-rolled, and hand-rolling is
where a reassociated sum gets in.

So one crate is worth asking for, and the last two rows are not candidates for
either: parallelising them means changing what they compute.

**The ranking is now measured**
([#627](https://github.com/michaelJwilson/snakes_and_ladders/issues/627)).
`cargo bench --locked` under `OMP_NUM_THREADS=1` on an idle host --- 1-minute
load 0.07, no agent and no suite running, the condition #598's withdrawn
profile lacked. Criterion medians of 100 samples:

| kernel | median | independent items, from the table above |
| --- | --- | --- |
| `pruning_log_likelihood/8taxa_200000sites` | **117.64 ms** | one per site |
| `external_field 200x5041 M=K=10` | 93.43 ms | one per site |
| `pruning_log_likelihood/4taxa_200000sites` | 58.12 ms | one per site |
| `sample_rows/2000000` | 26.69 ms | one per row |
| `class_posteriors 200x5041 M=K=10` | 23.06 ms | one per site |
| `sample_rows/200000` | 2.264 ms | one per row |
| `max_flow_expansion_network/32x32` | 385.9 us | **1**, Dinic is sequential |
| `max_flow_expansion_network/16x16` | 92.0 us | 1 |
| `max_flow_expansion_network/8x8` | 22.2 us | 1 |
| `double` | 704.8 ps | --- |

Pruning is the top of the ranking *and* the widest axis, so it is the one
kernel where a pool can pay, and it is what `rayon` should take first. The
maximum-flow network is three orders of magnitude below it and carries one
item, so it is a candidate on neither count --- which corroborates #598
independently: that half was kept for its layout, not for a speed claim.
`external_field` places second and is not in the table above; its shape is
counted before anything is written, not assumed.

**Still not measured:** whether a site-parallel pruning clears root
`CLAUDE.md`'s 2x bar against the NumPy reference at realistic `(sites, taxa)`.
No number is claimed here that was not taken.

## The stress-tier ranking, per problem family ([#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754))

**No family's top term is Rust's, and the one Python loop at 95% of its run
is the BCJR trellis.** `tests/benchmarks/profile_hotpaths.py --tier mid`,
which now covers the codes as well --- they were the one family with no
workload, so nothing ranked them. Two readings on an idle host, 4 cores,
1-minute load 0.06 and 1.00 at the two starts; module walls agreed to 0.02%
(`search` 56.18 and 56.19 s, `opt` 28.86 and 28.64 s). `docs/experiments/020`
carries the summary; this is the full ranking, five terms per workload.

| family | workload, `--tier mid` | wall | top five by self time | verdict |
| --- | --- | --- | --- | --- |
| trees | `search.infer` NNI, 20 taxa x 1,000 sites, 20 evaluations | 20.81 / 21.39 s | `run_backward` 50.5%, `pruning_torch._post_order` 17.5%, `LBFGS.step` 4.0%, `amax` 3.0% (41,058), `add_` 1.9% (124,204) | autograd is **not ours**; the post-order is the **algorithmic cut** taken below; the last two are a few µs each and **below the effect-size bar** |
| trees | `search.infer` SPR, same instance | 28.25 / 28.51 s | `run_backward` 53.8%, `_post_order` 19.0%, `LBFGS.step` 4.6%, `amax` 3.2% (59,004), `add_` 2.2% (191,897) | as above |
| Potts | `potts_mcmc` single-site, 32x32, 20 sweeps, `Backend.PYTHON` | 0.289 / 0.290 s | `_site_update` 29.1%, `heat_bath_log_weights` 13.9%, `cumsum` 10.0%, `searchsorted` 7.6%, `_wrapfunc` 7.2% | ~~default to flip~~ **stale, and corrected below**: [#599](https://github.com/michaelJwilson/snakes_and_ladders/issues/599) flipped it before this was read, and the row is the oracle route asked for by name |
| Potts | `SwendsenWangMove.propose`, 64x64 open, 8,064 edges | 42.19 / 42.28 ms, **0.32 ms** after two cuts and the port | `_swendsen_wang_sweep` 30.8%, `_recolour` 26.2% (24,372 calls, 2,437 clusters a sweep), `flatnonzero` 5.3%, `_find` 5.2%, ufunc `reduce` 4.5% | **cut, then ported**, [#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754): union-find and a Python loop per cluster. `WolffMove.propose` is 0.555 / 0.549 ms at the same lattice and stores its layout already, so the cluster moves are one ranked loop, not two --- the section below |
| HMM/coupled | `message_passing` tree schedule, chain 200 | 0.237 s, **0.031 s** after the port | `schedule.tree_passes` 29.8%, `_logsumexp_last` 10.6%, ufunc `reduce` 8.8%, `_send_from_variables` 7.9%, `_factor_terms` 4.4%; after the port `is_tree` 11.4%, `Layout.__init__` 10.4%, `find` 9.9% | **ported**, [#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754): 800 levels of NumPy dispatch, which is control flow and not arithmetic. 9.05x on the enclosing `sum_product` and the default flipped --- the section below |
| codes | `convolutional.bcjr`, K = 1,024; `turbo.decode_turbo`, 8 iterations | 47.0 ms / 151 ms, **3 / 5 ms** after the port | `bcjr` 95.8% and 95.5%; after the port the extension call, 53.5% and 84.2% | **ported**, [#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754): a four-state trellis walked forward and backward in Python, `cache=True` unavailable to it and no NumPy axis to vectorize over. 26.2x on the decode and the default flipped --- the section below |
| codes | `ldpc.decode` sum-product / min-sum, 996 bits, 50 iterations | 9.3 / 8.9 ms | `_tanh_rule` 31.6% / `_min_sum` 34.7%, `decode` 23.4 / 19.1%, `reduceat` 19.7 / 31.6%, `syndrome` 5.9%, `_clip` 4.5% | **below the effect-size bar**: already one `reduceat` per iteration over the edges, and the whole decode is 9 ms |
| mixtures/HMC | `sample.hmc.sample`, 1,000 draws, chain 64 | 24.57 / 24.31 s, **7.25 s** after the cut | `run_backward` 41.7%, `torch.logsumexp` 33.2% (512,000 calls), `log_partition` 10.2%, `unsqueeze` 6.0% (512,000), `Tensor.to` 1.5%; after the cut `run_backward` 40.4%, `logsumexp` 23.4% (96,000), `log_partition_by_squaring` 6.7%, `unsqueeze` 5.0% (136,000), `PottsObjective.__call__` 3.7% | **cut**, [#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754): the homogeneous transfer product reassociated by squaring, 12 `logsumexp` calls for 64 positions where there were 64. 3.57x on the enclosing `sample`, and the first term was **not** outside the package --- it is this loop's tape and it falls with the loop, 10.19 s to 2.93 s --- the section below |
| learn | `learn.reinforce`, 60 x 32 episodes, chain 8 | 5.60 / 5.50 s | `potts.features` 13.1%, `run_backward` 6.4%, `policy.sample` 6.2%, `surrogate_loss` 4.5%, `np.fromiter` 4.2% (68,706) | **below the effect-size bar**: #341 already cut `features` 1.32x, and what is left is 0.73 s spread over 34,353 calls with no term above 14% |
| search | `gibbs.sample_factor_graph`, 32x32, 20 sweeps | 2.03 / 0.96 s | `ffi.__call__` 15.6% / `templates.register_global` 16.1%, `marshal.loads` 3.1 / 6.6%, `abc.__new__` 8.1%, `isinstance` 2.2%, `ir._rec_list_vars` 1.8% | **not ours**: every term is `numba`, and the two readings differ by 2.1x because the first wrote the cache the second read |

**The Gibbs sweep's `numba` terms are a cold start, and it is already cached.**
All three kernels in `search/kernels.py` carry `cache=True` and the objects
are written beside the source, under the worktree's
`python/sal/search/__pycache__/`. Deleting them and measuring
the first call in a process separates the two costs: **1.108 s** cold against
**0.628 / 0.620 s** with the cache present, so the cache is worth 0.48 s and
is working. What remains is paid once per process and is not the sweep ---
every call after the first is **23.4 / 23.9 ms** at one sweep, **24.2 / 24.0 ms**
at twenty and **46.4 / 46.9 ms** at two hundred, which puts the sweep itself at
**0.123 ms** and the rest at a fixed 23.4 ms of graph flattening per call.
Nothing was changed: the enumerable tier's 12.5% `cpu_options.__init__` and
7.7% `marshal.loads` are a process paying its JIT once, and a ratio taken
against a 20-sweep run measures the harness rather than the sampler.

**One cut was taken, and the profile's share of it was an overstatement.**

| loop | pin | before | after |
| --- | --- | --- | --- |
| `pruning_torch.log_likelihood`: the post-order hoisted out of the evaluation. A topology fixes the traversal, so the schedule --- the nodes in post-order, each child's branch index baked in --- is built once per topology object and the evaluation is a flat loop over it, where it was a Python frame and two dictionary lookups per node per evaluation | log-likelihood, gradient and the `rescale=False` branch **bitwise** unchanged at 4, 6, 10 and 20 taxa (realized difference: zero, compared as `float.hex()`); `tests/regression/likelihood/test_pruning_torch.py` and `test_pruning_gradient.py` unchanged | value and gradient, `pytest-benchmark` median: 20 taxa x 500 sites **5.891 / 5.863 ms**, 20 x 2,000 **9.448 / 9.408 ms**, 50 x 500 **15.317 / 15.772 ms**, 50 x 2,000 **26.287 / 27.194 ms**. NNI search, 20 evaluations at 20 taxa: **19.40 / 17.94 s** | **5.798 / 5.724 ms** (1.02x), **8.971 / 8.980 ms** (1.05x), **14.343 / 14.647 ms** (1.07x), **24.032 / 24.367 ms** (**1.09x**, 2.54 ms). NNI **18.36 / 17.18 s** |

The ratio grows with the leaf count and not with the site count, which is
what a per-node cost predicts, and it is **far below the 17.5% the profile
attributed**. `cProfile` charges a frame per call, and the post-order was
38 frames per evaluation; removing them removes a cost the profiled run pays
and the real run largely does not. The honest number is the benchmark's:
2.54 ms of a 26.3 ms evaluation at 50 taxa, and 0.90 s of an 18.7 s NNI
search --- above the precedent root `CLAUDE.md` cites for leaving a term
alone (1.2 ms of 270 ms) by an order of magnitude, and bought with no second
language, no second implementation and no change to any number. **This is the
ranking's first correction: a self-time fraction is where to look, and it is
not the saving.** Every fraction in the table above is to be read that way,
and the walls beside them are what a port will be judged against.

**One ranked term was measured and not cut.** `_swendsen_wang_sweep` rebuilds
the edge endpoints per sweep with two `np.fromiter` passes over
`graph.edges`, where `PottsGraph.edge_index` has held them as one array since
[#623](https://github.com/michaelJwilson/snakes_and_ladders/issues/623). At
64x64 the pair is **0.900 / 0.674 ms** against `edge_index`'s **0.001 ms**,
which is **1.9%** of the 42.2 ms `propose` that encloses it. The store exists,
the change is one line and the values are the same `int64` indices --- and
1.9% is the effect size, so it goes with the port that takes the 26.2% beside
it rather than as a cut of its own (#754).

## The BCJR trellis in Rust ([#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754))

**26.2x on the eight-iteration turbo decode at the declared `K = 256`, which
saves 35.6 ms of 37.0, so the default flipped.** The ranking above put `bcjr`
at 95.8% of one pass and 95.5% of a decode, one Python loop over `K + m`
steps with four states to vectorize over; `src/bcjr.rs` runs the same
recursions and `likelihood.convolutional.rust` marshals the two trellis
tables and the three ratio vectors across once, contiguous and borrowed, with
the GIL released. `pytest-benchmark`, two readings, 1-minute load 1.40 and
1.79 on the shared 4-core host:

| mean, two readings | NumPy | Rust | ratio |
| --- | --- | --- | --- |
| `bcjr`, `K = 256` (`turbo/stress.yaml`) | 2.319 / 2.311 ms | 66.6 / 66.1 us | 34.8x / 35.0x |
| `bcjr`, `K = 1,024` | 9.023 / 8.873 ms | 275.1 / 272.7 us | 32.8x / 32.5x |
| `decode_turbo`, `K = 256`, 8 iterations | 37.00 / 37.19 ms | 1.410 / 1.412 ms | 26.2x / 26.3x |
| `decode_turbo`, `K = 1,024`, 8 iterations | 143.5 / 141.6 ms | 4.778 / 4.737 ms | 30.0x / 29.9x |
| the stress waterfall, 6 points x 50 frames | 11.37 s | 0.46 s | 24.7x |

**The ratio and the effect size point the same way, which is why this one was
taken and the LDPC decode beside it was not.** A pass is 2.32 ms of a 37.0 ms
decode and sixteen of them are 36.1, so the port is the decode; `ldpc.decode`
is 9 ms whole and stays NumPy. The residue after it is `_iterate`'s
interleaving at 4.7% and the wrapper at 1.5%: what is left to take is 0.2 ms
of 1.41, which is below the effect-size bar.

**Agreement is bitwise where the repository decodes, and a reassociation
above it.** `logaddexp` in the kernel is NumPy's `npy_logaddexp` branch for
branch and the log-sum-exp is `numerics.logsumexp`'s shift by the row
maximum, so at `memory` 1 and 2 --- four states, the `(7, 5)` register every
fixture and both benchmark lengths use --- the posterior, the extrinsic and
the log evidence are **equal to the last bit**, and the `ci` waterfall
returns the recorded 0.115, 0.061, 0.033 and 0.013 with the per-iteration
decisions identical frame by frame. At `memory` 3 and 4 NumPy's pairwise sum
switches to eight accumulators and this kernel stays left to right: realized
**2.3e-13** absolute and **2.2e-12** relative over `K` up to 1,024, inside
`CROSS_DEVICE_RTOL_FLOAT64`, with the log evidence still bitwise. The suite
asserts the equality and the bound separately rather than the looser one
everywhere. Both backends are pinned to `exact_bitwise_posterior` as well, so
the pair is not established by agreeing with each other.

`docs/experiments/024` carries the table.

## The Swendsen-Wang cluster pass ([#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754))

**2.04x from two algorithmic cuts and 63.4x more from the port: the
ranking's 42.2 ms `propose` is 0.32 ms.** The cuts come first because
root `CLAUDE.md` ranks them above a mechanical one, and because a ratio read
against an uncut reference is the reference's and not the port's. Both are
bitwise: the same members, the same order, the same values.

| the pass at 64x64, three states, at the transition | two readings |
| --- | --- |
| as the ranking read it (`perf_counter`, x10) | 41.59 / 41.19 ms |
| `beta * rows` hoisted out of the per-cluster loop --- a whole-field multiply and an allocation per cluster, and there is a cluster for every 1.7 sites | 30.66 / 31.01 ms (1.35x) |
| the clusters grouped by one stable `argsort` where each had scanned the whole labelling: `O(n_clusters * n_nodes)` to `O(n log n)` | 20.36 / 20.13 ms (**2.04x**) |
| the edge ends read from `PottsGraph.edge_index` instead of two `np.fromiter` passes (#623's store, the 1.9% #757 measured and left) | included above, 0.900 ms to 0.001 |

`pytest-benchmark`, two readings, 1-minute load 2.01 and 1.93 on the shared
4-core host, NumPy against Rust: `SwendsenWangMove.propose` **20.200 /
20.692 ms** against **318.5 / 327.6 us**, **63.4x / 63.2x**, which saves
**19.88 ms of the 20.20 ms pass that encloses it**; `sample_potts` over ten
Swendsen-Wang sweeps **168.9 / 173.6 ms** against **7.96 / 8.25 ms**,
21.2x / 21.0x. Ratio and effect size point the same way and the pass is
99% of the call that holds it, so there is nothing to weigh them against
each other over.

**The stream is not the oracle's, and that is why the default is not
flipped.** The oracle draws a cluster's colour and then, only where the field
difference is negative, its accept uniform --- what the next draw *is*
depends on the last one's outcome, so no array replays it and the kernel
cannot consume NumPy's generator. The Rust route draws the bond uniforms the
oracle draws, then one colour and one uniform per cluster in bulk: each is
independent and identically distributed as the oracle's own, so the chain is
of the same law and is not the same chain. `Backend.PYTHON` therefore stays
the default on `_swendsen_wang_sweep`, `SwendsenWangMove` and
`sample_potts`'s new `cluster_backend`, and flipping it is its own decision
against the recorded numbers (#754).

**Given the same draws it is the oracle bitwise, by construction rather than
by luck.** The bond probability `1 - exp(-beta J)` and the scaled field cross
as arrays NumPy evaluated, so the bond pass and the union-find are the
oracle's arithmetic exactly --- the labels are equal root for root, not
merely the same partition. What the kernel adds is a cluster's field sum,
which it takes left to right where NumPy takes it pairwise, and `exp` of the
difference. Both are thresholds, and both are guarded by `potts_mcmc._GUARD`
on the derivation the single-site sweep uses: two summation orders of `m`
terms differ by at most `2m` units of the last place of the sum of their
magnitudes, the width is eight times that, and a cluster inside it is handed
back and decided by NumPy before the kernel resumes. Over twelve passes at
16x16 nothing was handed back; the suite forces the path with a wide guard
and gets the same configuration.

The law is pinned where the stream cannot be: the enumerated 2x2 Boltzmann
distribution at two seeds, with and without a field, and the ablation that a
pass handed a zero field fails it. `docs/experiments/025` carries the table.


## The tree schedule in Rust ([#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754))

**9.05x on the `sum_product` that encloses it at the chain of 200, which
saves 21.6 ms of 24.3, so the default flipped --- and the algorithmic cut
beside it is 1.23x and was declined.** The ranking above put the tree
schedule's plan at 29.8% and its `logsumexp` at 10.6%: a chain carries one
message per level, so 200 positions are 798 steps of NumPy calls over a
single four-wide row. `src/message_passing.rs` walks the breadth-first order
and its reverse instead of grouping by height --- a message depends only on
messages of lower height, so the levels are a batching of the order and not a
constraint on it --- and `likelihood.message_passing.rust` marshals the
layout across once as offsets and flat arrays with the tables stacked,
borrowed, with the GIL released. `pytest-benchmark` mean, two readings,
1-minute load 3.05 and 3.51 on the shared 4-core host:

| mean, two readings | NumPy | Rust | ratio |
| --- | --- | --- | --- |
| `sum_product`, tree schedule, chain 200 | 24.310 / 23.778 ms | 2.687 / 2.563 ms | 9.05x / 9.28x |
| `max_product`, same chain | 21.171 / 18.598 ms | 2.763 / 2.723 ms | 7.66x / 6.83x |
| `sum_product`, chain 2,000, min of five | 213.3 / 212.7 ms | 21.14 / 20.95 ms | 10.1x |
| `--tier mid` self time, x5 | 0.245 s | 0.031 s | 7.9x |

**The cut was measured before the port, and it is the one this ranking asked
for first.** Root `CLAUDE.md` puts doing less work before doing the same work
faster, and the plan is rebuilt per call from the layout, so a plan stored
per graph is the cut the profile names. Removing the build entirely is its
upper bound: **21.003 / 20.544 ms to 17.123 / 16.663** at the chain of 200
and **213.3 / 212.7 to 163.0 / 163.6** at 2,000, which is **1.23x** and
**1.30x**. That is below the 2x bar on its own, it buys nothing on the single
call every caller in the package makes, and it is subsumed --- the kernel
builds no plan at all, and storing one would leave a derived copy of the
tables to keep in step with the graph. So it was not taken, and the number is
recorded rather than the reasoning alone.

**Two other numbers the port settles.** The general algorithm was **4.7x
behind** the specialised Torch forward recursion on this chain when #341
measured it; it is now **2.11x / 2.21x ahead** of it (5.661 / 5.675 ms), so
the cost of the generality is no longer paid. And the section's top term is
no longer the dispatch: it is `FactorGraph.is_tree` at 11.4% with its
union-find `find` at 9.9%, and `Layout.__init__` at 10.4% --- a check over
formatted strings and an incidence walk, both rebuilt per call, both left for
the ticket that now names them.

**The declared tolerance, and what is in the last place.** The kernel's
arithmetic is the oracle's operation for operation --- the same sums in the
same order, the reduced axes walked in the row-major order
`transpose(...).reshape(n, c, -1)` produces, a degree-one factor returned
unreduced so an indicator's `-inf` stays `-inf` --- and what is left is that
NumPy's vectorized `exp` and `log` differ from `libm`'s in the last place, on
3.4% and 1.0% of inputs here, which `sal.backend` already
states as the standing reason a compiled route is not bitwise. So the
comparison backs off one step: the max-product assignment, the guarantee and
the pass count are **equal**, and the marginals and `log Z` are held to
`CROSS_DEVICE_RTOL_FLOAT64`, realized **3.3e-15** absolute and 7.2e-15
relative on the marginals and **1.4e-14** absolute and 2.0e-16 relative on
`log Z`, over chains at three cardinalities, the Potts tree, its Forney form
and random trees of factor degree three and four.
`test_message_passing.py`'s bitwise pins against the dictionary reference
keep that claim by naming `Backend.PYTHON`, where it still holds exactly; the
one case that had left it was the Forney form, by 2.8e-17.
`docs/experiments/027` carries the table.

## The chain's transfer product by squaring ([#754](https://github.com/michaelJwilson/snakes_and_ladders/issues/754))

**3.57x on `hmc.sample` at the instance the ranking read, saving 17.0 s of
23.5 over 1,000 draws --- and the 41.7% the ranking called "not ours" was
this loop's tape, which falls with it.** The Potts chain is homogeneous: one
`J` and one `h` at every site, so the normalizer is a *power* of one matrix
and a power reassociates. `opt.potts.log_partition_by_squaring` takes
`T ** 63` in five squarings and six vector-matrix products where the
recursion took 63 in sequence, which is 12 `torch.logsumexp` calls per
evaluation against 64. The ticket predicted 6 products and 1.5–2x on the run;
both were low. Two readings, `perf_counter`, 1-minute load 4.72 to 6.32 on
the shared 4-core host:

| two readings | recursion | squaring | ratio |
| --- | --- | --- | --- |
| `hmc.sample`, 1,000 draws, chain of 64, q = 3 | 23.501 / 23.597 s | 6.601 / 6.511 s | 3.57x |
| the objective's forward + backward, one gradient | 3.352 / 3.394 ms | 0.948 / 0.923 ms | 3.60x |
| the objective's forward alone | 1.343 / 1.495 ms | 0.327 / 0.322 ms | 4.35x |
| `log_partition` alone, `pytest-benchmark` mean | 1,209.7 / 1,389.8 us | 245.4 / 246.9 us | 4.93x / 5.63x |
| the same with its gradient | 3,130.9 / 3,146.7 us | 717.7 / 742.5 us | 4.36x / 4.24x |
| `--tier mid` self time, `run_backward` | 10.191 s | 2.930 s | 3.48x |
| `--tier mid` self time, `torch.logsumexp` | 8.187 s, 512,000 calls | 1.693 s, 96,000 | 4.84x |

**The route reads the alphabet, not the chain length.** Squaring pays a cube
in `q` per step to buy a logarithm in the length, so which route costs less
is a statement about the two together: over `q` in 2..64 and lengths 2 to
1,024 it is **4.51x** at `q = 3, length = 64` and **0.04x** at
`q = 64, length = 4`. A rule reading the length alone would take both.
`squaring_is_cheaper` therefore counts the work: it takes squaring only where
`q * floor(log2 n) + popcount(n) <= n`, which is where it makes no more calls
*and* touches no more elements than the recursion and so cannot lose on
either term. That is conservative by construction --- it declines 1.25x at
`q = 3, length = 8` and 2.53x at `q = 32, length = 64` --- and never
regresses, because the recursion is what the function always did. The `ci`
fixture's own chain of 12 falls on the recursion side, so the gate-tier
numbers are unchanged, which is the policy: a speedup is established at
stress sizes alone.

**The reassociation reorders the sums, so agreement steps down one rung and
the step is measured.** It is **bitwise** at lengths 1 and 2, where both
routes run the same sequence of operations, and elsewhere the guard is
`CROSS_DEVICE_RTOL_FLOAT64`: realized **6.815e-16** relative at the stress
instance and **6.13e-14** the largest over `q` in 2..8 and lengths 1 to 129
where the route is taken, four and three orders inside the declared 1e-11.
Both routes are refereed by brute-force enumeration at lengths 1 to 8, the
squaring route by `likelihood.potts.strip_log_partition` on an open
`length x 1` strip at 2, 5, 16 and 64 --- the rung `infra/ladder.py` now
records below it --- and the gradient by central differences at three lengths
and three couplings. The HMC pins on the Potts posterior are `q = 2` on a
chain of 8 and so run the new route: the quadrature mean and spread and the
step-size pins hold at the values they held on the recursion.
`log_partition_by_recursion` stays as the oracle.
`docs/experiments/028` carries the table.
