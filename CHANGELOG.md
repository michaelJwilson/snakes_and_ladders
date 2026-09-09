# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

New entries are managed as [towncrier](https://towncrier.readthedocs.io)
fragments under `changelog.d/` (see `changelog.d/README.md`) and merged into
this file at release time; the `[Unreleased]` section below predates that
convention and is retained as history.

<!-- towncrier release notes start -->

## [0.5.0] - 2026-09-09

### Added

- Sum-product belief propagation over a Potts Markov random field (`snakes_and_ladders.likelihood.belief_propagation`), returning single-site and pairwise marginals and the Bethe free energy, with two exact oracles beside it in `snakes_and_ladders.likelihood.potts`: exhaustive enumeration, and a 2-D strip transfer matrix whose cost is exponential in the strip's width rather than its site count. Belief propagation is exact on a tree and pinned there against enumeration; on a loopy lattice it is approximate and its deviation from the exact strip is reported rather than asserted. Messages that do not converge raise instead of returning a number. (#172)
- Swendsen-Wang and Wolff cluster updates on the Potts lattice (`snakes_and_ladders.search.potts_mcmc`), beside single-site heat bath behind one move-set interface, with the external field handled by a Metropolis accept step on the cluster's field difference rather than ignored. Each is validated by a chi-square goodness-of-fit against the exact Boltzmann distribution at an enumerable size, with and without a field, and the test's power is itself pinned by an ablation that removes the accept step. `snakes_and_ladders.search.statistics` adds the chi-square tail probability and an integrated autocorrelation time, both written rather than taken from a new dependency and both pinned against published or closed-form values. (#174)
- A tree fixture NNI hill climbing does not already solve, at `tests/regression/fixtures/tree_search/release.yaml`. Seven taxa with internal branches an order of magnitude shorter than the pendant ones: hill climbing reaches the enumerated maximum of all 945 unrooted topologies from 24 of 50 seeded starts and stops at a genuine local optimum on the rest, while the generating topology remains that maximum. The 6-taxon fixture, where greedy succeeds from every start, cannot separate a learned policy from it in either direction. (#177)
- A QA figure, `rl_tree_policy`, reporting what a learned proposal policy does on the tree fixture hill climbing does not solve: the landscape's local optima, and the policy's success rate against greedy's over independent training seeds at a matched per-episode budget. (#178)
- Worked notebooks under `docs/nb/`, one per problem class -- the Potts chain, phylogenetic trees, and the discrete hidden Markov model. Each runs simulation, fitting, discrete search and, where one exists, policy learning against the oracles the regression suite establishes, and ends with a Further Work section naming what it could not demonstrate and the issue that carries it. (#180)
- Added a self-time profiling harness (`infra/profile_harness.py`,
  `tests/benchmarks/profile_hotpaths.py`) auditing `snakes_and_ladders.sim`, `snakes_and_ladders.search`
  and `snakes_and_ladders.learn` for Rust-port opportunities. `snakes_and_ladders.numerics.sample_rows`
  clears the bar -- 2.7x (200,000 sites) to 3.9x (2,000,000 sites) over NumPy
  in an oracle-pinned spike, filed as #187; NNI/SPR neighbourhood generation,
  the candidate fit, and the RL rollout do not, with measured reasons on #181. (#181)
- `snakes_and_ladders.learn.rollout.rollout` takes `stop_at_local_optimum`, and `snakes_and_ladders.learn.policy` gains `EpsilonGreedyPolicy` and the narrower `Policy` protocol it satisfies. Together they let an episode continue past a state no action improves and take the worsening move that leaves it — measured on the issue #177 fixture, where escape from a local optimum rises from 0.111 at epsilon 0 to 0.883 at 0.4. Both default to the previous behaviour. (#194)
- `snakes_and_ladders.learn` carries two more instances of `Environment`: the Potts landscape over an arbitrary graph, via `PottsLandscape.on_graph`, and `snakes_and_ladders.learn.hmm.StatePathLandscape`, a search over hidden state paths at known parameters. Both are pinned against the enumerated estimator oracle in `snakes_and_ladders.learn.exact` and against exhaustive enumeration of their own state spaces, and neither takes an application type — the no-application-imports rule holds, so a caller unpacks a `PottsGraph` or an `HmmParams` into plain arrays. (#195)
- `snakes_and_ladders.opt.potts.PottsLatticeObjective` fits a Potts model's coupling and external field on an arbitrary graph, against a normalizer enumerated exactly over every configuration. Interval coverage measured over 40 replicates is 0.981 at 100 samples and 0.956 at 400 and 1600, approaching the nominal 0.95 from above; the enumerated normalizer reduces to the chain's transfer matrix to machine precision. (#197)
- Continuous integration re-executes every notebook under `docs/nb/` and fails a pull request whose re-executed output disagrees with the committed one. `infra/check_notebooks.py` both checks and, with `--write`, regenerates, so the two always execute a notebook identically; it strips the per-cell wall-clock timestamps nbclient records, so regenerating an unchanged notebook rewrites nothing. Text outputs are compared exactly; figures are checked for still being produced rather than byte-compared, since rendered images embed metadata that is not stable across matplotlib builds. The kernel installs with `uv sync --extra notebooks` and is not part of a normal install. Its first run rejected `hmm.ipynb` and `potts_chain.ipynb` for printing a converged optimizer's residual gradient norm, which moves by orders of magnitude between machines while the parameters it reports agree to four decimals; both now print the tolerance it cleared. (#203)
- Alpha expansion for Potts MAP at any label count (`snakes_and_ladders.search.alpha_expansion`), built from `snakes_and_ladders.search.maxflow`'s exact minimum cut one label at a time, with the single-site descent baseline it has to beat. It carries the repository's first proved approximation bound — within a factor 2 of the global optimum for a uniform Potts coupling — which holds at every size rather than only where enumeration reaches. Measured over 40 runs at an enumerable size it found the global optimum 39 times; past enumeration it beat the best of eight single-site descents on every trial. (#207)
- Hamiltonian Monte Carlo over any `snakes_and_ladders.opt.Objective` (`snakes_and_ladders.opt.hmc`), so an interval can be a quantile of the posterior rather than the curvature at the mode. The leapfrog integrator is pinned where it is exact — reversible to `1e-15`, and second order in the step size at a measured ratio of exactly 4.00 across four halvings — before the chain is checked against an analytic Gaussian and against the Potts chain's posterior integrated on a grid. `WithGaussianPrior` makes the prior an explicit declaration, since a bare likelihood read as a density is a posterior under an improper flat prior. The chain reports its per-proposal energy error because acceptance rate alone does not detect a step size that biases the posterior spread. (#208)
- `snakes_and_ladders.sim.canonical` holds three problem instances whose answer is known from outside this repository, each consumed by more than one module. The triangular Ising antiferromagnet has an exact ground-state energy at every size — a double count over its `3N` edges and `2N` triangles fixes exactly `N` agreeing edges, attained at `N = 9, 12, 16` — which makes it the only discrete instance here whose optimum is known past enumeration rather than bounded. A planted Viana-Bray spin glass carries a state of known energy, an upper bound on the ground state at any size. An HMM with ambiguous emissions is the case where Viterbi and posterior decoding return different answers, both non-degenerate: the Viterbi path is unique by 0.3033 nats and every posterior marginal exceeds 0.6256. `snakes_and_ladders.likelihood.hmm_paths` enumerates all `k ** T` paths and reads both decodings off one enumeration, pinned against the forward recursion to 1e-12. (#209)
- An exact ground state for the two-state submodular Ising model, by Dinic maximum flow (`snakes_and_ladders.search.maxflow`, with a Rust kernel in `snakes_and_ladders.search.maxflow_rust` running 28-34x faster as a kernel and 6.6-10.6x as a caller sees it). It is the repository's first discrete optimum that is proved rather than enumerated, so a heuristic past the ~20 sites enumeration reaches now has a reference. Validated against enumeration at exact equality, against two analytic corners far past it, and by the max-flow min-cut theorem as a self-check. A negative coupling and more than two states are refused rather than approximated: the first is NP-hard, the second is alpha expansion. (#210)
- `snakes_and_ladders.learn.relaxed` implements the Gumbel-softmax half of `ROADMAP.md` Stage 3's differentiable-search bullet: continuous relaxations of Potts configurations and HMM state paths, whose optima are enumerable and can therefore referee the claim. The relaxation reduces to the discrete objective exactly at every corner, and `E_q[score] = score(q)` holds for any multilinear objective — which licenses a deterministic ascent with no sampling at all. Both Gumbel-softmax estimators are measured against the exact gradient rather than assumed accurate: over 20000 draws the bias falls from 0.598 at `tau = 2.0` to 0.036 at `tau = 0.1` while the standard deviation rises from 0.165 to 3.39. (#211)
- Rosenbrock, Rastrigin and Himmelblau as `snakes_and_ladders.opt.Objective` instances (`snakes_and_ladders.opt.testfunctions`), so the optimizer is checked against minimizers known in closed form rather than only against likelihood surfaces where an early stop and a weakly identified parameter look the same. The autodiff gradient is pinned against hand-written closed forms, all four of Himmelblau's equal minima are shown reachable, and Rastrigin measures what a single fit can claim: 0 of 200 uniform starts reached the global minimum, every one reporting convergence. (#213)
- `snakes_and_ladders.sim.graph.erdos_renyi_graph` draws `G(n, p)` beside `lattice_graph`, and belief propagation is now measured over an ensemble rather than on three hand-built fixtures. Acyclicity is checked per draw so the oracle stays exhaustive enumeration and no asymptotic threshold is invoked: BP is exact on every acyclic draw to 3.7e-15 relative, and the deviation on cyclic draws is reported. The ensemble reaches structures no committed fixture did — 104 of 120 draws carried an isolated vertex, the case on the boundary of the edgeless special path. (#214)
- Max-Cut as the antiferromagnetic Ising ground state (`snakes_and_ladders.search.max_cut`), with a Goemans-Williamson semidefinite relaxation and its 0.87856 guarantee. On random graphs with triangles at 12, 16 and 18 nodes the rounded cut reached the enumerated optimum every time, with the computable certificate at 0.95 to 0.98. The relaxation is solved approximately by Burer-Monteiro gradient ascent rather than by taking an SDP dependency, so the certificate is weaker than the theorem — a test pins the symptom, a ratio slightly above 1 on a bipartite graph, which an exact solve could not produce. (#215)
- Fitch parsimony (`snakes_and_ladders.likelihood.parsimony`), scoring a topology with no model and no branch lengths, pinned against exhaustive enumeration over internal-node labellings. It exists for the Felsenstein zone, where parsimony is statistically inconsistent: measured over 12 replicates it recovered the true topology 0 of 12 times at 200, 1000 and 5000 sites while likelihood went 10/12, 12/12, 12/12. The Farris zone is committed alongside as the control — parsimony 12/12 at every site count, likelihood 4/12, 6/12, 10/12 — because an implementation that were simply broken would fail both. (#216)
- An emission-family interface for hidden Markov models (`snakes_and_ladders.emissions`), with the categorical matrix as one implementation and univariate Gaussian emissions as the second. `HmmParams` now carries a family rather than a matrix; `GaussianHmmObjective` fits the continuous case, and `baum_welch_family` runs EM over any family. The Gaussian likelihood is unbounded, so its variance floor is derived from the data and reaching it raises rather than clamps. (#228)
- Negative binomial emissions for hidden Markov models, the first family whose Baum-Welch M step is an optimization rather than a formula. `Reestimate` now carries what the M step did — whether it settled, whether a parameter reached the bound the data identifies it over, and the weighted score at the answer — and Baum-Welch refuses an M step that did not settle. (#229)
- `snakes_and_ladders.qa.likelihood_footprint` states the memory a simulation and a likelihood
  evaluation hold across `ROADMAP.md` §1.2's declared scale, and reports the
  declared maximum against the 16 GB requirement, closing the `O(n x L x k)` row
  `STATUS.md` carried as unmeasured. The figures are computed from the arrays'
  shapes so they survive a change of machine, and the regression suite pins each
  one against the allocator's count (#232). (#232)
- A turbo code as the second member of the fourth problem class: a recursive systematic convolutional encoder from octal generator polynomials with its trellis as `(state, input)` arrays and its tail (`sim.convolutional`), a seeded random interleaver, the rate-1/3 unpunctured turbo encoder, and the same code as a `ParityCheck` so #340's enumeration reaches it; log-MAP BCJR with the observation on the edge, Viterbi over the same branch metrics, and enumeration over all `2 ** K` messages as the oracle (`likelihood.convolutional`); the `from_trellis` factor-graph adapter, so the general sum-product decodes the chain; and the serial extrinsic exchange between two BCJR passes with the uncoded closed form `Q(sqrt(2 E_b / N_0))` and the seeded waterfall measurement (`likelihood.turbo`). BCJR equals enumeration to 2.8e-14 in log-odds and the tree-schedule `sum_product` to 2.7e-15; at `K = 1024` the bit error rate falls from 5.1e-2 at 0 dB to below 1/204,800 at 1.5 dB over 200 frames per point. The GF(2) elimination `sim.ldpc` already owned is now one function, `sim.ldpc.null_space`: the parity-check encoder takes the null space of `H` and the turbo code takes the null space of its generator, which were two copies of it. The forward recursion reads a gather rather than a scatter, ranked by a `cProfile` that put 21.9% of a `K = 1024` decoding's self time in `numpy.ufunc.at`: one BCJR pass is 20.9% faster at `K = 1024` and eight turbo iterations 31.8%, with the enumeration oracle unmoved at 2.8e-14. (#233)
- Every test now says what it is checked against. Five kind markers — `oracle`,
  `simulated_truth`, `mathematical`, `edge_case`, `structural` — are registered
  in `pyproject.toml` and required of every test outside `tests/benchmarks/` by
  `tests/regression/test_test_kinds.py`, which is itself tested to fail on an
  unmarked test. `critical` is a second, independent axis marking the 76 tests
  that gate early: CI runs them in 3.0 s before selecting anything, so a broken
  invariant reports in seconds rather than after the rest of the suite. Kinds are
  tags rather than a partition, because a test may legitimately be refereed two
  ways (#237). (#237)
- `snakes_and_ladders.search.potts_mcmc_rust` runs the single-site heat-bath
  sweep in Rust, beside `potts_mcmc` rather than replacing it: **77x** the Python
  sweep at 64 nodes and **108x** at 1,024, with the ratio rising rather than
  decaying because the adjacency crosses once in compressed-row form and the
  uniforms cross as one array.

  Nothing switches to it, and that is deliberate. Rust's `f64::exp` agrees with
  NumPy's to within a unit in the last place rather than exactly, and
  `searchsorted` is a threshold, so one draw across a boundary that moved by 1 ulp
  sends the two chains apart permanently — which would move every autocorrelation
  figure `STATUS.md` pins and every notebook output that reads a chain. The two
  backends are compared by the distribution they converge to, against exhaustive
  enumeration, with a field and without (#246). (#246)
- `snakes_and_ladders.opt.initialize` makes where an optimization starts the
  caller's choice: `FromObjective` (today's behaviour), `Perturbed` for a
  surface whose nominal start is stationary, and `RandomRestart` drawing from a
  passed-in generator. `opt.fit.fit_from` runs every start and returns the best
  fit, all of them, and their spread — a multi-start fit that reported only the
  best would hide the multimodality it exists to expose.

  Multi-start is measured, not assumed. On Himmelblau's four equal minima a
  single fixed start reaches one basin and four restarts reach all four; on
  Rastrigin's dense local minima sixteen restarts reach the global minimum 2
  times in 30 against 0 in 30 from one start, and widening the draw does not
  help. **No default changes**: every number `STATUS.md` pins is still produced
  from the objective's own start (#251). (#251)
- Poisson, binomial and beta-binomial emissions, completing the dispersion axis around the negative binomial: under-dispersed, equidispersed and over-dispersed on a bounded support. Each is an exact limit of a neighbour, so `BetaBinomial(n, 1, 1)` and `Binomial(1, p)` are checked as equalities and the Poisson and binomial limits at the rate their truncation predicts. (#260)
- A finite Gaussian mixture as a problem class — `snakes_and_ladders.sim.mixture` generates, `snakes_and_ladders.opt.mixture` fits — whose component M step is the emission family's, unchanged from the hidden Markov case. `KMeansPlusPlus` is the first initializer that reads its objective's data, and refuses an objective whose parameters it cannot interpret. (#262)
- Yoshida's fourth-order symplectic integrator, selectable in `snakes_and_ladders.opt.hmc.sample` beside leapfrog. Both are compositions of the same kick-drift-kick sub-step, and `Integrator.force_evaluations` reports what a trajectory costs so the two can be compared at equal gradient evaluations rather than at equal steps. Leapfrog remains the default: it reaches the same acceptance for a quarter of the gradients on the objectives measured. (#266)
- `snakes_and_ladders.opt.schedule` — temperature schedules (`Constant`, `Linear`, `Exponential`, `Cosine`) mirroring `torch.optim.lr_scheduler`, with both endpoints reached exactly at the declared steps. `search.potts_mcmc.sample_potts` and `opt.hmc.sample` take a `temperature`; `search.potts_mcmc.anneal_potts` and `opt.hmc.anneal` run the same transitions on a schedule and return the best state visited; `search.potts_mcmc.parallel_tempering` runs replicas at fixed temperatures with Metropolis exchanges, each replica on its own spawned generator. At temperature 1 every existing chain is unchanged bitwise. (#267)
- `snakes_and_ladders.opt.fit.standard_errors_at` gives a fit an interval from the observed information at that point, whatever optimizer produced it, and `fit(include_intervals=True)` / `fit_from(include_intervals=True)` attach one to the fit they return. Every objective now implements `theta_from`, the inverse of its constraint map, so an expectation-maximization fit — which never builds an unconstrained vector — can be given the same interval a gradient fit gets. The refusals are unchanged and gain one: a point where the information is singular, indefinite or ill-conditioned is still refused, and so is an unconverged fit, rather than either being given a meaningless interval. (#268)
- A discrete search result states its support (`search.support`): the neighbourhood weight and margin under a move set, the enumerated weight over every topology where it fits, and Felsenstein's bootstrap per split, each named as what it is and pinned where enumeration reaches. (#270)
- Topology search fits each neighbour from its parent's branch lengths, matched by leaf split, and can rank a neighbourhood by one cached likelihood evaluation before fitting the top candidates (`infer(..., warm_start=, lazy_top=)`); `Inference` reports fits and likelihood evaluations separately. (#289)
- A factor graph (`sim.factor_graph`) with adapters from a tree, a Potts graph, a hidden Markov chain and the coupled spatio-sequential model, its Forney normal form, and one sum-product / max-product (`likelihood.message_passing`) pinned to pruning, the path enumeration and belief propagation. (#290)
- `PROBLEMS.md` catalogues the supported problems with the code behind each, resolved by a test; `CHECKS.md` lists every `oracle` and `simulated_truth` check, generated by `infra/checks_ledger.py` and compared by CI like a figure. (#291)
- The coupled spatio-sequential model of #290 has declared parameters, a simulator under one generator (`sim.spatio_sequential`), and an enumeration oracle for its evidence and posteriors (`likelihood.spatio_sequential`), pinned against the factor graph and against the per-class forward recursion. (#300)
- The coupled spatio-sequential model is fitted: forward–backward as an evaluator (`likelihood.forward_backward`, closing #173), the per-class E step and external field, block-coordinate ascent with alpha expansion, single-site descent or an annealed Wolff move as the label solver, `Emission_Mixture++` seeding and the `Graph_BurnIn++` annealed start (`search.spatio_sequential`, `opt.mixture`), with a notebook from fixture to fitted labels. (#306)
- Surrogates for the expensive evaluations, each carrying its claim: certified analytic bounds (plug-in and parsimony bounds on a topology's maximized log-likelihood; mean-field and spanning-tree bounds on a lattice's `log Z`, and the ground-state bracket they give), learned predictors and calibrated bounds on the bound features (`snakes_and_ladders.learn.surrogate`), and `infer(..., surrogate=)` to rank a neighbourhood by any of them before fitting the top candidates. (#308)
- One Gibbs sampler and one annealer over any factor graph (`search.gibbs`): a tempered heat-bath sweep, an exact block move for chain-shaped subsets, and a Metropolis move over tree topologies, each pinned against enumeration or exact marginals, with the generic sweep's cost against the Potts kernels measured. (#309)
- `snakes_and_ladders.log`: the run logger. Every line carries the elapsed minutes since the entry point's start and the run's current phase, shared across every logger in the process; `warning_once` and `info_once` say a thing once. The QA scripts, `qa.build` and `infra/check_notebooks.py` log through it, and fits and searches log at DEBUG through the standard logger. (#311)
- A shared plotting style beyond the series palette: colour utilities (`blend_with_white`, `with_opacity`, an eight-colour Okabe–Ito `STATE_PALETTE` with `discrete_palette`), a `notebook_style` at screen resolution, and `snakes_and_ladders.qa.layout` with the compositions more than one figure needs — a spatial grid of features over 2-D coordinates, grouped tracks with a gap between groups, a joint distribution with per-group marginals, and a discrete legend. (#312)
- A state-value critic (`snakes_and_ladders.learn.critic`), the actor–critic that uses it as a baseline (`learn.actor_critic`), proximal policy optimization with generalized advantage estimation and an optional scheduled epsilon-greedy behaviour policy (`learn.ppo`), PUCT search and expert iteration (`learn.planning`), an `MLPPolicy` beside the linear one, and exact action values and optimal values by enumeration (`learn.exact`) as the oracles they are held to (#313, parts 1 to 3 as far as `dev` supports them; the factor-graph environment and the surrogate reward wait on #296 and #308). (#313)
- The experiment ledger: `docs/experiments/` holds one file per measured comparison, written from `TEMPLATE.md` with its commit, feature under test, fixture and size tier, methods, budget, seeds, results, finding and actions; `infra/experiments.py` validates every file and generates the index, and a guard runs it per pull request. Seeded with three comparisons `STATUS.md` stated inline. (#314)
- A `frameworks` extra (`gymnasium`, `rustworkx`, `torchrl`, `torch_geometric`) and what it fronts: `snakes_and_ladders.search.gym.GymnasiumEnvironment`, a `gymnasium.Env` over any `learn.Environment` with a padded `Box` observation, `Discrete(n_max)` actions under `info["action_mask"]`, termination at a local optimum, truncation at the decision budget and `info["evaluations"]`, pinned by Farama's `check_env` and by an episode round-tripped through both interfaces; `PottsGraph.to_rustworkx` / `from_rustworkx`, with the lattice and `G(n, p)` generators refereed by `rustworkx.generators` and the ground-state cut by `networkx`; TorchRL's `GAE` and `ClipPPOLoss` as unit oracles for `learn.ppo`, and PyTorch Geometric's `GINConv` for `learn.surrogate.GraphSurrogate`; and `snakes_and_ladders.sandbox`, the oracle home a replaced implementation moves to, with a guard that only tests and QA import it (#322, closing #242 into it). (#322)
- The textbook's derivations appendix derives each of the three exact evaluators the main text cites rather than states: Felsenstein's pruning as the marginalization over internal states (`app:pruning`), forward–backward as the same marginalization on a chain (`app:forward-backward`), and sum-product on a tree with the Bethe free energy of a factor graph as its loopy stationary point (`app:sum-product`, `eq:bethe-factor`); the pruning, forward–backward, path-enumeration and message-passing modules cite them. (#326)
- `FeatureSet.FULL` for the tree environment: per move, the improvement it buys, the change in Fitch parsimony score, the pattern support of the split it breaks and of the split it makes, and the sizes of the two subtrees it exchanges, each standardized within the neighbourhood; the single improvement column stays the default. `split_pattern_support` and `pattern_support` in `search.support`, `sign_test_p_value` in `search.statistics`, and the known-parameter reward under GTR from a given rate matrix. On the hard 7-taxon fixture the full set reaches the enumerated maximum from more starts than the single feature and than hill climbing, recorded in `docs/experiments/005-tree-policy-features.md`. (#328)
- `search.support` reports the neighbourhood and enumerated weights for a labelling of any factor graph -- a Potts configuration or a hidden path -- over single-site changes, held to `enumerate_potts` at `beta = 1` and to the enumerated path posterior; a tempered ensemble over topologies and over labellings (`search.tempered`, replica exchange from the moves of `search.gibbs` on the ladder of `opt.schedule`) gives a marginal weight over the whole space with its exchange acceptance and autocorrelation time beside it; and the calibration of the neighbourhood weight and the bootstrap is re-measured at seven and eight taxa behind the release gate. (#331)
- Parallel tempering over a continuous `Objective` (`opt.hmc.parallel_tempering`, replicas of the Hamiltonian transition exchanging positions by Metropolis), the exact McNemar test on two methods' paired hits (`opt.budget.mcnemar`, `Comparison.paired_p`) with a relative tolerance on `Comparison.hits`, and the five-component mixture comparison at equal likelihood evaluations — multi-start EM, annealing and tempering over 40 shared starts — recorded as `docs/experiments/004` and cited from `STATUS.md`. (#332)
- Adaptive samplers, opt-in and reported. `snakes_and_ladders.opt.hmc.sample` takes an `Adaptation(warmup, target_acceptance, step_jitter)`: a warm-up sets a diagonal mass matrix from the warm-up sample variance and the step size by dual averaging to the target acceptance (Hoffman & Gelman 2014, §3.2), then the chain is drawn at those fixed values; `HmcChain.adapted` reports the step, the mass diagonal and the warm-up acceptance, and `HmcChain.force_evaluations` what the chain cost. `snakes_and_ladders.opt.hmc.effective_sample_size` estimates the effective sample size by Geyer's initial positive sequence. `snakes_and_ladders.opt.schedule.adapt_ladder` chooses a parallel-tempering ladder from measured exchange acceptance, inserting, removing and moving temperatures until every neighbouring pair exchanges inside a stated band, and `snakes_and_ladders.search.potts_mcmc.adapt_ladder_potts` runs it on a Potts instance. The fixed-parameter samplers are unchanged bitwise. (#333)
- Sankoff's weighted-step-matrix parsimony (`snakes_and_ladders.likelihood.parsimony.sankoff_score`), of which Fitch is the unit-cost case — equal to `fitch_score` on every topology of the five-taxon fixture, and to a brute force over internal labellings under an asymmetric matrix — and large parsimony (`snakes_and_ladders.search.infer.parsimony_search`): the hill climb of `infer` on the Fitch or Sankoff score under NNI and SPR, with the same budget accounting, reaching the enumerated minimum from every start at five and six taxa and returning the wrong tree on the Felsenstein-zone fixture where the likelihood optimum is the right one. (#335)
- A low-density parity-check code as the fourth problem class: Gallager's regular ensemble as offsets into one edge array (`sim.ldpc.gallager_code`), the binary symmetric, erasure and Gaussian channels behind one log-likelihood-ratio interface, a GF(2) encoder at `n <= 512` asserting `H c = 0`, the `from_parity_check` factor-graph adapter, a vectorized flooding decoder with the `tanh` rule and min-sum and a syndrome stop (`likelihood.ldpc.decode`), codeword enumeration for exact bit posteriors and the ML codeword at `n <= 24`, and density evolution on the erasure channel. The decoder is pinned to the general sum-product on six loopy codes and to enumeration on a cycle-free one; the 19,998-bit (3,6) code brackets the published threshold 0.4294 at the release gate. (#340)
- `snakes_and_ladders.parallel.map_tasks` is the one seam for CPU parallelism over independent tasks: results in input order, one generator per task spawned from the caller's so a run at four workers is bitwise the run at one, `spawn` start method, an explicit intra-op thread count per worker, and a task's exception re-raised with the item that raised it. `opt.fit.fit_from`, `opt.budget.compare` and `search.support.bootstrap_support` take an explicit `workers=` (`1` is serial; no default); a bootstrap replicate now draws from its own spawned generator rather than the shared one, so its frequencies at a given seed change once. Measured on a 4-core host, no site reaches 2× at 4 workers at the mid-size tier — the pool's spawn and import exceed the work — so all three are recorded in `STATUS.md` as negative results and callers pass `1`. (#344)
- The tree's data-driven starts, the equivalent of the HMM's spectral method (#364).
  `snakes_and_ladders.likelihood.distance` estimates pairwise distances from an
  alignment — the Jukes–Cantor closed form and the log-det distance — each with
  its delta-method variance; `snakes_and_ladders.search.neighbor_joining` builds
  the tree from them, exact on additive distances, with the four-point condition
  and Atteson's radius beside it; `snakes_and_ladders.likelihood.hadamard` is the
  Hadamard conjugation for the two-state symmetric model at up to 12 taxa, with
  the four-state Jukes–Cantor alignment reduced to it exactly. `FromDistances`
  and `FromHadamard` in `snakes_and_ladders.search.initialize` are the fourth and
  fifth `Initializer`, each also naming a start topology for the search. No
  default changes: the objective's own start remains what every pinned number
  was produced from. (#364)
- `snakes_and_ladders.sandbox.gym_vector.vector_environment` and `rollout_batch`: a batch of
  episodes stepped together through `gymnasium.vector.SyncVectorEnv`, with
  `TimeLimit` carrying the decision budget and `RecordEpisodeStatistics` the
  episode boundary. At one copy it equals `learn.rollout.rollout` under the same
  generator, draw for draw. It is slower per episode than the sequential rollout
  at every batch size measured and is not the default; the numbers are in
  `docs/experiments/007-batched-rollout-and-torchrl-ppo.md`. (#392)
- `snakes_and_ladders.likelihood.mixture_assignments.enumerate_mixture_assignments`,
  the mixture's evidence and E step summed over every component assignment, on
  the footing `likelihood.hmm_paths` occupies for a chain. With it, the three
  cells the applicability tables marked as refereed by the simulated truth alone
  and enumerable at the CI size are now pinned to enumeration: the HMM block
  sweep, the coupled model's burn-in, and the mixture's evidence, objective and
  k-means++ start. (#393)
- Two rendered instance figures beside the textbook's hand-drawn sketches, which state a structure and show no instance. `snakes_and_ladders.qa.tanner_graph` draws the enumerable (3, 6) Gallager code -- 12 bits, 6 checks, 36 edges under seed 340 -- as a Tanner graph and as the matrix it comes from, the band covering consecutive bits drawn heavy in both. `snakes_and_ladders.qa.coupled_labelling` draws the coupled model's planted labelling on the 10x10 lattice beside the one block ascent recovers from the annealed start, and the external field at the fit, ringed at the 15 of 100 nodes the field alone mislabels. Both lay their nodes out from the instance rather than from a randomized graph layout, and a test pins the coordinates, because a layout an input stamp cannot see is one a re-render does not reproduce. Rendering in 2.7 s and 3.0 s against the 30 s cap on a cited figure, both cited by the textbook's LDPC and coupled sections. (#394)
- `snakes_and_ladders.emissions.CountPairEmission`, a two-channel count emission: a negative binomial for a total and a beta-binomial for the successes within it. Its two forms are selected by a keyword-only `joint` with no default, because they are different generative models — the independent form fixes the beta-binomial's trial count and adds the two log-densities, the joint form takes the trial count from the drawn total, as a coverage and its allele count.

  A mixture of count emissions as a supported problem — `snakes_and_ladders.sim.emission_mixture` generates, `snakes_and_ladders.opt.emission_mixture` fits — whose component M step is the emission family's, unchanged from the hidden Markov and Gaussian-mixture cases, and whose start is the `Emission_Mixture++` seeding rule that had no model to seed until now.

  The coupled spatio-sequential model at 5,041 vertices, with two-channel count
  emissions. `snakes_and_ladders.sim.count_pairs` declares a negative-binomial
  total beside a beta-binomial success count per hidden state, simulates the
  71 x 71 triangular instance once at 20,000 positions from the seed in
  `tests/regression/fixtures/spatio_sequential_counts/stress.yaml`, and bins it
  by 1, 5 and 10 along the sequential direction. Aggregation is exact for the
  negative-binomial channel and a declared misspecification for the other.
  `snakes_and_ladders.sim.fixtures.fixture` takes `key` beside a tier, naming
  the instance a study of a problem defaults to.

  The Rust crate gains the coupled model's E step and its simulator.
  `oxi_snakes_and_ladders.class_posteriors` and `...external_field` read the
  emission log-density from tables indexed by the integer counts instead of
  calling `lgamma` per count: at the 5,041-vertex instance's coarsest bin
  factor, 20.1 s to 0.55 s and 191.3 s to 1.10 s against the NumPy oracle they
  are pinned to. `oxi_snakes_and_ladders.simulate_count_pairs` draws the
  instance's 1.0e8 count pairs from a stream per vertex keyed by the fixture's
  seed and the vertex, so the draw does not depend on the traversal.
  A third test tier, `key`, runs one problem's declared instance end to end
  under a 120 s cap of its own. (#399)
- Baseline records beside every fixture: `tests/regression/fixtures/<problem>/<tier>.baseline.json` holds what a reference algorithm achieves on the instance — the enumerated maximum, the hill-climbing and untrained-policy rates, the exact targets a surrogate is fitted against — with the seed and budget that produced each and a digest over the fixture, the computing code and the library versions. `infra/baselines.py --write` computes them, `snakes_and_ladders.sim.fixtures.baseline` reads one and refuses a stale one, and `infra/release.sh` recomputes them all. Three claims that left the per-pull-request tier for the release tier have fast siblings again, each reading a record rather than recomputing 8.3 s of rollouts, 7.5 s of likelihood fits or 1.9 s of exact expected returns. (#401)
- A problem no baseline solves, declared and refereed (#406). `tests/regression/fixtures/planted_glass/ci.yaml` is a planted Viana-Bray spin glass at 18 sites and mean degree 4, frustration 0.30, on which single-site descent reaches the enumerated ground state from 0.079 of starts (95% interval 0.056 to 0.101 over 16 seeds of 50 restarts) against the 0.4 `STATUS.md` now states as the bar for solved. Its optimum comes from enumerating all 262,144 configurations, not from the planted state, which scores 7.0 above it. Twenty-seven candidates were measured to choose it and their numbers are in `STATUS.md`. No learned policy is claimed on it: `learn.potts.PottsLandscape.on_graph` takes one scalar coupling across every edge and the instance's difficulty is its per-edge signs, which `TICKETS.md` now carries. (#406)
- Site-pattern compression (`snakes_and_ladders.likelihood.patterns`): identical alignment columns collapse to distinct patterns with integer weights, and `pruning`, `pruning_torch` and `pruning_rust` take those weights. The log-likelihood is unchanged -- the sum is reassociated, not approximated -- and the column work falls by the compression ratio, 78x at the four-taxon 20 000-site fixture and 10x at the eight-taxon 200 000-site one.

  The block-frequency bound (`snakes_and_ladders.likelihood.blocks`): the alignment partitions into blocks of N consecutive sites, blocks occurring at least `min_count` times are evaluated exactly through the pattern compression of their columns, and the rare tail is bounded above and below by the extreme per-site log-likelihood the substitution model admits at those branch lengths. The interval contains the exact log-likelihood, checked on the ci and stress fixtures and on random alignments over four seeds. Its width is (bounded sites) x (per-site extreme range), which at the tree fixtures is 2.6 to 2.8 times |log-likelihood| when the whole alignment falls in the tail -- so neither end ranks topologies, and the surrogate's POINT claim, the frequent blocks' own log-likelihood, is what does.

  The interval is a cheaper fit, not a cheaper forward pass: it costs 4.0x the uncompressed evaluation at five taxa by 2 000 sites and 5.8x at four taxa by 20 000, and raising the cutoff from 1 to 32 buys 17% and 2.5% because the block partition's sort dominates and does not depend on the cutoff. Against a 254 ms branch-length fit the 3.65 ms interval still wins, which is what the ranked search's 2 fits against 13 measures.

  The tropical Grassmannian relaxation of topology search (`snakes_and_ladders.sandbox.tropical`), which `ROADMAP.md` Stage 3 had recorded as blocked on an oracle. A tree is a point of `Gr(2, n)` -- a vector of pairwise distances satisfying the tropical Plucker relation, which is the four-point condition -- and a quartet's resolution is the argmin of its three pairing sums, so softening that argmin at temperature `tau` makes the topology differentiable. `quartet_table` fits the three resolutions of every quartet once, `relaxed_score` is the softmin-weighted sum, `optimize` ascends it with Adam and reads the topology back by neighbor joining, and `corner_bound` and `temperature_for` are the certificate that at a tree metric the relaxed value is the discrete quartet score.

  Two oracles referee it. Enumeration at 5 to 8 taxa: at the metric of every one of the 15, 105 and 945 topologies of the three tree_search fixtures the relaxed value equals the discrete score to 3.8e-16 relative -- the softmin's own leakage is certified under 1e-11 and what remains is float64 rounding of a sum over quartets, so the agreement is pinned relatively; the argmin reading of a quartet agrees with the tree's leaf bipartitions over all 1,065 of those topologies; and the quartet surface shares its maximizer with the fitted likelihood where both enumerate. The Hadamard conjugation of the exact two-state spectrum gives a metric whose four-point violation is under 1e-12 and which resolves every quartet as the generating tree does, by a route sharing no algebra with a walk over the tree.

  It is not shown to beat anything. Annealed ascent from 8 random metrics reaches the enumerated maximum 8/8 at five and six taxa and 7/8 at eight, the miss 15.8 below in 302,287; neighbor joining on the estimated distances reaches the same maximum at no gradient steps on every fixture measured, so no budget-matched claim is made against the bounded-radius search. On the 7-taxon fixture built for hill climbing to fail, the top two quartet scores differ by 3.4e-8 relative -- inside the convergence of the fits that produced them -- and both methods return the generating topology. Ascent leaves the variety, at four-point violations of 0.24, 0.05 and 2.15 where the metric has mean 1, and at seven taxa its relaxed value exceeds every corner's by 0.96: the outer relaxation's gap, measured. It therefore ships in `sandbox/` rather than in `search/`: measured, declined, and conserved with the 37 tests that declined it and the figure `fig:tropical-relaxation` renders from it, because deleting the subject of a measurement leaves the measurement with nothing to refer to. No live module imports it, asserted by `tests/regression/test_sandbox.py`. (#408)
- The Potts external field varies per site: `h` is `(n_states,)` or `(n_nodes, n_states)`, widened once by `snakes_and_ladders.sim.potts.site_field`, and the two samplers, `enumerate_potts` and `strip_log_partition` index one shape. Three fixtures follow it. `potts_spots` declares the coupled model's spatial half with a covariate-driven field, `h[n, m] = alpha[m] * log(size[n] / size_bar)`, at 3x3 where enumeration is exact, on a 12x6 strip where the column transfer matrix is, and at 71x71 triangular where neither is; `alpha` is recovered inside its 95% intervals at the first, the sampler matches the exact mean field energy to 0.005 at the second, and the mean `alpha` of a label rises monotonely across size quartiles at the third. `potts_lattice/stress` is the 12x12 open square at the exact 3-state transition, whose coupling `snakes_and_ladders.sim.potts.critical_coupling` computes from `ln(1 + sqrt(q))` rather than storing; `tests/regression/search/test_potts_mcmc.py` and section 9 of `docs/nb/potts_chain.ipynb` read it instead of each building their own copy. (#413)
- `infra/new_worktree.sh`, the agent worktree convention as one command: the branch from a named base, the shared `.venv` symlink, the gitignored compiled extension, an import check that refuses a resolve outside the new worktree, and the `PYTHONPATH` to export. It replaces five steps repeated in every agent brief that failed four times on 2026-09-08 — a sibling worktree's code imported and judged (#401), 164 collection errors from a real `.venv` directory (#404), and two worktrees without the extension. (#425)
- Three benchmarks decomposing the Rust pruning backend's FFI boundary into argument marshalling, the binding call and the whole Python-visible call, at two taxon counts and two site counts an order apart; and `snakes_and_ladders.sandbox.compiled_pruning`, the `torch.compile` front of the pruning forward that those measurements declined, kept so the decline is re-checkable against a later torch. (#443)
- `snakes_and_ladders.sandbox.pruning_problem.PruningProblem` holds a pruning alignment
  across passes behind a Rust `#[pyclass]`. It is a declined implementation
  kept with the measurement that refused it: the FFI marshalling it removes is
  1-5% of the call (`STATUS.md`). `snakes_and_ladders.likelihood.pruning_rust` is unchanged. (#444)
- `snakes_and_ladders.likelihood.pruning_analytic`: the Felsenstein pruning log-likelihood behind one `torch.autograd.Function`, with the two-pass analytic backward in place of the taped one. The value is `pruning_torch`'s and the gradient agrees with it to 2.1e-13 relative; the autograd graph is 2 nodes rather than 115 at 8 taxa, and one L-BFGS fit of 8 taxa by 20,000 sites takes 454.2 ms against the taped path's 694.5. `pruning_torch` stays the oracle. (#449)

### Changed

- Fixture reading has one implementation, and tests have time budgets (issue #132). Four loaders opened with the same four lines — a required-field set, `yaml.safe_load`, a set difference and an identical error — differing only in the per-field validation after it; that preamble is `snakes_and_ladders.fixtures.load_declared`, and each model keeps the checks that are genuinely its own.

  Tests now run in three tiers against two budgets, stated in `DEV.md`: the per-pull-request tier inside **5 minutes**, a `stress` tier inside **10**, and the release gate outside both. A test's tier is decided by what its size is *for* — a size that keeps an exact oracle available is a CI size even when slow, a size that shows behaviour at scale is a stress size even when fast. `tests/_scale.py`'s `at_scale` runs one test body at both sizes rather than duplicating it into two tests that drift apart.

  Measured, uncontended on one machine: the CI tier went from 263 s to **141 s** over 825 tests, and the stress tier holds 10 tests in 51 s. The budget was 37 s from being breached and rising; `DEV.md`'s previously documented 138 s over 540 tests had gone stale by 1.5x in tests and 1.9x in wall clock. `infra/measure_test_budget.sh` reports both tiers against their budgets, and `tests/regression/test_scale_tiers.py` asserts the structure a clock cannot — that the stress tier is reachable and non-empty, that the two selections partition, and that an unregistered marker fails rather than silently selecting nothing. (#132)
- A pull request runs the tests its change can affect. `infra/select_tests.py`
  turns the changed files into test paths and coverage targets, and
  `python-tests` calls it: a documentation-only change runs nothing, a `learn`
  change runs learn's tests and its one benchmark, a lockfile change runs
  everything.

  Measured with the coverage gate, before and after: documentation only
  174.0 s to 0.0 s, `learn/` to 17.1 s, `qa/` to 59.1 s, `search/` to 115.0 s,
  `opt/` to 163.7 s. The saving is uneven because the import graph is: `snakes_and_ladders.qa`
  imports four of the other five modules and is 33.6% of the suite's time, so
  most changes reach it, while `snakes_and_ladders.learn` is imported by nothing.

  Three rules keep it honest. Dependents come from the import graph, so a change
  to `snakes_and_ladders.likelihood` runs `snakes_and_ladders.search`, which imports it. A change that
  cannot be attributed to one module selects everything. And coverage is measured
  against what was selected — the claim narrowing from "the package is 90%
  covered" to "every module this pull request touched is 90% covered by its own
  tests" — with the package-wide gate still running on every push to `main` and
  in `infra/release.sh`.

  Benchmarks are selected by the regression module each pairs with, read off the
  filenames rather than listed, so a `learn` change times one benchmark instead
  of thirteen. (#161)
- Added `snakes_and_ladders.sim.graph` (a general Potts graph, with N-D lattices as a
  constructed case) and `snakes_and_ladders.sim.potts` (spin-configuration simulation on it:
  exact on a 1-D open chain, single-site Gibbs MCMC otherwise). (#170)
- Promoted the HMM from an `snakes_and_ladders.opt` fitting fixture to a first-class
  `snakes_and_ladders.sim.hmm` generator that retains the hidden state path alongside the
  emitted observations, on the footing the tree simulator already has. (#171)
- `snakes_and_ladders.sim` and `snakes_and_ladders.opt` draw their fixtures through a Rust categorical sampler, `snakes_and_ladders.numerics_rust.sample_rows`, backed by `snakes_and_ladders.oxi_snakes_and_ladders`. Issue #181's audit measured the NumPy original at 94-96% of `simulate_alignment`'s self time; the port is 2.6x faster at 200,000 draws and 1.7x at 2,000,000, and bit-identical to it — the uniforms are still drawn by the caller's `numpy.random.Generator`, so a seeded simulation produces exactly the alignment it did before. `snakes_and_ladders.numerics.sample_rows` stays as the oracle the port is pinned against. (#187)
- `snakes_and_ladders.numerics_rust.sample_rows` borrows its arrays instead of copying them, through `rust-numpy`. The Python-visible speedup over the NumPy oracle rises from 1.3x to 2.6x at 2,000,000 draws and from 2.2x to 2.8x at 200,000, and no longer shrinks as the array grows. The result is unchanged: it remains bit-identical to the oracle for the same generator state. (#202)
- Three duplications now have one home each, and a guard that keeps it that way (issue #230). `logsumexp` was written four times in two spellings, in `snakes_and_ladders.sim.potts`, `snakes_and_ladders.opt.potts`, `snakes_and_ladders.likelihood.potts` and `snakes_and_ladders.likelihood.belief_propagation`; it lives in `snakes_and_ladders.numerics` beside `sample_rows`, which arrived there by the same route. Twelve sites across eight modules walked a graph's edges and couplings with a hand-written `zip(..., strict=True)`; that pairing is an invariant of `PottsGraph` and is now `PottsGraph.weighted_edges()`. Four enumeration caps in three units — 200,000 configurations, 200,000 paths, 20 nodes, and a docstring-only `n <= 6` that nothing enforced — are one policy in `snakes_and_ladders.enumeration`, in configurations, with one refusal message.

  `snakes_and_ladders.likelihood.brute_force` now **refuses** an oversized tree rather than attempting it: its limit had been a sentence asking callers to keep to six taxa, so an oversized call ran until the kernel stopped it and read as infrastructure breaking rather than as a stated limit.

  `infra/duplication_survey.py` records the query behind each of the survey's seven counts, so the before-and-after claim is reproducible rather than recalled. (#230)
- `snakes_and_ladders.oxi_snakes_and_ladders.pruning_log_likelihood` now takes the alignment as one
  `(n_leaves, n_sites)` `int64` array with a per-node row index, rather than
  nested Python lists, and releases each partial likelihood once its parent has
  consumed it. Caller-visible speedup against the NumPy oracle at 200 taxa by
  11,000 sites rises from 1.00x to 2.5x, and peak memory falls from 150 MB to
  23 MB. `snakes_and_ladders.likelihood.pruning_rust.log_likelihood` is unchanged in
  signature and returns the same value; only the extension module's own binding
  moved (#232). (#232)
- Module `CLAUDE.md` files now carry principles and name where the detail lives, per root `CLAUDE.md`'s rule 6. Root grew four lines over 26 commits while the eight module files grew three- to six-fold over seven to ten each; 41 measurements had accumulated in them, each a second copy of a number `STATUS.md`, a docstring or a test already owned. The measurements are gone (0 remain, 826 lines total against 1109), and `tests/regression/test_claude_md_pointers.py` now enforces both halves — no measurement, and a 120-line budget — each tested to fail on a violating input rather than only to pass on the current tree.

  Root `CLAUDE.md`'s **Altitude** paragraph said the opposite of rule 6 for exactly these files, which is how the growth happened without any pull request breaking a rule. It now governs `DEV.md` and `INSTALL.md`, which are followed step by step, and states the `CLAUDE.md` exception explicitly. (#235)
- Randomness enters the public API as a generator, not a seed.
  `simulate_alignment`, `simulate_potts`, `sample_potts`,
  `iterated_conditional_modes` and `infer` take an `rng: np.random.Generator`
  where they took `seed: int`; `np.random.default_rng(seed)` at the call site
  reproduces the stream each function built internally, so every number
  `STATUS.md` pins is unchanged.

  `sim/CLAUDE.md` states the reason: seeding inside a call makes every draw of an
  ensemble identical, which looks like a passing test over many draws and is one
  draw. `tests/regression/sim/test_generator_signatures.py` now asserts the
  property per converted function and guards the rule, distinguishing a
  function's `seed` parameter from a fixture's declared `seed` field, which
  stays.

  Two consequences. `infer` refuses rather than inventing a generator when it
  must draw a starting topology and none was given: an unseeded one makes the run
  irreproducible and a constant-seeded one is a second seed nobody declared. And
  the `SimulatedDataset`, `SimulatedPottsDataset` and `SimulatedHmmDataset`
  records drop their `seed` field, which had no consumer — the seed is declared
  in the fixture and read from there, and a generator cannot be asked which seed
  made it (#240). (#240)
- The technical document is split in two: `docs/paper.pdf` reports results and `docs/textbook.pdf` states the problem formulations, the algorithms and the properties each is validated against, sharing one notation file so a symbol cannot mean two things. The infrastructure material moves to `DEV.md`, which the Altitude rule already makes its single copy. The per-pull-request figure selection is now the union of what both documents cite, so a figure only one of them uses is still regenerated. (#249)
- The Python package is now `snakes_and_ladders` and the Rust extension
  `oxi_snakes_and_ladders`, replacing `phylo` and `oxiphylo`. The old names
  referred to phylogenetics, one of the three problem classes this project
  supports on equal footing. The console script `run_phylo` becomes
  `run_snakes_and_ladders`. This is an API break with one consumer and no
  deprecation path: every import changes, and nothing else does — all thirteen QA
  figures were re-rendered from the renamed code and eleven came back
  byte-identical.

  Separately, 85 links pointed at the repository's former path under this owner,
  which this project does not live at: 82 pull-request citations in `STATUS.md`, the
  `README.md` CI badge, and `Cargo.toml`'s `repository` field. All now name the
  repository, and `tests/regression/test_repository_links.py` keeps it that way
  (#250). (#250)
- `CLAUDE.md` gains **Runtime Optimization Opportunities**: a ten-line list — profiling, cache, layout, SIMD, branches, inlining, allocation, double buffering, the FFI boundary, compiled backends — each naming the reference chapter, plus three Python performance references in the routing table. `DEV.md` gains a five-step **Profiling a Hot Path** procedure. (#257)
- `docs/tex/textbook.tex` now carries, labelled, every equation and algorithm the code cites: the Jukes–Cantor closed form and its normalization, site independence, the root marginalization, forward simulation, the belief-propagation message and the Bethe free energy, the episode return and the REINFORCE estimator, and the cross-device tolerance. Every docstring and test cites a label rather than a title or a number, and `tests/regression/test_document_labels.py` fails on a label no document defines. (#274)
- Every equal-budget comparison runs through `opt.budget.compare`: a budget in one declared unit, a restart baseline whose count is derived from the budget, refusal of any method that spends past it, and one table; the glass, Rastrigin and mixture-seeding studies are re-expressed through it. (#281)
- Audit of the integrated tree against `CLAUDE.md`'s **Runtime Optimization Opportunities** (#287). `PottsGraph.compressed_adjacency()` gives every kernel a contiguous layout; `search.kernels` carries a `numba` single-site-descent sweep that `iterated_conditional_modes` now runs by default, bitwise equal to the Python oracle (7x at 32x32); `anneal_potts` and `parallel_tempering` take `backend=Backend.RUST` to run the extension's sweep (5.8x to 7.8x), opt-in because the agreement is distributional; the torch pruning recursion builds every branch's transition matrix in one call (a 6-taxon hill climb 3.99 s to 2.32 s, values bitwise unchanged); `PottsLandscape.features` is one NumPy pass (REINFORCE 1.51 s to 1.00 s, exact); `spr_neighbours` deduplicates on leaf bitmasks without building a tree per candidate (1.41 s to 0.59 s at 30 taxa, same neighbours in the same order). `numba` becomes a dependency. (#287)
- Every pull request's title starts with its base branch in brackets, `[main]` or the parent branch of a stacked change, and a `pr-title` CI job checks the title against the pull request's base. (#292)
- The textbook applies one notation and one skeleton — model, factor graph, algorithm, pin — to every problem and algorithm, gives the discrete solvers and the samplers labelled sections, adds the coupled spatio-sequential model's section with its figure and algorithms, and moves four derivations into an appendix. (#298)
- `snakes_and_ladders.sandbox` is the home of the second implementation, whichever
  side of a measurement won: the hand-rolled code a framework replaced on a hot
  path, or the framework front that lost. The three fronts issues #391 and #392
  declined move in — the batched rollout through `gymnasium.vector`, and the
  generalized advantage and clipped surrogate through TorchRL — each still pinned
  against ours at 1e-10 and still timed beside it, so a library release that moves
  the ratio is re-measured rather than assumed. The Gymnasium adapter over
  `learn.Environment` was not declined and stays in `snakes_and_ladders.search.gym`. (#322)
- `snakes_and_ladders.oxi_snakes_and_ladders.max_flow` and `ising_ground_state` borrow their arguments as `float64` and `int64` NumPy arrays through `rust-numpy` instead of copying Python lists, and `ising_ground_state` returns an `int64` array instead of a list. `snakes_and_ladders.search.maxflow_rust` is unchanged in signature and returns the same configuration and energy, bitwise, at extents 16, 32 and 64. The caller-visible speedup over the NumPy reference moves from 6.1-10.5x to 6.3-10.7x; the kernel with its arrays built is 26-32x, and the remaining difference is measured to be the Python `energy` evaluation, not the boundary (#336). (#336)
- The `torch` stream takes a generator, not a seed. `opt.hmc.sample`,
  `opt.hmc.anneal` and `search.max_cut.goemans_williamson` take a
  `generator: torch.Generator` where they took `seed: int`;
  `torch.Generator().manual_seed(seed)` at the call site reproduces the stream
  each function built internally, so every pinned HMC and Goemans-Williamson
  number is unchanged. These were the last three `seed` signatures in the
  package, deferred from #254, and
  `tests/regression/sim/test_generator_signatures.py` now checks the property
  for them too and guards every public signature with no exemption (#337). (#337)
- Runtime-optimization audit of every module (#341). `likelihood.message_passing` holds its messages as rows of two preallocated edge arrays and runs one vectorized pass per group of like-shaped factors; the dictionary-per-message implementation moves to `likelihood.message_passing_reference` as the oracle, and the two agree bitwise on every schedule. Flooding on the 8x8 lattice goes from 75x `belief_propagation` to 1.8x (514.6 ms to 12.4 ms); the tree schedule on a 200-step chain from 9.5x the forward recursion to 4.7x (29.8 ms to 14.8 ms), and it no longer recurses, so a 2,000-step chain runs. `search.maxflow.energy` sums its edge terms in one gather (7.7-8.6x on one configuration; now 1e-12 relative to `log_weights` rather than bitwise). `learn.PottsLandscape.features` is one gather from a padded neighbour table (REINFORCE 2.43 s to 1.85 s on the length-8 chain, exact). `tests/benchmarks/profile_hotpaths.py` ranks every module's workload at two tiers. (#341)
- Release audit for 0.4.0 (#358). `STATUS.md`'s summary states the roadmap progress since 0.3.0 per milestone with the pull requests that moved it, and gains a consistency audit listing the seven annealing entry points and eight enumerators as one algorithm under different types or as copies, each with the oracle that would pin a merge, and the methods refereed by the simulated truth alone with the oracle each would need. `ROADMAP.md` names low-density parity-check decoding as the fourth problem class (#340) and states the two-way validation standard: an oracle where one is affordable, recovery of the simulated truth where none is. `TICKETS.md` drops the landed work and re-points the ticketed items. The textbook gains a problem statement per row of `PROBLEMS.md` — large parsimony, the frustrated lattices, the Gaussian mixture, the continuous test functions, and a placeholder for the parity-check class — each with a QA figure, and two generated tables of which algorithm and which referee applies to each problem, written by `infra/problems_tables.py` from the catalogue and the suite's kind markers; the paper's placeholders are replaced by the budgeted comparisons and the thread-scaling result, with the external-tool comparison stated as ticketed (#126). The release template gains three precondition checkboxes. (#358)
- The reference routing table in `CLAUDE.md`, the textbook's Reference Taxonomy appendix and `docs/tex/references.bib` now carry 42 further entries — books, reviews and papers separated per row, with a new row for learning in combinatorial optimization — and a guard asserts the three name the same works. `docs/blue_sky.md` inventories every earlier Stage 3 proposal with its state and lists 15 candidates, each with the oracle that referees it and the tier it runs at; `ROADMAP.md` Stage 3 records which halves of its five items landed and points at the candidates. (#360)
- Pull requests change `docs/tex/` and never the committed PDFs, which a "Rebuild the documents" ticket's pull request rebuilds (new issue template; CI refuses a PDF change elsewhere and no longer compares the PDFs); `main`'s CI runs are no longer cancelled by the next merge; the test-selection step fetches the base branch's history; the document build runs against the synced environment instead of rebuilding the Rust extension; `infra/measure_test_budget.sh` measures the critical tier unless `--full`. (#369)
- Per-pull-request validation under 300 s (#372). `infra/validate.sh` runs lint, types, the critical gate, the selected tests, and only the Sphinx, notebook and document checks the diff calls for, each timed. A figure is rendered only when its inputs changed: `snakes_and_ladders.qa.inputs` digests a renderer's source, its import closure, its fixtures and the drawing libraries, and a stamp beside each committed figure (`docs/tex/figures/<stem>.inputs`) records the render; the notebooks carry the same stamps and the checker skips an unchanged one. `infra/select_tests.py` selects the guards that read a changed prose file rather than nothing. A test over 10 s on the reference host carries `release` or `stress` or fails the validation; `tests/conftest.py` pins one BLAS thread per process; every script exports `UV_NO_SYNC=1`. A cited figure renders in at most 30 s or carries a waiver naming its ticket; `rl_tree_policy` is waived. (#372)
- Release audit for 0.5.0 (#376). `STATUS.md` states the roadmap progress since the 0.4.0 audit per milestone with the pull request that moved it — #373 for the tree's data-driven starts (#364), #378, #379 and #380 — and gains a consistency audit section; its header records that the 0.4.0 cut is deferred and that the repository carries no tag. Every problem statement in the textbook now carries the same five parts: the model, the sizes it is supported at read from the fixtures and the size tiers, the model as a factor graph with a hand-drawn TikZ sketch in its own file, the algorithms cited from a new appendix grouped as initializers, samplers, optimizers and surrogates, and a validation subsection naming each referee with what it does and does not establish; `tests/regression/test_problem_statements_complete.py` fails a section missing any of the five, and an algorithm environment nothing cites. `infra/problems_tables.py` gains a third generated table, in four parts, pairing each problem with each method family — the tier at which the pairing is validated and the kind of referee, read from the suite, and a note on when the family wins and when it does not, from the committed `docs/tex/generated/method_notes.yaml`; a pairing with no test is marked untested rather than omitted, and a note citing an experiment that does not exist fails the generation. Five hand-rolled implementations gain referee tests against the frameworks that duplicate them: `learn.reinforce`'s loss against TorchRL's `ReinforceLoss`, `search.topology`'s equality against `rustworkx.is_isomorphic`, and — skipping until `scipy` is declared — `search.neighbor_joining` against SciPy's UPGMA, `likelihood.hadamard` against `scipy.linalg.hadamard`, and `opt.fit` against `scipy.optimize.minimize`. `ROADMAP.md` names the data-driven start as a Milestone 1.3 deliverable and states what makes a problem statement complete; the release template gains the per-problem checklist, the frameworks table and two further preconditions. `DEV.md`'s restated counts are re-measured and dated. (#376)
- The single "technical document" is renamed throughout to the two documents `docs/tex/` has built since #249: `infra/build_technical_doc.sh` is now `infra/build_documents.sh`, the CI job `technical-doc` is `documents` (`Documents (paper and textbook)`, which the branch protection's required check has to be renamed to), and the 77 remaining references in prose, headings and docstrings name the documents, the paper or the textbook. `tests/regression/test_documents_name.py` fails any live file that names the retired artifact again; `changelog.d/`, `CHANGELOG.md` and `STATUS.md`'s dated audits keep the old name as history. (#377)
- Every supported problem now declares its instance as a fixture. One file per problem and size tier under `tests/regression/fixtures/<problem>/`, each stating the size, the seed, the model that reads it and the oracle available at that size; `snakes_and_ladders.sim.fixtures` loads one through that model's loader, and `tests/_scale.at_fixture` parameterizes a test over the tiers a problem declares. The Gaussian mixture, the coupled spatio-sequential model, the LDPC code, the frustrated lattices and the continuous test functions gain a file and a loader; the nine existing fixtures move into the layout unchanged, so no pinned value moves. Every QA figure now takes its instance as `--params`, the notebooks read fixtures, and `infra/experiments.py` requires an experiment's `fixture:` field to name one. Guards hold `PROBLEMS.md` and the registry to each other, refuse a figure or notebook that builds its own instance, and list every (fixture, method family) pairing no test makes, each with its reason in `docs/tex/generated/method_notes.yaml`. (#382)
- Three declined framework fronts are kept as code rather than as a paragraph.
  `snakes_and_ladders.sandbox` gains `pyg_surrogate` (PyTorch Geometric's
  `GINConv` and `global_add_pool` computing `learn.surrogate.GraphSurrogate`'s
  forward), `rustworkx_clusters` (`rustworkx.connected_components` labelling the
  Swendsen-Wang bond graph) and `scipy_mincut`
  (`scipy.sparse.csgraph.maximum_flow` solving the ground state's minimum cut,
  with the `int32` capacity scaling the real-valued reduction needs). Each was
  measured against the implementation it would have replaced and lost; each
  arrives with the test that pins it against that implementation, so a library
  release which moves the numbers fails a test rather than dating a finding.

  Pinning `scipy_mincut` narrowed a claim: over 20 random fields at lattice
  extent 16, rounding the capacities at scale 1e2 selects a strictly worse cut
  on three of them, 7.18e-4 to 4.31e-3 above the minimum, while 1e3 to 1e8 agree
  with both Dinic implementations exactly. (#388)
- The Swendsen-Wang sweep labels its clusters by squaring the union-find parent
  map to a fixed point and groups them by one stable sort, rather than walking
  `_find` once per site and scanning for each cluster's members: 1.52x to 1.98x
  per sweep at lattice extents 8 to 48, and 1.60x and 1.80x on the committed
  benchmark at extents 8 and 16, with the chain unchanged entry for entry. (#389)
- `learn.ppo` scores each batch's neighbourhoods once per iteration rather than
  once per epoch, and clips the whole batch in one pass rather than looping over
  episodes: 8.06 s to 5.83 s over the 1,920-episode Potts-chain budget, with the
  training curve identical iteration for iteration. TorchRL's `GAE` and
  `ClipPPOLoss` were measured as replacements and not adopted. (#391)
- `SEAMS.md`, regenerated by `infra/seams_survey.py` and held by a guard, inventories every protocol and shared contract with its implementers, consumers and problem classes; the seam rule (three consumers) enters `CLAUDE.md` and `DEV.md`; the fixture registry is the problem classes' contract and no `Problem` protocol is added; one EM driver over the HMM and the mixture was measured and declined. `TICKETS.md` drops five bullets whose work has landed and re-points five to the tickets that now carry them. The applicability table is regenerated after the tier move. Three notebooks' Further Work sections stop naming landed features as unbuilt. (#400)
- `docs/nb/potts_chain.ipynb` covers the Potts lattice as well as the chain: the `potts_lattice/ci` instance and its graph, belief propagation's Bethe normalizer against enumeration over all 19,683 configurations (1.47e-02 absolute, 4.67e-03 relative, single-site marginals within 4.25e-03), single-site flips against Swendsen-Wang and Wolff by the energy autocorrelation time at equal sweeps, and the lattice coupling and field recovered inside all four intervals. Its Further work section now names the gaps the lattice pass leaves rather than #404, which carried the pass. (#404)
- Root `CLAUDE.md` states the fixture stance (#408): every claim here is refereed by an oracle or by the parameters that generated the data, an empirical alignment supplies neither, so the repository works on simulated fixtures and takes no real data today. External phylogenetic software is surveyed and considered rather than adopted — nothing installed, no lockfile change — and admitting one needs a role our own oracles cannot fill, an OSI-approved licence, and the Dependencies & External Tools rules satisfied. `docs/external_tools.md` is that survey: IQ-TREE 2, RAxML-NG, Biopython, DendroPy, ETE, scikit-bio and TreeSwift with licence, OSI status, star count against the 1,000-star rule, and the role each could fill, closing on IQ-TREE 2 for benchmarking as the first adoption if the stance changes. `ROADMAP.md` §1.2's external-parity requirement, its Milestone 2.3 deliverable, the `STATUS.md` ledger row and the `TICKETS.md` bullets for #126 are restated as deferred rather than withdrawn, naming this repository's own exact oracles and simulation truth as what referees large `n` in the meantime. The parallel-agent cap rises from three to five in `DEV.md`, set by review bandwidth and host-lock contention rather than core count.

  Three cheaper ways to run the same topology search, each opt-in, each measurable alone and each held to the enumerated optimum (`snakes_and_ladders.search`): a parsimony starting tree, which is `parsimony_search(...).topology` passed as `infer`'s `topology`; `spr_neighbours(..., radius=r)` and `infer(..., radius=r)`, which regraft only within `r` of the pruning point for an `O(n r)` neighbourhood, where radius 1 is exactly the NNI neighbourhood and any radius from the leaf count up is the unbounded search candidate for candidate; and `infer(..., partial_reoptimization=True)`, which fits a candidate over only the branches whose bipartition the move changed and refits the accepted move in full, so the reported log-likelihood stays a maximized value. At eight taxa the three together reach the unbounded search's log-likelihood for 1,930 forward passes against 16,600. Defaults are unchanged.

  The Rust pruning kernel keeps a partial as `(state, site)` and walks the sites in tiles, so every inner loop is a contiguous run the compiler can vectorize and one tile's rows stay in cache: 28% off the kernel at four taxa by 200,000 sites and 35% at eight, and nothing measurable through the binding at 11,000 sites. The arrays crossing the FFI boundary are unchanged. (#408)
- The per-pull-request test job bounds its fallback rather than running the whole suite for any change it cannot attribute, and carries a timeout; the whole suite runs on the push to `main` and at the release gate. A change touching a likelihood kernel, the Rust sources or a lockfile refuses the bound. (#409)
- The Potts notebook's Further Work now names an issue for every gap it lists: a
  lattice at the transition (#413) and a learned policy on the lattice (#414),
  both filed and carried in `TICKETS.md`. (#413)
- The single host build lock splits in two (`infra/locks.sh`): `measure` for
  anything whose number reaches a pull request, and `validate` for correctness
  runs nobody times, three at a time. A measurement still runs alone, so timings
  stay comparable, while three validations now overlap where they used to
  serialize. A measurement stays at one thread, since the recorded baselines were
  taken there. (#418)
- `CHECKS.md`, `SEAMS.md` and the textbook's applicability tables are no longer committed (#425). All three are written from the tree, and committing them made a machine-written file a merge participant: on 2026-09-08 `CHECKS.md` was resolved in 13 merges across 9 branches, `SEAMS.md` in 2, `docs/tex/generated/problems_tables.tex` in 3, and a conflict between two machine writings carries nothing to resolve. `infra/ledgers.sh` writes all three — in `infra/build_documents.sh`, at the release gate, in `infra/review_gates.sh`, and in CI's `python-tests` job, where `--check` fails if regenerating rewrote a tracked file, so the guarantee that the ledger matches the tree is checked rather than remembered. CI writes both Markdown ledgers to its run summary, which replaces browsing them on GitHub. `docs/tex/generated/method_notes.yaml` is hand-written and stays committed, moving to `docs/tex/method_notes.yaml` so nothing hand-written lives under `generated/`. (#425)
- Timed and documented the document build: `DEV.md` now carries its wall clock beside the other budgets (230.7 s from a clean checkout, 13.9 s with every figure stamp current, 1.7 s incremental), the set of paths it writes and which of them a pull request may commit, and the reproducibility result — both PDFs byte-identical across two clean builds. `INSTALL.md` says to revert the two PDFs after a local build. (#429)
- `burn`'s taped gradient, which issue #449 measured and declined, is conserved rather than deleted: `snakes_and_ladders.sandbox.pruning_burn` over `src/pruning_burn.rs`, with `burn-ndarray`, `burn-autodiff` and `burn-tensor` optional behind a new `sandbox` Cargo feature. The default `maturin develop --release` compiles 34 crates in 37 s and no `burn`; `--features sandbox` compiles 102 in 86 s, five of them `burn`. `infra/release.sh` compiles the feature under both `cargo clippy --all-targets` and `cargo test` (33 s and 34 s from clean, beside 34 s and 38 s for the same two without it), which is what stops a feature-gated route rotting unbuilt — the conserved file needed three `clippy` fixes and a `cargo fmt` to compile under the current toolchain. The Python module imports either way and raises `ImportError` when called against an extension without the feature, so `tests/regression/likelihood/test_pruning_burn.py` skips rather than falling back to the route it referees. (#449)
- A pull request is judged by the critical gate; the `ci` tier and its coverage gate run on the push to `main` and at the release gate, and the cited-figure stamp check moves to the release gate with them. The `pytest` job took 17 to 32 minutes on a pull request while the other nine finished inside three, and a merge of `main` cost a branch a six-minute figure render that produced byte-identical output. What a pull request no longer proves, and the measurement that restores each check, is issue #455. (#455)
- An experiment file under `docs/experiments/` is three sections — question,
  numbers, finding — in a body of at most ten non-blank content lines, the front
  matter, the title and the section headings excluded;
  `infra/experiments.py --check` fails a file over the cap. (#458)
- Release 0.5.0 (#468), cut against the union of the eight open branches rather than against `main`, so the version's contents are the work that exists rather than the subset that had merged; [#467](https://github.com/michaelJwilson/snakes_and_ladders/issues/467) reconciles the two. `Cargo.toml` moves from `0.3.0` to `0.5.0` and `0.4.0` is never spent: it was audited (#358) and its cut held, so its fragments are consumed into this section alongside #376's. The consistency audit corrects three claims a document made and a measurement disproved --- `DEV.md`'s nine review-gate rows against the eight #456 left, and `DEV.md` and `changelog.d/425.removed.md` both stating that #422's reachability walk saved nothing, which #445 measured as true of the twelve branches sampled and false in general, at 19 of 19 cited figures staled by the closure digest against the walk's 1 and no figure's bytes moved. The branch-protection precondition `DEV.md` recorded as outstanding is confirmed met: `main` has taken merges since #377's rename and `Documents (paper and textbook)` reports green, so that paragraph is corrected too. The tag precondition is recorded as not met --- the repository has never carried a tag --- and `origin/dev` is recorded as diverged from `main` in both directions rather than as the integration branch the release template assumes. (#468)
- A release ticket's pull request opens as a draft as soon as its branch exists and is updated every 10 minutes with the remaining work and the estimated time to completion (`CLAUDE.md`, Conventions).
- Measured against single-flip hill climbing over 40 shared seeds on a chain whose optimum needs coordinated flips: greedy reaches it 5/40, the deterministic relaxation 18/40 (McNemar `p = 0.00098`), and every sampled variant — soft, straight-through, annealed — 11/40 (`p = 0.18`), a tie. The sampling is what costs, not the relaxation, and the deterministic run is also 15% cheaper. The tropical Grassmannian half of the same roadmap bullet stays unbuilt and `TICKETS.md` now records why: no tree instance at an interesting size has a known optimum to referee it.
- The LaTeX intermediates `latexmk` leaves beside `docs/paper.pdf` and `docs/textbook.pdf` (`.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.toc`) and the stale `docs/draft.*` set are untracked and ignored by glob; only the `.tex` sources and the two PDFs are committed, so a document rebuild no longer conflicts with every other one on files nobody reads.
- Throughput rules in `CLAUDE.md` and `DEV.md`: agents run in parallel with one CPU-bound step at a time behind the host lock; tickets of one plan shape land as one pull request and a plan with independent steps is split across agents; local validation before a push is the changed-file lint, the type check and the critical tier, with the pull request's CI as the final validation; review starts from `infra/review_gates.sh`; a test over the per-PR duration cap moves to the `release` tier.
- `CLAUDE.md` states the model routing: Claude Fable plans, implements, writes and reviews pull requests and takes review tickets; mechanical work is delegated to Claude Opus subagents and validated before it lands.

### Fixed

- A figure's input stamp now hashes the definitions the renderer actually
  reaches, parsed as an AST with docstrings stripped, rather than the source text
  of every module in its import closure. Rewording a docstring or declaring an
  unread constant in `qa/runner.py` staled all 19 cited figures and now stales
  none, while editing a function they call still stales exactly those that call
  it. `infra/review_gates.sh` gains three checks and runs nine in 29 s. (#418-stamps)
- Fixed the root-detection assertion in `test_the_search_finds_the_generating_tree_and_rejects_a_worse_one`: it now walks every node of both fitted trees via `snakes_and_ladders.sim.tree.preorder` instead of checking only the two root objects, unblocking `infra/release.sh`'s release gate. (#168)
- `STATUS.md`'s summary row for Milestone 1.3 read "Potts lattice not started"
  while its own Requirements Ledger recorded the lattice **Met** at 0.981
  coverage; the paragraph carrying that evidence sat under Milestone 2.1, a
  continuous-optimization result filed under reinforcement learning. The row and
  the paragraph now agree with the ledger, and row 1.4 names the exact-baseline
  family its own section documents.

  `ROADMAP.md` §1.2 required parity "under equivalent wall-clock time", which
  `DEV.md` forbids measuring on CI hardware and `search/CLAUDE.md` contradicts
  outright; it is now an equal budget of objective evaluations. Milestone 1.4
  names Max-Cut, which landed and was absent. The nine required checks are
  enumerated once, in `DEV.md`, rather than in three places.

  `TICKETS.md` dropped two bullets naming closed issues, gained the twelve filed
  issues it had never listed, and annotated four bullets that already described
  one (#244). (#244)
- Every notebook's Further work section said "no job re-runs it"; the notebooks CI job has since #203. The sentence is gone, `potts_chain.ipynb`'s coverage gaps now name the ticket that adds them, and `infra/check_notebooks.py` fails a notebook whose last cell is not a Further work section or whose line names no issue. (#278)
- `docs/tex/generated/method_notes.yaml` no longer states a reason for two pairings that are now tested; the Potts lattice's samplers and optimizers gained tests and kept their untested entries, which `tests/regression/docs/test_problems_tables.py` refuses. (#411)
- A committed figure's input stamp is a digest of the definitions its renderer
  executes, read from the AST with docstrings and comments removed, rather than
  of the text of every module it imports. A reworded docstring or an unread
  constant in `qa/runner.py` stales none of the 19 cited figures where it
  staled all of them; an edit to a function a renderer calls still stales
  exactly the figures that call it, and a module the analysis cannot see
  through is hashed whole and named in the digest. (#418)
- The bounded merge gate's 15-minute cap now applies only where the selection is
  actually bounded. A change to a likelihood kernel, the Rust sources or a
  lockfile is deliberately unbounded and runs the whole suite, which the cap
  cancelled — indistinguishably from a failure. (#423)
- A measurement no longer holds the host lock while it waits for one (`infra/locks.sh`, #427). `measure` took its own lock first and then blocked collecting three validation slots, so a measurement that had not started could hold a slot a validation was queued for: 3.31 s of a queued validation's wait, measured, behind a job that never ran. The four locks become one readers-writer lock — `measure` takes it exclusively and nothing else, `validate` takes it shared plus one of three slots — which removes the slot-collecting re-entry, the recursion through the script and the execute-bit fallback that re-entry needed, a third of the file. `with_lock measure --wide` goes with them, having had no consumer since it was added. Which kind a job is now decides on two questions stated in the script and in `DEV.md`: would a busy host change a number this run reports, and does the job use more than one core. One property is traded rather than won: `flock` grants a fresh shared request while an exclusive one waits, so a stream of validations now starves a measurement (18 s of an 18 s stream, 5 runs of 5, where the previous lock let it in within 0.06 s). `SAL_LOCK_WAIT` bounds it at 1800 s and the choice is recorded as #431, because the turnstile that fixes it costs the guarantee this change exists to establish. (#427)
- Made `main` green after #346 merged: `CHECKS.md` regenerated for the tests #346, #352 and #354 added, and the reference kernel's Raises section names `message_passing.ConvergenceError` in full so Sphinx no longer finds two targets.
- Made `main` green after #352, #353 and #354 merged: `test_opt_hmc_adaptive.py` and `profile_hotpaths.py` call `hmc.sample` with the generator #345 introduced, `profile_hotpaths.py` passes `compare` the `workers` #353 requires, and the docs index lists `likelihood.message_passing_reference`.
- Made `main` green, part six: the relaxed-optimization benchmark passes a `torch.Generator` rather than a seed, and the `technical-doc` job checks out full history so its PDF rule can find the base branch's merge base.
- Made `main` green: the HMM/samplers pairing gained a test when #420 pinned the block sweep to the enumerated path posterior, and a pairing loses its entry in the untested block by gaining one, so the reason it carried is removed. `tests/regression/docs/test_problems_tables.py` failed on `main` for both of its untested-pairing assertions.
- Made `main` green: the textbook cites Algorithm `alg:block-frequency` where the bracket is stated, so every algorithm it defines is reachable from the text.
- Made `main`'s whole-suite run green: three QA runner tests read the written paths from the stream the run logger writes to.
- The planted Viana-Bray spin glass proposed in #209 does not referee a Stage 2 claim, and `STATUS.md` now says so with the measurement. Against 20-restart iterated conditional modes at `n = 100` and mean degree 4, single-site descent lands on the planted energy below frustration 0.2 and beats it above, where the planted state is no longer near-optimal; raising connectivity does not open a window. The fixture supplies a known-energy reference past enumeration, which is a real and separate thing, and `TICKETS.md` keeps the search for an instance no baseline solves open.
- Three failures on `main` that predated any open pull request: the `seed=` calls left by #240 in two tests, one QA script and the tree notebook; the kind markers missing from `tests/regression/test_document_labels.py`; `search/CLAUDE.md` over its 120-line budget; and #310's Gibbs benchmark importing a backend module that only #287's audit provides, which now skips instead of failing.
- Three more `main` failures that predated every open pull request: `tests/regression/test_select_tests.py` expected `learn` to be a leaf module and its benchmark selection to be one file, which `search.rl` (#178) and the surrogates (#308) changed; the HMM notebook's committed outputs; and a stale `docs/paper.pdf`.
- `docs/source/index.rst` was missing five modules — `snakes_and_ladders.sim.graph`, `snakes_and_ladders.sim.potts`, `snakes_and_ladders.sim.hmm`, `snakes_and_ladders.scripts` and `snakes_and_ladders.scripts.run_snakes_and_ladders` — the third drift of an invariant `docs/CLAUDE.md` states and that `sphinx-build -W` structurally cannot catch, since it fails on a broken entry and never on an absent one. All five are added, and a millisecond test now enforces the invariant in both directions.

### Removed

- The merge gate's bounded selection and the figure stamp's reachability walk, both measured against what they saved.

  `infra/select_tests.py` loses `--budget`, `UNBOUNDABLE` and the `bounded` output (#409, #424). A bound on the selection and a fixed timeout cannot coexist: the 15-minute cap fired on exactly the pull requests the bound refused — every `likelihood/` one — and a cancelled job is indistinguishable from a failing one (#423). The gate now runs the selection under the job's own cap. The selection itself stays: `pytest -m "not release and not stress"` is **1,098 s over 2,121 tests** on the 4-core reference host, 3.7x the 5-minute budget, so running everything per pull request is not available.

  `snakes_and_ladders.inputs` loses the AST reachability walk (#418, #422) and hashes the renderer's whole import closure, still as ASTs with docstrings removed. Over the twelve branches merged before 2026-09-09 the walk and the closure hash staled the same **26** figures, which is the measurement #425 removed the walk on; #445 has since measured the case that sample never contained. On a change reaching only some of the figures the closure digest stales **19 of 19** cited figures where the walk stales **1**, and **no figure's bytes move** — so the claim that the walk saved nothing is true of the branches sampled and false in general, and what the closure hash costs is a re-render of every figure on any edit inside a shared module. The over-approximation is still in the safe direction: a figure re-renders that need not have, rather than a figure published against code that no longer produces it. (#425)


## [0.3.0] - 2026-09-03

### Changed

- The thirteen QA scripts share one command line (`snakes_and_ladders.qa.runner`). Each
  declared its own `ArgumentParser`, `--output-dir`, load, write and
  `try/finally` around `plt.close`, so a fix to one reached only that one; a
  script now declares its output stem, the parameters files it takes, and the
  builder that turns them into a figure or a `tabular` body.

  Two behaviours were inconsistent and are now uniform. Every script reports what
  it wrote, where three did and ten were silent. Every figure is closed even when
  writing refuses it, where the twelve hand-written `finally` blocks each had to
  get that right separately.

  `sim_tree`, `sim_example` and `sim_problem_sizes` lose their
  `render_*_figure`/`render_problem_sizes_table` wrappers, which loaded and wrote
  around a build step; the build step is now `build_figure`/`build_table` and the
  loading and writing are the runner's. `--n-sites-shown` keeps its default of 10.
  Every one of the thirteen committed figures regenerates byte-identically. (#150)
- `ROADMAP.md` is restructured around three problem classes rather than one:
  phylogenetic trees, N-D Potts models in an external field, and HMMs each carry
  a deliverable and a validation under every Stage 1 and Stage 2 milestone, and
  milestones are keyed `N.M` so a reference resolves. It gains a first section
  stating the agentic development loop — ticket, plan, pull request, validation,
  and the record the loop writes into — each as a deliverable and the gate that
  holds it.

  Milestone status leaves the roadmap for `STATUS.md`, which states what landed,
  the oracle that established it, and the pull request that carries it; a
  requirements ledger against §1.2; and what is not claimed. `TICKETS.md` states,
  as titles, the tickets remaining between the two. A roadmap that also tracked
  its own progress could not be edited without re-litigating both. (#152)
- `README.md` leads with the three problem classes the roadmap now states —
  phylogenetic trees, Potts models in an external field, and HMMs — rather than
  phylogenetics alone, and gains a Features section stating each capability with
  the measurement behind it, a table of the nine `CLAUDE.md` contracts and what
  each governs, and a reference table routing the literature by concern.

  The technical document builds again. Two citations named keys absent from
  `references.bib` (`cormen2022`, `hwu2022` against the bib's `clrs2022` and
  `pmpp2022`), which `latexmk` reports while still exiting 0 — CI's log check
  catches it, so `docs/draft.pdf` could not be regenerated. The backend-agreement
  figure read its generated caption into a macro and then discarded it for a
  hardcoded restatement quoting a stale measurement; it now uses the caption the
  QA script wrote.

  `infra/build_technical_doc.sh` exports `FORCE_SOURCE_DATE=1`. `SOURCE_DATE_EPOCH`
  fixes the PDF's `/CreationDate` but not `\today`, which reads pdftex's date
  primitives: the committed PDF's title page carried the day it was built, so the
  staleness check would have failed on any pull request opened the following day. (#153)
- The accuracy figure's per-PR test no longer re-runs the sweep at its committed
  size. It ran 6 site counts x 8 replicates = 48 searches, 27.7 s, 23% of the
  whole per-PR suite, to assert four things about a caption — while the
  technical-document build rendered the same figure again and the release gate
  asserted the scientific claim. It now sweeps two sizes by two replicates: the
  pipeline still runs end to end, at 4.1 s.

  The suite is 138.0 s over 540 tests, against 165.6 s over 525 before — faster
  with fifteen more tests in it. (#154)
- The technical-document build regenerates only the QA figures
  `docs/tex/main.tex` cites, and the release gate regenerates the rest. The
  figure list lived as thirteen invocations in `infra/build_technical_doc.sh`
  that nothing connected to the document, so when the document stopped citing
  eleven of them the build kept rendering all thirteen and no check noticed:
  that job spent 281.6 s to run a 1.5 s LaTeX build, and 98.4% of it produced
  figures nothing included. A full build is now 5.9 s, and `docs/draft.pdf` and
  all thirteen committed figures are byte-identical to before.

  `snakes_and_ladders.qa.manifest` is the single statement of which figures exist and what
  renders each one. `snakes_and_ladders.qa.build` reads it and renders a selection: what the
  document cites (per pull request), the whole manifest (`--all`, which
  `infra/release.sh` now runs with `--check`), or named stems (`--only`).
  Citing a figure the manifest cannot render is refused rather than skipped.

  `snakes_and_ladders.qa.build` pins `SOURCE_DATE_EPOCH` for the figures it renders.
  matplotlib embeds it, so without it two rebuilds of an unchanged figure
  differ and a comparison reports every figure stale — which a caller had to
  know to prevent. `infra/build_technical_doc.sh` reads the value back from
  there rather than keeping a second copy.

  `infra/measure_build.sh` times the build stage by stage, so a claim that it
  got faster is a pair of numbers. (#154)
- `tests/regression/` is split by submodule — `sim/`, `likelihood/`, `opt/`,
  `learn/`, `search/`, `qa/` — which is `DEV.md`'s own rule once a kind outgrows
  one flat directory, reached at 39 modules. Tests belonging to no submodule stay
  at the top level. Pure moves; 540 tests pass before and after.

  CI caches the TeX Live packages the `technical-doc` job installs, the one
  install still paying full price on every run while `uv` and Cargo were both
  cached.

  `DEV.md`'s suite budget said 131 s over 140 tests and 954 s over 141. It is
  138 s over 540 and 989 s over 550.

  Selecting tests by the files a pull request changed is recorded as unavailable
  rather than built: `python-tests` runs `--cov-fail-under=90` on the same
  invocation, and a subset cannot meet it — `tests/regression/sim` alone measures
  12% — so scoping would mean weakening a gate `CLAUDE.md` forbids weakening. (#154)
- Root `CLAUDE.md`'s **Writing Style** section states what it governs: every
  document in the repository, each module's `CLAUDE.md` included, and every
  docstring, comment, commit message and pull-request body. The rules bound all
  of that already; nothing said so, and the eight module files carried a generic
  "these are local" line that never named the section, so an agent reading one
  alone had no way to know.

  Each of the eight now names **Writing Style** and states that it binds that
  file. Referenced, not copied: the section changed three times on the day this
  was written, and nine copies would already disagree. **Expected Reader** stays
  in `docs/CLAUDE.md` alone, being a contract about the technical document.

  A regression test enforces it, rather than leaving the invariant stated and
  unchecked — the failure mode of `docs/source/index.rst`, which claimed to
  cover every submodule while missing eighteen. (#155)
- A generated plan has a stated shape: 2–5 steps, each saying how it will be
  validated, ending with an **Open Questions** section that carries every
  question on the desired behaviour. A reviewer now finds the questions in one
  place instead of reading the prose for them, and a plan with none says so
  rather than omitting the heading.

  Root `CLAUDE.md` states which documents are read at which altitude, and what
  that means for repetition. `ROADMAP.md`, `STATUS.md` and `TICKETS.md` plan and
  track. `CLAUDE.md`, `DEV.md`, `INSTALL.md` and the module files are worked in
  and carry their detail in full rather than as pointers into each other, so
  someone following one of them need not assemble the answer from three. Detail
  may repeat between them; where it repeats it must agree, and root `CLAUDE.md`
  settles which reading is right. The Writing Style stays the exception — one
  text, referenced — because it binds every file at once.

  The plan's shape is therefore stated three times, at the altitude each
  document is read at: `ROADMAP.md` §0.2 has it as part of the loop,
  `DEV.md` and `infra/CLAUDE.md` add what the step's validation must name.
  `DEV.md` also states a rule that was written nowhere — a pull request
  implements a plan already approved — and says which document holds the loop's
  intent when the two disagree.

  Root `CLAUDE.md`'s Writing Style already governed every document, docstring,
  comment, commit message and pull-request body; the list now names plans and
  ticket comments too, which is what makes a plan subject to it. (#162)
- `STATUS.md` is read at `0.3.0`. Between `0.2.0` and `0.3.0`, six pull requests
  refined the development loop and its record — `ROADMAP.md`'s restructuring,
  the QA figure-manifest and script-runner refactor, the test-layout split, and
  the writing-style and plan-shape pointers — and no roadmap milestone moved.

  The Release issue template drops its now-redundant one-ticket-per-version
  notice, defaults its title and target-version field to the next expected
  version, gains a blank-by-default Suggested work section, and its consistency
  audit prompt now names the template itself as something the audit checks. (#165)


## [0.2.0] - 2026-09-03

### Added

- A model-agnostic optimization interface (`snakes_and_ladders.opt`): an unconstrained
  parameter vector, a differentiable scalar objective, and a map back to named
  constrained parameters, with the shared constraint maps that keep every
  parameter feasible by construction. Two reference instances ship with it — a
  1-D Potts chain in an external field and a discrete HMM — each validated
  against an exact independent oracle (transfer matrix and brute-force path
  enumeration respectively) and a central-difference derivative check. The
  interface carries no application knowledge, and a test asserts it imports
  nothing from `snakes_and_ladders.sim`, `snakes_and_ladders.likelihood` or `snakes_and_ladders.search`.

  Gradient-based fitting for any `Objective` (`snakes_and_ladders.opt.fit`): L-BFGS with a
  strong-Wolfe line search, convergence judged on the gradient relative to the
  objective's own magnitude, and confidence intervals from the observed Fisher
  information pushed through the constraint map by the delta method. Validated
  by parameter recovery on both instances — the Potts chain's 95% intervals
  cover the truth at exactly the nominal rate over 60 replicates, and the HMM's
  gradient fit is cross-checked against Baum–Welch, an independent fitting
  algorithm sharing no optimizer, parameterization or constraint map with it.
  Two QA figures report the recovery and how fast the intervals converge on
  their nominal coverage. (#63)
- Branch-length fitting for a fixed topology (`snakes_and_ladders.likelihood.objective`),
  behind the same model-agnostic interface as the Potts and HMM instances — the
  optimizer needed no phylogenetic special-casing, which is the evidence issue
  \#63's abstraction was not shaped by one model. Branch lengths are recovered
  within their confidence intervals on both the unrooted and rooted fixtures,
  and `ROADMAP.md`'s sub-second gradient update at n=100 is now measured (203 ms
  at 1000 sites) rather than asserted.

  The two branches below a rooted binary tree's root are fitted as one
  parameter and reported as their sum, because only the sum is estimable: under
  a reversible model the likelihood is unchanged by moving length between them.
  `snakes_and_ladders.opt.fit.parameter_covariance` now rejects an ill-conditioned observed
  information rather than inverting it, since a numerically singular matrix
  inverts successfully and yields a meaningless interval.

  The general time-reversible model (`snakes_and_ladders.sim.gtr`), so `Q` and `π` have free
  parameters to fit at all — Jukes–Cantor has none. Exchangeabilities and the
  stationary distribution are fitted alongside the branch lengths and recovered
  within their intervals, and `simulate_alignment` takes an optional
  `rate_matrix` (omitting it keeps the Jukes–Cantor closed form unchanged).
  Equal exchangeabilities with a uniform `π` reproduce `jc_rate_matrix` and
  `jc_transition_probabilities` to machine precision, which is how the model is
  validated. Its three normalizations are gauges rather than conventions: each
  removes an exactly flat direction that would otherwise leave every parameter
  without a confidence interval. (#104)
- Device dispatch for the likelihood engine (`snakes_and_ladders.likelihood.device`):
  availability-based selection preferring CUDA, then Metal/MPS, then CPU, and
  the cross-device agreement tolerance `CLAUDE.md` had long promised was stated
  somewhere. `pruning_torch` takes dtype and device from the branch-length
  tensor it is given rather than hardcoding `float64` in six places, so a caller
  moves the whole recursion by moving one tensor; `float64` remains the default.
  The tolerance is relative and keyed on the lowest precision in the comparison
  — `1e-11` for `float64` on both sides, `1e-6` where either side is `float32`,
  since Metal cannot do `float64`. Both are derived from measured agreement
  rather than chosen, and the `float32` figure is exercised on CPU, so runners
  without an accelerator still check it. (#106)
- Topology search (`snakes_and_ladders.search.infer`): a user-facing `infer(alignment, k,
  ...)` that hill-climbs over NNI or SPR neighbourhoods, fitting the continuous
  parameters of every candidate, and reduces to a plain continuous fit when a
  topology is supplied. This is the first user of the seam issue #63 recorded
  and left unbuilt — a discrete move builds a new `Objective` rather than
  stepping inside a fit — and `snakes_and_ladders.opt` needed no change to serve it, which
  is the evidence that abstraction was not shaped by one model.

  Budgets are counted in candidate fits rather than seconds, so a run is
  reproducible from its seed; a topology is scored at most once per search,
  keyed on `leaf_bipartitions`; and `snakes_and_ladders.search.topology.random_topology`
  draws a seeded starting tree that reaches every topology on its leaf set.

  Exhaustive enumeration of unrooted topologies
  (`snakes_and_ladders.search.topology.enumerate_topologies`), which gives search quality
  its first independent oracle: below 8 taxa every topology can be scored, so
  "did hill climbing find the best tree" has an answer rather than an opinion.
  Measured on a 6-taxon fixture, both NNI and SPR reach the enumerated maximum
  and recover the generating topology from all 12 starting points — at a median
  of 14 candidate fits for NNI against 48 for SPR.

  Two QA figures for search: a trajectory against the exhaustive landscape, and
  the tree the search selected beside the highest-scoring one it rejected. The
  second reports a detail worth having in writing — the rejected topology fits
  the internal branch that would create the wrong grouping at essentially zero
  length, which is what rejecting a topology looks like from inside the
  continuous fit. `BranchLengthObjective.fitted_tree` is new, the inverse of
  `theta_from_truth`, so a fitted tree can be drawn or serialized as a tree. (#117)
- Reinforcement learning (`snakes_and_ladders.learn`): the `Environment` interface, a
  softmax-over-scored-actions policy, REINFORCE with a baseline, and an exact
  trajectory-enumeration oracle. Model-agnostic on the terms `snakes_and_ladders.opt` is —
  it imports nothing from `snakes_and_ladders.sim`, `snakes_and_ladders.likelihood` or `snakes_and_ladders.search`,
  asserted by an import-graph test — so the phylogenetic environment will live
  in `snakes_and_ladders.search` rather than here.

  The reference environment is single-flip local search over the same 1-D Potts
  chain `snakes_and_ladders.opt` fits, appearing once as an objective and once as a search
  problem. Its reward decomposes exactly into the two features the policy
  scores, which puts hill climbing *inside* the policy class as the weight
  vector proportional to `(J, 1)`, so a comparison against it is a statement
  about learning rather than about two unrelated algorithms.

  Rewards are closed forms at known parameters, with no inner optimization —
  issue #131's simplification, and what makes an episode cost microseconds
  rather than one L-BFGS solve per action.

  Because the action set and horizon are finite, the expected return is a
  closed form and its gradient follows by differentiating it. That is the
  oracle every claim here is pinned to, rather than to a training curve: the
  enumerated gradient agrees with central finite differences to 1.5e-11
  relative, and the sampled estimator with the enumerated gradient to 9.9e-03
  over 6000 episodes, while a myopic variant that credits each action with only
  its own reward is rejected at 71%.

  On that landscape the learned policy beats hill climbing at a matched
  decision budget, reaching the enumerated optimum from 86.6% of the 81 starts
  against greedy's 80.2%, in 8 of 8 training seeds.

  The phylogenetic RL environment (`snakes_and_ladders.search.rl`): tree search as the MDP
  the technical document specifies — a state is a topology, an action is an NNI
  or SPR neighbour, the reward is the improvement in log-likelihood. It lives in
  `snakes_and_ladders.search` rather than `snakes_and_ladders.learn` for the reason the phylogenetic
  `Objective` lives in `snakes_and_ladders.likelihood`: the model-agnostic module may import
  no application code.

  Two reward models, and the comparison between them is the deliverable.
  `FITTED` maximizes over branch lengths per candidate; `KNOWN` evaluates at one
  fixed branch length with no optimization, which is what makes an episode
  affordable — measured at 352 us against 113.7 ms per candidate.

  The substitution is validated rather than assumed
  (`snakes_and_ladders.qa.rl_reward_surface`). "The known parameters" cannot transfer across
  topologies at all — a branch length belongs to an edge, and a different
  topology has different edges — so the cheap reward is a different surface, not
  an approximation of the fitted one. On the 6-taxon fixture the two score the
  generating topology highest, agree on the best of all 105 topologies, and
  correlate at 0.9568; across a 50-fold range of the fixed branch length they
  still agree on the best topology every time, with the correlation never below
  0.8719.

  Measuring that turned up a property of the fitted surface worth recording: it
  does not totally order topologies. Many candidates share a maximized
  log-likelihood to within the optimizer's convergence, because the branch that
  would distinguish them is fitted to zero and the tree collapses to the same
  polytomy. Their order is therefore not a property of the model, and a rank
  correlation — which depends on it — moves by up to 0.04 under a perturbation
  of one part in 1e9, so it is not a measurement and cannot appear in a
  committed document that CI rebuilds.

  `snakes_and_ladders.qa.figure` gains `pearson_correlation`, which is continuous in the
  scores, written here rather than pulling in `scipy`.

  Not claimed: that a learned policy beats hill climbing on trees. The 6-taxon
  fixture cannot support that claim in either direction, because greedy already
  reaches the enumerated optimum from every start. (#131)

### Changed

- Cut the `0.1.0` release: `towncrier build --version 0.1.0` merged the fragments above into this file, `ROADMAP.md` records Milestone 5's NNI/SPR generators as landed, and `STATUS.md` (deleted from the repository but still described elsewhere as a live ledger) is dropped from `CLAUDE.md`, `DEV.md`, and the PR template — GitHub issues and labels (`infra/TICKETING.md`) are the project board now. `ROADMAP.md`'s remaining milestones (1, 2, 4, 6) gain the same landed/not-started status notes that 3 and 5 already carried; the fixtures directory, previously spelled out in eleven test modules under two names, moves to a single `tests/_fixtures.py`; and `DEV.md` states the measured cost of the release-gated suite so `pytest -m "not release"` is the obvious default while developing. (#101)
- CI skips the benchmark suite unless the change touches code a benchmark
  measures — `src/`, `python/snakes_and_ladders/{sim,likelihood,opt,search}/`,
  `tests/benchmarks/`, or a lockfile. Benchmarks are half the suite's wall clock
  (36 s of 71 s), and a documentation or QA change cannot alter what they
  measure. The job still runs and reports, since it is a required check, and
  coverage is unaffected because every line a benchmark reaches is also reached
  by the regression module it pairs with. (#109)
- Tolerances on log-likelihoods and gradients are now relative rather than
  absolute. Both quantities are sums over sites, so an absolute bound fixed at
  one site count does not transfer: the backends agree to ~8e-13 relative at
  every size, but the same agreement reads as 7.4e-07 absolute at 200,000 sites
  and would fail the previous 1e-9 bound. The suite's fast tests ran at tens of
  sites because that is cheap, not because the tolerance required it — a
  release-gated test now checks the bound at full fixture scale, and asserts
  that the previous absolute bound would have failed there. Absolute tolerances
  are kept where the quantity does not scale: transition probabilities, rate
  matrix row sums, and Monte Carlo frequencies. (#111)
- The technical document is restructured around the optimization abstraction
  rather than the phylogenetic application: notation is split into a
  model-agnostic half and an application half, the substitution model and its
  reversibility move to an appendix, code paths are gone from the body, and the
  reinforcement-learning section carries the theory needed to implement it.
  Two QA figures — backend agreement and analytic-versus-finite-difference
  gradients — are dropped with their scripts, since both reported checks the
  regression suite performs rather than performing any.

  Fixed a preamble setting that suppressed the space at *every* source line
  break in the document, so text wrapped mid-sentence set as "substitutionmodel".

  `snakes_and_ladders.qa.figure` gains `write_qa_table`, which emits a LaTeX `tabular`
  fragment instead of a matplotlib image, and `latex_integer`, which separates
  large numbers as `200\_000`. Caption safety is now enforced at the point of
  writing rather than asserted per test, so a caption that would break the
  LaTeX build fails in the script that wrote it. `render_problem_sizes_figure`
  is replaced by `render_problem_sizes_table`, and `state_label` moves from
  `snakes_and_ladders.qa.sim_example` to `snakes_and_ladders.qa.figure`. (#118)
- The technical document's prose is tightened globally: the quality-assurance
  section drops from 968 to 654 words and the body as a whole from 5986 to
  5581, with no claim, number or reference removed. The cut is mostly one
  duplication — a figure's body paragraph restating the caption the figure
  already carries, and the drift guarantee stated twice within twenty lines.
  The rule now applied throughout is that the caption says what a figure shows
  and the body says why it is there. (#125)
- `docs/` gains its own `CLAUDE.md`, on the pattern of the other module files.
  It carries the rules that keep a committed, CI-regenerated artifact true:
  what is committed against what is generated, that a figure is regenerated
  rather than edited, that the document reads captions and never restates them,
  and that a generated caption may report only quantities continuous in their
  inputs — since CI rebuilds `docs/draft.pdf` and byte-compares it, a
  discontinuous statistic breaks the build and was never a measurement anyway.
  The formatting contract stays in root `CLAUDE.md` under **Expected Reader**,
  referenced rather than restated. (#136)
- Docstrings and comments are revised against the writing style. The corpus
  needed five edits: of 50 flagged hedges, all ten uses of "strictly" are
  mathematical ("strictly positive", "strictly bifurcating"), and most of the
  rest are contrastive — "not merely wide", "exact rather than merely
  convenient", "obviously correct, not fast".

  Two magnitude claims are replaced by their measurements. `snakes_and_ladders.search.rl`
  called the fitted reward "roughly two orders of magnitude" more expensive
  than the known one; it is 113.7 ms against 352 us, a factor of 323. The
  search benchmark said a search spends "essentially all" of its time in
  candidate fits, and now says what the two benchmarks beside each other
  measure. (#138)
- Root `CLAUDE.md`, `DEV.md` and the module `CLAUDE.md` files are revised
  against the writing style, correcting three defects along the way.

  Root `CLAUDE.md` named the project `snakes_and_ladders`; the package is
  `snakes_and_ladders`. Its "Check known math properties/Invariants" rule was a label with no
  body, so the invariants it names — transition rows summing to 1, detailed
  balance, gradients against finite differences, a monotone likelihood — are
  restored. Its reference table announced a count of 25 that nothing checks, and
  now announces none.

  `DEV.md`'s `technical-doc` row said the job fails on undefined references or
  citations; it also fails on a multiply-defined label. `qa/CLAUDE.md` said it
  validates `search/` "later", where `snakes_and_ladders.qa` has validated `search/` and
  `learn/` since issues #117 and #131.

  No rule was added, dropped or re-scoped: the section headings and rule labels
  of both files are unchanged, checked by diff, except the empty rule above. (#138)
- `README.md`, `INSTALL.md` and `ROADMAP.md` are revised against the writing
  style in `CLAUDE.md`, along with one hedge in `docs/tex/main.tex`.

  The pass corrected more than it compressed. `README.md`'s opening sentence
  said "modern optimization discrete/continuous optimization" and misspelled
  "reference"; `ROADMAP.md`'s objective and numerics requirements were
  ungrammatical; `README.md` linked to `infra/TICKETING.md`, deleted in
  `aae9e74`. `INSTALL.md` carried two claims that had gone stale: that one
  `snakes_and_ladders.qa` script renders the technical document's figures, where eleven now
  do, and that CI's LaTeX job fails on undefined references or citations, where
  it also fails on a multiply-defined label and on a committed `docs/draft.pdf`
  that differs from the rebuild.

  `ROADMAP.md`'s milestones now state a specification and then a **Status:**
  line, so a reader can find what landed without reading a paragraph.
  `README.md`'s "What exists" gains the measurements behind its claims — `n =
  5..8` for the neighbour counts, 12 of 12 starting points at 6 taxa, 203 ms per
  gradient update at n=100, L=1000. (#138)
- The technical document is restructured as an academic letter: an abstract, an
  introduction stating the contributions and what is *not* claimed, methods,
  results grouped by claim rather than by the order the build emits figures,
  a discussion with threats to validity and outstanding work, and conclusions.

  Two figures answer requirements `ROADMAP.md` states and nothing had measured.
  `snakes_and_ladders.qa.backend_agreement` puts all three pruning backends against
  brute-force marginalization, which shares no code with the recursion it
  checks: worst relative deviation 4.0e-14 across four site counts spanning a
  factor of 30. `snakes_and_ladders.qa.topology_accuracy` measures the normalized
  Robinson-Foulds distance from the inferred topology to the generating one
  against the site count, and finds the margin: the 0.05 accuracy requirement
  is met from 125 sites upward, with 8 of 8 replicates recovering the topology
  exactly at 2000 sites against 5 of 8 at 60.

  `snakes_and_ladders.search.topology` gains `robinson_foulds` and
  `normalized_robinson_foulds`. The normalizer counts internal splits only:
  every tree over the same leaves induces all the trivial ones, so including
  them would shrink every distance by a taxon-count-dependent factor and
  silently weaken the bound.

  The competitiveness comparison against IQ-TREE 2 and RAxML-NG, and the
  learning curve of a trained phylogenetic agent, are stated as outstanding
  rather than drawn: neither measurement exists, and a figure with invented
  data in a committed document is worse than a stated gap. (#144)
- `snakes_and_ladders.numerics` holds the vectorized categorical sampler that
  `snakes_and_ladders.sim.simulate`, `snakes_and_ladders.opt.potts` and `snakes_and_ladders.opt.hmm` each carried a
  private copy of. The copies had drifted: two omitted the clamp on the last
  cumulative column that the third had, so a probability row summing to
  `1 - 4e-16` after rounding could return a category one past the end of the
  alphabet. The surviving copy carries the guard, and a test constructs the
  draw that triggers it rather than waiting for a 4e-16 event.

  `DEV.md` no longer restates `CLAUDE.md`'s Performance and Testing rules or
  `docs/CLAUDE.md`'s technical-document rules, and its release worked example
  no longer describes cutting `0.1.0` — a version whose changelog section was
  already built.

  `CHANGELOG.md`'s legacy `[Unreleased]` section is labelled as older than the
  dated sections above it rather than newer. `ROADMAP.md` records that the
  accuracy requirement's first half is now measured and its second half is not. (#146)


## [0.1.0] - 2026-09-02

### Added

- `k`-state Jukes–Cantor sequence simulator in `snakes_and_ladders.sim`: generates an
  alignment and the ancestral tree in Newick from a typed
  `simulation_params.yaml`, and retains the parameters that generated them.
  Simulated substitution frequencies are validated against the closed-form JC
  transition probabilities within a yaml-declared Monte Carlo tolerance across
  several site and taxa sizes. Promotes `numpy` and `pyyaml` to core
  dependencies. (#55)
- Added `snakes_and_ladders.sim.newick`: topology counting (`count_topologies`), Newick string validation (`validate_newick`), and state-labelled Newick serialization (`to_newick`), now the package's single source of Newick functionality. (#60)
- Added `snakes_and_ladders.qa`, quality-assurance figure scripts for the technical document, starting with `snakes_and_ladders.qa.sim_tree` (renders the assumed simulation tree with branch lengths). `infra/build_technical_doc.sh` regenerates these figures and builds `docs/draft.pdf`. (#61)
- Vectorized NumPy Felsenstein pruning in `snakes_and_ladders.likelihood`, computing
  `ln L(alignment | tau, Q, t, pi)` under the k-state Jukes-Cantor model with
  per-node rescaling accumulated in log space. Ships with an independent
  brute-force marginalizer used only as the test oracle at `n <= 6` taxa.
  Validated against brute-force marginalization to machine precision,
  rescaled/unrescaled agreement, the pulley principle (root-position
  invariance), and scoring the generating topology above random wrong
  topologies on simulated data. This is the reference every future backend
  (Rust, PyTorch, CUDA, Metal) is pinned against. (#62)
- Added `snakes_and_ladders.qa.sim_example` and `snakes_and_ladders.qa.sim_problem_sizes`: a worked
  4-taxon simulation example (Newick topology and aligned sequences) and a
  cross-fixture table of problem-size parameters (taxa, sites, seed,
  tolerance), read directly from the `simulation_params.yaml` fixtures. Wired
  into `infra/build_technical_doc.sh` and `docs/tex/main.tex`. (#67)
- Differentiable PyTorch Felsenstein pruning (`snakes_and_ladders.likelihood.pruning_torch`),
  taking branch lengths as a `torch.float64` CPU tensor separate from the
  topology so `torch.autograd` differentiates through them. Validated against
  the NumPy oracle and brute-force marginalization to `atol=1e-9`, against
  `torch.autograd.gradcheck` and central finite differences of the NumPy
  likelihood to `atol=1e-6`, and rescaled/unrescaled agreement. A general
  `rate_matrix` path (`torch.matrix_exp`) is exercised by a benchmark fitting a
  Jukes-Cantor rate matrix Q, alongside a forward-pass benchmark against the
  NumPy reference. (#70)
- Rust CPU Felsenstein pruning backend (`oxi_snakes_and_ladders.pruning_log_likelihood`,
  exposed via `snakes_and_ladders.likelihood.pruning_rust`), implementing the same
  recursion as the NumPy oracle in `src/pruning.rs` and exposed through PyO3.
  Validated against the NumPy oracle and independent brute-force
  marginalization at `n <= 6` taxa, and against the NumPy oracle at realistic
  (taxa, site) sizes to `abs_tol=1e-9`. Ships a `criterion` benchmark
  (`benches/oxi_snakes_and_ladders_bench.rs`) and a paired `tests/regression/test_pruning_rust.py`
  / `tests/benchmarks/test_pruning_rust_bench.py` module, reporting Rust vs.
  NumPy timings at 4- and 8-taxon, 200,000-site fixtures. (#77)
- NNI and SPR neighbourhood generators over unrooted binary topologies
  (`snakes_and_ladders.search.topology`), behind one `Topology -> Iterator[Topology]`
  interface. Validated exhaustively against `2 * (n - 3)` (NNI) and
  `2 * (n - 3) * (2 * n - 7)` (SPR) at `n = 5..8` -- every one of the
  `count_topologies(n - 1)` distinct topologies, cross-checked for neighbour
  validity, symmetry, and NNI-in-SPR containment (`n = 8` gated to
  `pytest -m release`, ~2.5 minutes). `snakes_and_ladders.sim.newick` gains
  `validate_unrooted_newick` for the trifurcating-root convention this reuses.
  The random-walk connectivity test is deferred to issue #73's canonical
  Newick key. (#79)
- Added a Release issue template (`.github/ISSUE_TEMPLATE/release.yml`) that
  drives the repository-consolidation audit ahead of a release, and
  `infra/release.sh`, a local release gate running every per-PR CI check plus
  the release-gated `pytest` tests, `sphinx-build -W`, and the technical
  document build. `DEV.md` documents the release procedure, including the
  version-bump and tag/publish steps. (#90)

### Changed

- `DEV.md` states the `tests/` layout convention: organized by kind
  (`regression/`, `benchmarks/`) at the top level and by subject within it, with
  rules for benchmark/regression pairing, fixture placement, and when a kind
  splits into submodule subdirectories. (#45)
- The rendered technical document (`docs/draft.pdf`) is now committed to the
  repository instead of being a gitignored build artifact. CI's
  `technical-doc` job fails a PR whose rebuilt PDF differs from the committed
  one, catching a `docs/tex/` or QA-figure change that wasn't regenerated. (#71)
- PR template gained a "Follow-up / Deferred Work" section for TODOs left to a tracking issue. (#84)
- `docs/tex/main.tex` now states its intended reader (a developer with
  baseline scientific/performance-computing background but no phylogenetics
  expertise) and the formatting contract that follows from it: streamlined
  main text, standard non-snakes_and_ladders-specific background (e.g. NNI, SPR) moved to
  a new appendix and cited from the point of use. `CLAUDE.md` records the
  same contract for anyone editing the document. (#85)
- PR template's Benchmark section gained a second table for scientific/tolerance regression tests, so contributors report the realized value alongside the reference and tolerance it was checked against. (#88)
- Simplified the Release issue template (`.github/ISSUE_TEMPLATE/release.yml`):
  the consistency-audit field is no longer required — the ticket's job is to
  trigger the consolidation audit and surface follow-up tickets, not to gate
  submission on having written them out — and the `infra/release.sh` checkbox
  is dropped. That check is already enforced by the documented release
  procedure (`DEV.md`'s "Release" section): the gate is run, and only then is
  the version bumped and the tag cut. (#94)

### Fixed

- Embedded TrueType (not Type 3) fonts in QA figures, fixing `docs/draft.pdf`'s failure to render in GitHub's blob viewer. (#76)
- Fixed the Release issue template (`.github/ISSUE_TEMPLATE/release.yml`):
  `roadmap-progress`, `consistency-audit`, and `follow-up-tickets` moved their
  guidance from `description:` (static gray helper text below the box) into
  `value:` (the box's own prefilled, editable content), so filers answer the
  ask instead of retyping it. (#98)


## [Unreleased] — pre-0.1.0 history

This section predates the `towncrier` convention and is retained as history.
It is older than the dated sections above, not newer.

### Added

- Changelog Automation: Adopted [towncrier](https://towncrier.readthedocs.io) to manage `CHANGELOG.md` via fragments in `changelog.d/`, restoring the no-merge-conflict, CI-enforced workflow of the bespoke system removed below — as a maintained dependency instead of custom infra code.
- Project Scaffolding: Initialized uv-based Python 3.12 environment and Rust backend via maturin/PyO3 (snakes_and_ladders.oxi_snakes_and_ladders).
- Module Architecture: Established core package skeleton (sim, likelihood, opt, search, infra) enforcing a strict separation between infrastructure and domain-specific application logic.
- Documentation Suite: Deployed Sphinx API docs, a LaTeX technical document for scientific foundations, and strategic planning documents (ROADMAP.md, STATUS.md, DEV.md).
- CI/CD Pipeline: Implemented GitHub Actions for Python/Rust linting, testing, documentation building, and dependency auditing using strictly locked environments (uv.lock, Cargo.lock).

### Changed

- Scientific Modeling: Formalized the Canonical Newick form and k-state Jukes–Cantor transition probabilities within the technical documentation.
- Strategic Roadmap: Defined strict engineering requirements (problem scale n=10−1000, RF ≤0.05, sub-second gradients) and partitioned development into six distinct workstreams.
- Framework Selection: Designated PyTorch as the primary autodiff engine and Aim for experiment tracking.
- Code Quality: Enforced mypy strict mode across all modules, required explicit np.random.default_rng for benchmarking, and applied standard linting rules to test suites.
- CI Optimization: Streamlined CI by caching uv/Cargo environments, auditing only modified dependency graphs, and auto-canceling superseded branch runs.

### Fixed

- Resolved dependency resolution failures in CI linting jobs and pinned cargo-audit to prevent floating resolve breakages.

### Removed

- Deprecated automated changelog fragments (changelog.d/) and associated infra/changelog.py script in favor of a standard flat file. (Superseded: this file's fragment-based workflow is restored via towncrier, a maintained dependency, rather than the bespoke infra removed here — see the towncrier adoption entry above.)
- Removed unused pytest-xdist dependency and legacy infra/ scaffolding modules.

### Security

- Raised pytest dependency to >=9.0.3 to patch vulnerability PYSEC-2026-1845.
