# TICKETS

The work that stands between `STATUS.md` and `ROADMAP.md`, as titles. Each line
below is one filing through `.github/ISSUE_TEMPLATE/task.yml` — the outcome, the
non-goals, and how it will be validated are written there, not here. A title is
not a plan: the plan is posted to the thread and approved before any pull
request opens (§0.2).

Ordering within a milestone is by dependency, not priority; priority is a label.
A parenthesized number is an issue already filed; a bullet without one is work
this file names and nobody has filed yet, which is the honest state and not an
oversight to paper over. `tests/regression/test_planning_documents_agree.py`
keeps the milestone headings here, in `ROADMAP.md` and in `STATUS.md` naming the
same work, and keeps the parenthesis the only way a ticket is cited.

## Milestone 1.1 — Simulation & Ground Truth Engine

- Support the different lattice types (#231)
- A turbo code problem, fixture and belief-propagation example (#233)
- A linear-time LDPC encoder at full size, so a non-trivial codeword can be
  sent through the 19,998-bit code rather than the zero word (#340)
- Bicycle codes and quantum LDPC codes as a second code family (#362)
- Move `PottsParams`/`load_potts_params` out of `snakes_and_ladders.opt.potts`, so
  `simulate_chains` can call the general graph sampler instead of keeping
  its own copy of the exact open-chain recursion (#186)
- Additional evolutionary models (#107)
- Rate variation across sites, in the simulator and every backend (#323)
- Emission-family extensions: multivariate and tied Gaussian, zero-inflated
  counts, the Dirichlet-multinomial (#330)
- Simulate at the declared scale — `n` to 1000, `L` to 11000 — and report the
  memory footprint of the simulator alone beside the evaluator's

## Milestone 1.2 — Differentiable Likelihood & Energy Engine

- Belief propagation converges in two sweeps at zero field, so every
  zero-field benchmark measures the fixture (#245)
- LDPC decoding at size: bit and block error rates against `p`, `epsilon`
  and `sigma` at 19,998 bits over shared seeds, iterations to convergence,
  wall clock and peak memory against the 16 GB requirement, and the profile
  that decides whether a Rust kernel over the same offsets follows (#340)
- Coloured iterated conditional modes on CUDA and Metal through torch,
  measured against the 10× rule before any Triton kernel (#227)
- One energy/likelihood evaluator API across the problem classes,
  asserted by an import-graph test (#238)
- Device dispatch for the pruning backend, CUDA and Metal/MPS, held to the
  cross-device tolerance and its `float32` bound (#280)
- Evaluate a Triton or JAX kernel for the site-parallel recursion against the
  10× rule before porting
- Branch-and-bound over topologies with the certified bounds of #308, and a
  bound for a non-Jukes–Cantor model, whose transition matrix is not affine
  in one variable per branch (#329)
- Derive pruning, forward–backward and sum-product in the textbook appendix,
  cited from the code (#326)
- Put `stubtest` in CI — the type stub has already drifted (#37)

## Milestone 1.3 — Continuous Optimization via Autodiff

- Fit HMM transition and emission matrices to nominal interval coverage
- Refuse an unidentifiable fit rather than returning a meaningless interval
  (#122)
- Realize the tolerance helper rather than assume it is applied by hand (#91)
- Profile a gradient fit in memory and time across the declared `n × L × k`
  range (#232)
- A spectral initialization for trees, on the footing the HMM's moment
  estimators give (#364)
- Trajectory-length adaptation (NUTS), only if a posterior the #268
  comparison reaches is one the fixed trajectory length of #333 samples badly

## Milestone 1.4 — Discrete Move Sets & Classical Baselines

- Iterated conditional modes as a first-class solver across every lattice
  model (#226)
- MAP decoding of an LDPC code as energy minimization: sum-product against
  the Gibbs sampler, the annealer and single-site descent at matched
  evaluations on enumerable codes, the ML codeword as referee (#340)
- Iterated conditional modes over HMM state paths (#176)
- A discrete instance no baseline solves within budget — still open. #177's
  tree is solved by random-restart greedy at 1.000 (#198), and #209 measured
  single-site descent matching or beating the planted Viana-Bray state at
  every frustration and connectivity tried
- Make the rooted/unrooted distinction explicit and give topologies a canonical
  key (#114)
- Multi-SPR neighbourhoods, each stating in which sense it is complete and what
  it costs per step (#329)
- Establish the external reference tools to benchmark against, and how they are
  installed (#126)
- A classical baseline suite the applications are scored against under one
  budget — large parsimony under NNI and SPR is the tree baseline that exists
  and `opt.budget.compare` the budget; the HMM and Potts baseline suites remain
- One Potts model in code: one adjacency, one energy, one heat-bath sweep
  (#277)
- Max-Cut: the gradient norm at termination, and a certified SDP upper bound
  behind an approved dependency (#334)
- One annealing driver and one exchange step behind the eight annealing and
  tempering entry points, and one weighted enumeration behind the eight
  enumerators, each merge pinned to the oracle `STATUS.md`'s consistency
  audit names
- Transcribe the seven- and eight-taxon calibration table of #331 from a
  release-gate run into `STATUS.md`, where #350 left a placeholder

## Milestone 2.1 — RL Agent Formulation & Deployment

- A feature set for the tree environment beyond the improvement a move buys,
  with the unidentifiable-constant invariance pinned, and the comparison it
  changes (#328)
- The factor-graph environment over the Gibbs moves of #309, and the
  surrogates of #308 as a tree reward model — the parts of #313 that waited
  on #296 and #308 and have no ticket since it closed
- Truth as a terminal penalty, never a training signal
- A move set for the code, so decoding is an environment like the other
  three (#340)

## Milestone 2.2 — Curriculum Learning

- Weight transfer across problem sizes for a policy, and the schedule from
  `n = 10` to `n = 1000`; the surrogates of #308 transfer from 5 to 6 taxa
  and from 3×3 to 4×6 lattices, and a policy does not yet
- Batched episode rollout, so a budget at `n = 200` is affordable
- Measure zero-shot collapse against the curriculum, so the regimen is
  justified rather than assumed

## Milestone 2.3 — Empirical Validation & Benchmarking

- Ingest empirical alignments, with their provenance recorded
- Benchmark harness: budget-matched runs against IQ-TREE 2 and RAxML-NG on
  shared seeds, once #126 installs them
- Report RF and ΔlnL against known truth up to `n = 1000`
- A fixed-hardware benchmark runner, since CI hardware cannot rank performance
- GPU scaling for the site-parallel recursion, once a device exists to
  measure on (#280)

## Milestone 2.4 — Experiment Tracking, Ablations & Leaderboard

- Create a ledger of benchmarked and validated runs with Aim (#75)
- Reproduce a run from a single manifest, and assert it
- Budget-matched ablation leaderboard across shared seeds — the experiment
  ledger's index is the leaderboard; the matrix of problems × tiers × method
  families is four cells filled
- Paired significance test required before a variant is adopted as
  state-of-the-art — `opt.budget.mcnemar` exists; the adoption rule is not
  yet a gate

## Stage 3 — Research Extensions

- Differentiable topology search over the tropical Grassmannian — blocked on
  an oracle, not on effort: unlike the Gumbel-softmax half (#211), no tree
  instance at an interesting size has a known optimum to referee the claim
- Whether the deterministic multilinear relaxation extends past a chain, to
  the Potts lattice and to graphs with cycles — #211 established the identity
  holds for any objective with one factor per site per term, and measured the
  method only on chains
- Learned compound moves (#147)
- Transformer policy over canonical encodings
- References and blue-sky directions (#360)

## Cross-Cutting Infrastructure

- CPU parallelism past the first three sites: the candidate fits of
  `search.infer`, `learn.rollout` batches, tempering replicas (a `rayon`
  loop is the alternative, a dependency decision), `qa.build` and
  `check_notebooks` through the seam, and `pytest-xdist` against the test
  budget (#344)
- Assess the computational efficiency of the key algorithms for scaling
  fixtures through simulation, optimization and learning (#232)
- The targets the runtime-optimization audit left unmet: the tree schedule
  within 2x of the forward recursion on a chain, the reassociated
  transfer-matrix product in the Potts chain objective, `maxflow_rust` as
  alpha expansion's inner solver, and a compiled sweep for the factor-graph
  Gibbs sampler (#341)
- Adopt `rustworkx` on a hot path where a measurement says so —
  `search.topology._component`, `potts_mcmc._adjacency`, the spanning trees
  of the bound — moving the replaced implementation to `sandbox/`
- TorchRL `TensorDict` environments and a `SyncDataCollector` over the
  Gymnasium adapter for Milestone 2.2's batched rollout, adopted only if the
  collector beats `learn.rollout` on the 8 → 20 taxa scaling with `float64`
  forced throughout
- PyTorch Geometric `Batch.from_data_list` for surrogate training at 20+
  taxa, and `HeteroData` over `sim.factor_graph` for a learned message
  passing beside the exact one, gated on the training-time benchmark
- RL frameworks for validation and extension (#315)
- Audit `ROADMAP.md`, `STATUS.md` and `TICKETS.md` (#283) — done at 0.4.0 by
  #358, to be closed with #275 and #276 once it merges
- Release 0.4.0 (#358); #236 is the earlier filing of the same release
- Release-readiness check: the changelog section, the `Cargo.toml` version
  and the tags agree (#327)
- Run the required checks on every pull request, not only those against
  `main` (#273)
- Labels say what the pull requests say (#279)
- Re-key the milestone references in code and `docs/tex/` to the roadmap's
  `N.M` numbering (#324)
- Cite or retire the two QA figures the documents still do not cite,
  `sim_problem_sizes` and `topology_accuracy` (#325)
- One canonical list of the local checks (#40)
- Detect a merge at a stale head, which silently drops commits (#123)
- Abbreviate the package as `sal` (#301)
- Test hygiene: remove the QA tests that duplicate their figures'
  computations, and give the entry points one shape (#338)
- Adopt `qa.layout` in the surrogate joint-distribution and coupled-model
  field figures (#339)
