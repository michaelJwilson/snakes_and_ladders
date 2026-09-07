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

- An HMM with Gaussian emissions, and the unbounded likelihood that comes
  with it (#228)
- An HMM with negative binomial emissions, the family whose M-step has no
  closed form (#229)
- Support the different lattice types (#231)
- A turbo code problem, fixture and belief-propagation example (#233)
- Define one fixture API across trees, lattices and chains (#132)
- Convert the sixteen signatures that still take a seed where the rule says a
  generator (#240)
- Move `PottsParams`/`load_potts_params` out of `snakes_and_ladders.opt.potts`, so
  `simulate_chains` can call the general graph sampler instead of keeping
  its own copy of the exact open-chain recursion (#186)
- Additional evolutionary models (#107)
- Rate variation across sites, in the simulator and every backend
- Simulate at the declared scale — `n` to 1000, `L` to 11000 — and report the
  memory footprint against the 16 GB / 24 GB requirement

## Milestone 1.2 — Differentiable Likelihood & Energy Engine

- Belief propagation converges in two sweeps at zero field, so every
  zero-field benchmark measures the fixture (#245)
- Coloured iterated conditional modes on CUDA and Metal through torch,
  measured against the 10× rule before any Triton kernel (#227)
- Expose forward-backward as an evaluator, not as Baum-Welch's internals (#173)
- One energy/likelihood evaluator API across the three problem classes,
  asserted by an import-graph test (#238)
- CUDA dispatch for the pruning recursion, pinned against the NumPy oracle
- Metal/MPS dispatch, and the `float32` tolerance it forces
- Evaluate a Triton or JAX kernel for the site-parallel recursion against the
  10× rule before porting
- Bounded and approximate likelihoods for branch-and-bound pruning, with the
  proofs `ROADMAP.md` §1.3 requires
- Put `stubtest` in CI — the type stub has already drifted (#37)

## Milestone 1.3 — Continuous Optimization via Autodiff

- Define an initialization abstraction and the initializers each supported
  optimization needs (#251)
- Fit HMM transition and emission matrices to nominal interval coverage
- Refuse an unidentifiable fit rather than returning a meaningless interval
  (#122)
- Realize the tolerance helper rather than assume it is applied by hand (#91)
- Profile a gradient fit in memory and time across the declared `n × L × k`
  range (#232)

## Milestone 1.4 — Discrete Move Sets & Classical Baselines

- Iterated conditional modes as a first-class solver across every lattice
  model (#226)
- Port the single-site Gibbs sweep to Rust, beside the oracle rather than
  replacing it (#246)
- Viterbi decoding, pinned against brute-force path enumeration and against
  the fixture where it disagrees with posterior decoding (#175, #209)
- Posterior decoding, reported as the per-site marginal maximum it is and
  never as the most likely path
- Iterated conditional modes over HMM state paths (#176)
- A discrete instance no baseline solves within budget — still open. #177's
  tree is solved by random-restart greedy at 1.000 (#198), and #209 measured
  single-site descent matching or beating the planted Viana-Bray state at
  every frustration and connectivity tried
- Make the rooted/unrooted distinction explicit and give topologies a canonical
  key (#114)
- Multi-SPR neighbourhoods, each stating in which sense it is complete and what
  it costs per step
- Temperature schedules, and the likelihood-versus-temperature curves that
  judge exploration
- Establish the external reference tools to benchmark against, and how they are
  installed (#126)
- A classical baseline suite the three applications are scored against under
  one budget

## Milestone 2.1 — RL Agent Formulation & Deployment

- A feature set for the tree environment, with the unidentifiable-constant
  invariance pinned
- A tree fixture hard enough to separate a policy from greedy (#177)
- PPO and a learned state-value critic
- Truth as a terminal penalty, never a training signal
- Train a phylogenetic policy and report its learning curve against the
  enumerated expected return (#178)

## Milestone 2.2 — Curriculum Learning

- Weight transfer across problem sizes, and the schedule from `n = 10` to
  `n = 1000`
- Batched episode rollout, so a budget at `n = 200` is affordable
- Measure zero-shot collapse against the curriculum, so the regimen is
  justified rather than assumed

## Milestone 2.3 — Empirical Validation & Benchmarking

- Ingest empirical alignments, with their provenance recorded
- Benchmark harness: budget-matched runs against IQ-TREE 2 and RAxML-NG on
  shared seeds
- Report RF and ΔlnL against known truth up to `n = 1000`
- A fixed-hardware benchmark runner, since CI hardware cannot rank performance

## Milestone 2.4 — Experiment Tracking, Ablations & Leaderboard

- Create a ledger of benchmarked and validated runs with Aim (#75)
- Reproduce a run from a single manifest, and assert it
- Budget-matched ablation leaderboard across shared seeds
- Paired significance test required before a variant is adopted as
  state-of-the-art

## Stage 3 — Research Extensions

- Differentiable topology search over the tropical Grassmannian (#211)
- Gumbel-softmax relaxation of Potts and HMM discrete states
- Neural surrogate for the likelihood and energy, with exact re-scoring of the
  top-`K` candidates
- Learned compound moves (#147)
- Transformer policy over canonical encodings
- Stochastic escape: Metropolis-Hastings worsening steps and ratchet-style
  reweighting (#194)

## Cross-Cutting Infrastructure

- Impose a test grouping, so a class of check can be selected independently
  of the module a diff touched (#237)
- Assess the computational efficiency of the key algorithms for scaling
  fixtures through simulation, optimization and learning (#232)
- Scope rustworkx for efficiency and scaling (#242)
- Vet `ROADMAP.md`, `STATUS.md` and `TICKETS.md` for incompleteness,
  inconsistency and error (#244)
- Separate the technical document into infrastructure, textbook and paper
  (#249)
- Release 0.4.0 (#236)
- Fix the root-detection assertion blocking `infra/release.sh`'s full-suite
  check (#168)
- Re-key the milestone references in code and `docs/tex/` to the roadmap's
  `N.M` numbering
- One canonical list of the local checks (#40)
- Detect a merge at a stale head, which silently drops commits (#123)
- Make the public-facing reference to the work consistent (#250)
- Derive the belief-propagation and forward-backward sections of `docs/tex/`, so
  all three problem classes are documented to the same standard
- Re-include the eleven committed QA figures the technical document no longer
  cites, so CI rebuilds nothing the document does not rest on
- Restore the pruning derivation and the parameter-recovery evidence the
  document dropped, against `ROADMAP.md` §1.3's required contents
