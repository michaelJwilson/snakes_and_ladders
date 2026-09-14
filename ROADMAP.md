# ROADMAP: Mixed Discrete-Continuous Optimization for Graph-Structured Models

## 0. The Development Loop

Development is agent-assisted. The claim the loop supports is not just that an
agent wrote the code, but that the process validates it. Each stage below is a gate,
ordered so that an approach is rejected before it is written and a claim is pinned
before it is published.

### 0.1 The Ticket

- **Deliverable:** a unit of work filed through
  `.github/ISSUE_TEMPLATE/task.yml`, naming the desired outcome stated so it
  can be checked, what it unblocks, the non-goals, the submodule, and how it
  will be validated. Blank issues are disabled.
- Priority (`high`, `medium`, `low`) and submodule labels come from
  `.github/labels.yml` and are applied by a workflow.

### 0.2 The Plan

- **Deliverable:** a plan posted to the ticket thread before any code exists,
  at which point the issue is labelled `planned`.

  A plan is 2–5 steps, or more where the work needs them, and the plan says
  why, each stating how it will be validated. It ends with an
  **Open Questions** section carrying every question on the desired
  behaviour; a plan with nothing outstanding says so under that heading rather
  than omitting it. A plan is subject to `CLAUDE.md`'s Writing Style.
- **Gate:** a maintainer applies `approved`, and only then may a pull request
  open. The pull request must implement the plan already in the thread. A plan
  that turns out to be flawed gets a revised plan posted to the same thread,
  not a silent correction in the diff. Disjoint tickets run one at a time, each
  in its own worktree and its own pull request; coupled changes run as a
  single sequential chain.
- **Record the branch before the first commit.** Once a branch is created for
  an approved plan, the first thing posted is a single issue comment naming
  the branch (and, once opened, the PR number) — before any further commit is
  pushed, so an interrupted or deferred session leaves a ticket that already
  points at the in-flight branch.

### 0.3 The Pull Request

- **Deliverable:** a change answering the fixed checklist in
  `.github/pull_request_template.md`: the Definition of Done, regression baseline and new
  benchmark numbers for every hot path it touches, the *realized* value of
  every tolerance-based test beside its reference and the tolerance, the
  documents the change made untrue, and anything deferred with the tracking
  issue that carries it.
- **Gate:** ten required checks, stated in full in `DEV.md`, spanning a title
  naming the base branch, lint and types in both languages, both test suites,
  the documentation and document builds, the notebook re-execution, and the
  dependency audits. Documentation Sync is part of the diff, not a follow-up:
  a change that makes `README.md`, `DEV.md`, `INSTALL.md`, a `CLAUDE.md`,
  `STATUS.md`, `ROADMAP.md` or `docs/tex/` untrue corrects it in the same pull
  request, and adds a `changelog.d/` fragment if it is user-visible.

### 0.4 Validation

- **Deliverable:** for every new functionaltiy, an oracle that shares no code
  with it — an analytic result, a brute-force on a small problem, an
  exhaustive enumeration, or an independently implemented algorithm — and a
  regression test pinning the claim to it within a stated tolerance.  Further,
  recovery of known parameters and states from a dedicated simulation, across a
  range of problem sizes.
- **Gate:** three rules constrain what the suite may contain. Coverage
  theatre is forbidden: a test asserting only shapes, or only that nothing
  raised, does not count, and a gap is has a placeholder  and a ticket.
  Every accelerated path keeps its reference, e.g. vectorized NumPy for Rust,
  Correctness comes from independent sources, generalization across problems,
  validation of analytic properties, external frameworks and agreement across
  devices and precisions is a declared relative tolerance keyed on the lowest
  precision, never bitwise. The standard is two-way: where an oracle is affordable
  at a size the claim is pinned to it, and where none is, recovering the simulated truth.
  The validation is a ladder, bootstrapping from simple algorithms on small problems to
  efficient ones on large, and validated against each other along the way.  Recovery of
  simulation serves as an independent, final validation.

### 0.5 The book

- **Deliverable:** the textbook and the paper (`docs/tex/`, contents in §1.3).  The textbook
  contain the formulation of all supported problems in a common notation, the problem size/fixture
  definitions, all supported algorithms, their applicability and efficiency for each problem,
  the analytic results used for validation.  The tone is concise and follows the writing style
  in Claude.md at root.  Every plot and table in is rendered by from declared fixtures by `snakes_and_ladders.qa`
  using the same code it reports on.  The paper is in the academic style, advertising the work in a
  measured, authoriative tone.  Plots show key evidence for conclusions drawn, e.g. the optimization performance
  for key problems and the relative efficiency gains.  The model as a factor graph with a sketch of that structure,

## 1. Project Objectives & Specifications

### 1.1 Core Objective

Develop and deploy solvers for mixed discrete-continuous optimization across problems
defined by inference for graphical models (for a range of problem sizes).  Namely, 
continuous optimization, GMMs, HMMs, phylogenetics, ND Potts models with a site-based
external field, low-density parity-check codes, Bicycle codes and qLDPC. A derived
instances exists:  the coupled spatio-sequential model.  The framework integrates automatic
differentiation for continuous parameters with reinforcement learning (RL) to
learn proposal policies that score discrete structural candidates using exact,
approximate, or bounded likelihoods/energies.

### 1.2 Technical Requirements

- **Accuracy & Validation:**
  - *Phylogenetics:* normalized Robinson–Foulds (RF) distance ≤0.05 against
    simulated ground-truth topologies.  Recovery of known parameters and state configurations.
  - *HMMs/Potts:* recovery of true coupling/transition parameters within 95%
    confidence intervals; precise state-sequence decoding.
  - *Spatio-sequential  model:* recovery of knwoen class labels up to permutation, and
    of the emission parameters, on key instances that are too large for enumeration.
  - *Codes:* agreement with exhaustive maximum-likelihood decoding on codes
    short enough to enumerate, and the block error rate reported against the
    channel where they are not (#340).
  - *Performance parity:* convergence metrics (ΔlnL or ΔE) must match or exceed
    exact oracles on small `n`, and on large `n` the referees this repository
    has — its own exact oracles where they still reach, and the simulated truth
    that generated the fixture, meaning the RF bound above and recovery of the
    generating parameters — under an **equal budget of objective evaluations**.
    Comparison against a state-of-the-art classical framework (e.g. IQ-TREE 2
    for trees) is deferred rather than dropped: `CLAUDE.md` admits no external
    solver today, and this requirement returns, at the same equal budget, if
    one is adopted (`docs/external_tools.md` surveys the candidates). The
    budget is counted in evaluations rather than seconds because `DEV.md`
    forbids ranking performance on CI hardware. A wall-clock comparison belongs
    on the fixed-hardware runner, reported beside the evaluation count and never
    as the gate.
- **Computational Scaling & Hardware:**
  - *Memory footprint:* bounded to 16 GB unified memory (Apple Silicon) or 24 GB VRAM (NVIDIA).
  - *Hardware dispatch:* native support for CUDA, Metal/MPS, and CPU.
  - *Numerical precision:* cross-device tensor operations evaluated against a
    declared float tolerance rather than bitwise equality. Use `float64` where
    required for recursive stability (e.g. partition functions, pruning),
    falling back to `float32` for Metal compatibility.


## Stage 1: Mathematical Foundations & Baseline Infrastructure

Establish the simulations, algorithms, and optimization backends for all
problems / fixtures for a range of sizes determined by necessity and runtime
budget.

- **Milestone 1.1: Simulation & Ground Truth Engine**
  - *Deliverable:* data generators for every problem.
    - *C(\theta):*
    - *GMMs:*
    - *HMMs:* hidden state paths and emitted observation sequences.
    - *Phylogenetics:* `k`-state evolutionary models on simulated topologies.
    - *Potts models:* ND lattices and Markov random fields (MRFs) with
      specified coupling constants and external fields.
    - *Codes:* Gallager's regular parity-check ensemble and the bicycle
      construction from a circulant, the binary symmetric, erasure and
      Gaussian channels behind one log-likelihood interface, and an encoder
      where elimination is affordable.
    - *Canonical cases:* instances whose answer is known externally, a closed form,
      a published result.
  - *Validation:* tests against generated simulations, analytic, canonical cases.
- **Milestone 1.2: Differentiable Likelihood & Energy Engine**
  - *Deliverable:* high-performance evaluators implemented in
    PyTorch/Triton/JAX (GPU) and Numpy/njit/Rust (CPU), e.g.
    - *Phylogenetics:* Felsenstein's pruning algorithm.
    - *Potts models:* belief propagation and transfer matrix methods.
    - *HMMs:* the forward-backward algorithm.
    - *Codes:* log-domain sum-product and min-sum on the Tanner graph with a
      syndrome stop, held to the general sum-product and to enumeration.
  - *Validation:* brute-force marginalization on small (`n ≤ 10`) graphs
    within the specified floating-point tolerance, with the API
    application-agnostic.
- **Milestone 1.3: Continuous Optimization via Autodiff**
  - *Deliverable:* gradient-based solvers to fit continuous parameters.
    - *Phylogenetics:* branch lengths `t`, rate matrices `Q`, root
      distributions `π`.
    - *Potts models:* coupling strengths `J`, per-site external field `h`.
    - *HMMs:* transition matrices `A`, emission matrices `B`.
  - *Deliverable:* posterior sampling over the same interface, so an interval
    can be a quantile of the posterior rather than the curvature at the mode,
    and the two can be compared.
  - *Deliverable:* starts read from the data rather than from the objective,
    where a method of moments supplies one — pairwise distances and a joining
    for a tree, a spectral inversion where the model admits it, a seeding for
    a mixture — each carrying the guarantee it comes with rather than a claim
    about the optimum it leads to.
  - *Validation:* autodiff gradients against central finite differences. A
    sampler is validated where it is exact before where it is statistical —
    integrator reversibility and its order of accuracy — then against a target
    whose normalizer is known by quadrature. A start is validated against the
    structure it claims to recover exactly — the path lengths of a known tree,
    the cost of a known clustering — and then against the objective's own start
    at an equal budget of evaluations, reported whichever way it falls.
- **Milestone 1.4: Discrete Move Sets & Classical Baselines**
  - *Deliverable:* strict structural neighborhoods for classical sampling.
    - *Phylogenetics:* nearest-neighbor interchange (NNI) and subtree
      prune-and-regraft (SPR).
    - *Potts models:* Swendsen-Wang and Wolff cluster update algorithms.
    - *HMMs:* Viterbi decoding and structural state-space updates (e.g.
      iterated conditional modes).
    - *Exact baselines:* where a discrete optimum is computable in polynomial
      time, compute it — a minimum cut for the two-state submodular Ising
      ground state, and alpha expansion above two states, with its proved
      approximation bound.
    - *The NP-hard side of the same model:* Max-Cut, with a semidefinite
      relaxation and the certificate it yields, so the boundary between what
      is solved exactly and what is only bounded is drawn rather than
      assumed.
- **Milestone 1.5: Continous samplers / HMC / Parallel tempering.

## Stage 2: Complete support for Reinforcement Learning & Variational Search methods

Replace hand-designed search heuristics with classical learned proposal policies and
exact or (differentiable) surrogates, e.g. neural networks.

- **Milestone 2.0: RL definition**
  A comprehensive formulation of RL in the textbook, defining classical methods and
  a discussion of their suitability for the supported problems.

- **Milestone 2.1: RL Agent Formulation & Deployment**
  - *Deliverable:* define the MDPs across all problems.
    - *State:* the current discrete structure (topology, lattice
      configuration, or state path), its fitted continuous parameters, and
      observation summaries.
    - *Action:* valid structural transformations from the classical
      neighborhoods (e.g. NNI/SPR, cluster flips, path mutations).
    - *Reward:* TBC. E.g. improvement in the objective function, ΔlnL.
  - *Validation:* train a policy that strictly outperforms classical baselines
    on held-out simulated validation sets under a similar evaluation budget.
- **Milestone 2.2: Curriculum Learning**
  - *Deliverable:* a progressive training regimen, since RL policies collapse
    when exposed to combinatorial spaces zero-shot.
- **Milestone 2.3: Empirical Validation & Benchmarking**
  - *Deliverable:* the RL agents benchmarked on high-dimensional simulated
    datasets past the size enumeration reaches.
  - *Validation:* compare convergence speed and final objectives against the
    classical domain heuristics this repository implements — large parsimony
    under NNI and SPR, and hill climbing — since no external solver is admitted
    (`docs/external_tools.md`).


- **Milestone 3.1: model surrogates and bounds for supported problems**
- **Neural Surrogate Modeling:** train graph neural networks (GNNs) or
  transformers to approximate the Felsenstein likelihood, Potts energy, or HMM
  likelihood. The RL agent queries the surrogate 10,000× faster to filter
  proposal batches, computing the exact evaluation only on the top-`K`
  candidates.
  - *Landed:* as certified analytic bounds plus learned predictors on the gap
    above them, ranking a neighbourhood for exact re-scoring of the top-`K`
    (#317); the filter's cost ratio at large `n` is unmeasured.

- TODO bounds.


- **Milestone 4.1: Experiment Tracking, Ablations & Leaderboard**
  - *Deliverable:* a localized tracking manifest (e.g. Aim) logging git
    commits, objective traces, compute budgets, and QA figures.
  - *Validation:* an ablation leaderboard ranking algorithmic variants by
    budget-matched metrics across shared random seeds, with statistical
    significance by paired test before a new state-of-the-art is adopted. The
    leaderboard is the generated index of `docs/experiments/` (#314), one file
    per experiment against its commit.


## Stage 5: Research Extensions (Blue Sky)

Architectural extensions that amortize the cost of discrete structural search
and reduce the number of exact evaluations.

- **Differentiable Topology Search:** formulate continuous relaxations of the
  discrete graph spaces, e.g. the tropical Grassmannian (for phylogenetic trees)
  or Gumbel-softmax relaxations (for HMM/Potts/HMM) to enable gradient-based optimization.
  - *Validation:* the relaxation must reduce to the discrete objective exactly
    at the corners of the simplex; the gradient estimator's bias and variance
    are measured against the exact gradient rather than assumed small; and any
    claim to beat a classical baseline needs the budget-matched paired test
    §4.1 requires.
  - *Landed:* the Gumbel-softmax half (#225); the tropical Grassmannian half
    as a softmin over quartet resolutions in tree-metric coordinates, refereed
    by enumeration at 5 to 8 taxa and by the Hadamard closed form (#408). It
    does not beat the bounded-radius search: on every fixture where both are
    affordable neighbor joining reaches the same optimum at no gradient steps,
    so no budget-matched claim is made. The module is conserved as
    `snakes_and_ladders.sandbox.tropical` on that measurement, with the tests
    and the figure that referee it.

- **Learned Compound Moves:** replace single atomic actions with extended macro-actions,
  e.g. whose number is sampled via a Dirichlet process, for efficiency and to tunnel.
  - *Landed:* nothing learned (#147); an exact block move over a chain-shaped
    subset exists (#310).
- Attention as applied to Newick strings.
- *HMMs:* spectral initialization against the k-means++ start; variational
  EM against Baum-Welch at equal evaluations.
- *LDPC (#340):* neural belief propagation with learned message weights,
  refereed by maximum-likelihood decoding below 24 bits and the
  density-evolution threshold above.