# ROADMAP: Mixed Discrete-Continuous Optimization for Graph-Structured Models

## 0. The Development Loop

Development is agent-assisted. The claim the loop supports is not that an agent
wrote the code; it is that the process establishes whether the code is right.
Each stage below is a gate, and the ordering is deliberate: an approach is
rejected before it is written, a claim is pinned before it is published.
`CLAUDE.md` is the contract every stage is read against, and it is
authoritative — where it and any other document disagree, it wins.

### 0.1 The Ticket

- **Deliverable:** a unit of work filed through
  `.github/ISSUE_TEMPLATE/task.yml`, naming the desired outcome stated so it
  can be checked, what it unblocks, the non-goals, the submodule it lands in,
  and — the field that does the work — how it will be validated. Blank issues
  are disabled.
- **Gate:** a ticket whose validation field names only output shapes, or only
  that execution did not raise, is rejected at filing rather than at review.
  Priority (`high`, `medium`, `low`) and submodule labels come from
  `.github/labels.yml` and are applied by a workflow, so the taxonomy cannot
  drift from the documentation that describes it.

### 0.2 The Plan

- **Deliverable:** a plan posted to the ticket thread before any code exists,
  at which point the issue is labelled `planned`. Review of an approach costs
  a comment; review of an implementation costs the implementation.

  A plan is 2–5 steps, or more where the work needs them and the plan says
  why, each stating how it will be validated. It ends with an
  **Open Questions** section carrying every question on the desired
  behaviour, so a reviewer finds them in one place rather than reading prose
  for them; a plan with nothing outstanding says so under that heading rather
  than omitting it. Being written for a reviewer whose time and context are
  finite, a plan is subject to the Writing Style in `CLAUDE.md` like anything
  else here.
- **Gate:** a maintainer applies `approved`, and only then may a pull request
  open. The pull request must implement the plan already in the thread. A plan
  that turns out to be flawed gets a revised plan posted to the same thread,
  not a silent correction in the diff. Disjoint tickets run as parallel
  worktrees and parallel pull requests; coupled changes run as a single
  sequential chain.
- **Record the branch before the first commit.** Once a branch is created for
  an approved plan, the first thing posted is a single issue comment naming
  the branch (and, once opened, the PR number) — before any further commit is
  pushed, so an interrupted or deferred session leaves a ticket that already
  points at the in-flight branch.

### 0.3 The Pull Request

- **Deliverable:** a change answering the fixed checklist in
  `.github/pull_request_template.md`: the Definition of Done, baseline and new
  benchmark numbers for every hot path it touches, the *realized* value of
  every tolerance-based test beside the tolerance it was checked against, the
  documents the change made untrue, and anything deferred with the tracking
  issue that carries it.
- **Gate:** nine required checks — `ruff`, `mypy --strict`, `clippy`,
  `cargo fmt`, the Rust suite, the Python suite under its coverage floor, the
  Sphinx build with warnings as errors, the technical-document build, the
  re-execution of every committed notebook, and the dependency audits. Documentation Sync is part of the diff, not a follow-up:
  a change that makes `README.md`, `DEV.md`, `INSTALL.md`, a `CLAUDE.md`,
  `STATUS.md`, `ROADMAP.md` or `docs/tex/` untrue corrects it in the same pull
  request, and adds a `changelog.d/` fragment if it is user-visible.

### 0.4 Validation

- **Deliverable:** for every claim, an oracle that does not share code with
  the thing it checks — an analytic result, a brute-force marginalization, an
  exhaustive enumeration, or an independently implemented algorithm — and a
  regression test pinning the claim to it within a stated tolerance.
- **Gate:** three rules constrain what the suite may contain. Coverage
  theatre is forbidden: a test asserting only shapes, or only that nothing
  raised, does not count, and a gap is left unwritten and tracked as an issue
  instead. Every accelerated path keeps its vectorized NumPy implementation as
  the oracle the Rust, PyTorch and GPU backends are pinned against — deleting
  the slow path removes the only thing that says the fast path is right.
  Correctness comes from an independent source rather than from a second
  backend, and agreement across devices and precisions is a declared relative
  tolerance keyed on the lowest precision in the comparison, never bitwise
  equality.

### 0.5 The Record

- **Deliverable:** the technical document (`docs/tex/`, contents specified in
  §1.3) is the record a validated claim is written into. Every figure and
  table in it is rendered by `phylo.qa` from the code it reports on, carries a
  caption naming the seed, the sizes and the model that produced it, and is
  committed so a changed plot is visible in review rather than after a build.
  `CHANGELOG.md` is assembled by `towncrier` from per-pull-request fragments;
  `STATUS.md` states what has landed against each milestone below, with the
  pull request that carries it; `TICKETS.md` states, as titles, what has not.
- **Gate:** continuous integration regenerates the figures and rebuilds the
  document, and fails a pull request whose rebuilt PDF differs from the
  committed one, so a figure cannot drift from the code that produced it. A
  generated caption may report only quantities continuous in their inputs — a
  discontinuous statistic breaks that build and was never a measurement. A
  release is itself a ticket, gated on `infra/release.sh` passing before a
  version is tagged.

## 1. Project Objectives & Specifications

### 1.1 Core Objective

Develop and deploy modern solvers for mixed discrete-continuous optimization
across three primary classes of graphical models: phylogenetic trees (the large
parsimony problem), N-dimensional Potts models in an external field, and hidden
Markov models (HMMs). The framework integrates automatic differentiation for
continuous parameters with reinforcement learning (RL) to learn proposal
policies that score discrete structural candidates using exact, approximate, or
bounded likelihoods/energies.

### 1.2 Technical Requirements

- **Accuracy & Validation:**
  - *Phylogenetics:* normalized Robinson–Foulds (RF) distance ≤0.05 against
    simulated ground-truth topologies.
  - *Potts/HMMs:* recovery of true coupling/transition parameters within 95%
    confidence intervals; precise state-sequence decoding.
  - *Performance parity:* convergence metrics (ΔlnL or ΔE) must match or exceed
    exact oracles on small `n`, and state-of-the-art classical frameworks
    (e.g. IQ-TREE 2 for trees) on large `n` under equivalent wall-clock time.
- **Computational Scaling & Hardware:**
  - *Memory footprint:* bounded to `O(n×L×k)` — `n` nodes/taxa, `L`
    sequence/chain length, `k` alphabet/state size — strictly fitting within
    16 GB unified memory (Apple Silicon) or 24 GB VRAM (NVIDIA).
  - *Hardware dispatch:* native support for CUDA, Metal/MPS, and CPU (for
    CI/CD).
  - *Numerical precision:* cross-device tensor operations evaluated against a
    declared float tolerance rather than bitwise equality. Use `float64` where
    required for recursive stability (e.g. partition functions, pruning),
    falling back to `float32` for Metal compatibility.

### 1.3 The Technical Document

The `docs/tex/` directory serves as the authoritative, version-controlled
mathematical record of the project. The application logic is strictly bound to
this LaTeX documentation. It must contain:

- Complete mathematical formulations of all substitution models, energy
  landscapes, and transition probabilities across the three problem classes.
- Rigorous derivations of the exact inference algorithms (Felsenstein's
  pruning, belief propagation, forward-backward).
- Formal definitions of the Markov decision process (MDP) formulations for the
  RL agents.
- Proofs for any proposed upper/lower bounds used for branch-and-bound pruning.
- Dynamically generated QA figures and tables (parameter recovery, convergence
  curves) sourced directly from tracked CI runs to guarantee empirical
  reproducibility.

## Stage 1: Mathematical Foundations & Baseline Infrastructure

Establish the simulation, exact inference, and optimization backends for the
three distinct problem classes. Target scales span `n ∈ [10, 1000]`
nodes/taxa/states, with sequence/lattice lengths `L ∈ [100, 11000]`.

- **Milestone 1.1: Simulation & Ground Truth Engine**
  - *Deliverable:* data generators for all problem classes.
    - *Phylogenetics:* `k`-state evolutionary models (Jukes-Cantor, GTR) on
      simulated topologies.
    - *Potts models:* N-D lattices and Markov random fields (MRFs) with
      specified coupling constants and external fields.
    - *HMMs:* hidden state paths and emitted observation sequences.
    - *Canonical cases:* instances whose answer is known from outside this
      repository — a closed form, a published result, or an enumeration
      sharing no code with what it tests — admitted only when more than one
      module consumes them.
  - *Validation:* verify generated sequence/spin distributions against
    analytic, closed-form transition probabilities and partition functions.
    A canonical case is validated against the outside answer it was admitted
    for, never against a run of the method it is meant to referee.
- **Milestone 1.2: Differentiable Likelihood & Energy Engine**
  - *Deliverable:* high-performance evaluators implemented in
    PyTorch/Triton/JAX (GPU) and Rust (CPU).
    - *Phylogenetics:* Felsenstein's pruning algorithm.
    - *Potts models:* belief propagation and transfer matrix methods.
    - *HMMs:* the forward-backward algorithm.
  - *Validation:* match brute-force marginalization on small (`n ≤ 10`) graphs
    within the specified floating-point tolerance. Ensure the API remains
    application-agnostic.
- **Milestone 1.3: Continuous Optimization via Autodiff**
  - *Deliverable:* gradient-based solvers to fit continuous parameters.
    - *Phylogenetics:* branch lengths `t`, rate matrices `Q`, root
      distributions `π`.
    - *Potts models:* coupling strengths `J`, external fields `h`.
    - *HMMs:* transition matrices `A`, emission matrices `B`.
  - *Deliverable:* posterior sampling over the same interface, so an interval
    can be a quantile of the posterior rather than the curvature at the mode,
    and the two can be compared.
  - *Validation:* validate autodiff gradients against central finite
    differences. Validate a sampler where it is exact before where it is
    statistical — integrator reversibility and its order of accuracy — then
    against a target whose normalizer is known by quadrature.
- **Milestone 1.4: Discrete Move Sets & Classical Baselines**
  - *Deliverable:* implement strict structural neighborhoods for classical
    sampling.
    - *Phylogenetics:* nearest-neighbor interchange (NNI) and subtree
      prune-and-regraft (SPR).
    - *Potts models:* Swendsen-Wang and Wolff cluster update algorithms.
    - *HMMs:* Viterbi decoding and structural state-space updates (e.g.
      iterated conditional modes).
    - *Exact baselines:* where a discrete optimum is computable in polynomial
      time, compute it — a minimum cut for the two-state submodular Ising
      ground state, and alpha expansion above two states, with its proved
      approximation bound. A heuristic past the size enumeration reaches has
      otherwise nothing to be checked against.

## Stage 2: Reinforcement Learning & Variational Search

Replace fixed, hand-designed search heuristics with learned proposal policies
parameterized by neural networks.

- **Milestone 2.1: RL Agent Formulation & Deployment**
  - *Deliverable:* define the MDPs across all three problem classes.
    - *State:* the current discrete structure (topology, lattice
      configuration, or state path), its fitted continuous parameters, and
      observation summaries.
    - *Action:* valid structural transformations from the classical
      neighborhoods (e.g. NNI/SPR, cluster flips, path mutations).
    - *Reward:* improvement in the objective function (ΔlnL for
      phylogenetics/HMMs, ΔE for Potts models).
  - *Validation:* train a policy that strictly outperforms classical baselines
    on held-out simulated validation sets under a fixed evaluation budget.
- **Milestone 2.2: Curriculum Learning**
  - *Deliverable:* implement a progressive training regimen. RL policies
    frequently collapse when exposed to massive combinatorial spaces
    zero-shot. The agent must train on `n = 10` nodes/taxa, transferring
    weights and progressively scaling to fine-tune on `n = 50`, `n = 200`, and
    `n = 1000` environments.
- **Milestone 2.3: Empirical Validation & Benchmarking**
  - *Deliverable:* benchmark the RL agents on high-dimensional, empirical
    datasets.
  - *Validation:* compare convergence speed and final objectives against
    state-of-the-art domain heuristics.
- **Milestone 2.4: Experiment Tracking, Ablations & Leaderboard**
  - *Deliverable:* deploy a localized tracking manifest (e.g. Aim) logging git
    commits, objective traces, compute budgets, and QA figures.
  - *Validation:* maintain an ablation leaderboard ranking algorithmic variants
    using budget-matched metrics across shared random seeds. Require
    statistical significance via paired tests before adopting a new
    state-of-the-art.

## Stage 3: Research Extensions (Blue Sky)

Advanced architectural extensions aimed at aggressively amortizing the cost of
discrete structural search and minimizing expensive exact evaluations.

- **Differentiable Topology Search:** formulate continuous relaxations of the
  discrete graph spaces. Utilize representations like the tropical Grassmannian
  (for phylogenetic trees) or Gumbel-softmax relaxations (for Potts/HMM
  discrete states) to enable end-to-end, gradient-based optimization of the
  discrete structure alongside continuous parameters, bypassing discrete RL
  moves entirely.
  - *The two halves differ in what can referee them, and that decides the
    order.* Potts configurations and HMM state paths are enumerable, so the
    exact optimum, the exact expected score and the exact gradient are all
    computable and "does the relaxation find what discrete search finds" is
    falsifiable. Tree topologies at any interesting size are not. The
    Gumbel-softmax half is therefore built first and the tropical Grassmannian
    waits on an oracle rather than on effort.
  - *Validation:* the relaxation must reduce to the discrete objective exactly
    at the corners of the simplex; the gradient estimator's bias and variance
    are measured against the exact gradient rather than assumed small; and any
    claim to beat a classical baseline needs the budget-matched paired test
    §2.4 requires.
- **Neural Surrogate Modeling:** train lightweight graph neural networks (GNNs)
  or transformers to directly approximate the Felsenstein likelihood, Potts
  energy, or HMM likelihood. The RL agent queries the surrogate 10,000× faster
  to filter massive proposal batches, calculating the exact, expensive
  evaluation only on the top-`K` highest-probability candidates.
- **Learned Compound Moves:** replace single atomic actions (e.g. one SPR move,
  one cluster flip) with temporally extended macro-actions, sampled dynamically
  via a Dirichlet process, to efficiently tunnel through local optima.
- **Transformer Policy over Canonical Encodings:** serialize discrete graphs
  (e.g. canonical Newick strings for trees, canonical adjacency sequences for
  lattices) into tokenized sequences. Train an autoregressive or policy-gradient
  transformer model directly on the structural sequence.
- **Stochastic Escape Mechanisms:** implement Metropolis-Hastings
  accepted-worsening steps or ratchet-style site reweighting (e.g. simulated
  annealing) to force the RL agent out of suboptimal valleys in heavily ridged
  landscapes.
