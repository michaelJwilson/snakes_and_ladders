# Blue sky: Stage 5 candidates

What `ROADMAP.md` Stage 5 has proposed so far, the state of each proposal, and
the candidates that would follow — each stated with the oracle that referees
it and the tier it runs at (`DEV.md`: CI is the size an exact oracle still
holds at, stress the same claim at a size CI cannot hold, release unbounded).
A candidate here is sized to become one `task.yml` filing; none is filed by
this file, and the issue named beside each row is the one that carries it. The sources each candidate
rests on are in `CLAUDE.md`'s routing table and the textbook's Reference
Taxonomy appendix. Issue #360.

The rule the roadmap sets and this file keeps: a proposal without an oracle is
blocked on the oracle, not on effort. Every candidate below names one, and the
three that have none at the size that matters say so.

## 1. Prior proposals and their state

| Source | Proposal | State | Evidence |
| --- | --- | --- | --- |
| Roadmap Stage 3, differentiable topology | Gumbel-softmax relaxation of Potts and HMM states | Landed (#225) | Relaxation exact at every corner to `1e-11`; deterministic ascent 18/40 against greedy's 5/40, McNemar `p = 0.00098`; the relaxation on the Potts lattice and joint structure-plus-parameter optimization not built (`STATUS.md` Stage 3) |
| Roadmap Stage 3, differentiable topology | Tropical Grassmannian relaxation over trees | Landed (#408); not shown to beat a classical baseline | Enumeration at 5 to 8 taxa and the Hadamard closed form referee it; neighbor joining reaches the same optimum at no gradient steps on every fixture measured (`STATUS.md` Stage 3) |
| Roadmap Stage 3, neural surrogates | Networks approximating the likelihood or energy, exact re-scoring of the top-`K` | Landed as certified bounds plus learned predictors on the gap (#317, closing #308) | Plug-in lower and parsimony upper bounds with 0/15 violations; surrogate-ranked SPR search reaches the full search's optimum 4/4 at 277 likelihood evaluations against 20,718; learned gap R² 0.98 held out at eight taxa. The `10,000×` filter at large `n` is unmeasured |
| Roadmap Stage 3, compound moves | Macro-actions sampled dynamically | Not started (#147) | #310's chain block move is an exact macro-move with no learning; multi-SPR neighbourhoods are #329's third item |
| Roadmap Stage 3, transformer policy | Autoregressive or policy-gradient model over canonical encodings | Not started | #317's per-branch and per-node tokens and one-block attention model are the encoding half, fitted as a surrogate rather than trained as a policy; #114 gave topologies a canonical key |
| Roadmap Stage 3, stochastic escape | Accepted-worsening steps, annealing, ratchet reweighting | Landed in three parts (#267, #310, #352) | Declared schedules, annealing and parallel tempering with the swap kernel pinned against enumeration (#267); one Gibbs sampler and annealer over the factor graph, and Metropolis over topologies reaching the enumerated best 6/6 (#310); adaptive HMC and a tempering ladder from measured exchange, inside the band 20/20 seeds (#352). Epsilon-greedy escape in the tree environment raised matched-budget success 0.560 to 0.908 (`STATUS.md` 2.1). Ratchet-style site reweighting not built |
| `docs/nb/phylo_tree.ipynb` Further Work | A fixture separating a policy from greedy | Landed (#177), and the answer is a tie | Learned policy 0.485 against greedy's 0.480, sign test `p = 1.0` (`STATUS.md`, What Is Not Claimed); random-restart greedy reaches 1.000 at the same 60-decision budget |
| `docs/nb/phylo_tree.ipynb` Further Work | A richer feature set | Open (#328) | Every algorithm lands on greedy's 0.48 with one feature (`STATUS.md` 2.1) |
| `docs/nb/phylo_tree.ipynb` Further Work | Comparison against established software | Open (#126), no external tool installed | `STATUS.md` What Is Not Claimed |
| `docs/nb/phylo_tree.ipynb` Further Work | Scaling past eight taxa | Open | Enumeration is the oracle below eight taxa; a certified bound (#329) is the candidate replacement |
| `docs/nb/hmm.ipynb` Further Work | Viterbi, iterated conditional modes, forward-backward as an evaluator | Viterbi and the posteriors landed (#175, #173); iterated conditional modes open (#176) | The notebook still decodes by enumeration of `3**8` paths (#407) |
| `docs/nb/potts_chain.ipynb`, `spatio_sequential.ipynb` Further Work | The lattice pass, cluster-move autocorrelation, equal-budget solver rows | Open (#278, #281, #290) | The chain is shown; the lattice, belief propagation and cluster moves exist and are not in the notebook |
| #329 | Branch-and-bound over the certified bounds, a non-JC bound, multi-SPR | Open (#329) | Bounds certified, used only for ranking |
| #334 | The Max-Cut SDP certificate behind an approved dependency | Open (#334) | `search/max_cut.py` states its bound is not rigorous |
| #756 | NUTS | Open, conditional on #268 | #352's fixed trajectory length adapted instead |
| #227 | CUDA and Metal dispatch, a Triton or JAX kernel | Open (#227 for coloured ICM) | GPU dispatch specified and not built (`STATUS.md` What Is Not Claimed) |
| #340 | An LDPC problem class at `10,000 × 20,000` | Open, parts 2 and 3 | The 19,998-bit erasure bracket runs at the release gate; `ldpc.ipynb` stops at 996 bits and decoding as optimization is unbuilt |

## 2. Candidates

Each: what is built, the oracle, the tier, and what the roadmap item it serves.

### 2.1 Trees

- **Subsplit-network variational inference** (Zhang & Matsen 2018, 2019) as
  the differentiable-topology route that has an oracle. A subsplit Bayesian
  network parameterizes a distribution over topologies; fitted variationally
  to the fitted likelihoods, its mode is the search's answer and its
  distribution the support #270 estimates. *Oracle:* the flat-prior weight
  over fitted likelihoods, enumerated at 5 to 8 taxa (15 to 10,395
  topologies), the quantity #310's topology move is already pinned against.
  *Tier:* CI at 5 and 6 taxa; stress at 8. *Serves:* a second
  differentiable-topology route beside the tropical Grassmannian one #408
  landed, which parameterizes a point rather than a distribution.
- **A graph-network policy over the tree**, #317's graph model as the encoder
  and #328's features as its input, trained by #313's PPO. *Oracle:* the
  enumerated maximum on #177's 7-taxon fixture, where greedy reaches it from
  0.48 of 50 starts and the linear policy from 0.485; the paired test §2.4
  requires over 16 training seeds; and the random-restart baseline of 1.000 at
  60 decisions, reported beside. *Tier:* CI. *Serves:* Milestone 2.1's
  validation and the transformer-policy item, whose encoder this is.
- **Metropolis-coupled MCMC over topologies** (Altekar et al. 2004): replicas
  of #310's topology move on #352's adapted ladder, exchanging as #267's
  Potts replicas do. *Oracle:* at `T = 1` the cold chain samples the
  enumerated flat-prior weight (the chi-square pin #310 carries); the swap
  kernel's stationary distribution against enumeration (#267's third
  check); fits to the enumerated best against #310's annealed move at equal
  fits. *Tier:* CI at 5 taxa and 300 sites; stress at 8. *Serves:* stochastic
  escape's unbuilt half on trees.
- **The external-baseline comparison at 50 to 200 taxa** (Minh et al. 2020;
  Kozlov et al. 2019). *Oracle:* none at that size; the referee is simulated
  truth — RF `≤ 0.05` per §1.2 — under an equal budget of likelihood
  evaluations. *Tier:* release. *Blocked:* on #126's dependency decision, which
  currently admits no external tool.

- **Sequential Monte Carlo over topologies**, with #310's topology move as
  the mutation kernel and the fitted likelihood as the weight. A particle set
  carries topologies rather than a single state, so the estimate is a
  distribution and the normalizing constant comes with it. *Oracle:* the
  enumerated flat-prior weight at 5 to 8 taxa --- the same quantity the
  subsplit-network candidate and #310's move are pinned against --- and the
  marginal likelihood against that enumeration, which nothing in the tree
  estimates for trees today. *Tier:* CI at 5 and 6 taxa; stress at 8.
  *Serves:* #756's blue-sky line, and the support estimate #270 asks for,
  from the particle weights rather than from a bootstrap.

### 2.2 Potts models and Markov random fields

- **Tree-reweighted max-product** (Wainwright, Jaakkola & Willsky 2005) as a
  certified upper bound on the MAP value, tighter than the bracket
  `ground_state_energy_bounds` derives from the spanning-tree bound on
  `log Z`. *Oracle:* `bound.certify` against the enumerated ground state at
  3-state 3x3 and 2-state 4x4; the minimum cut for the two-state
  ferromagnet at any size; the closed-form triangular-antiferromagnet ground
  state at 9x9. *Tier:* CI. *Serves:* neural surrogates' certified half, and
  #329's branch-and-bound on lattices.
- **Population annealing** (Machta 2010) beside parallel tempering.
  *Oracle:* the closed-form ground state on the 9x9 periodic triangular
  antiferromagnet #352 tuned its ladder on, 20 seeds; at 32x32 the referee is
  agreement with parallel tempering's best energy at equal sweeps, each
  reported. *Tier:* CI at 9x9; stress at 32x32. *Serves:* stochastic escape,
  and the leaderboard's population row.
- **A replication of the physics-inspired GNN ground-state result** (Schuetz,
  Brubaker & Katzgraber 2022) against its greedy critique (Angelini &
  Ricci-Tersenghi 2023), on Max-Cut, recorded as a `docs/experiments/` file
  rather than a feature. *Oracle:* enumeration at `n ≤ 12` (#334's size); the
  Goemans–Williamson certificate ratio `≥ 0.878` above it; single-flip
  greedy from `search` as the baseline it is measured against. *Tier:* CI at
  `n ≤ 12`; release above. *Expected:* negative, which is the result the
  roadmap's scepticism of surrogates asks to have in writing.
- **Boundary matrix-product contraction** (Schollwöck 2011) as an exact
  oracle past the width the transfer matrix holds (`k**W` states). *Oracle:*
  the transfer matrix itself at every width it holds, to `1e-10` relative,
  and convergence in the bond dimension beyond, reported as a curve. *Tier:*
  CI at width `≤ 6`; stress at 32x32. *Serves:* every lattice claim above
  enumeration, including the Gumbel-softmax relaxation on the lattice
  `STATUS.md` records as unmeasured.

### 2.3 Hidden Markov models

- **Spectral initialization** (Hsu, Kakade & Zhang 2012) as the start with a
  guarantee, against the k-means++ start #262 measured. *Oracle:* recovery of
  `(A, B)` up to permutation within the 95% intervals §1.2 requires, and the
  log-likelihood reached at equal Baum-Welch iterations, checked against the
  enumerated likelihood at `L ≤ 10`. *Tier:* CI. *Serves:* Milestone 1.3's
  initializer abstraction (#251).
- **Variational EM against Baum-Welch at equal evaluations.** *Oracle:* the
  variational bound never exceeds the exact log-likelihood from the forward
  recursion, rises monotonically per iteration, and the gap at convergence is
  reported beside the parameters recovered. *Tier:* CI.

### 2.4 Low-density parity-check codes

- **Neural belief propagation** (Nachmani et al. 2018) with learned message
  weights over the Tanner graph of #340's code, the one class where a learned
  proposal has an exact oracle at every size it trains at. *Oracle:* maximum-
  likelihood decoding by enumeration below 24 bits; above it the
  density-evolution threshold of the regular (3, 6) ensemble (Gallager 1962),
  which plain belief propagation must reach and the learned decoder must not
  fall below. *Tier:* CI below 24 bits; release at `10,000 × 20,000`.
  *Serves:* neural surrogates, with a referee at every size.

### 2.5 Cross-cutting

- **Macro-actions as options** (Sutton, Precup & Singh 1999; Bacon, Harb &
  Precup 2017) for the compound-moves item: an option is a sequence of atomic
  moves with a termination condition, learned by option-critic. *Oracle:*
  `learn.exact`'s enumerated expected return and optimal value, re-pinned for
  the extended action set on the 81-configuration chain; the antiferromagnetic
  chain of #225's comparison, whose optimum needs coordinated flips (greedy
  5/40). *Tier:* CI.
- **A learned dynamics model for the planner** (Schrittwieser et al. 2020):
  #313's PUCT search over predicted rewards and features rather than the
  environment. *Oracle:* the true-model planner's visit distribution on every
  chain state, and the argmax-of-`Q*` pin it already passes with the exact
  value as leaf. *Tier:* CI.
- **CMA-ES and a genetic algorithm as population baselines** (Hansen &
  Ostermeier 2001) the leaderboard lacks: CMA-ES against the gradient fit on
  branch lengths and `(J, h)`; a genetic algorithm over topologies and
  configurations against random-restart greedy. *Oracle:* the same maximum
  and the enumerated optimum, at equal objective evaluations. *Tier:* CI,
  recorded under `docs/experiments/`.
- **Neural transport for the continuous posterior** (Hoffman et al. 2019): a
  normalizing flow fitted in warm-up, with HMC run in the transported space,
  which is the answer to the correlated targets a diagonal metric handles
  badly --- the case #756 reaches for a dense metric for. *Oracle:* the
  analytic Gaussian `test_opt_hmc.py` already pins, and the enumerated
  mixture posterior (#749); the flow may not move either. *Measured as:*
  effective samples per gradient against HMC and against MALA and slice
  sampling, the tuning-free baselines #756 landed, at equal gradients.
  *Tier:* CI on the Gaussian and the mixture; stress on the tree posterior,
  where the correlation is real. *Serves:* Milestone 1.5's efficiency half,
  which the ladder pins for correctness and nothing pins for cost.
- **Training across a distribution of fixtures** as the mechanism for §2.2's
  curriculum: a policy trained on seeds and sizes drawn from a declared
  distribution, evaluated on held-out fixtures at each size. *Oracle:* the
  enumerated optimum at `≤ 8` taxa; #317's zero-shot-against-transferred
  comparison (set model R² 0.68 to 0.94 at 5 to 6 taxa) as the precedent
  for the measurement. *Tier:* CI at `≤ 8` taxa; stress at 20.

## 3. Candidates proposed 2026-09-20

Ten, each with the referee that would decide it and the tier it runs at; four
are filed (#823 to #826) and the rest wait on a ticket.

| Candidate | What it buys | Referee | Tier | Ticket |
| --- | --- | --- | --- | --- |
| Simulated bifurcation for Potts ground states | A relaxation reaching the planted glass's ground state from 0.562 of seeds against descent's 0.079, on NumPy and torch | enumeration on the declared glass; `sim.potts.energy` on every labelling | ci, stress | #823, in review (#833) |
| Sequential Monte Carlo over topologies | A population over trees with resampling, where a single tempered chain round-trips slowly | enumeration at 7 taxa; `round_trips` against the tempered ensemble | ci | #824 |
| Residual belief propagation | An adaptive message order: 176 sweeps against 254 on the critical lattice, worse on the codes | the reference flooding; `strip_log_partition` | ci | #825, in review (#845) |
| Polar codes as a problem row | 683 conserved lines promoted, then CRC-aided list decoding | the reference recursion; `exact_decoding` at `N = 16` | ci, stress | #826 |
| Tensor-network contraction for lattice `log Z` | An exact-to-tolerance `log Z` on lattices past the strip's width, by boundary MPS with a stated bond dimension | `strip_log_partition` where it reaches; the bond-dimension sweep's convergence where it does not | stress | none |
| Belief-propagation-guided decimation for the frustrated lattice | A ground-state route that fixes the surest spin and re-runs, against alpha-expansion and the max-cut SDP | enumeration on the planted glass; `minimum_frustrated_edges` on the triangular lattice | ci | none |
| A learned temperature ladder | The rung placement `adapt_ladder` tunes by round trips, learned across instances of one family | `round_trips` and `up_fraction` at a matched sweep budget against the geometric and adapted ladders | stress | none |
| Gumbel-top-k sampling of topologies | Sampling without replacement from a policy's scores, so a neighbourhood's top-`K` is drawn once rather than re-scored | the exact expectation from `enumerate_topologies` at 6 taxa | ci | none |
| CRC-aided list decoding on the turbo code | The polar step's CRC, applied where a decoder already exists | `exact_decoding` on the caterpillar code; the turbo waterfall's committed error counts | ci, release | none |
| Coloured Gibbs sweeps on the lattice | A two-colour update that vectorises every odd then every even site, the shape a GPU kernel wants, before any device is targeted | `sample_potts` single-site chains: same law by the goodness-of-fit test, same autocorrelation time | ci, stress | none |

