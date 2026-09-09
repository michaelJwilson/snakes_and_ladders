---
id: 007
date: 2026-09-09
commit: 6790315257458a3b772a879374eb1196304ea15c
branch: claude/likelihood-443-profile
pr: 0
tickets: [443, 436, 341, 232]
problem: tree
fixture: tests/regression/fixtures/tree_jc/, swept around the declared instance --- caterpillar topologies at 4 to 20 taxa (branch lengths uniform on [0.05, 0.3], seed 0) and random topologies from `search.topology.random_topology`, alignments simulated at seed 1
size: release
methods: [eager, torch-compile, rust-binding]
budget: 0
seeds: [0, 1]
hardware: Linux-x86_64
status: confirmed
---

# What does one branch-length fit cost, and which of the three candidate ports would change it?

## Feature under test

Issue #443 proposes three changes against the per-fit cost and states a
mechanism for each. Each is falsifiable here, before any of them is written:

1. `torch.compile` on `pruning_torch.log_likelihood` closes the gap on its
   own, at a compile cost a search of 300 candidates can amortize.
2. The autograd graph carries **one node per tree node**, so the backward
   cost tracks the tree and level-synchronous batching would collapse it.
3. The L-BFGS step is under #341's 10% bar at the sizes this ticket cares
   about, not only at the two already measured.
4. The FFI boundary is most of a through-binding call, so the licensed Rust
   change is **crossing it less often** rather than a faster kernel.

## Setup

Caterpillar topologies at 4, 6, 8, 12, 16 and 20 taxa, branch lengths uniform
on [0.05, 0.3] under `default_rng(0)`, alignments simulated at 1,000 sites
under `default_rng(1)` --- the fixture shape `tests/benchmarks/profile_hotpaths.py`
already profiles, so the numbers are comparable with #341's and #436's. The
level counts and the compile-reuse test also use topologies drawn from
`search.topology.random_topology`, because a search's candidates differ in
shape and a caterpillar at different branch lengths would answer a different
question.

Four workloads:

- **one fit**: `opt.fit.fit(BranchLengthObjective(...), max_iterations=500)`,
  every run converged;
- **one SPR search**: `search.infer(moves=MoveSet.SPR, max_evaluations=20)`
  under `default_rng(4)`, the workload #436 profiled;
- **one forward, and one forward with its backward**, through
  `pruning_torch.log_likelihood`, beside the NumPy oracle and the Rust
  backend on the same instance;
- **one through-binding call** to `oxi_snakes_and_ladders.pruning_log_likelihood`,
  split into the wrapper's argument marshalling, the binding call, and the
  return.

Self-time fractions come from `infra/profile_harness.self_time_table`, the
repository's harness, medians over 5 profiled runs. Wall clocks are medians
over 15 to 25 repeats after one untimed warm-up, with the range reported.
Taken under `with_lock measure` on the 4-core reference host
(Linux-x86_64, Intel Xeon at 2.80 GHz, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`,
torch 2.13.0), so every number is comparable with `STATUS.md`'s.

The harness carries the self-time half and not the boundary half: `cProfile`
cannot see inside a PyO3 call, so the decomposition of (4) is `perf_counter`
around each part, and the parts are checked to sum rather than assumed to.

## Results

**(1) `torch.compile` does not close the gap.** Forward and backward,
1,000 sites, median of 15 with the range:

| taxa | eager (ms) | compiled (ms) | compiled / eager | autograd nodes, eager | compiled |
| --- | --- | --- | --- | --- | --- |
| 8 | 3.46 [3.07, 4.49] | 4.18 [3.59, 5.38] | **1.21** | 105 | 94 |
| 20 | 9.41 [8.62, 11.55] | 10.24 [8.70, 12.48] | **1.09** | 273 | 262 |

Whole fits agree: 8 taxa 0.104--0.113 s eager against 0.117--0.184 s
compiled; 20 taxa 0.505--0.584 s against 0.494--0.560 s. Dynamo compiles 4
frames with **0 graph breaks** and the value agrees with eager to 1.8e-16
relative, so this is a compiled recursion and not a silent fallback --- it is
simply not faster. The compile costs **42.8 s** on a cold inductor cache and
**3.7--3.8 s** in a later process on the warm one, and it **is** amortized:
12 distinct random topologies at 8 and at 20 taxa were served by the same 4
frames, no recompilation, the first call 3.77 s and every later one
3.5--4.6 ms. Amortization is therefore not what stops it; there is no
steady-state win to amortize.

**(2) The autograd graph carries 7 nodes per tree node, not 1, and it tracks
the tree exactly.** Nodes reachable from the loss of one
`BranchLengthObjective` forward:

| taxa | tree nodes | graph nodes | per tree node | forwards per fit | graph nodes per fit | fit (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 6 | 51 | 8.50 | 20 | 1,020 | 0.0411 [0.0389, 0.0445] |
| 6 | 10 | 79 | 7.90 | 25 | 1,975 | 0.0765 [0.0736, 0.0782] |
| 8 | 14 | 107 | 7.64 | 27 | 2,889 | 0.0998 [0.0978, 0.2394] |
| 12 | 22 | 163 | 7.41 | 40 | 6,520 | 0.2651 [0.2417, 0.2920] |
| 16 | 30 | 219 | 7.30 | 48 | 10,512 | 0.4332 [0.3823, 0.4546] |
| 20 | 38 | 275 | 7.24 | 47 | 12,925 | 0.5119 [0.4741, 0.5333] |

The count is **7 x (tree nodes) + 9** at every size, exactly. By operation at
8 taxa, where the 107 are 13 branches and 6 internal nodes: **four per
branch** (`Select` the branch's transition matrix, `Permute` it, `Mm` the
message, `Mul` it into the parent, 13 of each) and **five per internal node**
(the rescaling's `Amax`, `Where`, `Unsqueeze`, `Div` and `Log`, 6 of each),
with the remaining 25 the shared transition-matrix build, the root `Mv`, the
`log_scale` additions and the objective's own terms. So the mechanism #443
names holds and its constant is 7x what it assumed.

**(3) The L-BFGS step is under the bar at both sizes and falls with size.**
Self-time fraction, median of 5 profiled runs with the range:

| workload | taxa | autograd backward | Torch post-order | `lbfgs.py:step` | all of `lbfgs.py` |
| --- | --- | --- | --- | --- | --- |
| SPR search | 8 | 40.85 [40.64, 41.15] | 19.92 [19.80, 20.04] | **6.63** [6.59, 6.84] | 8.93 |
| SPR search | 20 | 47.26 [47.18, 47.57] | 24.05 [23.81, 24.09] | **3.57** [3.48, 3.60] | 4.84 |
| one fit | 8 | 45.13 [44.13, 46.23] | 21.14 [20.37, 21.54] | **3.18** [3.05, 3.34] | 5.70 |
| one fit | 20 | 50.30 [39.30, 50.66] | 23.70 [23.27, 40.53] | **2.08** [1.62, 2.08] | 3.16 |

The 8-taxon search row reproduces #436 (40.5% / 19.7% / 6.7%) to within 0.4
points, which is what licenses reading the 20-taxon row beside it.

**(4) The boundary is 1--5% of a through-binding call; the kernel is
90--100%.** Median of 25 with the range, `pruning_rust.log_likelihood`:

| cell | whole (ms) | argument marshalling | binding call | return | residual |
| --- | --- | --- | --- | --- | --- |
| 8 taxa x 1,000 | 0.3443 | 0.0170 (4.93%) | 0.3087 (89.68%) | 0.000098 (0.03%) | 0.0184 (5.35%) |
| 8 taxa x 10,000 | 3.8828 | 0.0433 (1.11%) | 3.7426 (96.39%) | 0.000121 (0.00%) | 0.0968 (2.49%) |
| 20 taxa x 1,000 | 0.9609 | 0.0397 (4.13%) | 0.8771 (91.28%) | 0.000123 (0.01%) | 0.0440 (4.57%) |
| 20 taxa x 10,000 | 11.6924 | 0.1674 (1.43%) | 11.6962 (100.03%) | 0.000115 (0.00%) | -0.1713 (-1.47%) |

**The three do not quite sum, and the missing term is named.** The residual
is the wrapper's own validation --- the `pi` shape check, the missing-leaf
scan over `preorder`, `check_weights`, and the weight branch --- which is
0.018--0.097 ms and independent of the site count, so it is 5% of the call at
1,000 sites and inside the spread at 10,000, where the negative residual at
the largest cell is the +-10% run-to-run spread on an 11.7 ms call rather
than a term unaccounted for.

The per-crossing floor, measured at one site so the kernel's own work is one
column:

| taxa | nodes | binding at 1 site | marshalling at 1 site | wrapper checks | total per crossing |
| --- | --- | --- | --- | --- | --- |
| 8 | 14 | 0.0032 ms | 0.0281 ms | 0.0184 ms | **0.0497 ms** |
| 20 | 38 | 0.0074 ms | 0.0341 ms | 0.0440 ms | **0.0855 ms** |

The binding's floor is 0.2 µs per node, which is PyO3 extracting `children`
and `leaf_row` from Python lists; the two NumPy arrays are borrowed and cost
nothing. Marshalling grows at 1.6e-5 ms per site (8 taxa) --- one `int64` per
observed state --- against the kernel's 1.1e-3 ms per site, a factor of 68.

**(5) What amortizing the crossings would save, calculated from (4) and (2).**
*This is arithmetic on the measurements above, not a benchmark: nothing in
the fit path crosses the boundary today, because `search.infer` and
`BranchLengthObjective` evaluate through `pruning_torch` and
`pruning_rust` is reached only by `qa.backend_agreement` and the tests.*

If the pruning recursion inside a fit were the Rust kernel, the fit would
cross once per forward pass:

| taxa | crossings per fit | per crossing | boundary tax per fit | fit | share |
| --- | --- | --- | --- | --- | --- |
| 8 | 27 | 0.0497 ms | 1.34 ms | 99.8 ms | **1.34%** |
| 20 | 47 | 0.0855 ms | 4.02 ms | 511.9 ms | **0.79%** |

One crossing per fit, or one per batch of candidates, therefore recovers at
most 1.34% and 0.79% of a fit --- under #341's 10% bar by an order of
magnitude, before any of the work of making the whole fit live behind one
call.

**What each candidate has to work with.** One forward and one backward,
1,000 sites, median of 25 with the range:

| taxa | NumPy forward | Rust forward | torch forward, no grad | torch forward, taped | backward | backward / taped forward |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 1.14 [0.97, 1.44] | 0.45 [0.37, 0.61] | 1.25 [1.14, 1.66] | 1.52 [1.35, 2.23] | 2.01 [1.83, 2.51] | **1.32** |
| 20 | 2.85 [2.51, 4.06] | 1.25 [1.15, 1.48] | 3.17 [2.85, 3.94] | 4.44 [3.50, 6.48] | 6.04 [5.38, 7.83] | **1.36** |

And how far a level-synchronous schedule could collapse the recursion, since
it batches one operation per *level* and not per node:

| taxa | tree nodes | caterpillar levels | random-topology levels, median (range of 20) | nodes / levels |
| --- | --- | --- | --- | --- |
| 8 | 14 | 7 | 5 (4--6) | 2.0 caterpillar, 2.8 random |
| 20 | 38 | 19 | 10 (7--16) | 2.0 caterpillar, 3.8 random |

## Figures

none

## Finding

`torch.compile` does not close the gap and PR 2 and PR 3 are not refused by
it: the compiled recursion is **1.21x and 1.09x slower** than eager at 8 and
20 taxa, with 0 graph breaks and a 1.8e-16 relative agreement, and the
autograd graph falls only 105 to 94 nodes. The compile is cached across 12
distinct random topologies at both sizes, so the 42.8 s cold compile is paid
once per process and is not the obstacle; there is no steady-state win to
amortize.

The graph-node count tracks the tree exactly, at **7 x (tree nodes) + 9** and
not the 1 the ticket assumed, so the diagnosis in #443's body holds in
mechanism and understates the constant by 7x. A fit builds 1,020 graph nodes
at 4 taxa and 12,925 at 20.

The L-BFGS step is **6.63%** of an 8-taxon SPR search and **3.57%** of a
20-taxon one, and 3.18% and 2.08% of a bare fit. It does not cross #341's 10%
bar at the larger size --- it falls, because the pruning recursion grows with
the tree and a step over 37 branch lengths does not. Closed.

The FFI boundary is **not** where #436's kernel win went. Argument
marshalling is 1.1--4.9% of a through-binding call and return marshalling is
0.03% or less, against a kernel that is 90--100%; the whole per-crossing
floor is 0.0497 ms at 8 taxa and 0.0855 ms at 20. Crossing once per fit
instead of once per forward pass would recover **1.34%** and **0.79%** of a
fit. The reading in #443's body --- "if the boundary is most of the call, the
licensed change is crossing it less often" --- is refuted by its own test:
the boundary is not most of the call, and this closes that half of the ticket
with a number rather than a port. It also agrees with what #436 already
reported from the other side, that the kernel is 93.2% of the call at 20 taxa
by 11,000 sites.

What the three candidates have to work with instead: the backward pass costs
**1.32--1.36x** the taped forward, and a level-synchronous schedule can
collapse the recursion by **2.0x** on a caterpillar and **2.8--3.8x** on a
random topology --- not to `O(log n)`, because a caterpillar's depth is its
node count over two and the topologies a search visits sit between the two.

## Conclusion and actions

- **#443 PR 2 (level-synchronous batching): build it.** It is the only
  candidate with a number above the bar. The per-node Python post-order is
  19.9--24.1% of a search's self time and the backward that walks the same
  graph is 40.9--47.3%; batching cuts node visits by the nodes-to-levels
  ratio, 2.0x on the caterpillar profiled and 3.8x on a random 20-taxon
  topology, so the reachable share is `(1 - levels/nodes)` of those two
  terms: **30% to 53% of a search's self time as an upper bound**, and it is
  an upper bound because the arithmetic per site is unchanged and only the
  per-node dispatch is removed. The floor is the same ratio applied to the
  post-order term alone, **10% to 18%**, which is still above the bar.
- **#443 PR 3 (the analytic gradient): do not build it as scoped.** The taped
  backward costs 1.32--1.36x the taped forward, and the two-pass analytic
  derivative is a second traversal whose cost is a forward's, so the ceiling
  is `0.34 / 2.34` of a forward-and-backward pair --- **14.5% of an
  evaluation, and 12% of a search's self time** --- against the candidate the
  ticket itself calls the one with the most to get wrong. It is also the
  wrong order: PR 2 shrinks the same taped backward, and PR 3's remaining
  ceiling after it is smaller than 12%. Re-file it against a measurement
  taken after PR 2, not before.
- **#443's Rust question: do not build it.** The boundary is 1.1--4.9% of a
  through-binding call and the whole per-crossing tax on a fit is 1.34% at 8
  taxa and 0.79% at 20, so "cross it less often" has under 1.4% to win. What
  the numbers do say is that the Rust forward is **3.4x and 3.6x** the torch
  taped forward (0.45 against 1.52 ms, 1.25 against 4.44 ms), so a port that
  paid would have to carry the *gradient* as well as the forward --- PR 3
  written in Rust rather than in torch --- and that is a different ticket
  from either the kernel #436 declined or the amortization #443 proposed.
  Neither is licensed by these numbers.

## What is not claimed

Nothing about `torch.compile` in a mode other than the default: `fullgraph`,
`mode="reduce-overhead"` and CUDA graphs were not measured, and the result
here is that the default compiles cleanly and does not pay, not that no mode
could.

Nothing about sizes past 20 taxa or 10,000 sites. The boundary decomposition
was taken at 8 and 20 taxa by 1,000 and 10,000 sites; #436's declared-scale
cells (20 and 200 taxa by 11,000 sites) are not re-measured here, and the
200-taxon row of `test_pruning_rust_bench.py` is where that claim lives.

Nothing about what PR 2 will actually realize. The 30--53% and 10--18% above
are arithmetic on the measured self-time fractions and the measured
nodes-to-levels ratio, not a benchmark of code that does not exist, and the
same reassociation of the site sums that batching introduces is a tolerance
question `likelihood/CLAUDE.md` settles rather than something these numbers
bear on.

Nothing about the identification `#443` makes between its own target and
`TICKETS.md`'s unmet line, *"the tree schedule within 2x of the forward
recursion on a chain"*. That line is `likelihood.message_passing`'s
level-ordered schedule against `opt.hmm`'s forward recursion on a chain
(`STATUS.md`, #341), a different recursion from `pruning_torch`'s post-order;
this experiment neither closes nor supersedes it.

Nothing about memory. Peak resident size was not recorded for any of the
runs above, and the level-synchronous schedule holds a level's partials live
where the post-order holds a path's.
