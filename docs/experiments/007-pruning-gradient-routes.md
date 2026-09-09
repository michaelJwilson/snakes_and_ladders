---
id: 007
date: 2026-09-09
commit: d5c6cc8e7efb1baaf8fcc26f5ac36d1b3b4168af
branch: claude/likelihood-449-autodiff
pr: 0
tickets: [449, 443]
problem: tree
fixture: tests/regression/fixtures/tree_jc/ci.yaml and tests/regression/fixtures/tree_jc/release.yaml, both at 20,000 sites
size: ci
methods: [taped, burn, analytic]
budget: 60
seeds: [20260903, 20260904]
hardware: Linux-x86_64
status: confirmed
---

# Does a Rust autodiff framework, or a closed-form backward, beat PyTorch's tape on the pruning gradient?

## Feature under test

Issue #443 measures the taped backward at 40.5% of a fit and attributes the
cost to the number of autograd graph nodes, one per tree node per operation.
Two routes claim to remove it: `burn`'s `Autodiff<NdArray<f64>>` over the
recursion rebuilt in Rust (A), and the analytic two-pass backward of
`alg:pruning-backward` behind one `torch.autograd.Function` (B). The
prediction stated in advance on issue #449 is that **A rebuilds the tape B
removes**: `burn` tapes the way PyTorch does, so A relocates the cost rather
than removing it, while B collapses the recursion to one graph node.

## Setup

Two topologies from the committed fixtures --- 4 taxa (5 branches) and 8 taxa
(14 branches) --- each simulated at 20,000 sites under Jukes-Cantor from the
fixture's own seed. One measurement is a forward pass and a `backward()`,
which is what a fitting step costs; the budget is 60 such rounds per route,
reported as the median and the inter-quartile range. Wall times taken under
`with_lock measure` on the 4-core reference host, one BLAS thread. The `burn`
kernel alone is Criterion's, 100 samples, on a balanced tree of the same taxon
count. The oracle refereeing every route is PyTorch's taped backward, which is
not deleted.

## Results

**One gradient, ms, median (IQR) over 60 rounds.**

| size | taped (oracle) | A: `burn` | B: analytic |
| --- | --- | --- | --- |
| 4 taxa, 20,000 sites | 9.82 (1.11) | 14.24 (2.57) | **5.68 (0.74)** |
| 8 taxa, 20,000 sites | 33.06 (3.56) | 46.80 (3.86) | **16.36 (1.37)** |

**One L-BFGS fit of the branch lengths, 8 taxa, 20,000 sites, ms, median (IQR)
over 11 rounds.** All three converge, and all three reach 148940.229159.

| taped | A: `burn` | B: analytic |
| --- | --- | --- |
| 694.5 (52.3) | 1179.1 (144.1) | **454.2 (38.7)** |

**Graph nodes per evaluation.** PyTorch's count is distinct `grad_fn` nodes
reachable from the value; `burn`'s is `NodeID`s allocated on its own tape,
read as the difference between two fresh ids on a single-threaded run.

| taxa | branches | taped | A: `burn` tape | A: PyTorch side | B: analytic |
| --- | --- | --- | --- | --- | --- |
| 4 | 6 | 59 | 66 | 2 | 2 |
| 8 | 14 | 115 | 134 | 2 | 2 |
| 16 | 30 | 227 | 270 | 2 | 2 |

**A, kernel alone and through the binding, ms.** Criterion reports
[lower, upper]; the Python numbers are median (IQR) over 40 rounds.

| size | kernel alone (Criterion) | the PyO3 call, from Python | through `autograd.Function` |
| --- | --- | --- | --- |
| 4 taxa, 20,000 sites | 18.00 [17.74, 18.28] | 12.45 (1.24) | 14.43 (1.66) |
| 8 taxa, 20,000 sites | 48.01 [46.94, 49.27] | 47.06 (6.09) | 52.50 (3.86) |

The 4-taxon rows are not the same tree: Criterion's balanced tree has 6
branches against the fixture's 5, which is why its kernel number is the larger
one. At 8 taxa both have 14 branches, and 47.06 (6.09) against 48.01 is inside
the spread --- the boundary costs nothing measurable here.

**The forward pass, ms, median (IQR) over 60 rounds, 8 taxa, 20,000 sites.**

| taped, graph built | the same recursion under `no_grad` | B's forward | NumPy oracle |
| --- | --- | --- | --- |
| 12.38 (1.78) | 8.10 (0.65) | 10.96 (0.96) | 14.96 (1.12) |

**Agreement.** Both routes pass `torch.autograd.gradcheck` in `float64`.
Against the taped gradient, worst relative difference over 4 and 8 taxa at
2,000 and 20,000 sites: B 2.1e-13, A 5.9e-13, both inside
`CROSS_DEVICE_RTOL_FLOAT64` = 1e-11. Against central differences, sweeping the
step over 1e-4, 1e-5, 1e-6, 1e-7: the worst deviation is 9.5e-4 at h = 1e-4 and
1.007e-6 at h = 1e-6, and **all three routes deviate by the same amount to four
significant figures at every step**, which says the deviation is the difference
quotient's and not any gradient's. `search.infer` returns the same topology,
trace, evaluations, fits and convergence flag under each route.

## Figures

none

## Finding

**The prediction held.** `burn`'s tape is 66, 134 and 270 nodes at 4, 8 and 16
taxa against PyTorch's 59, 115 and 227 --- linear in the tree either way, and
12 to 19% larger. A relocated the cost and did not remove it: 46.80 ms against
33.06 at 8 taxa, 1.42x slower per gradient and 1.70x slower per fit. The
boundary is not the reason. `burn`'s kernel *alone*, 48.01 ms, is already 1.45x
PyTorch's entire Python-level evaluation, and the crossing itself is inside the
spread; a faster binding would change nothing.

**B removes the term.** 16.36 ms against 33.06 at 8 taxa is 50.5% of an
evaluation, and 454.2 ms against 694.5 is 34.6% of a fit. Two graph nodes at
every tree size, against 227 at 16 taxa.

**Half of B's win is in the forward pass, which is why it exceeds the cap a
backward-only accounting predicts.** The taped backward is 20.68 ms of the
33.06, 1.67x its forward; removing it alone would buy 62.6%. But building the
graph costs 4.28 ms of the taped forward's 12.38 (12.38 against 8.10 under
`no_grad`), and a `torch.autograd.Function` does not build it. B pays 2.86 ms
back to keep the messages its backward reads, giving a 10.96 ms forward and a
5.40 ms backward.

`burn`'s `f64` path is sound, which was the axis the dependency was adopted on:
the tape agrees with the taped `float64` gradient to 5.9e-13, where a narrowing
to `f32` would show at 1e-7. It is not why the route lost.

## Conclusion and actions

- #449: closed by this measurement. B stays as
  `snakes_and_ladders.likelihood.pruning_analytic`, a backend beside
  `pruning_rust` with its own tests and benchmark. A and the `burn`
  dependency were removed from `Cargo.toml` in the same pull request that
  added them, per the ticket's condition; the implementation is in that pull
  request's history.
- #443: the fitted path is that ticket's. `likelihood.objective` still calls
  `pruning_torch`, and switching it is one line against the 34.6% above ---
  taken there, beside PR 2's level-synchronous batching, which attacks the
  same term from the same side.

## What is not claimed

Nothing about a general rate matrix under A: `burn` exposes no matrix
exponential, so the route implemented Jukes-Cantor only and refused a `Q`. B
does differentiate the `matrix_exp` path, in `t` alone; a gradient in `Q` or in
the root distribution is `pruning_torch`'s and stays there, which is why the
GTR objective is untouched. Nothing about `candle`, `dfdx`, forward-mode duals
or Enzyme --- issue #449's survey costs those out and none was built. Nothing
about a GPU: every number here is one CPU thread. Nothing about site counts
above 20,000 or taxon counts above 16; the graph-node counts run to 16 taxa,
the wall times to 8.
