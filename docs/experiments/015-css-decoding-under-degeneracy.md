---
id: "015"
date: 2026-09-15
commit: e1b8588d27218c73f8813f770562fe88d054aeeb
branch: claude/sim-362-qldpc-css
pr: 568
tickets: [362]
problem: bicycle-css
fixture: tests/regression/fixtures/bicycle_css/ci.yaml
size: ci
methods: [degenerate-ml, single-error-ml, sum-product]
budget: 4000
seeds: [11, 12, 13]
hardware: Linux-x86_64
status: confirmed
---

# What does degeneracy buy a decoder, and what does belief propagation lose to the cycles the CSS condition forces?

## Question

A CSS code's distinct errors can share a syndrome and act identically on the codespace, so scoring a decode by `e == e_hat` is the wrong question. Scored on the coset instead: how much does summing over a coset beat the likeliest single error, and where does sum-product sit against that floor?

## Numbers

| logical error rate, n = 16, m = 7, rank 7, k = 2, 30 four-cycles, p = 0.05 | rate | failures: logical / undecoded |
| --- | --- | --- |
| exact degenerate ML, the floor | **0.070297** | --- |
| exact single-error ML | 0.072735 | --- |
| belief propagation, 3 seeds x 4,000 trials | 0.178750 +- 0.005788 | 403 / 1,742 |
| belief propagation at n = 96, k = 16, 160 four-cycles, p = 0.02 | 0.287583 +- 0.006276 | 2,128 / 1,323 |

## Finding

Degeneracy is worth **3.4%** of the logical error rate --- 0.070297 against 0.072735 --- and changes the returned coset at **24** of the 128 syndromes with no tie broken, so the two are genuinely different rules and not one rule reported twice.
Sum-product sits at **2.54x** the floor, and the CSS condition is why: `H H^T = 0` forces even row overlaps, an overlap of exactly two is a four-cycle, and the graph carries 30 at n = 16 and 160 at n = 96 --- a decoder assuming a tree, run on a graph the code's defining property fills with short cycles.
The two failures invert with size, which one block-error number would hide: non-convergence is 81% of failures at n = 16 and 38% at n = 96. The quotient is pinned by its partition, every reachable syndrome carrying exactly 2^k = 4 cosets. Reproduce with `pytest tests/regression/likelihood/test_css.py`; no actions.
