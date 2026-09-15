---
id: "014"
date: 2026-09-15
commit: a753441d981e4d9d319629b75001ca404456c3e0
branch: claude/sim-582-tree-ladder
pr: 583
tickets: [582, 443]
problem: tree
fixture: tests/regression/fixtures/tree_scale/ci.yaml
size: ci
methods: [before, after]
budget: 12
seeds: [20260914]
hardware: Linux-x86_64
status: confirmed
---

# What do the post-order optimizations buy on the fixture #582 asks for, and does the fixture recover its own truth?

## Question

#582 wanted a leaf-count ladder. Step 1 found no rung where the baselines fail, so the ladder is kept as a cost fixture: at its default rung, what do this branch's leaf-partial cache and dropped allocations save, and is the answer unchanged?

## Numbers

| `tree_scale` at 2,000 sites, median of 3 | before | after | ratio |
| --- | --- | --- | --- |
| 20 leaves, one gradient, ms | 8.389 | **7.032** | 1.19x |
| 20 leaves, the test's own search, s | 4.035 | **3.457** | 1.17x |
| 20 leaves, one full re-score, s | 0.141 | **0.117** | 1.21x |
| 50 leaves, one gradient, ms | 22.217 | **17.255** | 1.29x |

## Finding

The answer does not move: every run returns `log_likelihood` -23642.15298199 at 20 leaves and -43729.71042328 at 50, on 12 evaluations and 5 fits either side, which is what the two optimizations claim --- one is a cache of a constant and the other drops an allocation, and neither touches a value.
The saving is 1.17x to 1.29x here against the 12.1% and 10.6% measured separately on a generated 20-taxon tree, so the fixture reproduces them rather than revising them; `before` is `main` at `9407fb7` and `after` this branch, the two timed in separate processes on one alignment.
Neighbour joining recovers the declared topology exactly at every rung --- normalized RF 0.0000 at 20, 50 and 200 --- which is the step 1 result now pinned as a test rather than recorded in a sweep. No actions.
