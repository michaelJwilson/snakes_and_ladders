---
id: "013"
date: 2026-09-14
commit: a5ad18f33c33c4c2941bb8c83077d645426c90bf
branch: claude/sim-582-tree-ladder
pr: 583
tickets: [405, 344]
problem: tree
fixture: tests/regression/fixtures/tree_jc/release.yaml
size: ci
methods: [serial, threads, processes]
budget: 60
seeds: [0]
hardware: Linux-x86_64, 4 cores
status: confirmed
---

# What does fanning one neighbourhood's candidate fits across cores buy, and does it move the answer?

## Question

The candidates of an NNI neighbourhood are independent fits. Run through `snakes_and_ladders.parallel`, do they return the search the serial loop returns, and which backend pays?

## Numbers

| one `infer` run at 4 workers, seconds | serial | 4 threads | 4 processes |
| --- | --- | --- | --- |
| 8 taxa, 20,000 sites, 56 candidates | 30.25 | **25.27** | 49.42 |
| 20 taxa, 2,000 sites, 34 candidates | 14.94 | 29.74 | **9.92** |
| 50 taxa, 2,000 sites, 60 candidates | 58.87 | --- | **21.67** |

## Finding

The answer does not move: every run returns `log_likelihood` bitwise equal to the serial run's and the same topology, which input-ordered results over independent candidates guarantee and `test_parallel_candidate_fits_reproduce_the_serial_search_exactly` pins.
Which backend pays inverts with the regime, so neither may be defaulted: at few taxa and many sites the per-candidate work sits inside GIL-releasing `torch` kernels and threads win 1.20x while processes lose 1.63x to spawning a pool per neighbourhood and pickling a 20,000-site alignment per task; at many taxa and few sites the fit is Python-bound in the post-order and processes win 1.51x at 20 taxa and 2.72x at 50 while threads lose 1.99x.
The two rows below the fixture are a seeded random topology at root-to-tip height 0.25, which #582 has not yet landed as a fixture; they are the sizes the fan-out is for, and the pull request carries them. No actions.
