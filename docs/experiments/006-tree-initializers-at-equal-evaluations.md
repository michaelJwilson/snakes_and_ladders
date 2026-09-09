---
id: 006
date: 2026-09-08
commit: ebf977a8554d46cc4238b618df3bbb37de85f985
branch: claude/opt-364-spectral-init
pr: 0
tickets: [364, 251, 281]
problem: tree
fixture: tests/regression/fixtures/tree_search/ci.yaml, tests/regression/fixtures/tree_search/stress.yaml, and a random 20-taxon tree (lengths uniform on [0.02, 0.15], seed [364, 20]) at 1,000 sites
size: release
methods: [FromObjective, Perturbed, RandomRestart, FromDistances, FromHadamard]
budget: 2000
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
hardware: Linux-x86_64
status: confirmed
---

# Do the data-driven starts reach the tree's optimum from more starts, or in fewer evaluations, than the objective's own?

## Question

At equal likelihood evaluations, do the data-driven starts --- neighbor joining on pairwise distances, and the closest tree of the Hadamard conjugation --- reach the branch-length optimum in fewer evaluations, and the enumerated topology from more datasets, than the objective's constant start?

## Numbers

| mean fit evaluations / NNI candidates (datasets reaching the referee) | FromObjective | Perturbed | RandomRestart | FromDistances | FromHadamard |
| --- | --- | --- | --- | --- | --- |
| 5 taxa, 20 datasets | 23.0 | 23.1 | 24.1 / 7.3 (20) | 22.3 / 4.0 (20) | 21.8 / 4.0 (20) |
| 6 taxa, 10 datasets | 24.4 | 24.3 | 26.9 / 16.3 (10) | 23.5 / 6.0 (10) | 24.2 / 6.0 (10) |
| 20 taxa, 5 datasets | 34.0 | 33.4 | 41.6 / 60 (0) | 25.2 / 34 (5) | --- |

## Finding

No start separates on the fit below 20 taxa, where the estimator saves 9 evaluations of 34; on the search it buys the topology, one neighbourhood against the random start's 7.3 and 16.3 candidates and 5 of 5 datasets against 0 of 5 at 20 taxa (McNemar p = 0.062). From `pytest tests/regression/search/test_search_initialize.py -m release`; closes #364, no default changes.
