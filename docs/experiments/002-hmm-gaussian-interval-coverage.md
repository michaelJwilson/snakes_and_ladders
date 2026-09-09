---
id: 002
date: 2026-09-05
commit: 755c2aad09d32effb7b6a8b46f1e4c5b95847a97
branch: main
pr: 259
tickets: [228, 122]
problem: hmm-path
fixture: tests/regression/fixtures/hmm/, with the categorical family replaced by univariate Gaussian emissions --- two states, 24 replicates of 240 observations per separation
size: ci
methods: [gradient-fit-wald-intervals]
budget: 24
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
hardware: Linux-x86_64
status: confirmed
---

# Do the Wald intervals of a Gaussian-emission fit cover at their nominal rate, and from what separation?

## Question

Do the 95% Wald intervals from the observed information cover the generating parameters at their nominal rate once the emitting means are separated enough for the states to be identifiable, and under-cover below that?

## Numbers

| separation, standard deviations | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 6.0 |
| --- | --- | --- | --- | --- | --- | --- |
| intervals covering | 18/28 | 46/56 | 82/88 | 90/96 | 93/96 | 92/96 |
| rate | 0.643 | 0.821 | 0.932 | 0.938 | 0.969 | 0.958 |
| replicates at the variance floor, no interval | 17/24 | 10/24 | 2/24 | 0/24 | 0/24 | 0/24 |

## Finding

Coverage is nominal from two standard deviations of separation upward and degrades below it twice over: the intervals that exist under-cover, and an increasing fraction of replicates reach the variance floor and report none. From `pytest tests/regression/opt/test_opt_hmm_gaussian.py -m release`, figure `opt_coverage`; no actions.
