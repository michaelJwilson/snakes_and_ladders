---
id: 002
date: 2026-09-05
commit: 755c2aad09d32effb7b6a8b46f1e4c5b95847a97
branch: main
pr: 259
tickets: [228, 122]
problem: hmm-path
fixture: two-state HMM with univariate Gaussian emissions, 24 replicates of 240 observations per separation
size: ci
methods: [gradient-fit-wald-intervals]
budget: 24
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
hardware: Linux-x86_64
status: confirmed
---

# Do the Wald intervals of a Gaussian-emission fit cover at their nominal rate, and from what separation?

## Feature under test

The 95% Wald intervals from the observed information at a Gaussian-emission
fit cover the generating parameters at their nominal rate once the emitting
means are separated enough for the states to be identifiable, and under-cover
below that separation.

## Setup

Two hidden states with univariate Gaussian emissions one common standard
deviation wide, with the means separated by 0.5 to 6 standard deviations; 24
replicates of 240 observations per separation, each fitted by L-BFGS and given
Wald intervals from the observed information. A replicate at the variance
floor is a refusal, not an interval, and is counted as such. The oracle is the
generating parameter: an interval covers or it does not.

## Results

From `tests/regression/opt/test_opt_hmm_gaussian.py` and the coverage figure
`opt_coverage` at the commit above:

| separation | intervals covering | rate | replicates with no interval at all |
| --- | --- | --- | --- |
| 0.5 | 18/28 | 0.643 | 17/24 |
| 1.0 | 46/56 | 0.821 | 10/24 |
| 2.0 | 82/88 | 0.932 | 2/24 |
| 3.0 | 90/96 | 0.938 | 0/24 |
| 4.0 | 93/96 | 0.969 | 0/24 |
| 6.0 | 92/96 | 0.958 | 0/24 |

## Figures

`opt_coverage`

## Finding

Coverage reaches nominal from two standard deviations of separation upward and
degrades below it, seen twice over: the intervals that exist under-cover, and
an increasing fraction of replicates reach the variance floor and report no
interval at all.

## Conclusion and actions

none

## What is not claimed

Nothing about coverage at other sequence lengths or state counts; the
asymptotic argument behind the Wald interval is not tested past the two-state
case, and 24 replicates bound a coverage rate to roughly ±0.1.
