---
id: 004
date: 2026-09-08
commit: 42ce20690fc6a1db9cb610a792d33e08e0a247ce
branch: claude/opt-332-mixture-budget
pr: 0
tickets: [332, 284, 303]
problem: mixture
fixture: tests/regression/fixtures/mixture/ci.yaml, the five-component Gaussian mixture --- means 1.5 standard deviations apart, weights (0.30, 0.10, 0.25, 0.15, 0.20), 500 observations, seed 20260908
size: release
methods: [restarts, anneal, tempering]
budget: 3000
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
hardware: Linux-x86_64
status: confirmed
---

# Do annealing or parallel tempering reach the mixture's best-known optimum from more starts than restarts of EM, at equal likelihood evaluations?

## Question

At 3,000 likelihood evaluations per start, do annealing or parallel tempering reach the five-component mixture's best-known optimum, 1111.596 nats, from more of 40 shared starts than multi-start EM does?

## Numbers

| method, 40 starts, one shared L-BFGS polish | reaching the optimum | mean gap | McNemar against restarts |
| --- | --- | --- | --- |
| restarts, 7 polished EM restarts per start | 7 | 2.6 nats | --- |
| parallel tempering, ladder (1, 2, 4, 8) | 4 | 4.2 nats | p = 0.549 |
| annealing, temperature 8 to 1 | 1 | 8.0 nats | p = 0.031 |

## Finding

Restarts are not beaten: tempering does not separate from them and annealing is worse, as on Rastrigin and opposite to the frustrated lattice, because these basins are separated by relabelling and component overlap rather than by energy barriers. From `pytest tests/regression/opt/test_opt_mixture_budget.py -m release`. Actions #333, #321.
