---
id: 004
date: 2026-09-08
commit: 42ce20690fc6a1db9cb610a792d33e08e0a247ce
branch: claude/opt-332-mixture-budget
pr: 0
tickets: [332, 284, 303]
problem: mixture
fixture: five-component Gaussian mixture from `sim.mixture`, means 1.5 standard deviations apart, weights (0.30, 0.10, 0.25, 0.15, 0.20), 500 observations, seed 20260908
size: release
methods: [restarts, anneal, tempering]
budget: 3000
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
hardware: Linux-x86_64
status: confirmed
---

# Do annealing or parallel tempering reach the mixture's best-known optimum from more starts than restarts of EM, at equal likelihood evaluations?

## Feature under test

At 3,000 likelihood evaluations per start, simulated annealing with
Hamiltonian proposals or parallel tempering reaches the best-known optimum of
the five-component mixture from more of 40 shared starts than multi-start EM
does, with a McNemar p-value below 0.05.

## Setup

The fixture is the five-component Gaussian mixture #262 measured by hand and
did not commit, built from `sim.mixture` under seed 20260908: means
(-3, -1.5, 0, 1.5, 3) with unit scales, so adjacent components are 1.5
standard deviations apart, weights (0.30, 0.10, 0.25, 0.15, 0.20), 500
observations. Release tier: the 8-start version of the same code runs per
pull request.

The budget is 3,000 evaluations per start, held equal by `opt.budget.compare`.
One evaluation is one pass over the observations' per-component log
densities — an EM iteration, an objective value, or an objective gradient
each count one (0.6 ms, 0.4 ms and 1.5 ms single-threaded). Every method ends
with the same L-BFGS polish of at most 8 outer steps, reserved at 218
evaluations and charged, because EM converges linearly on components this
close and a raw EM value 500 iterations in sits 4 to 6 nats above its
basin's optimum:

- **restarts**: EM for 200 iterations from uniform weights, means at five
  distinct observations and pooled scales, then the polish; 418 evaluations
  per restart, so 7 restarts per start, best kept.
- **anneal**: one Hamiltonian chain from the same random start on an
  exponential schedule from temperature 8 to 1, step 0.02, 10 leapfrog
  steps, 198 proposals of 14 evaluations, then the polish from the best point
  visited.
- **tempering**: four replicas at temperatures (1, 2, 4, 8) from the same
  random start, exchanging after every round, 49 rounds, then the polish from
  the best point visited.

Step 0.02 is the largest at which every temperature tried accepted at least
0.85 of proposals on tuning starts (seed 1000, not among the 40); the ladder
is ratio 2 with four replicas as the Potts study's. Start `i` draws from
`np.random.default_rng([0, i])` whichever method runs, so all three see the
same random start.

The referee is the best-known negative log-likelihood: the lower of the
polished simulated parameters and the best of 1,000 polished restarts (EM for
500 iterations, seeds `[20260908, i]`). A start reaches it within `1e-6`
relative, 0.0011 nats. The McNemar p-value is exact, on the discordant starts
against restarts.

## Results

RESULTS_PLACEHOLDER

## Figures

none

## Finding

FINDING_PLACEHOLDER

## Conclusion and actions

ACTIONS_PLACEHOLDER

## What is not claimed

Nothing about other budgets: the tempered methods run 198 proposals or 49
rounds here, and the ordering can differ where they run longer. Nothing about
the step size, ladder or schedule beyond the one configuration fixed a priori.
Nothing about EM's own convergence rate, which the shared polish removes from
the comparison. Nothing about wall time as a cost: the seconds per start are
from a shared host under a load average above 30 and are reported for scale
only; the budget is in evaluations.
