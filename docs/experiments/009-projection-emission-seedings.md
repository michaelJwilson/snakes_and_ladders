---
id: 009
date: 2026-09-11
commit: 1a3420dc992a97035a639e15532f1430d571ea05
branch: claude/opt-541-seeding
pr: 553
tickets: [541, 554, 555, 559]
problem: coupled
fixture: tests/regression/fixtures/spatio_sequential_counts/stress.yaml, the key instance's model at bin factor 5, projected onto its emission mixture
size: release
methods: [prior, data, kmeans++, emission++, burn-in, hmc, anneal, tempering, restart]
budget: 6
seeds: [0, 1, 2, 3, 4, 5]
hardware: Linux-x86_64, 4 cores at load 15 throughout, so no wall clock is reported and cost is counted in passes over the data
status: confirmed
---

# Which seeding of each state's emission parameters should the coupled fit start from?

## Question

At an equal six-pass budget on the key model in projection --- 100 components, 4,000 observations, 40 a component --- does any seeding that reads the data or samples a surrogate surface reach a better projected optimum than a draw from the prior, and does the likelihood's ordering agree with the simulated truth's?

## Numbers

| over 6 paired instances | prior | kmeans++ | emission++ | burn-in | anneal | hmc | tempering | data | restart |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best of 6 at 1e-4 relative, 4.7 nats (McNemar against prior) | **6** | 4 (0.50) | 3 (0.25) | 1 (0.06) | 1 (0.06) | 0 (**0.03**) | 0 (**0.03**) | 0 (**0.03**) | 0 (**0.03**) |
| mean gap below the best, nats of 46,800 | **0.0** | 3.6 | 7.6 | 8.3 | 8.9 | 9.4 | 10.2 | 12.1 | 12.9 |
| seeding cost, passes of the fit's 6 | 0 | 1 | 1 | 1 | **144** | **144** | **144** | 0 | 0 |
| recovery of the generating component, ceiling 0.109 (McNemar against prior) | 0.060 | 0.069 (0.69) | --- | --- | --- | --- | **0.089** (**0.03**) | **0.082** (**0.03**) | --- |

## Finding

The two orderings are opposite. The prior draw reaches the best projected value on all six instances and no candidate beats it, yet it recovers the least truth: `data` and `tempering` beat it on recovery 6 of 6 while losing to it on the likelihood, and the mean relative error in a component's negative-binomial mean runs 0.884 for `prior` against 0.691 and 0.303 for those two. At 40 observations a component the projected likelihood is not a proxy for recovery, so a seeding cannot be chosen on it here and **no default moves** (#559).
Cost settles the rest: `tempering` buys 0.007 of recovery over the free `data` baseline for 144 passes against the fit's 6, a 24x bill for 8% of the remaining ceiling, and its ladder never exchanges (lowest swap acceptance 0.00) while every chain accepts every proposal --- at 4,000 observations the surrogate is sharp and these are optimizers, not samplers.
From `python infra/seeding_sweep.py --out sweep.json --workers 6` and the same with `--recovery prior data kmeans++ tempering`; actions #554, #555, #559.
