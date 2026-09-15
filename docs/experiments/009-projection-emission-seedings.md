---
id: 009
date: 2026-09-11
commit: 7823f251f25be230d2734f46c4306391699fbcff
branch: claude/opt-560-bregman-divergence
pr: 553
tickets: [541, 554, 555, 559, 560]
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
| best of 6 at 1e-4 relative, 4.7 nats (McNemar against prior) | 5 | 3 (0.50) | **6** (1.00) | — | — | — | — | — | — |
| mean gap below the best, nats of 46,800 | 3.2 | 6.8 | **0.4** | 11.5 | 12.1 | 12.6 | 13.4 | 15.3 | 16.1 |
| seeding cost, passes of the fit's 6 | 0 | 1 | 1 | 1 | **144** | **144** | **144** | 0 | 0 |
| recovery of the generating component, ceiling 0.109 (McNemar against prior) | 0.060 | 0.069 (0.69) | --- | --- | --- | --- | **0.089** (**0.03**) | **0.082** (**0.03**) | --- |

## Finding

The two orderings are opposite. `emission++` reaches the best projected value on all six instances and leads at 0.4 nats, ahead of the prior draw it displaced when #560 corrected its scoring to the Bregman divergence --- but only on the likelihood, and the paired test does not separate the two (5 hits against 6, p = 1.00). The prior draw still recovers the least truth: `data` and `tempering` beat it on recovery 6 of 6 while losing to it on the likelihood, and the mean relative error in a component's negative-binomial mean runs 0.884 for `prior` against 0.691 and 0.303 for those two. At 40 observations a component the projected likelihood is not a proxy for recovery, so a seeding cannot be chosen on it here and **no default moves** (#559).
Cost settles the rest: `tempering` buys 0.007 of recovery over the free `data` baseline for 144 passes against the fit's 6, a 24x bill for 8% of the remaining ceiling, and its ladder never exchanges (lowest swap acceptance 0.00) while every chain accepts every proposal --- at 4,000 observations the surrogate is sharp and these are optimizers, not samplers.
From `python infra/seeding_sweep.py --out sweep.json --workers 6` and the same with `--recovery prior data kmeans++ tempering`; #560 re-ran `emission++` and the two candidates its ordering rests on, and re-based the six unchanged fits on the best that move produced, 3.18 nats lower, without re-counting their hits. Actions #554, #555, #559.
