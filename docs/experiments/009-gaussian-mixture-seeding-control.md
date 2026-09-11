---
id: 009
date: 2026-09-11
commit: ce54d860b7195cbc507afde243db86b6f63f992b
branch: claude/opt-548-gmm-control
pr: 550
tickets: [548, 541]
problem: mixture
fixture: tests/regression/fixtures/mixture/ci.yaml, five components 1.5 standard deviations apart over 500 observations, the rung optimal_clustering_cost referees; and tests/regression/fixtures/mixture/release.yaml at bin factor 40, ten components over 504,100 two-channel observations, mirroring spatio_sequential_counts/stress.yaml at factor 5
size: release
methods: [random-restart, kmeans++, emission-d2, burn-in, family-sample, spectral, hmc, tempering, anneal]
budget: 40
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
hardware: Linux-x86_64
status: confirmed
---

# On a Gaussian, where the family's Bregman divergence is the squared Euclidean distance, does emission-aware D-squared sampling beat k-means++, tie it, or lose?

## Question

#541's candidate 3 rests on the claim that squared Euclidean is right for isotropic Gaussians and wrong for counts, so a Gaussian fixture should show no gain; does it?

## Numbers

| seeding rule, 200 seedings of `mixture/ci.yaml` | cost / exact optimum, mean | worst | mean gap of the fit it seeds, ci | key |
| --- | --- | --- | --- | --- |
| squared Euclidean D-squared, and the Gaussian Bregman divergence: identical draws | 1.8496 | 6.21 | 0.72 nats | 424 nats |
| the family's negative log density, which `plus_plus_start` scores with | 3.9111 | 33.80 | 1.38 nats | 668 nats |
| uniform, the random-restart baseline | 4.3470 | 38.18 | 5.30 nats | 3,030 nats |

Orderings, best first — ci, 40 starts: hmc, tempering, kmeans++ = spectral, emission-d2, anneal, family-sample, random-restart, burn-in. Key, 8 starts: spectral = kmeans++, hmc = tempering = anneal, emission-d2, family-sample, random-restart, burn-in.

## Finding

**The divergence ties and the implementation loses**, on both rungs and by 1.6x at the larger: the negative log density is the divergence plus the log normalizer, D-squared sampling normalizes rather than shifts, and the constant dilutes the rule nine tenths of the way to uniform. So the third outcome of #548 — the claim refuted — does not follow, and neither does the first: #541's candidate 3 measures a normalizer as well as a divergence, and its conclusion is unsafe as stated until it scores with one or the other. #541's own ordering does not exist yet, so both orderings above are this experiment's. From `pytest tests/regression/opt/test_opt_mixture_seeding.py -m release`; `STATUS.md` carries the full tables and what the 20,164,000-observation rung costs. Actions #541, and two bullets in `TICKETS.md`.
