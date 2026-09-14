---
id: "010"
date: 2026-09-11
commit: 7823f251f25be230d2734f46c4306391699fbcff
branch: claude/opt-560-bregman-divergence
pr: 550
tickets: [548, 541, 560]
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

| seeding rule | #541, counts: nats below the best of 8, 4 components, 4,000 draws, 6 passes | here, Gaussian ci: nats below the truth-started fit, 5 components, 500 draws | here, Gaussian key: the same, 10 components, 504,100 two-channel draws | cost / exact optimum over 200 seedings, ci |
| --- | --- | --- | --- | --- |
| squared Euclidean D-squared | 20.4 | 0.723 | 424 | 1.8496 |
| the family's Bregman divergence, which `plus_plus_start` scores with since #560 | 200.0 | 0.723, on identical draws | 749 | 1.8496 |
| the negative log density, which it scored with before | 20.5 | 1.38 | 668 | 3.9111 |
| the uniform control | 12.0 | 5.30 | 3,030 | 4.3470 |

Orderings, best first — Gaussian ci: hmc, tempering, kmeans++ = emission-d2 = spectral, anneal, family-sample, random-restart, burn-in. Gaussian key: spectral = kmeans++, hmc = tempering = anneal, family-sample, emission-d2, random-restart, burn-in.

## Finding

**It ties on one rung and loses on the other, and the correction decided which.** On the one-channel rung the divergence *is* squared Euclidean, so candidate 3 is k-means++ draw for draw — 1.8496 times the exact optimum against the negative log density's 3.9111, with uniform seeding at 4.3470, so the rule it shipped with sat nine tenths of the way to no rule at all (#560). On the two-channel key rung the two channels carry different pooled scales, the divergence divides each by its own, and that is a different rule from k-means++ on the raw pair: 749 nats against 424, further than the 668 it replaced. From `pytest tests/regression/opt/test_opt_mixture_seeding.py -m release`; `STATUS.md` carries the nine-row tables and what the 20,164,000-observation rung costs. Actions: the `TICKETS.md` bullet #548 filed on the chain step size, and #541.
