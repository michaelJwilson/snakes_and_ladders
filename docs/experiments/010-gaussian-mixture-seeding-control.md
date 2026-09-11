---
id: "010"
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

| seeding rule | #541, counts: nats below the best of 8, 4 components, 4,000 draws, 6 passes | here, Gaussian ci: nats below the truth-started fit, 5 components, 500 draws | here, Gaussian key: the same, 10 components, 504,100 two-channel draws | cost / exact optimum over 200 seedings, ci |
| --- | --- | --- | --- | --- |
| squared Euclidean D-squared | 20.4 | 0.72 | 424 | 1.8496 |
| the Gaussian Bregman divergence: identical draws | not applicable | 0.72 | 424 | 1.8496 |
| the negative log density, which `plus_plus_start` scores with | 20.5 | 1.38 | 668 | 3.9111 |
| the uniform control | 12.0 | 5.30 | 3,030 | 4.3470 |

Orderings, best first — #541, counts: tempering, hmc = anneal, burn-in, data, kmeans++, emission++, prior. Here, Gaussian ci: hmc, tempering, kmeans++ = spectral, emission-d2, anneal, family-sample, random-restart, burn-in. Here, Gaussian key: spectral = kmeans++, hmc = tempering = anneal, emission-d2, family-sample, random-restart, burn-in.

## Finding

**It wins on neither, and the reason is that it never ran the divergence.** On counts it ends 0.1 nats behind squared Euclidean at equal budget and on a Gaussian it is 1.6x further from the truth-started fit; the divergence itself ties *exactly*, seeding for seeding. `plus_plus_start` scores with the family's negative log density, which is the divergence plus the log normalizer, and D-squared sampling normalizes rather than shifts, so the constant dilutes the rule nine tenths of the way to uniform. #541's conclusion therefore stands on a normalizer as much as on a geometry. From `pytest tests/regression/opt/test_opt_mixture_seeding.py -m release`; `STATUS.md` carries the nine-row tables and what the 20,164,000-observation rung costs. Actions: the two `TICKETS.md` bullets #548 filed, and #541.
