---
id: "030"
date: 2026-10-08
commit: 9c1ff9272cc200e5e6fa2d3ced8db134aafdc5f1
branch: claude/gmm-anneal-1378b
pr: 0
tickets: [1378]
problem: mixture
fixture: tests/regression/fixtures/mixture/ci.yaml, the five-component mixture of experiment 004, best-known 1111.596410; and the seven-state joint count-pair HMM built in sal.qa.count_pair_hmm_anneal (8,000 positions, seed 1378), declared since #1416 as tests/regression/fixtures/count_pair_hmm/ci.yaml with the same draw
size: release
methods: [restarts, anneal]
budget: 3000
seeds: [0]
hardware: Linux-x86_64
status: confirmed
---

# Does tuned HMC annealing reach a known optimum from as many starts as restarts with the same polish, at equal passes?

## Question

With `step_size="auto"`, a ramp tuned on held-out starts and the polish from its best, does `hmc.anneal` match polished restarts on the mixture (3,000 passes) and on the count-pair HMM (600 passes)?

## Numbers

| per start, shared starts | mixture: hits of 40 / mean gap / s | HMM: positions missed (median, range) / mean gap / s |
| --- | --- | --- |
| restarts, same polish | 7 / 2.68 nats / 1.56 | 4.6% (0.5-32.1%) / best of 4 / 17.6 |
| tuned anneal, then polish | 5 (McNemar p = 0.774) / 5.37 nats / 0.47 | 48.7% (3.9-54.1%) / 243 nats / 10.9 |

## Finding

**Mixture: the anneal no longer trails significantly (1/40 at p = 0.031 in 004, 5/40 at p = 0.774 here; ramp inverse-linear 4 to 0.25 of 18). HMM: it trails, 0/4, its schedule ending 2,300-2,800 nats above the truth's 63,361; T0 = 1, 10, 100 x |log L|/n missed 48%, 49%, 68%, so no multiple transfers. The 10x-budget referee (one start) reached 63,391.6, above restarts' best.** `python -m sal.qa.mixture_hmc_anneal` and `python -m sal.qa.count_pair_hmm_anneal`. Actions #1393.
