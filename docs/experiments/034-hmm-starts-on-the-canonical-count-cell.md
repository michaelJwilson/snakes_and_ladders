---
id: "034"
date: 2026-10-08
commit: d00baaca4f4abcbbcac1f93924a8192f0820233d
branch: claude/hmm-anneal-remeasure-1393
pr: 0
tickets: [1393, 1378]
problem: hmm-path
fixture: tests/regression/fixtures/count_hmm_reference/stress.yaml, 8,000 positions, 8 generating levels fitted by 7 states, generating log L -60,947.364
size: stress
methods: [restart, anneal, sample, k-means++, quantile]
budget: 400
seeds: [1393]
hardware: Linux-x86_64
status: confirmed
---

# Does #1393's 48.7% missed after HMC annealing reproduce on the canonical count-pair cell, every start polished by Baum-Welch?

## Question

At 400 passes per start (gradients, scored draws and Baum-Welch iterations alike; samplers vary only log mu and logit p per state, chain at stay 1 - 1e-7), does the tuned anneal miss more positions after Baum-Welch than a Baum-Welch restart?

## Numbers

| 10 shared starts unless stated | missed after BW, median (range) | log L gap to best -60,916.2, median | grads + BW iters | s/start |
| --- | --- | --- | --- | --- |
| restart (random K rows, BW) / k-means++ / quantile (1 run) | 1.7% (1.1-30.0) / 1.8% (1.1-21.7) / 1.2% | 9.4 / 8.7 / 0.8 nats | 0+387 / 0+398 / 0+277 | 14.2 / 14.9 / 12.4 |
| anneal, step auto, T0 = 100 to 1 (T0 = 10, 1000 on 4 starts) | 1.2% (0.7-21.8) (12.9%, 4.8%) | 9.4 nats (7.5, 26.0) | 178+222 | 9.3 |
| sample at T = 100, 8 adapt + 13 draws, best (T = 10, 30, 300 on 4) | 5.0% (1.2-23.5) (1.2%, 1.8%, 4.3%) | 59.7 nats (13.5, 11.2, 79.0) | 184+216 | 7.7 |

## Finding

**No: the anneal at T0 = 100 misses a median 1.2% against restarts' 1.7% (better on 6 of 10 starts, worse on 3, sign test p = 0.51), at the restarts' median gap and 0.66x their wall; the fixed-T sampler at the downstream T = 100 misses 5.0%, and T = 10-30 1.2-1.8%.** Restarts, k-means++ and the anneal end a median 22 nats above the generating log L, the T = 100 sampler 28.5 below; 2/10 restarts and the quantile start come within 1 nat of the best. The sweeps ran on 4 starts to hold ~11 min on 2 threads; emission++ is not run. `python -m sal.qa.count_hmm_reference_starts`. Actions: close #1393.
