---
id: "038"
date: 2026-10-08
commit: d2c629821e547fc0b1a29158b8b0a3242ab137ce
branch: claude/hmc-tuning-1390q6
pr: 1409
tickets: [1390, 1393]
problem: hmm-path
fixture: tests/regression/fixtures/count_hmm_reference/stress.yaml, 8,000 positions, 8 generating levels fitted by 7 states; variants dominant_50, dominant_90, delta_0.1, delta_0.6
size: stress
methods: [restart, k-means++, anneal, sample]
budget: 400
seeds: [1393]
hardware: Linux-x86_64
status: confirmed
---

# What does step="auto" spend, does T0 in units of |log L|/n transfer, and do the rare levels survive, with Baum-Welch after every HMC start?

## Question

At 400 passes per start, Baum-Welch polishing every start: does a pilot larger than 49 gradients change the step or the outcome, does one T0 multiple of |log L|/n (the start's negative log-likelihood over n = 8,000) lead on the variants, and which methods keep the 1% and 0.2% levels as states?

## Numbers

| stress, 8 shared starts | missed after BW, median | starts missing > 10% | 1% levels kept (of 16) | tuner + grads + BW |
| --- | --- | --- | --- | --- |
| restart / k-means++ | 1.7% / 1.8% | 1 / 0 | 15 / 15 | 0+0+378 / 0+0+373 |
| anneal, auto, T0 = 1, 3, 10, 30, 100 x scale | 12.9, 1.5, 1.2, 1.2, 2.2% | 4, 2, 1, 2, 1 | 16, 16, 16, 16, 14 | 49+129+222 |
| anneal T0 = 10x, pilot 4 / 8 per step; given 3e-3, 1e-2, 3e-2 | 1.2, 27.4%; 12.1, 1.3, 1.8% | 2, 7; 4, 2, 1 | 16, 16; 16, 15, 15 | 97+129+174, 193+129+78; 0+129+271 |
| sample, T = 1, 3, 10, 30, 100 x scale | 1.2, 17.5, 3.8, 4.3, 3.9% | 3, 5, 2, 0, 0 | 16, 16, 11, 5, 2 | 66+118+216 (BW 87, 11 at 30x, 100x) |
| variants, 4 starts: restart / k-means++ / anneal 10x / sample 1x | d50 9.1/6.1/8.4/6.2%; d90 3.4/3.5/4.4/5.6%; dlm 0.1 8.9/15.0/19.7/23.8%; dlm 0.6 4.7/0.9/25.2/16.5% | | | |

## Finding

**The 2-proposal pilot (49 gradients, 12% of 400) already settles: its choice equals the 8-proposal pilot's on 6/8 starts (3e-3 on 6/8), and the 8-proposal pilot's 193 gradients cost Baum-Welch enough to miss 27.4%. No single multiple leads: anneal T0 = 3-100x and given steps 1e-2-3e-2 are indistinguishable at 8 starts (1-2 in the ~25% merge mode), and 10x trails k-means++ or restarts on all four variants, while |log L|/n moves only 7.72-8.01 across starts and variants, so these cells cannot test the unit's transfer.** Level 5 (20 positions) is kept by no stress run; the 1% levels are kept by every arm except the sampler at T >= 10x (11, 5, 2 of 16). Trimmed: 4 starts on variants, no per-variant multiple grid. `python -m sal.qa.count_hmm_hmc_tuning stress`; `... variants 10 1`; 18 min on 2 threads. Actions: none in defaults; proposals in #1409.
