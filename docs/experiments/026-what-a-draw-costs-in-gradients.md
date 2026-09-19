---
id: "026"
date: 2026-09-19
commit: 954b39fd33c39028f57f5c8737e32cd0506f2276
branch: claude/opt-756-baselines
pr: 0
tickets: [756]
problem: mixture
fixture: tests/regression/fixtures/mixture/ci.yaml names the problem and is not the instance; the two run here are the analytic Gaussian of tests/_objective_checks.py and the twelve-observation enumerable weight posterior of tests/_posteriors.py, neither of which the registry declares
size: stress
methods: [hmc, mala, slice]
budget: 0
seeds: [1, 2]
hardware: Linux-x86_64
status: confirmed
---

# Does a Hamiltonian trajectory pay for its gradients against a single Langevin step and a tuning-free slice sampler?

## Question

HMC spends `n_steps + 1` gradients a proposal where MALA spends two and slice sampling spends objective evaluations and no gradient: does the trajectory buy back enough effective samples to lead on either unit, or on the wall clock that is the one unit all three spend?

## Numbers

| effective samples per 1,000 evaluations, worst coordinate, seeds 1 / 2 | HMC (gradients) | MALA (gradients) | slice (objective) |
| --- | --- | --- | --- |
| analytic Gaussian, 2,000 draws | 25.3 / 22.0 | 59.1 / 71.8 | 30.8 / 29.4 |
| analytic Gaussian, 8,000 draws | 27.8 / 30.5 | 53.0 / 54.7 | 31.4 / 30.3 |
| enumerable mixture posterior, 800 draws | 19.4 / 20.5 | 191.8 / 166.5 | 100.8 / 116.5 |
| enumerable mixture posterior, 3,200 draws | 38.8 / 30.2 | 196.2 / 235.0 | 125.2 / 115.1 |
| wall, 8,000 / 3,200 draws | 17.5 s / 15.1 s | 3.8 s / 3.8 s | 2.6 s / 2.5 s |

## Finding

**No: the trajectory is not paid for at these dimensions, and neither unit decides it alone.** MALA takes 1.7x to 2.6x HMC's effective samples per gradient on the two-coordinate Gaussian and 5.1x to 7.8x on the one-coordinate mixture posterior, where a ten-step trajectory retraces a line it has already crossed, and slice sampling is ahead of HMC per evaluation on both targets — 30.3 to 31.4 against 22.0 to 30.5, and 100.8 to 125.2 against 19.4 to 38.8 — with no tuning beyond a width, whose twentyfold change moves a sweep from 11.3 to 16.0 evaluations. A gradient is a forward evaluation and a backward pass through the same tape, so the two columns are not one column and the ranking is settled on the wall instead, where it is the same: 2.6 s, 3.8 s and 17.5 s for 8,000 draws at 1-minute loads of 3.90 to 4.32 on a 4-core host shared with two other agents. Reproduced by `pytest tests/regression/opt/test_opt_sampler_efficiency.py -m release`; no actions.
