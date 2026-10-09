---
id: "041"
date: 2026-10-09
commit: 62b59ea457f68a31593076b1ffb113298765b3b6
branch: claude/combined-solver-figure-1414
pr: 0
tickets: [1414, 1413, 1412]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_labelling/ci.yaml and count_hmm_reference/ci.yaml (the CI pin); the figure runs on realizations 5-29 of the downstream package's sim/manifests/study15.toml at its commit a7ff616ed5488ec0f9b95d0115ea6f805d9b674a, both problems built by its stage functions (Potts 3,000 sites, q = 3-5; HMM 5,910-10,195 rows, K = 7, phased)
size: stress
methods: [argmax, icm, downstream-icm+floor+merge, icm+floor, ae+icm, trws, anneal-heat-bath, anneal-wolff+gibbs, anneal-sw+gibbs, merge, run-start, restart, k-means++, quantile, hmc-anneal, hmc-sample]
budget: 4000
seeds: [1414]
hardware: Linux-x86_64, Intel Xeon 2.80 GHz, 2 processes x 1 thread
status: confirmed
---

# On 25 downstream stream realizations, which sal solver reaches the bound, and which start the best fit?

## Question

Per realization, does each labelling arm reach TRW-S's bound, and does each HMM start, polished by the downstream fit and then to convergence, reach the best log-likelihood any run reached?

## Numbers

| median [IQR] over 25 realizations | gap, nats (after polish) | missed % | wall s |
| --- | --- | --- | --- |
| (a) icm / downstream icm+floor+merge / argmax+polish | 22.90 [5.66, 82.18] / 24.89 [5.66, 82.18] / 14.97 [6.01, 64.91] | 5.2 / 5.2 / 5.0 | 6e-4 / 9e-4 / 7e-4 |
| (a) ae+icm / trws / anneal heat bath, wolff+gibbs, sw+gibbs (4,000 sweeps of visits) | 0.06 [0, 0.53] / 0 [0, 0] / 0 [0, 0.26], 0 [0, 0.18], 0 [0, 0.45] | 3.6 / 3.6 / 3.6, 3.5, 3.5 | 0.005 / 0.027 / 1.16, 1.05, 0.99 |
| (b) downstream fit: run start / restart / k-means++ / quantile / hmc anneal / hmc sample | 114 / 146 / 137 / 122 / 106 / 103 | 61.6 / 49.8 / 36.2 / 61.3 / 45.7 / 50.6 | 1.0 / 1.0 / 1.0 / 1.0 / 3.0 / 3.3 |
| (b) converged: same order | 67.7 / 75.3 / 47.8 / 25.4 [0, 75.3] / 44.5 / 51.4 | 40.5 / 42.1 / 35.0 / 35.9 / 40.8 / 43.2 | 7.4 / 10.7 / 12.3 / 14.1 / 12.1 / 11.8 |

## Finding

TRW-S's decode meets its bound on all 25, and the anneals and ae+icm come within 0.06 nats in the median at 40-200x its wall. Index ICM and the downstream path stop 23-25 nats above it in the median. The downstream fit's 12 E steps leave every start 103-146 nats below the best run. Running on to convergence halves that, with quantile closest (25 nats), at 7-14x the wall. Capture takes 111 s per realization. Run with `python -m sal.qa.combined_solvers --source stream`. No actions.
