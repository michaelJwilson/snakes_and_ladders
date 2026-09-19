---
id: 20
date: 2026-09-19
commit: 048a342adf507ed45fd0831619accd66f2fafd45
branch: claude/opt-756-sampling
pr: 0
tickets: [756]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_lattice/, the 16x16 open square at the exact 3-state transition J_c = ln(1 + sqrt(q)) in zero field
size: stress
methods: [single-site, locally-balanced, gibbs-with-gradients]
budget: 0
seeds: [0, 1]
hardware: Linux-x86_64
status: confirmed
---

# Do gradient-informed proposals decorrelate the energy faster than the heat bath at the transition?

## Question

At the exact transition on a 16x16 open lattice, does a sweep of `n_nodes` locally balanced or Gibbs-with-gradients proposals buy an independent energy sample in fewer sweeps than a heat-bath sweep of `n_nodes` updates, and in less work?

## Numbers

| 4,000 sweeps, 400 burn-in | single-site | locally balanced | Gibbs with gradients |
| --- | --- | --- | --- |
| tau in sweeps, seed 0 / seed 1 | 12.93 / 8.07 | 5.12 / 4.79 | 5.12 / 4.79 |
| wall, NumPy sweep | 9.3 s | 75.7 s | 109.3 s |

## Finding

Yes in sweeps, no in work: 4.96 sweeps buy an independent sample against the heat bath's 10.50, **2.1x fewer**, at **8.1x** the wall per sweep in NumPy and 250x the Rust sweep that is the default --- so **3.9x behind** the NumPy heat bath at equal wall clock and 380x behind the shipped one. The two gradient-informed chains are one chain to four figures, the Taylor estimate being the exact single-flip difference on a pairwise energy. Read by `search.statistics.integrated_autocorrelation_time` over `sample_potts`'s energies, two readings at a 1-minute load of 0.96 to 1.09; `tests/regression/search/test_potts_mcmc.py`'s chi-square against enumeration licenses the comparison. Action #754 owns the port the gap would need.
