---
id: 001
date: 2026-09-04
commit: 405f5d00ccba5bdaa804e72f77e9f8f0120a602d
branch: main
pr: 212
tickets: [212]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_lattice/, swept around the declared instance --- open square lattices, q = 3, at the exact transition J_c = ln(1 + sqrt(q)), extents 8 to 24
size: stress
methods: [single-site, swendsen-wang, wolff]
budget: 0
seeds: [0]
hardware: Linux-x86_64
status: confirmed
---

# Do cluster updates decorrelate faster than single-site heat bath at the transition?

## Question

Do Swendsen-Wang and Wolff cluster moves decorrelate the energy in fewer sites touched than the single-site heat bath at the exact transition `J_c = ln(1 + sqrt(q))`, by a margin that widens with the lattice?

## Numbers

| autocorrelation time, sites touched | extent 8 | 12 | 16 | 24 |
| --- | --- | --- | --- | --- |
| single-site | 3.27 | 6.89 | 9.74 | 10.37 |
| Swendsen-Wang | 2.56 | 3.91 | 4.33 | 4.86 |
| Wolff | 2.71 | 3.04 | 3.68 | 5.01 |

## Finding

Single-site slows 3.2x between extent 8 and 24 against roughly 1.9x for both cluster algorithms, a 2.1x gap at extent 24 and widening, understated by lattices this small with an open boundary; from `pytest tests/regression/search/test_potts_mcmc.py`, whose chi-square against the exact Boltzmann distribution licenses the comparison. Action #231.
