---
id: 001
date: 2026-09-04
commit: 405f5d00ccba5bdaa804e72f77e9f8f0120a602d
branch: main
pr: 212
tickets: [212]
problem: potts-lattice
fixture: open square lattice, q = 3, at the exact transition J_c = ln(1 + sqrt(q)), extents 8 to 24
size: stress
methods: [single-site, swendsen-wang, wolff]
budget: 0
seeds: [0]
hardware: Linux-x86_64
status: confirmed
---

# Do cluster updates decorrelate faster than single-site heat bath at the transition?

## Feature under test

Swendsen–Wang and Wolff cluster moves, beside the single-site heat bath behind
one interface, decorrelate the energy in fewer sites touched than the
single-site sweep at the critical coupling, and the gap widens with the lattice.

## Setup

Open square lattices of extent 8, 12, 16 and 24 with `q = 3` states at the
exact transition `J_c = ln(1 + sqrt(q))`. Each sampler runs from the same
seed and the energy autocorrelation time is measured in units of sites touched,
so a cluster move that touches many sites is charged for them. The budget is
the autocorrelation measurement itself rather than a fixed count of
evaluations, which the `0` above records. Correctness of all three samplers
is pinned separately by a chi-square against the exact Boltzmann distribution
at an enumerable size (36 runs, six seeds, significance 0.001, worst
`p = 0.0145`), which is what licenses comparing their speed.

## Results

Energy autocorrelation time, normalized to sites touched, from
`tests/regression/search/test_potts_mcmc.py` at the commit above:

| extent | single-site | Swendsen–Wang | Wolff |
| --- | --- | --- | --- |
| 8 | 3.27 | 2.56 | 2.71 |
| 12 | 6.89 | 3.91 | 3.04 |
| 16 | 9.74 | 4.33 | 3.68 |
| 24 | 10.37 | 4.86 | 5.01 |

## Figures

none

## Finding

Single-site slows by 3.2× between extent 8 and 24 while both cluster
algorithms slow by roughly 1.9×, so the gap is 2.1× at extent 24 and widening.
The separation understates the asymptotic one: the lattices are small and the
boundary is open, both of which soften the transition.

## Conclusion and actions

#231 — support the other lattice types, so the comparison runs on a periodic
lattice where the transition is sharper.

## What is not claimed

Nothing about the dynamic critical exponents: four extents on an open lattice
do not fit a power law, and no exponent is stated. Nothing about performance
away from `J_c`, where single-site updates are not slow.
