---
id: "022"
date: 2026-09-19
commit: 1289daff10be249fb98b75351a80616a56e228ba
branch: claude/opt-756-clusters
pr: 0
tickets: [756]
problem: potts-lattice
fixture: tests/regression/fixtures/frustrated_lattice/ci.yaml, at 12x12 rather than the declared 3x3, and the file's own 60-site planted glass at frustration 0.25
size: stress
methods: [niedermayer, houdayer, single-site]
budget: 0
seeds: [0, 1, 2, 3, 4, 5, 6, 7]
hardware: Linux-x86_64
status: confirmed
---

# Does a cluster move buy a round trip on a frustrated lattice, where Wolff cannot run at all?

## Question

Niedermayer's rule runs where Wolff is refused, and Houdayer's move is free of an accept step: on the frustrated triangular antiferromagnet, does either buy anything — a cluster smaller than the lattice, a shorter round trip on the tempered ladder — or does the construction percolate and hand back a global spin reversal?

## Numbers

| 12x12 periodic triangular, J = -1 | Wolff | Niedermayer | Houdayer |
| --- | --- | --- | --- |
| cluster, sites of 9 (3x3, in a field) | refused | 8.31 | — |
| overlap defect / its largest component, of 144 | — | — | 65–72 / 43–59 |
| round-trip time, sweeps (8 seeds, paired) | — | — | 1,090.7 against 1,115.5 without, sign test p = 0.6875 |
| wall, 8 seeds x 2,000 sweeps x 10 rungs | — | — | 67.8 s against 17.4 s |

## Finding

**Both constructions percolate here and neither buys a round trip.** Niedermayer's cluster is 8.31 sites of 9 on the 3x3 antiferromagnet and 8.94 of 9 where the couplings are mixed — a single cluster that is the lattice is a global spin reversal — and Houdayer's is 43 to 59 sites of a 65 to 72-site defect region on the 12x12. The round-trip time is 1,090.7 sweeps with the move against 1,115.5 without, five seeds of eight shorter and two longer, **p = 0.6875**, at **3.90x and 4.01x the wall** over two readings at 1-minute loads of 1.77 and 2.11; on the yaml's 60-site planted glass at frustration 0.25, 772.3 against 955.9 at p = 0.2891 and 3.52x. Reproduced by `pytest tests/regression/search/test_search_tempered.py -m release`; the moves are exact, which is what `test_potts_mcmc.py`'s chi-square against enumeration says, so what is measured here is the instance and not the implementation. No actions: an instance below the overlap percolation threshold is what would separate them, and this lattice is not one.
