---
id: "016"
date: 2026-09-15
commit: ca877c1f1df68c2bf596203ba536d27003d13eb4
branch: claude/infra-586-sparse-incidence
pr: 0
tickets: [586]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_lattice/
size: ci
methods: [greedy, candidate]
budget: 0
seeds: [11]
hardware: Linux-x86_64
status: confirmed
---

# One incidence layout: does lifting it cost any consumer time?

## Question

Can three classes share one compressed-row layout without any of them getting slower?

## Numbers

| median, per call | before | after | ratio |
| --- | --- | --- | --- |
| Wolff cluster move, 64x64 periodic | 3.451 ms | 0.935 ms | 3.69x |
| `FactorGraph.degree`, 800 variables | 42.350 ms | 2.801 ms | 15.1x |
| `FactorGraph.neighbours`, 800 | 30.480 ms | 1.600 ms | 19.1x |
| `gallager_code`, 120 bits | 0.140 ms | 0.168 ms | 0.83x |
| `gallager_code`, 19,998 bits | 11.515 ms | 11.545 ms | 1.00x |
| Dinic row walk, 16,384 rows of degree 6, as lists / offsets / NumPy | 1.95 ms | 3.93 ms | 25.63 ms |

## Finding

Two consumers gain, one pays a fixed 28 us of calls that is gone by 3,000 bits, and the fourth was measured and left as lists: `pytest tests/benchmarks/test_incidence_bench.py`. Actions: `no actions`.
