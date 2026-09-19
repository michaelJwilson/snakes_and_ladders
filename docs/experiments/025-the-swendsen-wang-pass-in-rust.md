---
id: "025"
date: 2026-09-19
commit: bed58b65b8e2ccb2c26681ad628fb28d0fa476f1
branch: claude/potts-754-cluster-pass
pr: 0
tickets: [754]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_lattice/
size: stress
methods: [numpy, rust, hoisted-multiply, sorted-grouping]
budget: 0
seeds: [7, 4]
hardware: Linux-x86_64
status: confirmed
---

# The Swendsen-Wang pass in Rust: what is left of the ranking's two Potts terms?

## Question

Does the cluster pass earn 2x over the NumPy reference at 64x64 once the reference is cut, and is the single-site verdict beside it still live?

## Numbers

| two readings | Python | Rust | ratio |
| --- | --- | --- | --- |
| `SwendsenWangMove.propose`, 64x64 open, 8,064 edges, `pytest-benchmark` mean | 20.200 / 20.692 ms | 318.5 / 327.6 us | 63.4x / 63.2x |
| `sample_potts` Swendsen-Wang, 10 sweeps, the same lattice | 168.9 / 173.6 ms | 7.96 / 8.25 ms | 21.2x / 21.0x |
| the same pass as the ranking read it, before the cuts (`perf_counter`, x10) | 41.59 / 41.19 ms | --- | 129x / 126x end to end |
| the field multiply hoisted out of the per-cluster loop | 30.66 / 31.01 ms | --- | 1.35x |
| and the clusters grouped by one sort where each scanned the labelling | 20.36 / 20.13 ms | --- | 1.51x |
| single-site 32x32 x 20 sweeps, the oracle against the default route | 173.4 / 186.1 ms | 2.23 / 2.45 ms | 78x / 76x |

## Finding

Yes, by 32x the bar: **63.4x** on the enclosing `propose`, saving **19.9 ms of 20.2**, and the ranking's 42.2 ms row ends at 0.32 ms --- of which **2.04x is the two algorithmic cuts**, taken first because `CLAUDE.md` ranks them above a port and because a ratio read against an uncut reference is not the port's. Agreement is **bitwise** with the oracle's own pass replayed on the same draws, at 3x3 and 6x6 over three temperatures and two seeds, guarded per cluster as the sweep is per site; the stream is not the oracle's, so the default stays Python and the law is pinned by enumeration. The single-site verdict beside it is **stale**: #599 flipped that default before the ranking read it and the 0.289 s row is `Backend.PYTHON` passed on purpose. `pytest tests/benchmarks/test_potts_mcmc_rust_bench.py -k swendsen --benchmark-columns=mean`, two readings, 1-minute load 2.01 and 1.93 with two other agents on the host. Actions: `no actions`.
