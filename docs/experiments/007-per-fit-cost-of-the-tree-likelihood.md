---
id: 007
date: 2026-09-09
commit: 6790315257458a3b772a879374eb1196304ea15c
branch: claude/likelihood-443-profile
pr: 0
tickets: [443, 436, 341, 232]
problem: tree
fixture: tests/regression/fixtures/tree_jc/, swept around the declared instance --- caterpillar topologies at 4 to 20 taxa (branch lengths uniform on [0.05, 0.3], seed 0) and random topologies from `search.topology.random_topology`, alignments simulated at seed 1
size: release
methods: [eager, torch-compile, rust-binding]
budget: 0
seeds: [0, 1]
hardware: Linux-x86_64
status: confirmed
---

# What does one branch-length fit cost, and which of the three candidate ports would change it?

## Question

Each of #443's three candidates --- `torch.compile` on the pruning forward, the L-BFGS step, and a Rust port amortizing the FFI boundary --- clears #341's 10% bar on a fit.

## Numbers

| candidate | measurement | verdict |
| --- | --- | --- |
| `torch.compile` on `pruning_torch.log_likelihood` | forward and backward **1.21x** eager at 8 taxa, **1.09x** at 20; 0 graph breaks, 1.8e-16 relative; graph 105 to 94 nodes; 13.6 s cold compile, 3.7--3.8 s warm, one artifact over 12 random topologies | declined; conserved as `sandbox.compiled_pruning`, pinned by `tests/regression/likelihood/test_pruning_torch_compile.py` |
| autograd graph nodes per fit | **7 x (tree nodes) + 9** exactly at every size --- four per branch, five per internal node; 1,020 at 4 taxa, 12,925 at 20; nodes per level 2.0x caterpillar, 2.8--3.8x random | #443's mechanism at 7x its constant; licenses **#443 PR 2** |
| `lbfgs.py:step` self time against #341's 10% bar | **6.63%** of an 8-taxon SPR search, **3.57%** of a 20-taxon one; 3.18% and 2.08% of a bare fit | under the bar and falling with size; closed, nothing built |
| FFI boundary of `pruning_rust.log_likelihood`, decomposed | marshalling **1.1--4.9%** of a through-binding call, kernel 90--100%, return under 0.03%; per-crossing floor 0.0497 ms at 8 taxa and 0.0855 ms at 20, so **1.34%** and **0.79%** of a fit | refused; kept measurable by `tests/benchmarks/test_pruning_rust_bench.py` |

Reproduce under `with_lock measure` on the 4-core reference host (torch 2.13.0, one BLAS thread): self time from `infra/profile_harness.self_time_table` over `tests/benchmarks/profile_hotpaths.py`'s fixtures, boundary rows from `tests/benchmarks/test_pruning_rust_bench.py`, medians over 5 profiled runs and 15--25 timed repeats.

## Finding

Only level-synchronous batching clears the bar --- **#443 PR 2**, 10--18% of a search's self time as a floor and 30--53% as an upper bound; **#443 PR 3** re-filed against a measurement taken after it, since the taped backward is 1.32--1.36x the taped forward; the Rust port and the L-BFGS step are closed by 1.34% and 6.63%.
