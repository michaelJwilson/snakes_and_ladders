---
id: 008
date: 2026-09-10
commit: a1c2d551f0bf9cf3fa09dded394587fd8d72a089
branch: claude/infra-528-ffi-backends
pr: 0
tickets: [528, 447, 405]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_lattice, swept over open lattices at extents 8, 16 and 32
size: ci
methods: [python, rust]
budget: 0
seeds: [83, 85, 163, 165, 323]
hardware: Linux-x86_64, 4 cores, load 0.56-0.99 either side of every run
status: confirmed
---
# Which loop does `cProfile` rank first across the roadmap's paths, and does a compiled backend pay there?

## Question

`python tests/benchmarks/profile_hotpaths.py --tier mid` ranks every module's workload by self time. Is any uncompiled loop above the 10% rule, and is issue #447's claim --- `alpha_expansion` half in a Python minimum cut it has no compiled path for --- what the ranking shows?

## Numbers

| self time, mid tier, top uncompiled loop | fraction | ported? |
| --- | --- | --- |
| `search.alpha_expansion` 32x32: `maxflow._augment` 28.4% + `_levels` 20.6% | **49.0%** | yes, this ticket |
| `search.gibbs.sample_factor_graph` 32x32: `conditional` 44.8% | 44.8% | no: #341's, unchanged |
| `search.maxflow.ising_ground_state` 32x32: `_augment` + `_levels` 65.5%; every other module's first entry is `torch`, `numpy` or a Rust kernel | --- | no: `maxflow_rust` already carries it, and nothing else ranks |
| `alpha_expansion` median, 8x8 / 16x16 / 32x32 at 3 labels, Python -> Rust cut | 17.1 -> 3.5, 94.2 -> 14.2, 648.5 -> 60.3 ms | **4.82x, 6.66x, 10.76x** |

## Finding

#447 held: the ranking puts the Python Dinic solver at 49.0% of `alpha_expansion`, and the reason was never the boundary --- `oxi_snakes_and_ladders.max_flow` computed the source side and discarded it, so `expand`, which needs the cut and not the flow value, could not use the kernel at all.
Returning that side and hoisting a per-edge `set` construction out of the edge loop gives 4.82x to 10.76x as a caller pays; Criterion times the kernel alone at 15.2, 68.4 and 298.9 µs per cut, so the crossing is not the term at these sizes either.
From `python tests/benchmarks/profile_hotpaths.py --tier mid`, `pytest tests/benchmarks/test_alpha_expansion_bench.py` and `cargo bench -- max_flow_expansion_network`; no third backend and no other candidate above the 10% rule, which #528 reports as the outcome it is.
