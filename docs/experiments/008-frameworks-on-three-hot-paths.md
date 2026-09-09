---
id: 008
date: 2026-09-09
commit: 1d7a89a3787003087626391da2f45a7e21120301
branch: claude/opt-390-frameworks
pr: 0
tickets: [388, 389, 390]
problem: potts-lattice
fixture: open square lattices at J = 0.6 with a seeded per-node field, and the 5-taxon tree neighbourhoods of tests/regression/fixtures/tree_search/ci.yaml
size: stress
methods: [ours, scipy-csgraph, rustworkx, torch-geometric]
budget: 0
seeds: [0]
hardware: Linux-x86_64
status: confirmed
---

# Does a framework beat the implementation it would replace on any of the three hot paths #376 named?

## Feature under test

Each of `scipy.sparse.csgraph.maximum_flow` (#388), `rustworkx.connected_components`
(#389) and PyTorch Geometric's `GINConv`/`global_add_pool` (#390) is faster than
the code it would front, by enough that the implementation moves to
`sandbox/` and referees it from there. The three would be the first entries in
that directory.

## Setup

One thread (`OMP_NUM_THREADS=1`), the host's exclusive lock held for every
timing, best of 5 (3 at extent 64, 20 for a surrogate forward). `cProfile`
self time ranks the term first, per root `CLAUDE.md`'s profile-first rule and
#341's bar: a term under 10% of its run is recorded and not ported. The
oracle each framework is pinned against is the implementation it would
replace — the energy is a combinatorial minimum and is compared exactly, the
cluster labelling is compared as a partition, and the surrogate's forward is
compared on tied first-layer weights where the two architectures coincide.

## Results

**Self time, `cProfile`.**

| path | term the framework would front | share |
| --- | --- | --- |
| `search.maxflow.ising_ground_state`, extent 16 / 32 / 64 | `_augment` + `_levels` | 61.1% / 66.2% / 73.5% |
| `search.potts_mcmc` Swendsen–Wang, extent 8 / 16 / 24 | `_find` + `_union` | 10.9% / 14.8% / 15.6% |
| `search.potts_mcmc` Wolff, extent 16 | `_find` + `_union` | 0.0% |
| `learn.fit_surrogate` with `GraphSurrogate`, 120 examples | `index_add` (gather and pool) | 7.0% |

**#388, the ground state.** Wall clock per `ising_ground_state`, random
per-node field:

| extent | Python Dinic | `scipy.sparse.csgraph.maximum_flow` | Rust Dinic (#336) |
| --- | --- | --- | --- |
| 16 | 7.59 ms | 2.15 ms | 0.51 ms |
| 32 | 36.86 ms | 7.26 ms | 2.37 ms |
| 64 | 290.33 ms | 33.02 ms | 14.12 ms |

All three report the same energy to every digit printed (−317.489949509,
−1359.092598119, −5480.514606575). scipy's `maximum_flow` takes `int32`
capacities, so the real-valued reduction is scaled and rounded; at extent 16
the energy is unchanged at scales from 1e2 to 1e8.

**#389, the cluster labelling.** The labelling alone, on the bond set of a
seeded configuration:

| extent | sites | open bonds | union-find | `rustworkx` | `scipy.csgraph` |
| --- | --- | --- | --- | --- | --- |
| 8 | 64 | 24 | 54.6 µs | 28.5 µs | 139.6 µs |
| 16 | 256 | 73 | 172.7 µs | 101.9 µs | 160.8 µs |
| 24 | 576 | 193 | 431.1 µs | 262.9 µs | 174.1 µs |
| 48 | 2304 | 657 | 1699.8 µs | 1158.4 µs | 207.0 µs |

The whole sweep, per sweep, with the labelling replaced and the recolouring
grouped by one stable sort in every column but the first:

| extent | current | ours (pointer doubling) | `rustworkx` | `scipy.csgraph` |
| --- | --- | --- | --- | --- |
| 8 | 0.296 ms | 0.195 ms | 0.229 ms | 0.346 ms |
| 16 | 1.113 ms | 0.650 ms | 0.725 ms | 0.644 ms |
| 24 | 2.531 ms | 1.386 ms | 1.568 ms | 1.179 ms |
| 48 | 10.321 ms | 5.220 ms | 5.846 ms | 3.745 ms |

Only the second column reproduces the chain: 60 sweeps from one seed at
extents 8, 16 and 24 give an array equal to the current sampler's entry for
entry, where both framework columns give a different one.

The same change through the committed benchmark, which runs at the transition
`J_c = ln(1 + sqrt(3))` rather than at `J = 0.6` and so builds fewer, larger
clusters --- `tests/benchmarks/test_potts_mcmc_bench.py`, 60 sweeps, minimum:
15.778 to 9.850 ms at extent 8 (1.60x) and 58.017 to 32.177 ms at 16 (1.80x).

**#390, the surrogate.** `GraphSurrogate` against the same architecture built
from `GINConv` with `eps = 0` and `global_add_pool`, weights copied across:

| examples / nodes / edges | forward, ours | forward, PyG | 40-epoch fit, ours | 40-epoch fit, PyG |
| --- | --- | --- | --- | --- |
| 60 / 480 / 420 | 0.849 ms | 0.972 ms (1.14× slower) | 181.4 ms | 189.1 ms (1.04× slower) |
| 240 / 1920 / 1680 | 3.779 ms | 3.288 ms (1.15× faster) | 494.2 ms | 460.6 ms (1.08× faster) |

Largest entry-wise gap between the two forwards on tied weights: 1.33e-15.

## Figures

none

## Finding

**No framework is adopted.** #390 fails the profile bar outright: the
`index_add` the framework would front is 7.0% of the fit, under #341's 10%,
and PyG is 1.04× slower at the CI size and 1.08× faster at four times it.
#388 clears the profile bar — Dinic is 61.1% to 73.5% of the ground state —
and scipy is 3.5× to 8.8× the Python reference, but 2.3× to 4.2× *slower*
than the Rust Dinic that already fronts this path, so adopting it would add a
third implementation of one algorithm and slow the caller down. #389 clears
the bar at extent 16 and above, and `rustworkx` is 1.29× to 1.77× the current
sweep — but this module's own code reaches 1.52× to 1.98× over the same
range, beating `rustworkx` at every extent measured and reproducing the chain
exactly, which neither framework does.

`scipy.sparse.csgraph.connected_components` is the framework #389 did not
name, and it is the faster of the two past extent 16: 1.179 ms against this
module's 1.386 at extent 24 and 3.745 against 5.220 at 48, 1.18x and 1.39x,
while losing at extent 8 (0.346 against 0.195). It is declined on the chain
rather than on the clock — its labels number the clusters by first appearance
in the CSR scan, so the recolouring takes different draws in a different
order, and experiment 001's autocorrelation times and every seeded Potts
figure would move. Buying 1.39x at extent 48 with a re-based sampler is a
trade this experiment does not make; a follow-up that wanted it would have to
sort the components into the union-find's root order, which needs the
union-find run anyway.

The `_find`/`_union` share of a **Wolff** sweep is 0.0%: Wolff grows one
cluster from a seed by breadth-first search and calls neither, so the
labelling #389 names is not on that path at all.

## Conclusion and actions

- #389 — closed by the change measured here: pointer doubling for the roots
  and one stable sort for the grouping, 1.52× to 1.98× with the chain
  unchanged. `rustworkx` declined on being 1.12× to 1.17× slower than it.
- #388 — closed on the numbers; nothing moves. The Python Dinic stays where
  it is as the Rust path's oracle.
- #390 — closed on 7.0%; nothing moves. PyTorch Geometric stays the referee
  it already was in `test_learn_surrogate_pyg.py`.
- #450 — seen while running the whole `search` suite for this experiment and
  not caused by it: the seven-taxon support calibration fails its
  bin-occupancy precondition with all 16 runs in the lowest bin. `release`
  tier, and no import path reaches it from the sweep this experiment changed.
- #405 — the Python Dinic is 49.9% of `alpha_expansion` at extents 16 and 32,
  which has no compiled path: `maxflow_rust.max_flow` returns the flow value
  and not the cut, and `expand` needs the cut. scipy's 3.5× to 8.8× would
  reach that call site. Recorded there rather than acted on: #388 is scoped to
  `ising_ground_state`, and `expand`'s infinite-capacity arcs have no `int32`
  form, so the port is its own decision.

## What is not claimed

Nothing about `scipy.sparse.csgraph.maximum_flow` at a scale where its `int32`
capacities overflow, or on an instance where the rounding changes the
minimiser: the scale sweep above is one lattice at one seed, and a bound on
the rounding error against the energy gap is not derived here. Nothing about
either framework on more than one host, or on more than one thread. Nothing
about PyTorch Geometric above 240 examples, where its trend is still
improving. Nothing about `rustworkx.connected_components` reached through a
graph built once and mutated per sweep rather than rebuilt: the measurement
rebuilds it, which is what the bond set changing every sweep requires of the
`PyGraph` API used here.
