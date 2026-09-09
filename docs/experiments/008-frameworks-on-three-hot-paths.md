---
id: 8
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

## Question

Is `scipy.sparse.csgraph.maximum_flow` (#388), `rustworkx.connected_components` (#389) or PyTorch Geometric's `GINConv` (#390) fast enough on the term it would front to move that term's implementation to `sandbox/`?

## Numbers

| front | `cProfile` self time of the term it replaces | framework | what fronts the path today | verdict |
| --- | --- | --- | --- | --- |
| scipy min cut, `ising_ground_state` at extent 16 / 32 / 64 | 61.1 / 66.2 / 73.5% | 2.15 / 7.26 / 33.02 ms | Rust Dinic 0.51 / 2.37 / 14.12 ms | 2.3x-4.2x slower |
| `rustworkx` labelling, Swendsen-Wang sweep at extent 8 / 16 / 24 / 48 | 10.9 / 14.8 / 15.6% (0.0% for Wolff) | 0.229 / 0.725 / 1.568 / 5.846 ms | pointer doubling 0.195 / 0.650 / 1.386 / 5.220 ms | 1.12x-1.17x slower, and renumbers the clusters |
| PyG `GINConv`, `GraphSurrogate` forward at 60 / 240 examples | 7.0%, under #341's 10% bar | 0.972 / 3.288 ms | ours 0.849 / 3.779 ms | 1.14x slower, then 1.15x faster |

One thread under `with_lock measure`, best of 5 (3 at extent 64, 20 for a forward); PR #447 carries every column in full and `STATUS.md` the rows this is cited from.

## Finding

All three declined; the 1.52x-1.98x that landed instead is this repository's own pointer doubling and sorted grouping, and each front is conserved in `sandbox/` with the test that referees it (#388, #389, #390, #405).
