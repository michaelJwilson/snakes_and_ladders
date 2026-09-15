---
id: "017"
date: 2026-09-15
commit: 668e8a3ef77599b4b3e7380347a734ac05efb8ce
branch: claude/infra-586-sparse-incidence
pr: 591
tickets: [586, 592]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_lattice/
size: ci
methods: [greedy, candidate]
budget: 0
seeds: [3]
hardware: Linux-x86_64
status: confirmed
---

# Does message passing's list-of-lists layout cost anything worth porting?

## Question

Is `_Layout`'s plan a large enough share of `sum_product` to pay for a compressed layout?

## Numbers

| graph | `_Layout` | `+ tree_steps` | `sum_product` | layout share |
| --- | --- | --- | --- | --- |
| chain, 500 variables | 0.348 ms | 11.052 ms | 60.832 ms | 0.6% |
| chain, 2,000 variables | 1.482 ms | 62.564 ms | 256.776 ms | 0.6% |
| star, 2,000 leaves | 1.433 ms | --- | 592.255 ms | 0.24% |
| lattice 16x16, flooding | 0.264 ms | --- | 119.159 ms | 0.22% |

## Finding

No: the layout is under 1 per cent on every shape, and what the profile ranks instead is the tree schedule at 18.2 and 24.4 per cent of the two chains. Actions: `#592`.
