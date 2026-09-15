---
id: "012"
date: 2026-09-14
commit: a5ad18f33c33c4c2941bb8c83077d645426c90bf
branch: claude/sim-582-tree-ladder
pr: 583
tickets: [443, 449]
problem: tree
fixture: tests/regression/fixtures/tree_jc
size: ci
methods: [taped, analytic]
budget: 15
seeds: [20260914]
hardware: Linux-x86_64
status: confirmed
---

# Is the closed-form backward faster than the tape by an amount that depends only on the tree's size?

## Question

Experiment 007 measured the closed form 2.02x faster at 8 taxa and left the fitted path to switch as #443. Before switching it: is that ratio a property of the problem's size, so a default can be set from the size alone?

## Numbers

| one gradient, ms, median of 15, 20,000 sites | taped | analytic | taped/analytic |
| --- | --- | --- | --- |
| 4 taxa, the fixture's own branch lengths | 5.77 | 2.73 | **2.11** |
| 4 taxa, every branch at 0.1 | 3.17 | 5.60 | **0.57** |
| 8 taxa, the fixture's own branch lengths | 12.23 | 10.44 | 1.17 |
| 8 taxa, every branch at 0.1 | 11.47 | 7.74 | 1.48 |

## Finding

No: at one tree, one alignment and one site count, replacing the branch lengths with 0.1 moves the ratio from 2.11 to 0.57 --- the routes swap places --- so the ranking is not a function of size and no default follows from one. Flushing denormals to zero leaves the swing intact (2.11 to 1.70, 0.57 to 0.74), so that is not the mechanism and the cause is unidentified.
`BranchLengthObjective` therefore takes the route as `gradient=`, pinned to agree with the tape at 5.4e-16 relative and to the same optimum, and keeps the tape as its default; #443 does not switch the fitted path on this evidence, and the question it inherits is why the ranking moves with the branch lengths at all.
