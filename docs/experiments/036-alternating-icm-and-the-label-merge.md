---
id: "036"
date: 2026-10-08
commit: fc7e2168f3581f4a476e7177f99c57e7f92efa1f
branch: claude/icm-merge-alternation-1390
pr: 0
tickets: [1390, 1373]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_reference/stress.yaml and its variants margin_0.1, q8, imbalance_50, equal and forbidden
size: stress
methods: [icm-merge, icm-merge-to-joint-fixed-point]
budget: 0
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
hardware: Linux-x86_64
status: confirmed
---

# Does a second ICM pass after the label merge recover energy on the Potts reference cell?

## Question

After `Polish.ICM_MERGE`'s one ICM-then-merge pass, do ICM and merge alternated to a joint fixed point lower the energy by >= 0.1 nats in >= 5% of runs, from anneals at x1/x4/x16 alpha-expansion+ICM's visits, uniform labellings, or alpha-expansion?

## Numbers

| cell (3,000 sites; 32 seeds x 4 start kinds + ae) | round-1 merges | runs gaining >= 0.1 | extra visits per run, where merged | median gap after one pass, x1 / x16 / random |
| --- | --- | --- | --- | --- |
| stress, q8, imbalance_50, equal, forbidden | 0 of 645 | 0 | 0 | stress 11.24 / 1.35 / 40.72 nats |
| margin_0.1 (optimum: one label, at the TRW-S bound) | 128 of 129, each a collapse to one label | 0 | 21,000 (one clean ICM sweep; 4% to 23% of round 1) | 0.00 / 0.00 / 0.00 (ICM alone: 369.93 / 64.33 / 363.84) |

## Finding

One pass is the joint fixed point on every run: the merge never fires from an ICM fixed point on five cells, and on margin_0.1 it collapses to the one-label optimum (gap 0.00 to the bound), after which ICM's sweep is clean, so a collapse lowers the final energy, 64 to 370 nats below ICM alone. Round 1 equals `Polish.ICM_MERGE` bitwise (labelling, energy, `polish_spent`) on all 576 anneals. `python -m sal.qa.icm_merge_alternation`, 19 s on 2 threads. No actions: iterating `Polish.ICM_MERGE` would cost one sweep per collapse and recover nothing here.
