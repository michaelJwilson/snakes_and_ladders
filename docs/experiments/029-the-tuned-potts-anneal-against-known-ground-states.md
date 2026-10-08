---
id: "029"
date: 2026-10-08
commit: 9ad470c8161a34bffe6a6571807fe746eda1d8f5
branch: claude/known-ground-states-1378a
pr: 0
tickets: [1378, 1391, 1390]
problem: potts-lattice
fixture: tests/regression/fixtures/spatio_only/release.yaml at q = 2, tests/regression/fixtures/planted_glass/ci.yaml, and the seeded square-lattice sizing family of search.ground_state.lattice_rung at 144 to 4,096 sites
size: release
methods: [anneal-sw, anneal-wolff, anneal-gibbs, ae+icm, restart-icm, trws]
budget: 0
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
hardware: Linux-x86_64
status: confirmed
---

# Does the tuned Potts anneal reach a known ground state at x1, x4 and x16 alpha-expansion+ICM's site visits?

## Question

Does `anneal_potts` (tuned schedule, `Polish.ICM_MERGE`) hit the exact optimum where a closed form, a graph cut, enumeration or a proven MIP gives it?

## Numbers

| fixture, sites (32 seeds, x16 level) | anneal exact-hit | anneal median gap | ae+icm at x1 | restart-icm median gap |
| --- | --- | --- | --- | --- |
| zero field q = 3, -J\|E\|, 144 to 4,096 | 1.00 (SW and Wolff, every level) | 0 | exact | 0 |
| q = 2 in a field, graph cut, 144 to 5,041 | 0.09 at 144, 0.00 from 576 | 0.51% to 1.04% | exact | 2.90% to 6.21% |
| planted glass, enumerated -14.0 (planted -7.0), 18 | 1.00 (0.97 at x1) | 0 | refuses J < 0 | 0 (0.34 exact at x1) |
| q = 3 / q = 4 in a field, MIP-proven, 144 to 4,096 / 2,304 | 0.09 / 1.00 at 144, 0.00 from 576 | 0 to 1.29% | exact | 5.06% to 10.41% |
| workload: 3,000-site hex, q = 4, TRW-S decode = bound (8 seeds) | 0.00 | 0.83 nats | 0.96 nats | 30.50 nats |

## Finding

In a field the anneal misses its known optimum from 576 sites at every level, 0.29% to 1.29% at x16, where alpha-expansion+ICM is exact at 1/16 of the visits; at x1 the polish spends up to 11x the schedule's visits (64 x 64 zero field: 920,192 of 1,001,088). On the workload heat-bath Wolff alone is 917.5 nats off in 0.64 s and Wolff+Gibbs 0.52 in 0.93 s, Glauber at 4,000 sweeps 0.31 in 1.01 s. `python -m sal.qa.known_ground_states`, 70 s on 2 threads with the workload. Actions: #1391, the eleven missed cells in one ticket.
