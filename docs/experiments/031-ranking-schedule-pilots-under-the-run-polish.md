---
id: "031"
date: 2026-10-08
commit: 9c1ff9272cc200e5e6fa2d3ced8db134aafdc5f1
branch: claude/tuner-polish-1390q4
pr: 0
tickets: [1390]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_lattice/release.yaml
size: stress
methods: [lowest-energy, polished-gap-icm, polished-gap-icm-merge]
budget: 2304000
seeds: [0, 1, 2, 3, 4]
hardware: Linux-x86_64
status: confirmed
---

# Ranking schedule pilots under the run's polish: does `POLISHED_GAP` choose a different schedule from `LOWEST_ENERGY`, and a better one?

## Question

Over `SCHEDULE_GRID`, single-site, 5 tuning seeds on each of `potts_lattice` ci/stress/release, `planted_glass` ci and #1395's `potts_reference` stress (n = 3,000, q = 4): how often do the criteria disagree, and does the held-out `Polish.ICM_MERGE` run (4 seeds, 4x the pilot's steps) reach lower energy?

## Numbers

| 25 tunings per row | `LOWEST_ENERGY` | `POLISHED_GAP` + `ICM` | `POLISHED_GAP` + `ICM_MERGE` |
| --- | --- | --- | --- |
| choice differs from `LOWEST_ENERGY`, 8-sweep pilots | --- | 12 / 25 | 13 / 25 |
| choice differs from `LOWEST_ENERGY`, 64-sweep pilots | --- | 3 / 25 | 5 / 25 |
| held-out energy above TRW-S, `potts_reference` stress, 64 / 8 sweeps | 1.25 / 4.99 | 1.34 / 5.18 | 1.34 / 5.18 |
| held-out energy, `potts_lattice` release, 64 sweeps | -4970.44 | -4969.84 | -4969.95 |

## Finding

They disagree on 13 of 25 tunings at 8-sweep pilots and 5 of 25 at 64, but no disagreement buys energy: the held-out differences (0.09 to 0.6) are below the 1.06 spread across seeds on `potts_reference`, and on ci, stress and the glass every choice reaches one energy. A `Polish` pilot charges its polish to `spent`, +3.3% (`ICM`) and +4.9% (`ICM_MERGE`) on `potts_reference` at 64 sweeps, +105% at 8 sweeps on 144 sites. The script is in the pull request: 19 s for both pilot lengths on 2 threads. Actions: `no actions`.
