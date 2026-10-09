---
id: "040"
date: 2026-10-09
commit: 28a9e621692be98f96b2c619035529536e3f0a7d
branch: claude/potts-labelling-study-1413
pr: 0
tickets: [1413]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_labelling/stress.yaml and its variants easy, hard and dev_6000, mirroring the downstream stream's problems (3,000 sites, q = 5, degree 6, floor 50)
size: stress
methods: [argmax, icm, icm-floor, icm+floor-smallest, ae+icm, trws, icm+merge, anneal+icm_merge]
budget: 0
seeds: [0, 1, 2, 3, 4]
hardware: Linux-x86_64, 2 threads
status: confirmed
---

# Where do the downstream labelling step's solvers part from sal's, and by which cause?

## Question

From one field and start, does each arm reach the TRW-S bound on the downstream stream's problems and their mirror, and does switching one cause close each gap?

## Numbers

| E - bound, nats, argmax start | stream r0 / r1 / r2 / r5 / r6 | mirror (stress) |
| --- | --- | --- |
| the other package's ICM+floor+merge, 5 seeds; floor off | 396-407 / 26-48 / 60-74 / 28-77 / 502-666; 7.6-13.0 / same / same / same / 312-332 | 10.2-12.5 |
| its worklist index-sorted; sal icm; sal icm-floor | 6.6 / 32.8 / 40.6 / 65.6 / 162.0; 6.1 / 29.9 / 35.6 / 44.5 / 222.5; 394 (200 sweeps) / = / = / = / 428 (200) | 13.00; 13.00; 13.00 |
| the downstream patched path; sal ae+icm / TRW-S decode / anneal+icm_merge | 320 / 0.33 / 0 / 0 / 179; TRW-S decode = bound within 3e-9 on all five | 0.14; 2.08 / 0 / 0 |

## Finding

Four causes, each switched alone. The floor: r0 and r6 plant a label of 15-18 sites under the floor of 50, and dissolving it costs 390 and 190-330 nats; off, the gap falls to ICM's own, and sal's floored ICM oscillates to its 200-sweep budget. The sweep order: the shuffled worklist spreads 7-23 nats on r1-r5 and sal's random order spans the same; index-sorted it is 2-74 sites from sal's index ICM on the stream and bitwise it on the mirror. The merge: the other package halves the boundary gain, so on r6 it misses the one merge sal's full gain takes (-7.97 nats); doubled, its labelling equals sal's. The coupling convention: its cost + sal's energy = 1e-12 to 7e-9 on 2e6, no cause. Ties and the tolerance (0, a clean sweep) match; the field construction was not switched. Cost: sal's Rust ICM 1.6-2.4 ms against 30-75 ms (other) and 36-283 ms (patched) at 3,000 sites; icm-floor peaks at 5.26 MB of floor uniforms drawn up front. `python -m sal.qa.potts_labelling`, 40 s; downstream numbers from scratch glue, not committed. Actions: #1413, the solver API, the Rust backend and a floor that does not oscillate.
