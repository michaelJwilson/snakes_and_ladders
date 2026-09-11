---
id: 009
date: 2026-09-11
commit: 47292b4408fa5ab82ac033f97b2fbf17dc6f10d9
branch: claude/search-551-ground-state
pr: 552
tickets: [551]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_spots, at ci (9 sites, q = 3) and release (5,041 sites, q = 2 and q = 10)
size: release
methods: [greedy, icm, gibbs-T0, anneal, swendsen-wang, wolff, tempering, alpha-expansion, alpha-beta-swap, max-product]
budget: 2083260
seeds: [20260911]
hardware: Linux-x86_64, 4 cores, host not quiet -- eight agents running, so energies and counts only and no wall clock claimed
status: confirmed
---
# Which method recovers the Potts ground state at 5,041 sites, and does the fixture's simulated truth referee one?

## Question

Ranked on energy and on the generating parameters together: does any sampler reach the cut's exact ground state, and do the fixture's tilt, null class and occupancy separate methods where the energy bracket cannot?

## Numbers

| rung | winner | energy | gap to second | second |
| --- | --- | --- | --- | --- |
| ci, 9 sites, q = 3 | all ten, enumerated | -7.9364 | 0 | trivial: every entry reaches it |
| release, 5,041, q = 2 | alpha-expansion = alpha-beta-swap, exact | -10454.1563 | 219.03 | anneal -10235.13 |
| release, 5,041, q = 10 | alpha-expansion = alpha-beta-swap | -10454.1563 | 540.60 | anneal -9913.56 |

## Finding

Both cut-based minimizers reach the q = 2 graph-cut optimum exactly and no sampler comes within 219 of it; at q = 10 they return the *same* labelling, using 2 of 10 classes, because only the ladder's extremes are ever field-optimal.
The rung-3 bracket is **1067.42** wide against a 540.60 winner-to-second gap, so energy alone ranks nothing there --- and the structural referee refutes its own premise rather than rescuing it: the exact ground state's tilt is **0.2151**, below the fixture's thermal 0.4935, and its occupancy is 3286/1755 against the recorded 415-593, because a ferromagnet orders as `T` falls.
All three fixture statistics are finite-temperature: the method matching them best (Wolff, occupancy 484-567) is the worst on energy at -1141.56, and Swendsen-Wang confirms the cluster prediction --- from `T` = 2.000 to 0.068 its mean cluster grows 1.15 to 27.47 while the field accept rate falls 0.893 to 0.057, under 1.1% of clusters ever spanning the lattice.
From `python/snakes_and_ladders/search/ground_state.py` through `opt.budget.compare` at 2,083,260 site visits over 8 starts; `STATUS.md` carries the full tables. Actions: #551 closes, and the tilt's temperature dependence is ticketed.
