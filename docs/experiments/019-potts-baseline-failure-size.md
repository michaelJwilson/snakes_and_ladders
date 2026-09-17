---
id: "019"
date: 2026-09-17
commit: 2f75161570a6759ba3b017c478f456b041f7ac51
branch: claude/learn-596-potts-sizing
pr: 0
tickets: [596]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_lattice/stress.yaml
size: stress
methods: [icm, gibbs-T0, anneal, wolff, tempering, alpha-expansion]
budget: 200
seeds: [596]
hardware: Linux-x86_64
status: confirmed
---

# Where does the Potts baseline first fail, and which baseline is it?

## Question

At the critical coupling, is there a size where the classical baseline stops reaching the best known energy — and if so, which method is the baseline?

## Numbers

| excess over alpha-expansion, best of 8 starts | 144 sites | 576 sites | 576 at 25x budget |
| --- | ---: | ---: | ---: |
| ICM, single descent | 5.31% | 10.59% | 10.59% |
| random-restart ICM, matched budget | 3.06% | 6.09% | 3.69% |
| Wolff cluster | 3.38% | 34.35% | 8.74% |
| parallel tempering | 0.96% | 3.02% | 0.00% |
| simulated annealing | 0.04% | 0.00% | 0.00% |
| alpha-expansion vs the exact cut, q = 2, to 1,024 sites | 0.0 | 0.0 | --- |

## Finding

The baseline is annealing, not ICM: it reaches the expansion's energy at every size while single-site descent misses by 5–11% and is **flat in the budget**, so its failure is structural and restarts close only half of it; Wolff degrades with size in a field, 3.4% to 34.4%, being blind to the unary term. Zero field and q = 2 cannot host the gate at all: restart-ICM and Wolff reach the closed form `-J|E|` from every start to 1,024 sites, and the expansion equals the exact cut. Reproduce: `pytest tests/regression/search/test_potts_sizing.py -m 'not release'`, then `-m release`. Actions: `#596` plan revision — the gate is argued against annealing under a matched budget, and an exact-hit fraction is the wrong instrument for a continuous energy.
