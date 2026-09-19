---
id: "023"
date: 2026-09-19
commit: 473209dcfc9d477fdae0003fce8819d9aae75754
branch: claude/opt-756-tempering
pr: 0
tickets: [756]
problem: potts-lattice
fixture: tests/regression/fixtures/frustrated_lattice/ci.yaml, at 12x12 rather than the declared 3x3, beside a 16x16 open square at J_c = ln(1 + sqrt(2)) built in the test
size: stress
methods: [geometric, acceptance-placed, round-trip-placed]
budget: 0
seeds: [0, 1, 2, 3, 4, 5, 6, 7]
hardware: Linux-x86_64
status: confirmed
---

# Does placing a tempering ladder by its round trips beat placing it by its exchange acceptance?

## Question

Both criteria place the same number of rungs between the same endpoints: does the feedback-optimized placement (Katzgraber et al. 2006), which flattens the local diffusivity, buy a shorter round trip than the acceptance band does — and does either beat the geometric ladder they start from?

## Numbers

| round-trip time, recorded sweeps, 8 paired seeds | geometric | acceptance-placed | round-trip-placed |
| --- | --- | --- | --- |
| 16x16 open square at `J_c`, 10 rungs / 12 rungs | 1,690.3 | 1,194.4 | 1,053.7 (`p = 0.2891`) |
| 16x16, second warm-up seed, 13 rungs | 1,690.3 | 1,092.8 | 1,053.1 (`p = 1.0`) |
| 12x12 periodic triangular, 10 / 9 rungs | — | 360.1 | 366.6 (`p = 0.7266`) |
| 16x16, warm-up cut to 400 sweeps x 4 rounds | 1,690.3 | — | 1,469.5 and 5,357.1 |
| wall, 8 pairs (16x16 / 12x12) | — | 6.3 s / 3.5 s | 5.9 s / 3.5 s |

## Finding

**Neither criterion is established over the other, and the warm-up budget decides more than the criterion does.** At 1,000 warm-up sweeps over three rounds the round-trip-placed ladder is 1,053.7 sweeps per trip against the acceptance-placed 1,194.4 on the 16x16 at `J_c`, six seeds of eight shorter, `p = 0.2891`, and 1,053.1 against 1,092.8 at `p = 1.0` on a second warm-up seed; on the 12x12 triangular antiferromagnet 366.6 against 360.1 at `p = 0.7266` and 353.4 against 360.1 on the second reading, at 1-minute loads of 2.25 to 3.86. Both beat the geometric ladder they were placed from — 1,279.7 and 1,299.7 against 1,690.3 over two warm-up seeds, `p = 0.0078` and `0.2891` — so the placement is worth its warm-up, whichever criterion runs it. Cut the warm-up to 400 sweeps over four rounds and the feedback placement stops being reproducible: 1,469.5 at one warm-up seed and **5,357.1** at another, 3.2x *worse* than the geometric ladder at `p = 0.0078`, because `f` is then read from its own noise and the placement opens a gap at an end. Reproduced by `pytest tests/regression/search/test_search_tempered.py -m release -k placed`; no actions.
