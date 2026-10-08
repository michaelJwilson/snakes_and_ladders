---
id: "032"
date: 2026-10-08
commit: 0a46e429d9941c61a1b1ab1d93a70958f4992019
branch: claude/temperature-scale-1390q1
pr: 0
tickets: [1390]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_reference/stress.yaml
size: stress
methods: [tuned-ladder, c-absolute, c-coupling, c-margin, c-proposal]
budget: 4200000
seeds: [0, 1, 2, 100, 101, 102, 103, 104, 105, 106, 107]
hardware: Linux-x86_64
status: confirmed
---

# A start temperature in the instance's units: does T0 / s tuned on stress transfer?

## Question

Is `T0 / s` near-constant across `potts_reference`'s margin, q, graph and n variants, so that `c * s` from `stress` alone matches a per-variant tuned `T0` at equal visits?

## Numbers

| scale `s` (8 variants; `margin_3` flat at gap 0.00) | `T0 / s` spread, max/min | `c` from stress | held-out gap sum, `T0 = c * s`, nats |
| --- | --- | --- | --- |
| none: absolute `T0` | 4.0 | 2.0 | 57.62 |
| `J d`, coupling times mean degree | 4.0 | 0.333 | 57.94 |
| median top-two field margin | 20.3 | 0.668 | 92.76 |
| median `\|dE\|` of a uniform single-site proposal at a random start | 7.6 | 0.564 | 66.54 |

## Finding

`J d` transfers within the evidence: 57.94 nats against 54.72 for a per-variant tuned `T0` (seeds 0-2 racing `POLISHED_GAP`, ~12 runs' visits each) and 57.62 for absolute `T0 = 2`, the 3.2-nat excess inside `margin_0.1`'s standard error of 4.1; the margin scale does not (+38.0 nats, all `margin_0.1`). `J = 1` in every variant, so `J d` differs from absolute `T0` only by degree (4, 6, 7.06): the evidence cannot separate them. Gap is polished energy less the TRW-S bound, 8 held-out seeds, 1,400 visits per site, single-site heat bath then ICM; `python -m sal.qa.temperature_scale`, 162 s on 2 threads. Actions: `no actions`.
