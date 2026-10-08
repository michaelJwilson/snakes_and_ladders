---
id: "035"
date: 2026-10-08
commit: fc7e2168f3581f4a476e7177f99c57e7f92efa1f
branch: claude/budget-split-1390q5
pr: 0
tickets: [1390, 1392]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_reference/stress.yaml, with the stress variants margin_0.1 and q8
size: stress
methods: [long, restart-4, restart-16, tempering-auto, wolff-hb-alone, wolff-gibbs, sw-hb-alone, sw-gibbs, glauber, field-bond-wolff]
budget: 0
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
hardware: Linux-x86_64
status: confirmed
---

# At one site-visit budget, does one long anneal, k restarts or cluster tempering end nearest the TRW-S bound, and which move mix does?

## Question

At x1, x4 and x16 alpha-expansion+ICM's visits (polish `ICM_MERGE` counted in the budget), does any split beat one long anneal by more than its standard error, and does heat-bath Wolff alone trail Wolff + Gibbs and Glauber at equal visits?

## Numbers

| mean gap to the TRW-S bound, nats, x1 / x4 / x16 (16 seeds) | long | restart-4 | restart-16 | tempering, auto ladder |
| --- | --- | --- | --- | --- |
| stress, B = 273,000 visits x level | 14.51 / 2.81 / 1.77 | 29.76 / 9.69 / 1.66 | 22.87 (9.6 B) / 22.87 (2.4 B) / 6.37 | 37.46 / 51.97 / 41.73 |
| q8, B = 525,000 x level | 2.80 / 0.75 / 0.53 | 12.21 (1.3 B) / 1.67 / 0.50 | 12.19 (5.0 B) / 10.46 (1.3 B) / 0.91 | 21.63 / 31.16 / 32.83 |

| median gap, unpolished -> ICM_MERGE, stress (8 seeds) | Wolff-HB alone | Wolff + Gibbs | SW-HB alone | SW + Gibbs | Glauber |
| --- | --- | --- | --- | --- | --- |
| x16, 208 sweeps' visits | 3396.70 -> 24.75 | 0.98 -> 0.94 | 262.54 -> 16.81 | 2.25 -> 2.14 | 1.53 -> 1.40 |
| 4,000 sweeps' visits | 917.51 -> 23.37 | 0.52 -> 0.45 | 61.84 -> 10.24 | 0.34 -> 0.33 | 0.31 -> 0.30 |

## Finding

**One long anneal is not beaten:** restart-4 matches it at x16 (1.66 +- 0.30 against 1.77 +- 0.37 on stress, 0.50 against 0.53 on q8) and trails at x1 and x4; restart-16 polishes 16 times and overruns B up to 9.6x where marked; cluster tempering on the auto ladder (5 to 6 rungs, pilot 7.6 to 8.4 M visits apart) ends 21 to 52 nats off at every level; on margin_0.1 every arm polishes to the bound, the polish costing 2.5 B at x1. Polish outside B moves no conclusion (stress long 10.06 / 2.68 / 1.81). Experiment 029's 917.51 and 0.52 reproduce to its two decimals; Wolff + Gibbs leads Glauber at x16 and trails it by 0.15 nats at 4,000 sweeps, inside one standard error (0.44 +- 0.13 against 0.36 +- 0.11). The field-aware bond `1 - exp(-beta (J - dh)_+)` breaks detailed balance on an enumerated 4-cycle (flux gap 8.7e-3 against 1.0e-17 for the J-only control), so it was declined unmeasured (`sal.sandbox.field_bond_wolff`). `python -m sal.qa.budget_split`, 139 s on 2 threads; CI pin `tests/regression/qa/test_qa_budget_split.py`. Actions: no actions.
