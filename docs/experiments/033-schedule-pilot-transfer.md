---
id: "033"
date: 2026-10-08
commit: fc7e2168f3581f4a476e7177f99c57e7f92efa1f
branch: claude/pilot-transfer-1390q23
pr: 0
tickets: [1390, 1337]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_reference/stress.yaml, with ci.yaml and the stress variants margin_0.1, margin_3 and q8
size: stress
methods: [pilot-1/16, pilot-1/4, pilot-1, racing, common, racing+common]
budget: 0
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
hardware: Linux-x86_64
status: confirmed
---

# Does a short schedule pilot rank SCHEDULE_GRID as the full run does, and do racing or common random numbers pick the full-length best more often per pilot visit?

## Question

On `potts_reference` (SW + heat bath, 256-step run, ICM polish, `POLISHED_GAP`), is Kendall's tau between pilot and full-length rankings near 1 at 1/16 and 1/4 of the run, and does any racing x common combination raise P(full-length best) per pilot site visit by >= 2x at stress size?

## Numbers

| instance (reference: 12 candidates x 32 held-out seeds) | tau, f = 1/16 | tau, f = 1/4 | tau, f = 1 | split-half tau of the reference | best per-visit gain over both off, f = 1/16 / 1/4 |
| --- | --- | --- | --- | --- | --- |
| ci, 300 sites (grid spread 0.44 nats) | 0.17 (0.07) | 0.18 (0.07) | 0.12 (0.07) | 0.55 | none, <= 0.50x / racing 2.00x (P 0.08 to 0.12) |
| stress, 3,000 sites (spread 1.47) | 0.12 (0.04) | 0.22 (0.04) | 0.22 (0.04) | 0.45 | common 5.00x (P 0.02 to 0.10) / common 1.75x |
| stress margin 0.1 (spread 48.0) | 0.32 (0.05) | 0.21 (0.04) | 0.42 (0.04) | 0.85 | racing 1.52x / racing+common 1.60x |
| stress q 8 (spread 3.00) | 0.21 (0.05) | 0.62 (0.03) | 0.68 (0.02) | 0.76 | none, <= 0.84x / racing 1.11x |
| pilots' share of pilot + run visits; racing's spend | 43% | 75% | 92% | --- | racing 0.75x of the grid's, every instance |

## Finding

**Pilot transfer is bounded by the reference's own noise, and no combination earns a default flip.** Tau (mean over 16 tuning seeds, se in brackets) at 1/4 matches the run's own length within 2 se on ci, stress and q 8 (0.62 against 0.68) but not at margin 0.1 (0.21 against 0.42), and 1/16 loses most of it on q 8 (0.21); split-half tau of the 32-seed reference is 0.45 to 0.85, so the grid is near-flat after the polish except at margin 0.1, and at margin 3 every candidate polishes to the same energy (tau undefined, P(best) 1.00). Q3 at 48 seeds per combination: the one >= 2x at stress is common at 1/16 (1 to 5 of 48 seeds), not repeated at 1/4 or on any variant (0.43x, 0.84x), and racing's gain is its 0.75x spend where P is unchanged; mean regret moves by at most 2.1 nats on a 48-nat spread. Reproduced by `python -m sal.qa.schedule_pilots` (381 s, 2 processes); the CI-size pin is `tests/regression/qa/test_qa_schedule_pilots.py`; no actions.
