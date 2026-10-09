---
id: "039"
date: 2026-10-09
commit: 4d50f3a2bcdd6a01a454b5aa1825f2e223f66db2
branch: claude/hmm-fit-study-1412
pr: 0
tickets: [1412, 1410]
problem: hmm-path
fixture: tests/regression/fixtures/count_hmm_reference/ci.yaml
size: ci
methods: [sal, downstream, downstream-reference]
budget: 500
seeds: [1390, 1412]
hardware: Linux-x86_64
status: confirmed
---

# Does sal's Baum-Welch reproduce a downstream caller's count-pair HMM fit, and which differences decide recovery?

## Question

On one start, does `baum_welch_family` under sal's semantics (per-state dispersion, transition fitted) or a downstream caller's (shared dispersion and concentration, transition held) reach the downstream package's own fit, and at what cost per iteration?

## Numbers

| instance | sal: log L / missed / ms per it | downstream semantics: log L / missed / ms per it | downstream package's own fit |
| --- | --- | --- | --- |
| `count_hmm_reference/ci` | -7615.48 / 123 of 1,000 / 8.4 (87 it) | -7621.95 / 40 / 256.6 (115 it) | not run |
| `stress`, variant `downstream`, 60 it | -100472.81 / 0 of 10,240 / 36.9 | -100474.13 / 0 / 1495.1 (M step 98.8%) | not run |
| the downstream package's shipped easy instance, 4,688 positions | -43062.95 / 1,314 of 4,688 / 40 (69 it) | not run | -46426.10 / 3,076 / 7.45 s warm, 18.3 s first call; patched 0.51 s |
| a downstream stream instance, 10,496 positions | -98372.71 / 7,049 / 101 (65 it) | 7.5-10.7 s per it, not run to tolerance | -107076.40 / 1,386 / 10.6 s; patched 1.65 s; re-entered 40 times -105882.74 / 6,988 |

## Finding

Per-state dispersion fits higher on all four; it misses more on two (40 to 123, 1,386 to 7,049), fewer on the easy instance (3,076 to 1,314), and ties on one. The stream fit's recovery rests on its early stop: re-entered to an 8e-8 relative change it misses 6,988; the tied M step is 30-40x the untied one's cost. `python -m sal.qa.hmm_fit_semantics ci`; gaps on #1412.
