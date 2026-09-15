---
id: "018"
date: 2026-09-15
commit: 1b63d1f8d60f6981efeacb71055261bed056f414
branch: claude/likelihood-592-schedule
pr: 0
tickets: [592]
problem: tree
fixture: tests/regression/fixtures/tree_jc/
size: ci
methods: [greedy, candidate]
budget: 0
seeds: [3]
hardware: Linux-x86_64
status: confirmed
---

# What does the exact schedule buy, and what does building it cost?

## Question

Is the tree schedule's plan worth a fifth of the run, and is half of it worth half the answer?

## Numbers

| median | tree | upward | ratio |
| --- | ---: | ---: | ---: |
| `sum_product`, 1,000-variable chain | 146.4 ms | 85.8 ms | 1.71x |
| plan build, 2,000-variable chain, before / after | 64.2 ms | 52.9 ms | 1.21x |
| plan share of the run, 2,000-variable chain | 24.9% | 20.8% | --- |
| `log Z` against the two-pass schedule | --- | 1 ulp | --- |
| `log Z` against `likelihood.pruning`, 7 sites | --- | 1e-13 rel | --- |

## Finding

The upward pass gives all of `log Z` for 1.71x less work, and batching the group arrays recovers 1.21x of a plan whose residue is Python bookkeeping with no hotspot left: `pytest tests/benchmarks/test_schedule_bench.py`. Actions: `no actions`.
