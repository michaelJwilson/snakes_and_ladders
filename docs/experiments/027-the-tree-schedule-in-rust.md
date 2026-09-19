---
id: "027"
date: 2026-09-19
commit: acd175f1a230f3aa06c76ee31753062e54444efa
branch: claude/tree-754-schedule
pr: 0
tickets: [754]
problem: hmm-path
fixture: tests/regression/fixtures/hmm/
size: stress
methods: [numpy, stored-plan, rust]
budget: 0
seeds: [0]
hardware: Linux-x86_64
status: confirmed
---

# The tree schedule in Rust: does the per-level dispatch clear the 2x bar, and does the cut alone?

## Question

Is the ranking's 29.8% a plan to store or a pass to port, and what does either save of the `sum_product` that encloses it?

## Numbers

| chain of 200, four states | NumPy | stored plan | Rust |
| --- | --- | --- | --- |
| `sum_product`, `pytest-benchmark` mean | 24.310 / 23.778 ms | --- | 2.687 / 2.563 ms (**9.05x / 9.28x**) |
| `max_product`, same | 21.171 / 18.598 ms | --- | 2.763 / 2.723 ms (7.66x / 6.83x) |
| `sum_product`, same process, min of five | 21.003 / 20.544 ms | 17.123 / 16.663 ms (1.23x) | 2.572 / 2.482 ms (8.17x / 8.28x) |
| the same at a chain of 2,000 | 213.3 / 212.7 ms | 163.0 / 163.6 ms (1.31x / 1.30x) | 21.14 / 20.95 ms (10.1x) |
| `--tier mid` self time, x5 | 0.245 s, `_batched_variable_sends` 29.2% | --- | 0.031 s, `is_tree` 11.4% |

## Finding

A pass to port, by 4x the bar: **9.05x** and **21.6 ms of 24.3** at the chain of 200, where the stored plan alone is 1.23x and is subsumed --- the kernel walks the breadth-first order and builds no plan. The profiled section falls 7.9x and its top term is now the tree check, not the dispatch; the general algorithm is 2.11x the Torch forward recursion where #341 left it at 4.7x behind. Agreement is 3.3e-15 on the marginals and 1.4e-14 on `log Z`, inside `CROSS_DEVICE_RTOL_FLOAT64`, with the assignment equal: NumPy's vectorized `exp` against `libm`'s. `pytest tests/benchmarks/test_message_passing_bench.py --benchmark-columns=mean`, two readings, 1-minute load 3.05 and 3.51. Actions: `no actions`.
