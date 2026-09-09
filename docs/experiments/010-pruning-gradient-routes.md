---
id: "010"
date: 2026-09-09
commit: d5c6cc8e7efb1baaf8fcc26f5ac36d1b3b4168af
branch: claude/likelihood-449-autodiff
pr: 0
tickets: [449, 443]
problem: tree
fixture: tests/regression/fixtures/tree_jc/ci.yaml and tests/regression/fixtures/tree_jc/release.yaml, both at 20,000 sites
size: ci
methods: [taped, burn, analytic]
budget: 60
seeds: [20260903, 20260904]
hardware: Linux-x86_64
status: confirmed
---

# Does a Rust autodiff framework, or a closed-form backward, beat PyTorch's tape on the pruning gradient?

## Question

Does `burn`'s taped gradient in Rust (A), or a two-pass analytic backward behind a `torch.autograd.Function` (B), compute the pruning gradient below PyTorch's own tape, at the same value?

## Numbers

| 8 taxa by 20,000 sites unless stated, median (IQR) | taped (oracle) | A: `burn` | B: analytic |
| --- | --- | --- | --- |
| one gradient, ms, 4 taxa / 8 taxa | 9.82 (1.11) / 33.06 (3.56) | 14.24 (2.57) / 46.80 (3.86) | 5.68 (0.74) / 16.36 (1.37) |
| one L-BFGS fit, ms, all three reaching 148940.229159 | 694.5 (52.3) | 1179.1 (144.1) | 454.2 (38.7) |
| graph nodes per evaluation, 4 / 8 / 16 taxa | 59 / 115 / 227 | 66 / 134 / 270 | 2 / 2 / 2 |
| A's kernel alone (Criterion) against its PyO3 call | — | 48.01 [46.94, 49.27] against 47.06 (6.09) | — |
| forward, ms: graph built, under `no_grad`, B's, NumPy | 12.38 (1.78) / 8.10 (0.65) | — | 10.96 (0.96), against the oracle's 14.96 (1.12) |

## Finding

A is declined at 1.42x per gradient and 1.70x per fit, and the FFI boundary is not the reason --- its kernel alone is already 1.45x PyTorch's whole evaluation, while its `f64` path agrees with the taped gradient to 5.9e-13; B is adopted at 50.5% of an evaluation and 34.6% of a fit, half of that won in the forward, since a `torch.autograd.Function` does not build the graph (#449, #443); reproduced by `pytest tests/benchmarks/test_pruning_gradient_bench.py`, with PR #453 carrying every column.
