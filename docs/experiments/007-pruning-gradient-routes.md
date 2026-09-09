---
id: 007
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

Does `burn`'s `Autodiff<NdArray<f64>>` over the recursion in Rust (A), or the analytic two-pass backward of `alg:pruning-backward` behind one `torch.autograd.Function` (B), cost less than the taped gradient --- or does A, as issue #449 predicted before the measurement, rebuild the tape B removes?

## Numbers

| 8 taxa, 20,000 sites, except where the row says otherwise | taped (oracle) | A: `burn` | B: analytic |
| --- | --- | --- | --- |
| one gradient, ms, median (IQR) over 60 rounds | 33.06 (3.56) | 46.80 (3.86) | **16.36 (1.37)** |
| one L-BFGS fit, ms, median (IQR) over 11 rounds | 694.5 (52.3) | 1179.1 (144.1) | **454.2 (38.7)** |
| autograd graph nodes, 4 / 8 / 16 taxa | 59 / 115 / 227 | 66 / 134 / 270 | 2 / 2 / 2 |
| worst relative deviation from the taped `float64` gradient | --- | 5.9e-13 | 2.1e-13 |

## Finding

The prediction held: A's tape is 12 to 19% larger than the one it replaces, so it relocated the cost --- 1.42x per gradient and 1.70x per fit --- and the boundary is not the reason, since Criterion times its kernel alone at 48.01 ms [46.94, 49.27] against 47.06 (6.09) for the same call from Python, inside the spread.
B removes the term instead, cutting a gradient by 50.5% and a fit by 34.6%, half of it in the forward pass a `Function` does not tape (12.38 ms taped against 8.10 under `no_grad`); precision decided nothing, A's `f64` being sound.
From `pytest tests/regression/likelihood/test_pruning_gradient.py tests/regression/likelihood/test_pruning_analytic.py` and `tests/benchmarks/test_pruning_gradient_bench.py`; #449 adopts B as `likelihood.pruning_analytic` and conserves A behind the `sandbox` Cargo feature, and switching the fitted path to B is #443's.
