---
id: "024"
date: 2026-09-19
commit: bbebc5966d90acc6bb5a54db19673d88e6aa9775
branch: claude/codes-754-bcjr
pr: 0
tickets: [754]
problem: turbo
fixture: tests/regression/fixtures/turbo/stress.yaml
size: stress
methods: [numpy, rust]
budget: 0
seeds: [233, 9]
hardware: Linux-x86_64
status: confirmed
---

# The BCJR trellis in Rust: does the ranking's top term clear the 2x bar?

## Question

Does porting `bcjr` earn 2x over the NumPy reference at the declared `K`, and what does it save of the decode that encloses it?

## Numbers

| `pytest-benchmark` mean, two readings | NumPy | Rust | ratio |
| --- | --- | --- | --- |
| `bcjr`, `K = 256` (`turbo/stress.yaml`) | 2.319 / 2.311 ms | 66.6 / 66.1 us | 34.8x / 35.0x |
| `bcjr`, `K = 1,024` | 9.023 / 8.873 ms | 275.1 / 272.7 us | 32.8x / 32.5x |
| `decode_turbo`, `K = 256`, 8 iterations | 37.00 / 37.19 ms | 1.410 / 1.412 ms | 26.2x / 26.3x |
| `decode_turbo`, `K = 1,024`, 8 iterations | 143.5 / 141.6 ms | 4.778 / 4.737 ms | 30.0x / 29.9x |
| stress waterfall, 6 points x 50 frames | 11.37 s | 0.46 s | 24.7x |
| `--tier mid` self time, `bcjr` x5 / turbo | 47.0 / 151 ms | 3 / 5 ms | top term now the extension call, 53.5% / 84.2% |

## Finding

Yes, by 13x the bar: 26.2x on the enclosing eight-iteration decode at the declared `K = 256`, saving **35.6 ms of 37.0**, so the default flips and the release waterfall falls from 182 s to 6.0 s. Agreement is **bitwise** at memory 1 and 2 --- every declared register --- and 2.3e-13 absolute, 2.2e-12 relative at 3 and 4, where NumPy's pairwise sum over states reassociates and this kernel does not. `pytest tests/benchmarks/test_turbo_bench.py --benchmark-columns=mean`, two readings, 1-minute load 1.40 and 1.79 with two other agents on the host. Actions: `no actions`.
