---
id: "020"
date: 2026-09-19
commit: 048a342adf507ed45fd0831619accd66f2fafd45
branch: claude/opt-754-rust-ranking
pr: 0
tickets: [754]
problem: tree
fixture: tests/regression/fixtures/tree_search/
size: stress
methods: [profile, hoisted-post-order]
budget: 20
seeds: [0, 4]
hardware: Linux-x86_64
status: confirmed
---

# The stress-tier ranking: what does each family's hot path end in?

## Question

Which family's top term is a port, a default to flip, an algorithmic cut, or not ours?

## Numbers

| family | workload, `--tier mid` | wall | top self time | verdict |
| --- | --- | --- | --- | --- |
| trees | `infer` NNI / SPR, 20 taxa x 1,000 | 20.81 / 28.25 s | `run_backward` 50.5 / 53.8%, post-order 17.5 / 19.0% | not ours / cut, 1.05-1.09x realized |
| Potts | single-site 32x32 x 20; SW `propose` 64x64 | 0.289 s; 42.2 ms | `_site_update` 29.1%; `_recolour` 26.2% over 2,437 clusters | flip #599's default; port candidate |
| HMM/coupled | `message_passing` tree schedule, chain 200 | 0.237 s | `tree_passes` 29.8%, `_logsumexp_last` 10.6% | port candidate, per-level dispatch |
| codes | BCJR K=1,024; turbo, 8 iterations | 0.047 / 0.151 s | `bcjr` 95.8 / 95.5%, one Python trellis loop | port candidate, highest fraction measured |
| mixtures/HMC | `hmc.sample`, 1,000 draws, chain 64 | 24.57 s | `run_backward` 41.7%, `logsumexp` 33.2% over 512,000 calls | not ours; algorithmic cut first |
| learn | `reinforce`, 60 x 32 episodes, chain 8 | 5.60 s | `features` 13.1%, `policy.sample` 6.2% | below the bar, 0.73 s after #341's 1.32x |

## Finding

No family's top term is Rust's yet: two are Torch autograd, one is a default already measured, and the one loop at 95% of its run is the BCJR trellis. `python tests/benchmarks/profile_hotpaths.py --tier mid`, two readings, 1-minute load 0.06 and 1.00. Actions: `#754`.
