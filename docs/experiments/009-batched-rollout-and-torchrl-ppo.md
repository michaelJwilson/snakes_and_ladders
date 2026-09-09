---
id: 009
date: 2026-09-09
commit: c7bbd67c94f6019be0edbe456d774742715e1b1e
branch: claude/learn-392-torchrl
pr: 0
tickets: [391, 392]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_chain/ci.yaml, tests/regression/fixtures/tree_search/ci.yaml, tests/regression/fixtures/tree_search/release.yaml
size: ci
methods: [sequential, gymnasium-vector, torchrl]
budget: 1920
seeds: [0]
hardware: Linux-x86_64, one thread, exclusive host lock
status: confirmed
---

# Do `gymnasium.vector` and TorchRL make the rollout and the PPO update faster?

## Question

Does `SyncVectorEnv` collect an episode below `learn.rollout.rollout`'s wall clock at any batch size, and do TorchRL's `GAE` and `ClipPPOLoss` compute `learn.ppo`'s two equations below its own, at the same learned result?

## Numbers

| term, one thread under `with_lock measure` | ours | framework |
| --- | --- | --- |
| ms per episode, Potts chain at batch 1 / 4 / 16 | 0.795 | 1.381 / 1.018 / 0.938 |
| ms per episode, 5-taxon tree at batch 1 / 4 / 16 | 0.929 | 1.570 / 1.499 / 1.639 |
| ms per episode, 7-taxon tree at batch 1 / 4 / 16 | 4.139 | 6.314 / 6.261 / 6.654 |
| generalized advantage, ms per 32-episode batch | 0.029 | 6.384 |
| clipped surrogate, ms per 32-episode batch | 1.467 | 0.775 |
| the 1,920-episode PPO budget, s (96.30% of 81 starts, return 2.2779, both loops) | 5.83 | 6.31 |

## Finding

Both declined: `SyncVectorEnv` is a serial loop with no parallelism to amortize its bookkeeping against and costs 1.18x-1.75x at every batch size, `GAE` is 220x the recursion it would front, and the 1.28x TorchRL's whole loop showed is our own hoisted neighbourhood scoring (1.13x) and batch-wide clipping, which land in `learn.ppo`; reproduced by `pytest tests/benchmarks/test_learn_ppo_bench.py tests/benchmarks/test_search_gym_bench.py`, with PR #439 carrying every column (#391, #392).
