---
id: 007
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

## Feature under test

Two replacements from #376's frameworks table. `gymnasium.vector.SyncVectorEnv`
over `search.gym.GymnasiumEnvironment` collects episodes at a lower wall clock
per episode than `learn.rollout.rollout` does one at a time; TorchRL's `GAE`
and `ClipPPOLoss` compute `learn.ppo`'s two equations at a lower wall clock
than its own, at the same learned result.

## Setup

The three-state Potts chain of length four and the 5- and 7-taxon tree
fixtures, at a horizon of 6 decisions. The rollout comparison collects 128
episodes on the chain and 32 on each tree, at batch sizes 1, 4 and 16, against
the same count rolled sequentially under the same generator; the reported
number is the best of five runs (three on the trees). The PPO comparison is one
training run at the CI budget of 1,920 episodes (60 iterations of 32) on the
chain, best of three, scored by the enumerated optimum reached from all 81
starts and by the exact expected return, both of which enumeration referees.
Every timing was taken under the host's exclusive lock at one thread.

`tests/regression/search/test_search_gym_vector.py` referees the batched path:
at one copy it equals the sequential rollout draw for draw.
`tests/regression/learn/test_learn_ppo_torchrl.py` referees the two equations
against TorchRL's at 1e-10 and needed no code to move to do it.

## Results

Rollout, wall clock per episode:

| fixture | sequential | batch 1 | batch 4 | batch 16 |
| --- | --- | --- | --- | --- |
| Potts chain | 0.795 ms | 1.381 ms | 1.018 ms | 0.938 ms |
| tree, 5 taxa | 0.929 ms | 1.570 ms | 1.499 ms | 1.639 ms |
| tree, 7 taxa | 4.139 ms | 6.314 ms | 6.261 ms | 6.654 ms |

PPO, one training run of 1,920 episodes:

| loop | wall clock | starts reaching the optimum | exact expected return |
| --- | --- | --- | --- |
| `learn.ppo` as it stood | 8.06 s | 78/81 (96.30%) | 2.2779 |
| neighbourhoods scored once per iteration | 7.14 s | 78/81 (96.30%) | 2.2779 |
| TorchRL `GAE` + `ClipPPOLoss` | 6.31 s | 78/81 (96.30%) | 2.2779 |
| both changes, as landed | 5.83 s | 78/81 (96.30%) | 2.2779 |

The sampled return agreed iteration for iteration across every loop above: the
largest gap over the 60 iterations was 0.0.

The two equations timed alone on one 32-episode batch (164 decisions):

| term | ours | TorchRL |
| --- | --- | --- |
| generalized advantage | 0.029 ms | 6.384 ms, tensordicts prebuilt; 8.228 ms including their construction |
| clipped surrogate | 1.467 ms | 0.775 ms |

The clipped surrogate is 1.467 ms of a 175 ms PPO iteration, 0.8% --- too
small a term for a port in any language to pay for (Gorelick & Ozsvald ch. 2),
which is the first question `CLAUDE.md` puts to a proposed hot path.

## Figures

none

## Finding

`SyncVectorEnv` costs 1.18x to 1.75x per episode against the sequential
rollout at every batch size measured, on every fixture, and the cost does not
fall with the batch. It is a serial loop in one process, so it evaluates the same
environments in the same order and adds the vector API's observation stacking,
autoreset bookkeeping and the pad-to-`n_max` round trip through NumPy on top.
No batch size makes it pay, because there is no parallelism in it to amortize
the overhead against.

TorchRL's `GAE` is 220x slower than the recursion it would replace, 6.384 ms
against 0.029 ms per batch: the formula is 12 lines of Python over lists and
the framework's is a TensorDict rebuilt per episode.

Of the 1.28x TorchRL's whole loop showed, 1.13x is the neighbourhood scoring
hoisted out of the epoch loop and the rest is the clipped surrogate applied to
the concatenated batch rather than episode by episode. Both are changes to
`learn.ppo` itself; taking them leaves TorchRL behind, at 5.83 s against its
6.31 s.

## Conclusion and actions

#392 — the batched rollout lands as `search.gym.rollout_batch`, refereed and
not adopted anywhere: no training loop calls it, because it is slower than what
they call now. A `SyncVectorEnv` cannot deliver Milestone 2.2's "a budget at
`n = 200` is affordable"; only a collector with real parallelism could, which
is `AsyncVectorEnv` or a process pool and is its own measurement.

#391 — neither `generalized_advantages` nor `ppo_loss` moves to `sandbox/`, and
TorchRL stays what it already was here: the second implementation that referees
ours at 1e-10. The two improvements the comparison exposed land in `learn.ppo`.

## What is not claimed

Nothing about `AsyncVectorEnv` or any multi-process collector, which were not
run: the host is four cores shared with other agents and every baseline in
`STATUS.md` is a one-thread number. Nothing about batch sizes above 16, or
about environments whose step is expensive enough for the vector API's overhead
to vanish beside it — the 7-taxon fixture, the most expensive here at 4.139 ms
per episode sequentially, is still 1.51x slower batched. Nothing about TorchRL on a
GPU, where its vectorized loss would be timed against a different reference.
