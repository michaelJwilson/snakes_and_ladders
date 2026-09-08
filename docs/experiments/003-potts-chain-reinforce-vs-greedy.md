---
id: 003
date: 2026-09-03
commit: 59efe4ec1d23cf90121d8af9a7485bb1e4bd7cda
branch: main
pr: 135
tickets: [131]
problem: potts-lattice
fixture: tests/regression/fixtures/potts_chain/ci.yaml, the three-state chain of length four
size: ci
methods: [greedy, reinforce]
budget: 1920
seeds: [0, 1, 2, 3, 4, 5, 6, 7]
hardware: Linux-x86_64
status: confirmed
---

# Does a policy trained by REINFORCE reach the enumerated optimum from more starts than hill climbing?

## Feature under test

A softmax policy linear in the two features that span the reward, trained by
REINFORCE with a running-mean baseline, reaches the enumerated minimum energy
from a larger fraction of the 81 starting configurations than greedy hill
climbing at a matched decision budget.

## Setup

The three-state Potts chain of length four at known parameters; every one of
the `3^4 = 81` configurations is a start, and the optimum is enumerated. The
policy trains for 60 iterations of 32 episodes (1,920 episodes) with an
episode horizon of 6 decisions, in 8 training seeds; greedy hill climbing runs
from the same 81 starts at the same horizon. Both evaluate every action in the
neighbourhood per decision, so the per-decision cost is matched by
construction.

## Results

From `tests/regression/learn/test_learn_reinforce.py` at the commit above:

| method | starts reaching the optimum | training seeds beating greedy |
| --- | --- | --- |
| greedy hill climbing | 65/81 (80.2%) | — |
| REINFORCE, linear policy | 83.6%–87.1% (86.6% at seed 0) | 8/8 |

## Figures

none

## Finding

The learned policy reaches the enumerated optimum from 86.6% of the 81 starts
against greedy's 80.2%, in 8 of 8 training seeds. The reward decomposes exactly
into the two features, so hill climbing is inside the policy class as the
weight vector proportional to `(J, 1)`, and the comparison is a statement about
learning rather than about two unrelated algorithms.

## Conclusion and actions

#177 — a tree fixture hard enough to separate a policy from greedy, since the
chain's margin is six starts.
#313 — a critic, actor–critic and PPO measured at this budget, where PPO
reaches the optimum from 96.3% of the starts.

## What is not claimed

Nothing about the tree environment, where greedy already reaches the optimum
from every start on the available fixtures; nothing about budgets other than
1,920 episodes, where the ordering can differ.
