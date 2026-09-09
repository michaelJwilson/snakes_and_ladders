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

## Question

Does a softmax policy linear in the two features that span the reward, trained by REINFORCE with a running-mean baseline, reach the three-state chain's enumerated minimum energy from more of the 81 starts than greedy hill climbing at a matched decision budget?

## Numbers

| method, 6 decisions per episode | starts reaching the optimum | training seeds beating greedy |
| --- | --- | --- |
| greedy hill climbing | 65/81 (80.2%) | --- |
| REINFORCE, linear policy | 83.6%-87.1% (86.6% at seed 0) | 8/8 |

## Finding

The learned policy reaches the optimum from 86.6% of starts against greedy's 80.2%, in 8 of 8 training seeds; the reward decomposes exactly into the two features, so hill climbing is inside the policy class and this measures learning rather than two unrelated algorithms. From `pytest tests/regression/learn/test_learn_reinforce.py`. Actions #177, #313.
