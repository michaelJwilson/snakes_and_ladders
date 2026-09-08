---
id: 004
date: 2026-09-08
commit: b71b0b17d7b93458066db513e026a75d87201698
branch: claude/search-328-tree-features
pr: 0
tickets: [328, 178, 177]
problem: tree
fixture: tests/regression/fixtures/simulation_params_hard.yaml, the 7-taxon hard fixture of #177
size: release
methods: [greedy, reinforce-improvement, reinforce-full]
budget: 640
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
hardware: Linux-x86_64
status: confirmed
---

# Does a feature set beyond the improvement let a tree policy beat hill climbing?

## Feature under test

`FeatureSet.FULL` — per move, the improvement, the Fitch parsimony change,
the pattern support of the split broken and of the split made, and the sizes
of the two exchanged subtrees, standardized within the neighbourhood — lets a
linear softmax policy trained by REINFORCE reach the enumerated maximum
likelihood from more of the 50 starts than the single improvement feature
(#178's policy, which is an inverse temperature) and than greedy hill
climbing, at #178's budget.

## Setup

The hard 7-taxon fixture of #177, where NNI hill climbing reaches the
enumerated maximum from 24 of 50 starts (0.480). Budget per training seed:
640 episodes (40 iterations of 16), 30 decisions per episode, 16 rollouts
per start at evaluation over the same 50 starts; 16 training seeds shared
across the two feature sets; the enumerated maximum over all 945 topologies
as the referee. The paired test is the exact two-sided sign test over the 16
seeds (`search.statistics.sign_test_p_value`). The per-pull-request tier runs
one seed at half the budget (`test_the_full_set_is_ahead_of_the_single_feature_at_the_ci_budget`).

## Results

Typed from the release run of
`tests/regression/search/test_search_tree_policy.py::test_the_full_set_against_the_single_feature_over_sixteen_seeds`.

| method | mean rate over 16 seeds | range | ahead of the single feature | sign test |
| --- | --- | --- | --- | --- |
| greedy hill climbing | 0.480 | — | — | — |
| REINFORCE, improvement only | 0.487 | 0.465–0.535 | — | vs greedy p = 0.79 |
| REINFORCE, full set | 0.796 | 0.695–0.849 | 16 of 16 seeds | vs single p = 3.05e-5 |

Per-pull-request tier (seed 0, 320 episodes, 4 rollouts per start): greedy
0.480, untrained policy 0.02, improvement only 0.435, full set 0.595.

## Figures

none

## Finding

With the single feature the trained policy ties greedy (0.487 against 0.480,
p = 0.79), as #178 measured. With the full set it reaches the maximum from
0.796 of episodes, ahead of the single feature on every one of 16 training
seeds (p = 3.05e-5) and 0.316 above greedy: the fixture that separated
nothing from restarts separates the feature sets.

## Conclusion and actions

- #328: closed by this measurement.
- #313: the actor–critic, PPO and planner comparisons on the tree fixture
  re-run with `FeatureSet.FULL`, which was the named prerequisite.
- #329: the plug-in and parsimony bounds of #308 as further columns once
  their PR lands.

## What is not claimed

Not that the full set beats random-restart hill climbing at the same decision
budget (#194 measured restarts reaching the maximum from every start on this
fixture at 60 decisions; that comparison is #313's budget-matched harness).
Not that the result transfers to SPR or to larger fixtures; both are
unmeasured. Not that any column beyond the improvement is individually
necessary; no ablation was run.
