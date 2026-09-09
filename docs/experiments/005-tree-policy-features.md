---
id: 005
date: 2026-09-08
commit: b71b0b17d7b93458066db513e026a75d87201698
branch: claude/search-328-tree-features
pr: 0
tickets: [328, 178, 177]
problem: tree
fixture: tests/regression/fixtures/tree_search/release.yaml, the 7-taxon hard fixture of #177
size: release
methods: [greedy, reinforce-improvement, reinforce-full]
budget: 640
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
hardware: Linux-x86_64
status: confirmed
---

# Does a feature set beyond the improvement let a tree policy beat hill climbing?

## Question

Does `FeatureSet.FULL`, seven columns per move against #178's single improvement, let a linear softmax policy trained by REINFORCE reach the hard 7-taxon fixture's enumerated maximum likelihood from more of 50 starts than that policy and than greedy hill climbing, at #178's 640 episodes?

## Numbers

| method, 16 training seeds | mean rate | range | ahead of the single feature | sign test |
| --- | --- | --- | --- | --- |
| greedy hill climbing | 0.480 | --- | --- | --- |
| REINFORCE, improvement only | 0.487 | 0.465-0.535 | --- | vs greedy p = 0.79 |
| REINFORCE, full set | 0.796 | 0.695-0.849 | 16 of 16 seeds | vs single p = 3.05e-5 |

## Finding

The single feature ties greedy (0.487 against 0.480) and the full set reaches the maximum from 0.796 of episodes, 0.316 above greedy and ahead on every seed: the fixture that separated nothing from restarts separates the feature sets. From `pytest tests/regression/search/test_search_tree_features.py -m release`; closes #328. Actions #313, #329.
