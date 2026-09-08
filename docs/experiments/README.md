# docs/experiments/

One file per experiment, written from [`TEMPLATE.md`](TEMPLATE.md): the commit,
the feature under test, the fixture and its size tier, the methods compared at
one budget over shared seeds, the results, the finding, and the actions it
filed (issue #314). `STATUS.md` cites an experiment rather than restating its
table, and `tests/regression/test_experiments.py` holds every file to the
template and this index to the files.

**This index is generated.** Rewrite it with `python infra/experiments.py`;
`--check` fails when a file is invalid or the index is stale.

| id | problem | size | methods | budget | status | finding |
| --- | --- | --- | --- | --- | --- | --- |
| [1](001-potts-cluster-autocorrelation.md) | potts-lattice | stress | single-site, swendsen-wang, wolff | 0 | confirmed | Single-site slows by 3.2× between extent 8 and 24 while both cluster algorithms slow by roughly 1.9×, so the gap is 2.1× at extent 24 and widening. The separ... |
| [2](002-hmm-gaussian-interval-coverage.md) | hmm-path | ci | gradient-fit-wald-intervals | 24 | confirmed | Coverage reaches nominal from two standard deviations of separation upward and degrades below it, seen twice over: the intervals that exist under-cover, and... |
| [3](003-potts-chain-reinforce-vs-greedy.md) | potts-lattice | ci | greedy, reinforce | 1920 | confirmed | The learned policy reaches the enumerated optimum from 86.6% of the 81 starts against greedy's 80.2%, in 8 of 8 training seeds. The reward decomposes exactly... |
| [4](004-mixture-tempering-vs-restarts.md) | mixture | release | restarts, anneal, tempering | 3000 | confirmed | Restarts are not beaten on the mixture at 3,000 evaluations: 7 of 40 starts against tempering's 4 (p = 0.549, no separation) and annealing's 1 (p = 0.031, wo... |
| [5](005-tree-policy-features.md) | tree | release | greedy, reinforce-improvement, reinforce-full | 640 | confirmed | With the single feature the trained policy ties greedy (0.487 against 0.480, p = 0.79), as #178 measured. With the full set it reaches the maximum from 0.796... |
| [6](006-tree-initializers-at-equal-evaluations.md) | tree | release | FromObjective, Perturbed, RandomRestart, FromDistances, FromHadamard | 2000 | confirmed | On the branch-length fit no start separates: L-BFGS converges in 22 to 27 evaluations from every start at five and six taxa, the neighbor-joining start inclu... |
