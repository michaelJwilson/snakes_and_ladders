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
