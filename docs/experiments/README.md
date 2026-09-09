# docs/experiments/

One file per experiment, written from [`TEMPLATE.md`](TEMPLATE.md): front
matter carrying the commit, the fixture and its size tier, the methods
compared at one budget over shared seeds, the hardware and the status; then
three sections --- Question, Numbers, Finding (issue #314).

**The body is capped at ten non-blank content lines** after the front matter's
closing `---` (issue #458). Neither the title nor a section heading is one of
them, and the front matter is not counted at all: it is the reproducibility
record. What survives the cap is chosen, in this order: key metrics,
motivation, reproducibility. A number displaced by it moves to `STATUS.md`
where it is evidence for a milestone, or to the pull-request body where it is
the argument for a change; a number that fits neither was never evidence.
`STATUS.md` cites an experiment rather than restating its table, and
`tests/regression/test_experiments.py` holds every file to the template, the
cap, and this index to the files.

**This index is generated.** Rewrite it with `python infra/experiments.py`;
`--check` fails when a file is invalid, over the cap, or the index is stale.

| id | problem | size | methods | budget | status | finding |
| --- | --- | --- | --- | --- | --- | --- |
| [1](001-potts-cluster-autocorrelation.md) | potts-lattice | stress | single-site, swendsen-wang, wolff | 0 | confirmed | Single-site slows 3.2x between extent 8 and 24 against roughly 1.9x for both cluster algorithms, a 2.1x gap at extent 24 and widening, understated by lattice... |
| [2](002-hmm-gaussian-interval-coverage.md) | hmm-path | ci | gradient-fit-wald-intervals | 24 | confirmed | Coverage is nominal from two standard deviations of separation upward and degrades below it twice over: the intervals that exist under-cover, and an increasi... |
| [3](003-potts-chain-reinforce-vs-greedy.md) | potts-lattice | ci | greedy, reinforce | 1920 | confirmed | The learned policy reaches the optimum from 86.6% of starts against greedy's 80.2%, in 8 of 8 training seeds; the reward decomposes exactly into the two feat... |
| [4](004-mixture-tempering-vs-restarts.md) | mixture | release | restarts, anneal, tempering | 3000 | confirmed | Restarts are not beaten: tempering does not separate from them and annealing is worse, as on Rastrigin and opposite to the frustrated lattice, because these... |
| [5](005-tree-policy-features.md) | tree | release | greedy, reinforce-improvement, reinforce-full | 640 | confirmed | The single feature ties greedy (0.487 against 0.480) and the full set reaches the maximum from 0.796 of episodes, 0.316 above greedy and ahead on every seed:... |
| [6](006-tree-initializers-at-equal-evaluations.md) | tree | release | FromObjective, Perturbed, RandomRestart, FromDistances, FromHadamard | 2000 | confirmed | No start separates on the fit below 20 taxa, where the estimator saves 9 evaluations of 34; on the search it buys the topology, one neighbourhood against the... |
| [008](008-frameworks-on-three-hot-paths.md) | potts-lattice | stress | ours, scipy-csgraph, rustworkx, torch-geometric | 0 | confirmed | All three declined; the 1.52x-1.98x that landed instead is this repository's own pointer doubling and sorted grouping, and each front is conserved in `sandbo... |
