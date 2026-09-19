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
| [7](007-pruning-gradient-routes.md) | tree | ci | taped, burn, analytic | 60 | confirmed | The prediction held: A's tape is 12 to 19% larger than the one it replaces, so it relocated the cost --- 1.42x per gradient and 1.70x per fit --- and the bou... |
| [008](008-ffi-profile-and-alpha-expansion.md) | potts-lattice | ci | python, rust | 0 | confirmed | #447 held: the ranking puts the Python Dinic solver at 49.0% of `alpha_expansion`, and the reason was never the boundary --- `oxi_snakes_and_ladders.max_flow... |
| [009](009-projection-emission-seedings.md) | coupled | release | prior, data, kmeans++, emission++, burn-in, hmc, anneal, tempering, restart | 6 | confirmed | The two orderings are opposite. `emission++` reaches the best projected value on all six instances and leads at 0.4 nats, ahead of the prior draw it displace... |
| [010](010-gaussian-mixture-seeding-control.md) | mixture | release | random-restart, kmeans++, emission-d2, burn-in, family-sample, spectral, hmc, tempering, anneal | 40 | confirmed | **It ties on one rung and loses on the other, and the correction decided which.** On the one-channel rung the divergence *is* squared Euclidean, so candidate... |
| [011](011-potts-ground-state-recovery.md) | potts-lattice | release | greedy, icm, gibbs-T0, anneal, swendsen-wang, wolff, tempering, alpha-expansion, alpha-beta-swap, max-product | 2083260 | confirmed | Both cut-based minimizers reach the q = 2 graph-cut optimum exactly and no sampler comes within 219 of it; at q = 10 they return the *same* labelling, using... |
| [012](012-pruning-gradient-routes-after-the-post-order.md) | tree | ci | taped, analytic | 15 | confirmed | No: at one tree, one alignment and one site count, replacing the branch lengths with 0.1 moves the ratio from 2.11 to 0.57 --- the routes swap places --- so... |
| [013](013-parallel-candidate-fits.md) | tree | ci | serial, threads, processes | 60 | confirmed | The answer does not move: every run returns `log_likelihood` bitwise equal to the serial run's and the same topology, which input-ordered results over indepe... |
| [014](014-the-leaf-count-fixture.md) | tree | ci | before, after | 12 | confirmed | The answer does not move: every run returns `log_likelihood` -23642.15298199 at 20 leaves and -43729.71042328 at 50, on 12 evaluations and 5 fits either side... |
| [015](015-css-decoding-under-degeneracy.md) | bicycle-css | ci | degenerate-ml, single-error-ml, sum-product | 4000 | confirmed | Degeneracy is worth **3.4%** of the logical error rate --- 0.070297 against 0.072735 --- and changes the returned coset at **24** of the 128 syndromes with n... |
| [016](016-one-incidence-layout.md) | potts-lattice | ci | greedy, candidate | 0 | confirmed | Two consumers gain, one pays a fixed 28 us of calls that is gone by 3,000 bits, and the fourth was measured and left as lists: `pytest tests/benchmarks/test_... |
| [017](017-the-message-passing-layout.md) | potts-lattice | ci | greedy, candidate | 0 | confirmed | No: the layout is under 1 per cent on every shape, and what the profile ranks instead is the tree schedule at 18.2 and 24.4 per cent of the two chains. Actio... |
| [018](018-the-schedule-seam.md) | tree | ci | greedy, candidate | 0 | confirmed | The upward pass gives all of `log Z` for 1.71x less work, and batching the group arrays recovers 1.21x of a plan whose residue is Python bookkeeping with no... |
| [019](019-potts-baseline-failure-size.md) | potts-lattice | stress | icm, gibbs-T0, anneal, wolff, tempering, alpha-expansion | 200 | confirmed | The baseline is annealing, not ICM: it reaches the expansion's energy at every size while single-site descent misses by 5–11% and is **flat in the budget**,... |
| [021](021-potts-gradient-proposals.md) | potts-lattice | stress | single-site, locally-balanced, gibbs-with-gradients | 0 | confirmed | Yes in sweeps, no in work: 4.96 sweeps buy an independent sample against the heat bath's 10.50, **2.1x fewer**, at **8.1x** the wall per sweep in NumPy and 2... |
| [022](022-cluster-moves-for-frustrated-lattices.md) | potts-lattice | stress | niedermayer, houdayer, single-site | 0 | confirmed | **Both constructions percolate here and neither buys a round trip.** Niedermayer's cluster is 8.31 sites of 9 on the 3x3 antiferromagnet and 8.94 of 9 where... |
| [023](023-placing-the-tempering-ladder.md) | potts-lattice | stress | geometric, acceptance-placed, round-trip-placed | 0 | confirmed | **Neither criterion is established over the other, and the warm-up budget decides more than the criterion does.** At 1,000 warm-up sweeps over three rounds t... |
