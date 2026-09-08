---
id: 006
date: 2026-09-08
commit: ebf977a8554d46cc4238b618df3bbb37de85f985
branch: claude/opt-364-spectral-init
pr: 0
tickets: [364, 251, 281]
problem: tree
fixture: tests/regression/fixtures/tree_search/ci.yaml, tests/regression/fixtures/tree_search/stress.yaml, and a random 20-taxon tree (lengths uniform on [0.02, 0.15], seed [364, 20]) at 1,000 sites
size: release
methods: [FromObjective, Perturbed, RandomRestart, FromDistances, FromHadamard]
budget: 2000
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
hardware: Linux-x86_64
status: confirmed
---

# Do the data-driven starts reach the tree's optimum from more starts, or in fewer evaluations, than the objective's own?

## Feature under test

At equal likelihood evaluations, a start read from the data — neighbor
joining on pairwise distances (`FromDistances`) or the closest tree of the
Hadamard conjugation (`FromHadamard`) — reaches the branch-length optimum in
fewer evaluations than the objective's own constant start, and the topology
search begun from it reaches the enumerated optimum from at least as many
datasets as a random start does, in fewer candidates scored.

## Setup

Three instance sets of simulated datasets (seeds `[364, i]`): twenty of the
five-taxon fixture at 1,200 sites, ten of the six-taxon fixture at 1,500
sites, and five of a random twenty-taxon tree at 1,000 sites, where nothing
is enumerated. The counts keep each set inside ten minutes on one core: the
six-taxon referee is 105 fits per dataset and a twenty-taxon climb is 60
fits. The twenty-taxon tree's branches are uniform on [0.02, 0.15], longest
path 1.24, because at [0.02, 0.4] some pairs saturate at 1,000 sites --- three
quarters of sites differing, where no finite distance exists and the
estimator refuses. Release tier; the same code on three datasets of the
five-taxon fixture runs per pull request.

Two comparisons, both through `opt.budget.compare` with `workers=1`.

- **The branch-length fit** on the generating topology, from each of the five
  starts: `FromObjective` (0.1 on every branch), `Perturbed` (the default
  tilt of 0.1 in log coordinates), `RandomRestart` (one Gaussian draw of
  scale 1.0 in log coordinates, from `default_rng([0, i])`), `FromDistances`
  (Jukes–Cantor distances, neighbor joining, floor 1e-4) and `FromHadamard`
  (two-state recoding, closest tree, weights rescaled by 3/2; not offered at
  twenty taxa, above its cap of 12). Budget 2,000 evaluations per fit, one
  evaluation being one call of the objective, with or without a gradient.
  Every start converges to one optimum, so the number reported is the
  evaluations spent reaching it.
- **The NNI hill climb** from each start that names a topology: a random
  topology drawn from `default_rng([0, i])` (the restart baseline), the
  neighbor-joining tree, and the closest tree. Budget 60 candidates scored;
  warm starts on, lazy scoring off. The referee at five and six taxa is the
  enumerated optimum, every topology fitted (15 and 105 fits per dataset);
  at twenty taxa it is the best value any method found. A start reaches the
  referee within `1e-6` relative.

## Results

Typed from the release run of
`tests/regression/search/test_search_initialize.py::test_initializers_at_equal_evaluations`
(wall clock 108 s, 344 s and 355 s for the three cases, one thread).

Branch-length fit on the generating topology, evaluations to converge, mean
(min–max). Every dataset reaches the one optimum from every start:

| start | five taxa (20 datasets) | six taxa (10) | twenty taxa (5) |
| --- | --- | --- | --- |
| FromObjective | 23.0 (20–26) | 24.4 (23–26) | 34.0 (26–39) |
| Perturbed | 23.1 (21–25) | 24.3 (22–26) | 33.4 (26–39) |
| RandomRestart | 24.1 (21–35) | 26.9 (24–40) | 41.6 (25–51) |
| FromDistances | 22.3 (18–27) | 23.5 (20–27) | 25.2 (24–26) |
| FromHadamard | 21.8 (20–25) | 24.2 (21–27) | — |

NNI search, datasets reaching the referee, and candidates scored, mean
(min–max):

| start | five taxa (of 20) | six taxa (of 10) | twenty taxa (of 5) |
| --- | --- | --- | --- |
| RandomRestart | 20, 7.3 (4–10) | 10, 16.3 (14–22) | 0, 60 (60–60) |
| FromDistances | 20, 4.0 (4–4) | 10, 6.0 (6–6) | 5, 34 (34–34) |
| FromHadamard | 20, 4.0 (4–4) | 10, 6.0 (6–6) | — |

McNemar, `FromDistances` against `RandomRestart`: p = 1.000 at five and six
taxa (no discordant dataset), p = 0.062 at twenty (five discordant, all in
favour of the estimator, the smallest p-value five pairs can give). At
twenty taxa the random start's best value after 60 candidates sits 3,510
nats above the estimator's on the first dataset (18,089 against 14,576).

Atteson's radius on the fixtures, against the largest distance standard
error and the largest realized error at the declared sites and seed: five
taxa radius 0.035, standard error 0.040, error 0.046; six taxa 0.030, 0.039,
0.056; eight taxa 0.025, 0.0035, 0.0065. Neighbor joining recovers all three
topologies; only the eight-taxon fixture sits inside the guarantee.

## Figures

none

## Finding

On the branch-length fit no start separates: L-BFGS converges in 22 to 27
evaluations from every start at five and six taxa, the neighbor-joining
start included, though it begins within a few nats of the optimum and the
objective's constant 200 nats above it --- the stop is the gradient relative
to the objective and the curvature pairs cost the same from anywhere in the
basin. At twenty taxa the estimator saves 9 evaluations of 34 (25.2 against
34.0), and the random draw costs 8 more.

On the search the estimators buy the topology. At five and six taxa every
start reaches the enumerated optimum, and the estimator starts do so at the
cost of one neighbourhood --- 4 and 6 candidates, the start already being
the optimum --- against the random start's 7.3 and 16.3. At twenty taxa
the random start reaches the estimator's value on 0 of 5 datasets in 60
candidates and the neighbor-joining start is the fixed point of the climb
on 5 of 5, at 34 candidates. Neighbor joining and the Hadamard closest tree
gave the same topology on every fixture dataset.

## Conclusion and actions

- #364: closed by this measurement; no default changes, every number
  `STATUS.md` pins is still produced from the objective's own start. The
  search's `topology=None` still draws a random start; a caller who wants
  the estimator passes `FromDistances().tree(alignment, k)`.

## What is not claimed

Nothing about the general time-reversible model's fit: the log-det start is
pinned to the protocol and to its distances, not measured here. Nothing
about the Kimura three-parameter conjugation, which is not implemented; the
four-state alignments reach the Hadamard start through the two-state
recoding. Nothing about the hard fixture, where the estimated spectrum is
refused at 2,000 sites and neighbor joining is outside Atteson's radius.
Nothing about twenty taxa beyond five datasets of one tree at 60 candidates:
the random start had not converged, and a larger budget can close the gap.
Nothing about wall time: the seconds above are for scale; the budget is in
evaluations and candidates.
