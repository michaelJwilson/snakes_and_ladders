# Problems

The problems this repository supports, one row each. A row is **minimal by
design**: it names the problem, the marker its tests carry, the textbook
section that states it, and the code that defines it. Everything else about a
problem is written where it is refereed — the model in `docs/tex/textbook.tex`,
the instance in `tests/regression/fixtures/`, the progress in `STATUS.md`, the
checks in `CHECKS.md`. A fact restated here is a fact that can drift (issue
#640).

**Statement** is the LaTeX label of the textbook section stating the model.
It points one way only: the registry names the document, and the document names
no code (`docs/CLAUDE.md`), which is what keeps a module path out of the
textbook (issue #249). `tests/regression/docs/test_document_orphans.py` asserts
that direction on the hand-written sources.

**Key** is the fixture directory under `tests/regression/fixtures/` and the
marker its tests carry, which are one name and not two. A row with two keys is
one problem declared at two instances. The **family** is the statement's own
short name — `potts` over `potts_chain`, `potts_lattice` and `spatio_only` —
so a test selects broadly with `-m potts` and narrowly with `-m potts_lattice`,
and a test that runs on more than one type carries more than one marker.

**Defines** is the narrowest code that is this problem and no other: a module
where the module belongs to one problem, a symbol where the module is shared.
A test module importing any of it exercises the problem, which is how
`tests/_problems.py` marks it — no list of every symbol to maintain, and a
function added to a defining module is covered the day it is written.

**Shared machinery defines nothing.** `opt.fit`, `sample.hmc`, `opt.initialize`
and `sample.schedule` fit, sample, start and temper *every* row — `ROADMAP.md`
§1.1's claim — and `likelihood.message_passing`, `sim.factor_graph`,
`sample.gibbs`, `likelihood.potts`, `sim.ldpc`, `sim.graph`, `sim.potts`,
`sim.canonical`, `search.alpha_expansion`, `sample.potts_mcmc`, `search.infer`,
`sim.topology`, `emissions` and the rest of `likelihood/` are reached from
several. Importing one says nothing about which problem a test exercises, so
none of them defines one.

| Problem | Key | Statement | Defines |
| --- | --- | --- | --- |
| Phylogenetic tree, Jukes–Cantor | `tree_search`, `tree_scale` | `sec:phylo` | `sim.jc`, `sim.simulate`, `likelihood.pruning`, `likelihood.pruning_rust`, `likelihood.hadamard`, `search.neighbor_joining`, `learn.tree`, `sandbox.tropical` |
| Phylogenetic tree, general time-reversible | `tree_jc` | `sec:phylo` | `sim.gtr` |
| Phylogenetic tree, parsimony | `tree_jc` | `sec:parsimony` | `likelihood.parsimony` |
| Potts chain in a field | `potts_chain` | `sec:potts` | `sim.potts_chain`, `learn.potts`, `opt.potts` |
| Potts lattice and Markov random field | `potts_lattice` | `sec:potts` | `likelihood.belief_propagation`, `search.maxflow`, `search.max_cut`, `sim.graph.lattice_graph`, `sim.graph.erdos_renyi_graph` |
| Potts lattice in a per-site field | `spatio_only` | `sec:potts` | `sim.potts.spatio_only_field`, `sim.graph.triangular_lattice_graph` |
| Frustrated Potts: triangular antiferromagnet | `frustrated_lattice` | `sec:frustrated` | `sim.canonical.frustrated_triangular_lattice`, `sim.canonical.minimum_frustrated_edges` |
| Planted spin glass, enumerable ground state | `planted_glass` | `sec:frustrated` | `sim.canonical.planted_spin_glass`, `sim.canonical.PlantedSpinGlass` |
| Low-density parity-check code, Gallager (3,6) | `ldpc` | `sec:ldpc` | `sim.ldpc.gallager_code`, `likelihood.ldpc` |
| Low-density parity-check code, bicycle | `bicycle` | `sec:ldpc` | `sim.ldpc.bicycle_code` |
| Calderbank–Shor–Steane code over a bicycle matrix | `bicycle_css` | `sec:ldpc` | `sim.css`, `likelihood.css` |
| Turbo code, two (7,5) recursive systematic encoders | `turbo` | `sec:turbo` | `sim.convolutional`, `likelihood.convolutional`, `likelihood.turbo` |
| Polar code, successive cancellation and list decoding | `polar` | `sec:polar` | `sim.polar`, `likelihood.polar` |
| Hidden Markov model, six emission families | `hmm`, `ragged_hmm` | `sec:hmm` | `sim.hmm`, `opt.hmm`, `likelihood.hmm_paths`, `learn.hmm` |
| Coupled spatio-sequential model | `spatio_sequential`, `spatio_sequential_ragged` | `sec:coupled` | `sim.spatio_sequential` |
| Count-pair coupled spatio-sequential model | `spatio_sequential_counts` | `sec:coupled` | `sim.count_pairs`, `sim.count_pairs_rust`, `likelihood.spatio_sequential_rust` |
| Gaussian mixture | `mixture` | `sec:mixture` | `sim.mixture`, `likelihood.mixture_assignments` |
| Mixture of two-channel count emissions | `emission_mixture` | `sec:emissionmixture` | `sim.emission_mixture`, `opt.emission_mixture` |
| Continuous test functions | `test_functions` | `sec:testfunctions` | `opt.testfunctions` |

Every symbol above is rooted at `snakes_and_ladders`;
`tests/regression/test_problems_catalogue.py` resolves each one, so a row
cannot outlive what it names. Every key is a directory under
`tests/regression/fixtures/`, and `tests/regression/test_fixture_registry.py`
fails a row without a loadable CI-tier fixture and a fixture no row names.

## Structural moves and what survives them

A structural move constructs a new objective (`search/CLAUDE.md`, #815): the
parameter vector changes meaning and length, so nothing describing the old
space crosses the move, and what crosses is what the move leaves identical.
Each problem with such a move says what that is, or that the rule does not
apply. A list and not a table, so the two catalogue readers above see no row.

- **Phylogenetic tree** --- NNI and SPR on the topology (`search.infer`). A
  branch is the split it induces: the parent's fitted length on every kept
  split, the default on the splits the move made (`_warm_lengths`). Measured
  at #821 on `tree_search/ci`: the warm start saves 1.22x gradient evaluations
  and moves 9 of 60 optima by up to 3.55; with inherited collapsed branches
  started at the default it saves 0.98x and moves none. The saving was a
  defect of the fit (#839), and the learned environment stays cold.
- **Coupled spatio-sequential** --- relabelling of the sites
  (`search.spatio_sequential.label_step`). Not structural: the labels are
  discrete coordinates of the one objective `log p(x, l | theta)` and the
  emission parameters keep their meaning under a relabelling, so
  `fit_spatio_sequential` is block-coordinate ascent and the M step's
  parameters cross by right. A class permutation would be the structural move,
  and none is made.
- **Gaussian and emission mixtures** --- no move in the package. An assignment
  is the E step of one objective, not a move; `search.projection` and
  `kmeans_plus_plus` seed the fit once and nothing crosses after.
- **Hidden Markov model** --- no move in the package. The path is summed or
  maximized inside one objective; no segmentation is searched.
- **Potts lattice and planted glass** --- single-site, cluster and label moves
  on the configuration. Not structural: a configuration change leaves what
  `(J, h)` mean untouched, so there is nothing to carry and no warm start to
  invent.
- **Parsimony** --- NNI and SPR. Nothing continuous to fit, as
  `likelihood.parsimony` states.
