# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

New entries are managed as [towncrier](https://towncrier.readthedocs.io)
fragments under `changelog.d/` (see `changelog.d/README.md`) and merged into
this file at release time; the `[Unreleased]` section below predates that
convention and is retained as history.

<!-- towncrier release notes start -->

## [0.3.0] - 2026-09-03

### Changed

- The thirteen QA scripts share one command line (`snakes_and_ladders.qa.runner`). Each
  declared its own `ArgumentParser`, `--output-dir`, load, write and
  `try/finally` around `plt.close`, so a fix to one reached only that one; a
  script now declares its output stem, the parameters files it takes, and the
  builder that turns them into a figure or a `tabular` body.

  Two behaviours were inconsistent and are now uniform. Every script reports what
  it wrote, where three did and ten were silent. Every figure is closed even when
  writing refuses it, where the twelve hand-written `finally` blocks each had to
  get that right separately.

  `sim_tree`, `sim_example` and `sim_problem_sizes` lose their
  `render_*_figure`/`render_problem_sizes_table` wrappers, which loaded and wrote
  around a build step; the build step is now `build_figure`/`build_table` and the
  loading and writing are the runner's. `--n-sites-shown` keeps its default of 10.
  Every one of the thirteen committed figures regenerates byte-identically. (#150)
- `ROADMAP.md` is restructured around three problem classes rather than one:
  phylogenetic trees, N-D Potts models in an external field, and HMMs each carry
  a deliverable and a validation under every Stage 1 and Stage 2 milestone, and
  milestones are keyed `N.M` so a reference resolves. It gains a first section
  stating the agentic development loop — ticket, plan, pull request, validation,
  and the record the loop writes into — each as a deliverable and the gate that
  holds it.

  Milestone status leaves the roadmap for `STATUS.md`, which states what landed,
  the oracle that established it, and the pull request that carries it; a
  requirements ledger against §1.2; and what is not claimed. `TICKETS.md` states,
  as titles, the tickets remaining between the two. A roadmap that also tracked
  its own progress could not be edited without re-litigating both. (#152)
- `README.md` leads with the three problem classes the roadmap now states —
  phylogenetic trees, Potts models in an external field, and HMMs — rather than
  phylogenetics alone, and gains a Features section stating each capability with
  the measurement behind it, a table of the nine `CLAUDE.md` contracts and what
  each governs, and a reference table routing the literature by concern.

  The technical document builds again. Two citations named keys absent from
  `references.bib` (`cormen2022`, `hwu2022` against the bib's `clrs2022` and
  `pmpp2022`), which `latexmk` reports while still exiting 0 — CI's log check
  catches it, so `docs/draft.pdf` could not be regenerated. The backend-agreement
  figure read its generated caption into a macro and then discarded it for a
  hardcoded restatement quoting a stale measurement; it now uses the caption the
  QA script wrote.

  `infra/build_technical_doc.sh` exports `FORCE_SOURCE_DATE=1`. `SOURCE_DATE_EPOCH`
  fixes the PDF's `/CreationDate` but not `\today`, which reads pdftex's date
  primitives: the committed PDF's title page carried the day it was built, so the
  staleness check would have failed on any pull request opened the following day. (#153)
- The accuracy figure's per-PR test no longer re-runs the sweep at its committed
  size. It ran 6 site counts x 8 replicates = 48 searches, 27.7 s, 23% of the
  whole per-PR suite, to assert four things about a caption — while the
  technical-document build rendered the same figure again and the release gate
  asserted the scientific claim. It now sweeps two sizes by two replicates: the
  pipeline still runs end to end, at 4.1 s.

  The suite is 138.0 s over 540 tests, against 165.6 s over 525 before — faster
  with fifteen more tests in it. (#154)
- The technical-document build regenerates only the QA figures
  `docs/tex/main.tex` cites, and the release gate regenerates the rest. The
  figure list lived as thirteen invocations in `infra/build_technical_doc.sh`
  that nothing connected to the document, so when the document stopped citing
  eleven of them the build kept rendering all thirteen and no check noticed:
  that job spent 281.6 s to run a 1.5 s LaTeX build, and 98.4% of it produced
  figures nothing included. A full build is now 5.9 s, and `docs/draft.pdf` and
  all thirteen committed figures are byte-identical to before.

  `snakes_and_ladders.qa.manifest` is the single statement of which figures exist and what
  renders each one. `snakes_and_ladders.qa.build` reads it and renders a selection: what the
  document cites (per pull request), the whole manifest (`--all`, which
  `infra/release.sh` now runs with `--check`), or named stems (`--only`).
  Citing a figure the manifest cannot render is refused rather than skipped.

  `snakes_and_ladders.qa.build` pins `SOURCE_DATE_EPOCH` for the figures it renders.
  matplotlib embeds it, so without it two rebuilds of an unchanged figure
  differ and a comparison reports every figure stale — which a caller had to
  know to prevent. `infra/build_technical_doc.sh` reads the value back from
  there rather than keeping a second copy.

  `infra/measure_build.sh` times the build stage by stage, so a claim that it
  got faster is a pair of numbers. (#154)
- `tests/regression/` is split by submodule — `sim/`, `likelihood/`, `opt/`,
  `learn/`, `search/`, `qa/` — which is `DEV.md`'s own rule once a kind outgrows
  one flat directory, reached at 39 modules. Tests belonging to no submodule stay
  at the top level. Pure moves; 540 tests pass before and after.

  CI caches the TeX Live packages the `technical-doc` job installs, the one
  install still paying full price on every run while `uv` and Cargo were both
  cached.

  `DEV.md`'s suite budget said 131 s over 140 tests and 954 s over 141. It is
  138 s over 540 and 989 s over 550.

  Selecting tests by the files a pull request changed is recorded as unavailable
  rather than built: `python-tests` runs `--cov-fail-under=90` on the same
  invocation, and a subset cannot meet it — `tests/regression/sim` alone measures
  12% — so scoping would mean weakening a gate `CLAUDE.md` forbids weakening. (#154)
- Root `CLAUDE.md`'s **Writing Style** section states what it governs: every
  document in the repository, each module's `CLAUDE.md` included, and every
  docstring, comment, commit message and pull-request body. The rules bound all
  of that already; nothing said so, and the eight module files carried a generic
  "these are local" line that never named the section, so an agent reading one
  alone had no way to know.

  Each of the eight now names **Writing Style** and states that it binds that
  file. Referenced, not copied: the section changed three times on the day this
  was written, and nine copies would already disagree. **Expected Reader** stays
  in `docs/CLAUDE.md` alone, being a contract about the technical document.

  A regression test enforces it, rather than leaving the invariant stated and
  unchecked — the failure mode of `docs/source/index.rst`, which claimed to
  cover every submodule while missing eighteen. (#155)
- A generated plan has a stated shape: 2–5 steps, each saying how it will be
  validated, ending with an **Open Questions** section that carries every
  question on the desired behaviour. A reviewer now finds the questions in one
  place instead of reading the prose for them, and a plan with none says so
  rather than omitting the heading.

  Root `CLAUDE.md` states which documents are read at which altitude, and what
  that means for repetition. `ROADMAP.md`, `STATUS.md` and `TICKETS.md` plan and
  track. `CLAUDE.md`, `DEV.md`, `INSTALL.md` and the module files are worked in
  and carry their detail in full rather than as pointers into each other, so
  someone following one of them need not assemble the answer from three. Detail
  may repeat between them; where it repeats it must agree, and root `CLAUDE.md`
  settles which reading is right. The Writing Style stays the exception — one
  text, referenced — because it binds every file at once.

  The plan's shape is therefore stated three times, at the altitude each
  document is read at: `ROADMAP.md` §0.2 has it as part of the loop,
  `DEV.md` and `infra/CLAUDE.md` add what the step's validation must name.
  `DEV.md` also states a rule that was written nowhere — a pull request
  implements a plan already approved — and says which document holds the loop's
  intent when the two disagree.

  Root `CLAUDE.md`'s Writing Style already governed every document, docstring,
  comment, commit message and pull-request body; the list now names plans and
  ticket comments too, which is what makes a plan subject to it. (#162)
- `STATUS.md` is read at `0.3.0`. Between `0.2.0` and `0.3.0`, six pull requests
  refined the development loop and its record — `ROADMAP.md`'s restructuring,
  the QA figure-manifest and script-runner refactor, the test-layout split, and
  the writing-style and plan-shape pointers — and no roadmap milestone moved.

  The Release issue template drops its now-redundant one-ticket-per-version
  notice, defaults its title and target-version field to the next expected
  version, gains a blank-by-default Suggested work section, and its consistency
  audit prompt now names the template itself as something the audit checks. (#165)


## [0.2.0] - 2026-09-03

### Added

- A model-agnostic optimization interface (`snakes_and_ladders.opt`): an unconstrained
  parameter vector, a differentiable scalar objective, and a map back to named
  constrained parameters, with the shared constraint maps that keep every
  parameter feasible by construction. Two reference instances ship with it — a
  1-D Potts chain in an external field and a discrete HMM — each validated
  against an exact independent oracle (transfer matrix and brute-force path
  enumeration respectively) and a central-difference derivative check. The
  interface carries no application knowledge, and a test asserts it imports
  nothing from `snakes_and_ladders.sim`, `snakes_and_ladders.likelihood` or `snakes_and_ladders.search`.

  Gradient-based fitting for any `Objective` (`snakes_and_ladders.opt.fit`): L-BFGS with a
  strong-Wolfe line search, convergence judged on the gradient relative to the
  objective's own magnitude, and confidence intervals from the observed Fisher
  information pushed through the constraint map by the delta method. Validated
  by parameter recovery on both instances — the Potts chain's 95% intervals
  cover the truth at exactly the nominal rate over 60 replicates, and the HMM's
  gradient fit is cross-checked against Baum–Welch, an independent fitting
  algorithm sharing no optimizer, parameterization or constraint map with it.
  Two QA figures report the recovery and how fast the intervals converge on
  their nominal coverage. (#63)
- Branch-length fitting for a fixed topology (`snakes_and_ladders.likelihood.objective`),
  behind the same model-agnostic interface as the Potts and HMM instances — the
  optimizer needed no phylogenetic special-casing, which is the evidence issue
  \#63's abstraction was not shaped by one model. Branch lengths are recovered
  within their confidence intervals on both the unrooted and rooted fixtures,
  and `ROADMAP.md`'s sub-second gradient update at n=100 is now measured (203 ms
  at 1000 sites) rather than asserted.

  The two branches below a rooted binary tree's root are fitted as one
  parameter and reported as their sum, because only the sum is estimable: under
  a reversible model the likelihood is unchanged by moving length between them.
  `snakes_and_ladders.opt.fit.parameter_covariance` now rejects an ill-conditioned observed
  information rather than inverting it, since a numerically singular matrix
  inverts successfully and yields a meaningless interval.

  The general time-reversible model (`snakes_and_ladders.sim.gtr`), so `Q` and `π` have free
  parameters to fit at all — Jukes–Cantor has none. Exchangeabilities and the
  stationary distribution are fitted alongside the branch lengths and recovered
  within their intervals, and `simulate_alignment` takes an optional
  `rate_matrix` (omitting it keeps the Jukes–Cantor closed form unchanged).
  Equal exchangeabilities with a uniform `π` reproduce `jc_rate_matrix` and
  `jc_transition_probabilities` to machine precision, which is how the model is
  validated. Its three normalizations are gauges rather than conventions: each
  removes an exactly flat direction that would otherwise leave every parameter
  without a confidence interval. (#104)
- Device dispatch for the likelihood engine (`snakes_and_ladders.likelihood.device`):
  availability-based selection preferring CUDA, then Metal/MPS, then CPU, and
  the cross-device agreement tolerance `CLAUDE.md` had long promised was stated
  somewhere. `pruning_torch` takes dtype and device from the branch-length
  tensor it is given rather than hardcoding `float64` in six places, so a caller
  moves the whole recursion by moving one tensor; `float64` remains the default.
  The tolerance is relative and keyed on the lowest precision in the comparison
  — `1e-11` for `float64` on both sides, `1e-6` where either side is `float32`,
  since Metal cannot do `float64`. Both are derived from measured agreement
  rather than chosen, and the `float32` figure is exercised on CPU, so runners
  without an accelerator still check it. (#106)
- Topology search (`snakes_and_ladders.search.infer`): a user-facing `infer(alignment, k,
  ...)` that hill-climbs over NNI or SPR neighbourhoods, fitting the continuous
  parameters of every candidate, and reduces to a plain continuous fit when a
  topology is supplied. This is the first user of the seam issue #63 recorded
  and left unbuilt — a discrete move builds a new `Objective` rather than
  stepping inside a fit — and `snakes_and_ladders.opt` needed no change to serve it, which
  is the evidence that abstraction was not shaped by one model.

  Budgets are counted in candidate fits rather than seconds, so a run is
  reproducible from its seed; a topology is scored at most once per search,
  keyed on `leaf_bipartitions`; and `snakes_and_ladders.search.topology.random_topology`
  draws a seeded starting tree that reaches every topology on its leaf set.

  Exhaustive enumeration of unrooted topologies
  (`snakes_and_ladders.search.topology.enumerate_topologies`), which gives search quality
  its first independent oracle: below 8 taxa every topology can be scored, so
  "did hill climbing find the best tree" has an answer rather than an opinion.
  Measured on a 6-taxon fixture, both NNI and SPR reach the enumerated maximum
  and recover the generating topology from all 12 starting points — at a median
  of 14 candidate fits for NNI against 48 for SPR.

  Two QA figures for search: a trajectory against the exhaustive landscape, and
  the tree the search selected beside the highest-scoring one it rejected. The
  second reports a detail worth having in writing — the rejected topology fits
  the internal branch that would create the wrong grouping at essentially zero
  length, which is what rejecting a topology looks like from inside the
  continuous fit. `BranchLengthObjective.fitted_tree` is new, the inverse of
  `theta_from_truth`, so a fitted tree can be drawn or serialized as a tree. (#117)
- Reinforcement learning (`snakes_and_ladders.learn`): the `Environment` interface, a
  softmax-over-scored-actions policy, REINFORCE with a baseline, and an exact
  trajectory-enumeration oracle. Model-agnostic on the terms `snakes_and_ladders.opt` is —
  it imports nothing from `snakes_and_ladders.sim`, `snakes_and_ladders.likelihood` or `snakes_and_ladders.search`,
  asserted by an import-graph test — so the phylogenetic environment will live
  in `snakes_and_ladders.search` rather than here.

  The reference environment is single-flip local search over the same 1-D Potts
  chain `snakes_and_ladders.opt` fits, appearing once as an objective and once as a search
  problem. Its reward decomposes exactly into the two features the policy
  scores, which puts hill climbing *inside* the policy class as the weight
  vector proportional to `(J, 1)`, so a comparison against it is a statement
  about learning rather than about two unrelated algorithms.

  Rewards are closed forms at known parameters, with no inner optimization —
  issue #131's simplification, and what makes an episode cost microseconds
  rather than one L-BFGS solve per action.

  Because the action set and horizon are finite, the expected return is a
  closed form and its gradient follows by differentiating it. That is the
  oracle every claim here is pinned to, rather than to a training curve: the
  enumerated gradient agrees with central finite differences to 1.5e-11
  relative, and the sampled estimator with the enumerated gradient to 9.9e-03
  over 6000 episodes, while a myopic variant that credits each action with only
  its own reward is rejected at 71%.

  On that landscape the learned policy beats hill climbing at a matched
  decision budget, reaching the enumerated optimum from 86.6% of the 81 starts
  against greedy's 80.2%, in 8 of 8 training seeds.

  The phylogenetic RL environment (`snakes_and_ladders.search.rl`): tree search as the MDP
  the technical document specifies — a state is a topology, an action is an NNI
  or SPR neighbour, the reward is the improvement in log-likelihood. It lives in
  `snakes_and_ladders.search` rather than `snakes_and_ladders.learn` for the reason the phylogenetic
  `Objective` lives in `snakes_and_ladders.likelihood`: the model-agnostic module may import
  no application code.

  Two reward models, and the comparison between them is the deliverable.
  `FITTED` maximizes over branch lengths per candidate; `KNOWN` evaluates at one
  fixed branch length with no optimization, which is what makes an episode
  affordable — measured at 352 us against 113.7 ms per candidate.

  The substitution is validated rather than assumed
  (`snakes_and_ladders.qa.rl_reward_surface`). "The known parameters" cannot transfer across
  topologies at all — a branch length belongs to an edge, and a different
  topology has different edges — so the cheap reward is a different surface, not
  an approximation of the fitted one. On the 6-taxon fixture the two score the
  generating topology highest, agree on the best of all 105 topologies, and
  correlate at 0.9568; across a 50-fold range of the fixed branch length they
  still agree on the best topology every time, with the correlation never below
  0.8719.

  Measuring that turned up a property of the fitted surface worth recording: it
  does not totally order topologies. Many candidates share a maximized
  log-likelihood to within the optimizer's convergence, because the branch that
  would distinguish them is fitted to zero and the tree collapses to the same
  polytomy. Their order is therefore not a property of the model, and a rank
  correlation — which depends on it — moves by up to 0.04 under a perturbation
  of one part in 1e9, so it is not a measurement and cannot appear in a
  committed document that CI rebuilds.

  `snakes_and_ladders.qa.figure` gains `pearson_correlation`, which is continuous in the
  scores, written here rather than pulling in `scipy`.

  Not claimed: that a learned policy beats hill climbing on trees. The 6-taxon
  fixture cannot support that claim in either direction, because greedy already
  reaches the enumerated optimum from every start. (#131)

### Changed

- Cut the `0.1.0` release: `towncrier build --version 0.1.0` merged the fragments above into this file, `ROADMAP.md` records Milestone 5's NNI/SPR generators as landed, and `STATUS.md` (deleted from the repository but still described elsewhere as a live ledger) is dropped from `CLAUDE.md`, `DEV.md`, and the PR template — GitHub issues and labels (`infra/TICKETING.md`) are the project board now. `ROADMAP.md`'s remaining milestones (1, 2, 4, 6) gain the same landed/not-started status notes that 3 and 5 already carried; the fixtures directory, previously spelled out in eleven test modules under two names, moves to a single `tests/_fixtures.py`; and `DEV.md` states the measured cost of the release-gated suite so `pytest -m "not release"` is the obvious default while developing. (#101)
- CI skips the benchmark suite unless the change touches code a benchmark
  measures — `src/`, `python/snakes_and_ladders/{sim,likelihood,opt,search}/`,
  `tests/benchmarks/`, or a lockfile. Benchmarks are half the suite's wall clock
  (36 s of 71 s), and a documentation or QA change cannot alter what they
  measure. The job still runs and reports, since it is a required check, and
  coverage is unaffected because every line a benchmark reaches is also reached
  by the regression module it pairs with. (#109)
- Tolerances on log-likelihoods and gradients are now relative rather than
  absolute. Both quantities are sums over sites, so an absolute bound fixed at
  one site count does not transfer: the backends agree to ~8e-13 relative at
  every size, but the same agreement reads as 7.4e-07 absolute at 200,000 sites
  and would fail the previous 1e-9 bound. The suite's fast tests ran at tens of
  sites because that is cheap, not because the tolerance required it — a
  release-gated test now checks the bound at full fixture scale, and asserts
  that the previous absolute bound would have failed there. Absolute tolerances
  are kept where the quantity does not scale: transition probabilities, rate
  matrix row sums, and Monte Carlo frequencies. (#111)
- The technical document is restructured around the optimization abstraction
  rather than the phylogenetic application: notation is split into a
  model-agnostic half and an application half, the substitution model and its
  reversibility move to an appendix, code paths are gone from the body, and the
  reinforcement-learning section carries the theory needed to implement it.
  Two QA figures — backend agreement and analytic-versus-finite-difference
  gradients — are dropped with their scripts, since both reported checks the
  regression suite performs rather than performing any.

  Fixed a preamble setting that suppressed the space at *every* source line
  break in the document, so text wrapped mid-sentence set as "substitutionmodel".

  `snakes_and_ladders.qa.figure` gains `write_qa_table`, which emits a LaTeX `tabular`
  fragment instead of a matplotlib image, and `latex_integer`, which separates
  large numbers as `200\_000`. Caption safety is now enforced at the point of
  writing rather than asserted per test, so a caption that would break the
  LaTeX build fails in the script that wrote it. `render_problem_sizes_figure`
  is replaced by `render_problem_sizes_table`, and `state_label` moves from
  `snakes_and_ladders.qa.sim_example` to `snakes_and_ladders.qa.figure`. (#118)
- The technical document's prose is tightened globally: the quality-assurance
  section drops from 968 to 654 words and the body as a whole from 5986 to
  5581, with no claim, number or reference removed. The cut is mostly one
  duplication — a figure's body paragraph restating the caption the figure
  already carries, and the drift guarantee stated twice within twenty lines.
  The rule now applied throughout is that the caption says what a figure shows
  and the body says why it is there. (#125)
- `docs/` gains its own `CLAUDE.md`, on the pattern of the other module files.
  It carries the rules that keep a committed, CI-regenerated artifact true:
  what is committed against what is generated, that a figure is regenerated
  rather than edited, that the document reads captions and never restates them,
  and that a generated caption may report only quantities continuous in their
  inputs — since CI rebuilds `docs/draft.pdf` and byte-compares it, a
  discontinuous statistic breaks the build and was never a measurement anyway.
  The formatting contract stays in root `CLAUDE.md` under **Expected Reader**,
  referenced rather than restated. (#136)
- Docstrings and comments are revised against the writing style. The corpus
  needed five edits: of 50 flagged hedges, all ten uses of "strictly" are
  mathematical ("strictly positive", "strictly bifurcating"), and most of the
  rest are contrastive — "not merely wide", "exact rather than merely
  convenient", "obviously correct, not fast".

  Two magnitude claims are replaced by their measurements. `snakes_and_ladders.search.rl`
  called the fitted reward "roughly two orders of magnitude" more expensive
  than the known one; it is 113.7 ms against 352 us, a factor of 323. The
  search benchmark said a search spends "essentially all" of its time in
  candidate fits, and now says what the two benchmarks beside each other
  measure. (#138)
- Root `CLAUDE.md`, `DEV.md` and the module `CLAUDE.md` files are revised
  against the writing style, correcting three defects along the way.

  Root `CLAUDE.md` named the project `snakes_and_ladders`; the package is
  `snakes_and_ladders`. Its "Check known math properties/Invariants" rule was a label with no
  body, so the invariants it names — transition rows summing to 1, detailed
  balance, gradients against finite differences, a monotone likelihood — are
  restored. Its reference table announced a count of 25 that nothing checks, and
  now announces none.

  `DEV.md`'s `technical-doc` row said the job fails on undefined references or
  citations; it also fails on a multiply-defined label. `qa/CLAUDE.md` said it
  validates `search/` "later", where `snakes_and_ladders.qa` has validated `search/` and
  `learn/` since issues #117 and #131.

  No rule was added, dropped or re-scoped: the section headings and rule labels
  of both files are unchanged, checked by diff, except the empty rule above. (#138)
- `README.md`, `INSTALL.md` and `ROADMAP.md` are revised against the writing
  style in `CLAUDE.md`, along with one hedge in `docs/tex/main.tex`.

  The pass corrected more than it compressed. `README.md`'s opening sentence
  said "modern optimization discrete/continuous optimization" and misspelled
  "reference"; `ROADMAP.md`'s objective and numerics requirements were
  ungrammatical; `README.md` linked to `infra/TICKETING.md`, deleted in
  `aae9e74`. `INSTALL.md` carried two claims that had gone stale: that one
  `snakes_and_ladders.qa` script renders the technical document's figures, where eleven now
  do, and that CI's LaTeX job fails on undefined references or citations, where
  it also fails on a multiply-defined label and on a committed `docs/draft.pdf`
  that differs from the rebuild.

  `ROADMAP.md`'s milestones now state a specification and then a **Status:**
  line, so a reader can find what landed without reading a paragraph.
  `README.md`'s "What exists" gains the measurements behind its claims — `n =
  5..8` for the neighbour counts, 12 of 12 starting points at 6 taxa, 203 ms per
  gradient update at n=100, L=1000. (#138)
- The technical document is restructured as an academic letter: an abstract, an
  introduction stating the contributions and what is *not* claimed, methods,
  results grouped by claim rather than by the order the build emits figures,
  a discussion with threats to validity and outstanding work, and conclusions.

  Two figures answer requirements `ROADMAP.md` states and nothing had measured.
  `snakes_and_ladders.qa.backend_agreement` puts all three pruning backends against
  brute-force marginalization, which shares no code with the recursion it
  checks: worst relative deviation 4.0e-14 across four site counts spanning a
  factor of 30. `snakes_and_ladders.qa.topology_accuracy` measures the normalized
  Robinson-Foulds distance from the inferred topology to the generating one
  against the site count, and finds the margin: the 0.05 accuracy requirement
  is met from 125 sites upward, with 8 of 8 replicates recovering the topology
  exactly at 2000 sites against 5 of 8 at 60.

  `snakes_and_ladders.search.topology` gains `robinson_foulds` and
  `normalized_robinson_foulds`. The normalizer counts internal splits only:
  every tree over the same leaves induces all the trivial ones, so including
  them would shrink every distance by a taxon-count-dependent factor and
  silently weaken the bound.

  The competitiveness comparison against IQ-TREE 2 and RAxML-NG, and the
  learning curve of a trained phylogenetic agent, are stated as outstanding
  rather than drawn: neither measurement exists, and a figure with invented
  data in a committed document is worse than a stated gap. (#144)
- `snakes_and_ladders.numerics` holds the vectorized categorical sampler that
  `snakes_and_ladders.sim.simulate`, `snakes_and_ladders.opt.potts` and `snakes_and_ladders.opt.hmm` each carried a
  private copy of. The copies had drifted: two omitted the clamp on the last
  cumulative column that the third had, so a probability row summing to
  `1 - 4e-16` after rounding could return a category one past the end of the
  alphabet. The surviving copy carries the guard, and a test constructs the
  draw that triggers it rather than waiting for a 4e-16 event.

  `DEV.md` no longer restates `CLAUDE.md`'s Performance and Testing rules or
  `docs/CLAUDE.md`'s technical-document rules, and its release worked example
  no longer describes cutting `0.1.0` — a version whose changelog section was
  already built.

  `CHANGELOG.md`'s legacy `[Unreleased]` section is labelled as older than the
  dated sections above it rather than newer. `ROADMAP.md` records that the
  accuracy requirement's first half is now measured and its second half is not. (#146)


## [0.1.0] - 2026-09-02

### Added

- `k`-state Jukes–Cantor sequence simulator in `snakes_and_ladders.sim`: generates an
  alignment and the ancestral tree in Newick from a typed
  `simulation_params.yaml`, and retains the parameters that generated them.
  Simulated substitution frequencies are validated against the closed-form JC
  transition probabilities within a yaml-declared Monte Carlo tolerance across
  several site and taxa sizes. Promotes `numpy` and `pyyaml` to core
  dependencies. (#55)
- Added `snakes_and_ladders.sim.newick`: topology counting (`count_topologies`), Newick string validation (`validate_newick`), and state-labelled Newick serialization (`to_newick`), now the package's single source of Newick functionality. (#60)
- Added `snakes_and_ladders.qa`, quality-assurance figure scripts for the technical document, starting with `snakes_and_ladders.qa.sim_tree` (renders the assumed simulation tree with branch lengths). `infra/build_technical_doc.sh` regenerates these figures and builds `docs/draft.pdf`. (#61)
- Vectorized NumPy Felsenstein pruning in `snakes_and_ladders.likelihood`, computing
  `ln L(alignment | tau, Q, t, pi)` under the k-state Jukes-Cantor model with
  per-node rescaling accumulated in log space. Ships with an independent
  brute-force marginalizer used only as the test oracle at `n <= 6` taxa.
  Validated against brute-force marginalization to machine precision,
  rescaled/unrescaled agreement, the pulley principle (root-position
  invariance), and scoring the generating topology above random wrong
  topologies on simulated data. This is the reference every future backend
  (Rust, PyTorch, CUDA, Metal) is pinned against. (#62)
- Added `snakes_and_ladders.qa.sim_example` and `snakes_and_ladders.qa.sim_problem_sizes`: a worked
  4-taxon simulation example (Newick topology and aligned sequences) and a
  cross-fixture table of problem-size parameters (taxa, sites, seed,
  tolerance), read directly from the `simulation_params.yaml` fixtures. Wired
  into `infra/build_technical_doc.sh` and `docs/tex/main.tex`. (#67)
- Differentiable PyTorch Felsenstein pruning (`snakes_and_ladders.likelihood.pruning_torch`),
  taking branch lengths as a `torch.float64` CPU tensor separate from the
  topology so `torch.autograd` differentiates through them. Validated against
  the NumPy oracle and brute-force marginalization to `atol=1e-9`, against
  `torch.autograd.gradcheck` and central finite differences of the NumPy
  likelihood to `atol=1e-6`, and rescaled/unrescaled agreement. A general
  `rate_matrix` path (`torch.matrix_exp`) is exercised by a benchmark fitting a
  Jukes-Cantor rate matrix Q, alongside a forward-pass benchmark against the
  NumPy reference. (#70)
- Rust CPU Felsenstein pruning backend (`oxi_snakes_and_ladders.pruning_log_likelihood`,
  exposed via `snakes_and_ladders.likelihood.pruning_rust`), implementing the same
  recursion as the NumPy oracle in `src/pruning.rs` and exposed through PyO3.
  Validated against the NumPy oracle and independent brute-force
  marginalization at `n <= 6` taxa, and against the NumPy oracle at realistic
  (taxa, site) sizes to `abs_tol=1e-9`. Ships a `criterion` benchmark
  (`benches/oxi_snakes_and_ladders_bench.rs`) and a paired `tests/regression/test_pruning_rust.py`
  / `tests/benchmarks/test_pruning_rust_bench.py` module, reporting Rust vs.
  NumPy timings at 4- and 8-taxon, 200,000-site fixtures. (#77)
- NNI and SPR neighbourhood generators over unrooted binary topologies
  (`snakes_and_ladders.search.topology`), behind one `Topology -> Iterator[Topology]`
  interface. Validated exhaustively against `2 * (n - 3)` (NNI) and
  `2 * (n - 3) * (2 * n - 7)` (SPR) at `n = 5..8` -- every one of the
  `count_topologies(n - 1)` distinct topologies, cross-checked for neighbour
  validity, symmetry, and NNI-in-SPR containment (`n = 8` gated to
  `pytest -m release`, ~2.5 minutes). `snakes_and_ladders.sim.newick` gains
  `validate_unrooted_newick` for the trifurcating-root convention this reuses.
  The random-walk connectivity test is deferred to issue #73's canonical
  Newick key. (#79)
- Added a Release issue template (`.github/ISSUE_TEMPLATE/release.yml`) that
  drives the repository-consolidation audit ahead of a release, and
  `infra/release.sh`, a local release gate running every per-PR CI check plus
  the release-gated `pytest` tests, `sphinx-build -W`, and the technical
  document build. `DEV.md` documents the release procedure, including the
  version-bump and tag/publish steps. (#90)

### Changed

- `DEV.md` states the `tests/` layout convention: organized by kind
  (`regression/`, `benchmarks/`) at the top level and by subject within it, with
  rules for benchmark/regression pairing, fixture placement, and when a kind
  splits into submodule subdirectories. (#45)
- The rendered technical document (`docs/draft.pdf`) is now committed to the
  repository instead of being a gitignored build artifact. CI's
  `technical-doc` job fails a PR whose rebuilt PDF differs from the committed
  one, catching a `docs/tex/` or QA-figure change that wasn't regenerated. (#71)
- PR template gained a "Follow-up / Deferred Work" section for TODOs left to a tracking issue. (#84)
- `docs/tex/main.tex` now states its intended reader (a developer with
  baseline scientific/performance-computing background but no phylogenetics
  expertise) and the formatting contract that follows from it: streamlined
  main text, standard non-snakes_and_ladders-specific background (e.g. NNI, SPR) moved to
  a new appendix and cited from the point of use. `CLAUDE.md` records the
  same contract for anyone editing the document. (#85)
- PR template's Benchmark section gained a second table for scientific/tolerance regression tests, so contributors report the realized value alongside the reference and tolerance it was checked against. (#88)
- Simplified the Release issue template (`.github/ISSUE_TEMPLATE/release.yml`):
  the consistency-audit field is no longer required — the ticket's job is to
  trigger the consolidation audit and surface follow-up tickets, not to gate
  submission on having written them out — and the `infra/release.sh` checkbox
  is dropped. That check is already enforced by the documented release
  procedure (`DEV.md`'s "Release" section): the gate is run, and only then is
  the version bumped and the tag cut. (#94)

### Fixed

- Embedded TrueType (not Type 3) fonts in QA figures, fixing `docs/draft.pdf`'s failure to render in GitHub's blob viewer. (#76)
- Fixed the Release issue template (`.github/ISSUE_TEMPLATE/release.yml`):
  `roadmap-progress`, `consistency-audit`, and `follow-up-tickets` moved their
  guidance from `description:` (static gray helper text below the box) into
  `value:` (the box's own prefilled, editable content), so filers answer the
  ask instead of retyping it. (#98)


## [Unreleased] — pre-0.1.0 history

This section predates the `towncrier` convention and is retained as history.
It is older than the dated sections above, not newer.

### Added

- Changelog Automation: Adopted [towncrier](https://towncrier.readthedocs.io) to manage `CHANGELOG.md` via fragments in `changelog.d/`, restoring the no-merge-conflict, CI-enforced workflow of the bespoke system removed below — as a maintained dependency instead of custom infra code.
- Project Scaffolding: Initialized uv-based Python 3.12 environment and Rust backend via maturin/PyO3 (snakes_and_ladders.oxi_snakes_and_ladders).
- Module Architecture: Established core package skeleton (sim, likelihood, opt, search, infra) enforcing a strict separation between infrastructure and domain-specific application logic.
- Documentation Suite: Deployed Sphinx API docs, a LaTeX technical document for scientific foundations, and strategic planning documents (ROADMAP.md, STATUS.md, DEV.md).
- CI/CD Pipeline: Implemented GitHub Actions for Python/Rust linting, testing, documentation building, and dependency auditing using strictly locked environments (uv.lock, Cargo.lock).

### Changed

- Scientific Modeling: Formalized the Canonical Newick form and k-state Jukes–Cantor transition probabilities within the technical documentation.
- Strategic Roadmap: Defined strict engineering requirements (problem scale n=10−1000, RF ≤0.05, sub-second gradients) and partitioned development into six distinct workstreams.
- Framework Selection: Designated PyTorch as the primary autodiff engine and Aim for experiment tracking.
- Code Quality: Enforced mypy strict mode across all modules, required explicit np.random.default_rng for benchmarking, and applied standard linting rules to test suites.
- CI Optimization: Streamlined CI by caching uv/Cargo environments, auditing only modified dependency graphs, and auto-canceling superseded branch runs.

### Fixed

- Resolved dependency resolution failures in CI linting jobs and pinned cargo-audit to prevent floating resolve breakages.

### Removed

- Deprecated automated changelog fragments (changelog.d/) and associated infra/changelog.py script in favor of a standard flat file. (Superseded: this file's fragment-based workflow is restored via towncrier, a maintained dependency, rather than the bespoke infra removed here — see the towncrier adoption entry above.)
- Removed unused pytest-xdist dependency and legacy infra/ scaffolding modules.

### Security

- Raised pytest dependency to >=9.0.3 to patch vulnerability PYSEC-2026-1845.
