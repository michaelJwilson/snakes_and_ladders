# Developing snakes_and_ladders

Repository structure, CI enforcement, and contribution rules. For setup, see [INSTALL.md](INSTALL.md); for the project's trajectory and for the development loop these rules sit inside — ticket, plan, pull request, validation, record — see [ROADMAP.md](ROADMAP.md) §0. **`CLAUDE.md` is the authoritative source for conventions; in any conflict, `CLAUDE.md` prevails.**

This file is worked in, so it carries the mechanics in full — the layout, the checks, what the templates hold, how a release is cut — including where `ROADMAP.md` §0 has already stated the shape of them. Per `CLAUDE.md`'s altitude rule, detail may repeat between a planning document and a worked-in one; what it may not do is disagree. `ROADMAP.md` §0 is the loop's intent, and anything here that contradicts it is a defect here.

## Repository Layout

Infrastructure paths first, application paths after — the grouping below
carries the domain; `CLAUDE.md` states why keeping it liftable matters.

| Path | Contents |
| --- | --- |
| `benches/`, `tests/` | Criterion benchmarks (Rust), pytest suite, and integration tests. |
| `docs/source/` | Sphinx API documentation. |
| `PROBLEMS.md`, `CHECKS.md` | The problem catalogue, hand-written and resolved by a test; the checks ledger, written by `infra/checks_ledger.py --write` from the `oracle` and `simulated_truth` markers and compared by CI. |
| `python/snakes_and_ladders/` | Python package: re-exports, typed extension stubs, stub CLI. |
| `python/snakes_and_ladders/sim/` | Data generation and ground-truth retention. |
| `python/snakes_and_ladders/likelihood/` | Felsenstein pruning; CPU dispatch landed (NumPy, PyTorch, Rust), CUDA and Metal dispatch not yet implemented. Also the phylogenetic `Objective` (`objective.py`), which adapts the recursion to `opt/`'s fitting interface — it is here because `opt/` may import no application module. |
| `python/snakes_and_ladders/opt/` | Model-agnostic continuous parameter fitting via autodiff (PyTorch): the `Objective` interface, shared constraint maps, and the Potts and HMM reference instances. Imports nothing from `sim/`, `likelihood/` or `search/`, asserted by test. |
| `python/snakes_and_ladders/learn/` | Model-agnostic reinforcement learning: the `Environment` interface, the policy, REINFORCE, an exact trajectory-enumeration oracle, and a Potts-landscape reference instance. Imports nothing from `sim/`, `likelihood/` or `search/`, asserted by test. |
| `python/snakes_and_ladders/search/` | Move sets, temperature schedules, and the hill-climbing search (`infer.py`) that joins them to `opt/`. The phylogenetic RL environment (`rl.py`) lives here too, for the reason the phylogenetic `Objective` lives in `likelihood/`: `learn/` may import no application module. |
| `python/snakes_and_ladders/qa/` | QA figures/tables for the documents; renders, doesn't recompute. |
| `src/lib.rs` | Rust extension (`oxi_snakes_and_ladders`), exposed through PyO3. |
| `docs/tex/` | LaTeX source for the paper and the textbook, with the notation and preamble both share. |
| `infra/build_technical_doc.sh` | Regenerates QA figures, then builds `docs/paper.pdf` and `docs/textbook.pdf` (both committed; the `.aux`, `.bbl`, `.log` and other files `latexmk` leaves beside them are ignored, never committed). |

*Note: Each directory contains a localized `CLAUDE.md` defining specific constraints (e.g., `sim/` oracles, `search/` constraints). These append to, rather than override, the root `CLAUDE.md`.*

New issues are filed through `.github/ISSUE_TEMPLATE/task.yml`; blank issues are disabled via `.github/ISSUE_TEMPLATE/config.yml`.

## Test Layout

`tests/` is organized by **kind** at the top level and by subject within it. Where a new test goes follows from what kind of check it is, not from what it covers.

| Path | Holds |
| --- | --- |
| `tests/regression/` | Correctness. Asserts scientific validity against an independent oracle. |
| `tests/regression/{sim,likelihood,opt,learn,search,qa}/` | Split by submodule, the outgrown-flat-directory case below. |
| `tests/regression/` (top level) | Regression tests belonging to no submodule — `test_numerics.py`, `test_claude_md_pointers.py`, `test_pairwise_distance.py` (scaffolding). |
| `tests/benchmarks/` | `pytest-benchmark` timings. Asserts shape only; correctness is pinned by the regression counterpart. `profile_hotpaths.py` is the one exception — a `cProfile` self-time diagnostic, not a timing, and not `pytest`-collected. |
| `tests/regression/fixtures/` | Declarative test data (e.g. `simulation_params.yaml`). Data, not Python. |
| `tests/` (top level) | Whole-package and binding smoke tests, which belong to no single kind or submodule — `test_run_snakes_and_ladders.py`, `test_oxiphylo_bindings.py`. |

* **Every benchmark pairs with a regression module.** `benchmarks/test_<name>_bench.py` accompanies `regression/test_<name>.py`. A benchmark without a counterpart asserts nothing about correctness, which `CLAUDE.md`'s "No Coverage Theatre" rule forbids. `profile_hotpaths.py` is not a benchmark in this sense — it ranks self time for a Rust-port audit, asserts nothing, and pairs with no regression module — so the rule does not apply to it.
* **Split by submodule only when a kind outgrows one flat directory** — `tests/regression/likelihood/`, not a top-level `tests/likelihood/`. Kind stays the outer axis; a subject-first split would fight the two directories already there. `tests/regression/` reached 39 flat modules and was split under issue #154; `tests/benchmarks/` is 16 and stays flat.
* **A pull request runs the tests its change can affect, not all of them.** `infra/select_tests.py` turns the changed files into the test paths to run and the modules to measure coverage over; `python-tests` calls it (issue #161). Three rules make it safe. A module's dependents are derived from the import graph and run too, because `snakes_and_ladders.search` imports `snakes_and_ladders.likelihood` and a change to the latter can break the former. A change it cannot attribute to one module — a lockfile, a shared fixture, `src/`, the workflow itself — selects everything. And a diff that changed no code selects nothing, because the suite would then run identical tests over identical source to the last run on `main`.

  Benchmarks are selected the same way, by the regression module each pairs with: a `learn` change times `test_learn_reinforce_bench.py` and not the other twelve.

  Measured on one development machine, `pytest -m "not release"` with the coverage gate, before and after:

  | Diff | Before | After | Selected |
  | --- | --- | --- | --- |
  | documentation only | 174.0 s | **0.0 s** | nothing |
  | `learn/` | 174.0 s | **17.1 s** | learn |
  | `qa/` | 174.0 s | **59.1 s** | qa |
  | `search/` | 174.0 s | **115.0 s** | qa, search |
  | `opt/` | 174.0 s | **163.7 s** | learn, likelihood, opt, qa, search |
  | lockfile, shared fixture, `src/` | 174.0 s | 174.0 s | everything |

  The saving is uneven by design, and the reason is the import graph rather than the machinery. Attributed by `--durations=0` over one run: `qa` is 33.6% of the suite's time, benchmarks 29.6%, `search` 20.9%, and the rest under 6% each. `snakes_and_ladders.qa` imports four of the other five modules, so most changes reach the most expensive component; a change to `opt`, which everything depends on, saves almost nothing. `snakes_and_ladders.learn` is imported by nothing, so a change there saves 90%.

* **Coverage is measured against what was selected.** Where the whole suite runs, that is the package, as before. Where a subset runs, the claim narrows to *every module this pull request touched is at least 90% covered by that module's own tests* — stricter in one direction, since a module stops counting coverage it gets only incidentally from another module's tests, and weaker in another, since an untouched module is not re-checked. Measured on `main`: sim 100%, likelihood 100%, learn 100%, opt 99%, qa 99%, search 98%, so the gate holds without a new test. The package-wide gate still runs on every push to `main` and in `infra/release.sh`, which is what stops an unselected module rotting.
* **Fixtures follow their blast radius.** Used by one module: keep it in that module, or in a local `conftest.py`. Shared across modules: a top-level underscore-prefixed module such as `tests/_example_hotpath.py`, which is imported rather than collected.

---

## Infrastructure & Tooling

### Build System

`maturin` builds the Rust extension natively during `pip install .`.

* **Requirement:** A Rust toolchain is required for consumers.
* **Known Gap:** The typed stub `python/snakes_and_ladders/oxi_snakes_and_ladders.pyi` is hand-written. Run `python -m mypy.stubtest snakes_and_ladders.oxi_snakes_and_ladders` periodically to prevent drift.

### Continuous Integration

Ten required checks run via GitHub Actions (`.github/workflows/ci.yml`) on PRs against `main`:

| Job | Execution |
| --- | --- |
| `lint` | `ruff check`, `ruff format --check`, strict `mypy`, `towncrier check`, `infra/checks_ledger.py --check` |
| `rust-lint` | `cargo clippy -D warnings`, `cargo fmt --check` |
| `rust-tests` | `cargo test --locked`, `cargo bench` (informational) |
| `build` | `pip install .` (no lockfile, mimics fresh consumer), smoke import |
| `python-tests` | `pytest -m "not release"`, gated on minimum coverage; benchmarks skipped unless computational code changed |
| `docs` | Sphinx build (warnings as errors) |
| `technical-doc` | Regenerate the QA figures the documents cite (`infra/build_technical_doc.sh`), then LaTeX build. Fails on an undefined reference or citation, a multiply-defined label in either log, or a rebuilt PDF that differs from its committed copy |
| `notebooks` | Re-execute every notebook under `docs/nb/` (`infra/check_notebooks.py`) and fail on a re-executed output that differs from the committed one. Text is compared; a figure is checked only for still being produced. Regenerate with `--write` on the same script |
| `audit` | `pip-audit`, `cargo audit` (skips on cache hit if lockfiles are unchanged) |
| `pr-title` | The title starts with `[<base branch>]`, the branch the pull request targets (issue #292). One shell line; runs only where the workflow does, so on a stacked pull request it is the reviewer's until #273 lands |

`lint`, `python-tests`, `docs`, and `notebooks` restore a `~/.cache/uv` cache keyed on `uv.lock`'s hash before installing `uv`. `rust-lint`, `rust-tests`, `build`, and those same four jobs restore a shared `~/.cargo/registry`, `~/.cargo/git`, and `target/` cache keyed on `Cargo.lock`'s hash, so `oxi_snakes_and_ladders` (built via `maturin`/`pyo3` on every `uv sync` or `pip install .`) compiles from scratch only when a lockfile changes or no job has populated the cache yet. `audit`'s per-week marker cache (above) is unrelated and unaffected.

### CI & Performance Budget

* **Size Caps:** Restrict topological move tests to $n \le 10$ (exhaustive enumeration oracle).
* **No CI Profiling:** Do not rank performance on GitHub runners due to hardware variance. Benchmark on fixed hardware.
* **Three tiers, two budgets.** A test's tier is decided by *what its size is for*, never by how slow it happens to be: a size chosen so an exact oracle stays available is a CI size even when it is slow, and a size chosen to show behaviour at scale is a stress size even when it is fast.

  | tier | marker | budget | contents |
  | --- | --- | --- | --- |
  | CI | none (the default) | **5 minutes**, worst case | correctness at the smallest size that exercises the claim |
  | developer / stress | `stress` | **10 minutes** | the same claims at a size the CI budget cannot hold |
  | release | `release` | unbounded | long-running scientific validity, run by `infra/release.sh` |

  * **Use `pytest -m "not release and not stress"` while developing.** That is the CI tier, and the gate a pull request is judged against.
  * **The 5 minutes is the worst case, not the average.** `infra/select_tests.py` usually selects less, but it answers "everything" for any change it cannot attribute to one module — a lockfile, a shared fixture, `infra/` — so the full CI tier is the number that has to fit.
  * **A size that exists to show scaling is parameterized, never duplicated.** `tests/_scale.py`'s `at_scale` runs one test body at both sizes, so a change to the assertion reaches the large size by construction; two tests would let the large one drift until it asserted something the small one no longer did. `stress_only` is for a claim with no smaller size that still asserts it, and states the reason on the marker.
  * **Measured on one development machine, uncontended:** the CI tier runs in 141 s over 825 tests, the stress tier in 51 s over 10. Before issue #132 the same tests were one tier at 263 s, inside the 5-minute budget by 37 s and rising. `infra/measure_test_budget.sh` reproduces both and reports each against its budget. The previously documented figure — 138 s over 540 tests — had gone stale by a factor of 1.5 in tests and 1.9 in wall clock.
  * **The budgets are not asserted in the suite.** A wall-clock assertion would fail for the machine rather than for the change, which the "No CI Profiling" rule above forbids. What is asserted is structural: `tests/regression/test_scale_tiers.py` checks that the stress tier stays reachable and that no test carries a size marker it does not use.
  * **Plain `pytest` (no `-m` filter) is the release gate's job, not a development command.** `infra/release.sh` runs it as part of cutting a release; run it by hand only when you are cutting one, or when you have changed a release-gated test itself.
* **Two axes select tests, and they answer different questions.** `infra/select_tests.py` chooses **by module path** — what a diff could have broken. The *kind* markers choose **by what a test is checked against** — `oracle`, `simulated_truth`, `mathematical`, `edge_case`, `structural`, registered in `pyproject.toml` and required of every test outside `tests/benchmarks/` by `tests/regression/test_test_kinds.py`. They sit beside each other rather than one replacing the other: path selection carries the dependency reasoning issue #161 built, and the kinds are how you ask for a class of check independently of where the change landed. `--strict-markers` is on, so a misspelled marker fails collection instead of silently selecting nothing.
* **`critical` is the early gate, and it is a second axis rather than a kind.** It marks the tests whose failure invalidates everything after them — the import graph, the documentation index, the `CLAUDE.md` pointers, `select_tests` itself, the compiled extension, and the categorical sampler. **76 tests in 3.0 s**, measured uncontended on one development machine, against a `-m "not release"` suite of several minutes. CI runs `pytest -m critical` unconditionally before selection, so a broken invariant reports in seconds; run it locally the same way. A test is critical *and* a kind, never instead of one.
* **Benchmarks are conditional**, and are 29.6% of the suite's wall clock (40.2 s of the 136.0 s attributed to tests). They measure code a documentation or QA change cannot have altered, and since issue #161 they are selected per module rather than all together. The job itself always runs and always reports — it is a required check, and skipping the job rather than the step would leave it pending and block the merge. Coverage is unaffected, because every line a benchmark reaches is also reached by the regression module it pairs with.
* **The documents decide which figures a pull request rebuilds, and it is the *union* of what they cite.** `snakes_and_ladders.qa.manifest` states which QA outputs exist and what renders each one; `infra/build_technical_doc.sh` passes every document to the selection. Deriving it from one document would stop regenerating the other's figures and fail nothing, which is issue #154's defect in mirror image, so `cited_stems` refuses an empty set of documents and a test pins that leaving one out selects a smaller set. The figures neither cites are regenerated and compared at the release gate instead (`infra/release.sh` runs `snakes_and_ladders.qa.build --all --check`), so the check moves rather than disappearing. The cost is the reason the split is scoped: rendering all thirteen figures costs 281.6 s, the two documents together cite seven, and a full build of both PDFs is **60.1 s** against **5.9 s** when one document cited two. `topology_accuracy` is the reason it is seven and not eight — at **124.0 s** alone it is more than twice the rest of the build, so it stays at the release gate with the other five. Citing a figure the manifest cannot render fails the build rather than skipping it. `infra/measure_build.sh` reproduces these numbers on fixed hardware.
* **Tolerances on a quantity that scales with problem size are relative.** The log-likelihood is a sum over sites, so an absolute bound fixed at one site count does not transfer to another: the backends agree to ~8e-13 relative at every size, but that same agreement is 7.4e-07 absolute at 200,000 sites. Absolute bounds are correct for quantities that do not scale — a transition probability, a row sum, a Monte Carlo frequency — and are kept there.
* **Concurrency:** Superseded CI runs on the same branch are automatically cancelled.

### The Continuous Optimization Contract

Moved here from the technical document (issue #249): it is a statement about
the code's architecture rather than about a model, and the Altitude rule makes
this file the single copy. The *mathematics* of the constraint map is the
textbook's; what follows is what the implementation guarantees.

* **One optimizer, three model classes.** An objective is a differentiable
  scalar over an unconstrained vector, with a map back to the parameters the
  model is stated in. Nothing in `snakes_and_ladders.opt` may import
  `snakes_and_ladders.sim`, `.likelihood` or `.search`, and a test asserts it:
  a single convenience import turns a model-agnostic optimizer into a
  phylogenetics-specific one, and neither `ruff` nor `mypy` would notice.
* **Feasibility by construction, never by projection.** Positive parameters
  through a log or softplus map, distributions through a softmax on one fewer
  free value than the distribution has entries. Every point in the
  unconstrained space is a legal model, so no iterate has to be pushed back.
* **A structural move constructs a new objective.** It changes what the
  parameter vector means and how long it is, so it cannot be a step inside a
  fit over a fixed-length vector. The loop proposing moves owns that
  construction and fits per candidate.
* **Intervals come from the observed information, pushed through the
  constraint map by the delta method**, and are refused where the information
  is singular or worse-conditioned than a stated bound — an interval around a
  parameter the data does not identify summarizes nothing.

### Run Logs

Every entry point — a QA script, `snakes_and_ladders.qa.build`,
`infra/check_notebooks.py` — logs through `snakes_and_ladders.log` (issue
#311), and a line reads

```
2026-09-07 16:20:01 - 1.50m - INFO (render sim_tree) - snakes_and_ladders.qa.runner.figure_main:190 - wrote docs/tex/figures/sim_tree.pdf and docs/tex/figures/sim_tree_caption.txt
```

* **The second field is elapsed minutes** since the entry point started, so a
  slow step is located by subtracting neighbours rather than by wall-clock
  arithmetic.
* **The parenthesis is the run's phase**, set with `phase("...")` around the
  step and shared by every logger in the process; a line between phases has
  none.
* **Logs go to stderr**; stdout carries only what another script reads, such
  as the stems `qa.build --list` prints.
* **Libraries emit, entry points configure.** `opt.fit` and `search.infer` log
  at DEBUG through `logging.getLogger(__name__)` and install no handler; an
  entry point that wants those lines passes `level=logging.DEBUG` to
  `get_logger`. `warning_once` and `info_once` say a repeated thing once per
  logger, for a warning inside a loop.

### Profiling a Hot Path

`CLAUDE.md`'s **Runtime Optimization Opportunities** lists what to look for; this is the order to look, on fixed hardware per **No CI Profiling** above.

1. `python tests/benchmarks/profile_hotpaths.py` — a `cProfile` self-time ranking for `sim`, `search` and `learn` at a CI-sized fixture and one larger size. Not collected by `pytest`; run by hand and read. A candidate not near the top does not proceed.
2. `pytest tests/benchmarks/test_<name>_bench.py` — the NumPy or PyTorch baseline at the size the port would run at. The 10x rule is stated against realistic sizes, not the smallest that fits CI.
3. Time the port **alone** (`cargo bench`, Criterion, in `benches/`) **and through its binding** (`tests/benchmarks/`); the difference is the FFI boundary, and the pull request reports both.
4. For anything recursive, report peak memory beside time. No helper exists yet (`STATUS.md` records the memory requirement as not measured; #232 closes it), so take `tracemalloc` peaks by hand and say so.
5. Pin the port against the NumPy oracle within its tolerance before reporting the speedup.

`cProfile` cannot see inside a NumPy call or a Rust kernel; `pytest-benchmark` reports wall clock and nothing about cache, branches or vector width; Criterion times a kernel with its inputs already in Rust. Each ranks or times, none explains — the explanation is a change and its measured effect.

### Running With Workers

`snakes_and_ladders.parallel.map_tasks` is the one seam for CPU parallelism over
independent tasks (issue #344); no module keeps a pool of its own. Three sites
go through it — `opt.fit.fit_from` (starts), `opt.budget.compare` (cells of
method × instance × seed) and `search.support.bootstrap_support` (replicates) —
and each takes `workers=` explicitly.

* **`workers` is an argument, never a default or an environment variable.**
  `workers=1` is the serial loop; the QA runner and a test pass `1`; a run on
  a bigger machine passes its core count and gets the same numbers faster. A
  default read from the machine would change a run nobody edited.
* **A parallel run is bitwise the serial run.** Randomness is one generator
  per task, spawned in item order from the caller's with
  `numpy.random.Generator.spawn`, so task `i` draws the same stream at every
  worker count; results return in input order; and every worker runs at the
  intra-op thread count the site names, as the serial path does, so the same
  kernels reduce in the same order. Each site pins `workers=1` against
  `workers=4` with `==` or `torch.equal`, never `allclose`.
* **The thread rule.** `torch` and BLAS multithread inside a kernel, so a
  pool of workers each at the default thread count oversubscribes the cores;
  where a pool pays, pin `intra_op_threads` to cores divided by workers.
  `map_tasks` sets it per worker from that argument (`None` leaves the
  process's setting alone) and restores the caller's afterwards. The three
  sites pass `None`, by measurement: pinning one thread slowed the serial
  multi-start fit 2.9× (1.40 s against 0.49 s at 8 taxa × 1000 sites),
  because torch's intra-op parallelism over the sites is the parallelism that
  pays there, and no pool reached 2×. Serial and workers then run at the
  same count on one machine, which is what keeps the two bitwise equal. A
  site whose task body is a `torch` op above the parallel grain, or a Rust
  kernel under `allow_threads`, is the case for `backend="threads"`; a Python
  loop that holds the GIL is the case for `"processes"`, at the cost of
  pickling the task and result and of spawning the pool (the `spawn` start
  method, the one that works with `torch` and on Apple Silicon: workers
  import the package afresh, 1.81 s for 4 workers on the 4-core host, a
  fixed cost per call that `STATUS.md` reports beside each speedup).
* **The hardware.** Speedups are measured on fixed hardware per **No CI
  Profiling** above, with the core count and the 1-minute load stated beside
  every number; `STATUS.md` §0 carries the inventory of loops and the
  speedup matrix at 1, 2 and 4 workers with intra-op threads at 1 and at the
  default. A site under 2× at 4 workers is recorded there as a negative result
  and left serial — all three sites are, on the 4-core host, so a caller
  passes `workers=1` until a measurement on its own hardware says otherwise.
  Time only on an uncontended machine: a shared host at load above its core
  count reports the contention, not the code.
* **Not yet through the seam** (`TICKETS.md`, #344): the candidate fits of
  `search.infer`, `learn.rollout` batches, tempering replicas, `qa.build` and
  `infra/check_notebooks.py`, and `pytest-xdist` for the suite.

### Core Development Standards

* **Reproducibility:** Pin the environment. Use `--locked` for CI installs, pin runner images (`ubuntu-24.04`), and seed every generator through `np.random.default_rng(seed)`.
* **Versioning:** Lives in `Cargo.toml` (`[package].version`), and nowhere else.
* **Definition of Done:** Follow `CLAUDE.md`'s checklist.
* **A PR implements a plan already approved.** The ticket carries a plan comment before any code exists, the issue is labelled `planned`, and a maintainer applies `approved`; only then may the pull request open, and it implements that plan. A plan that turns out to be flawed gets a revised plan posted to the thread, not a silent correction in the diff.
* **Record the branch before the first commit.** Once a branch is created for an approved plan, the first thing posted is a single issue comment naming the branch (and, once opened, the PR number) — before any further commit is pushed, so an interrupted or deferred session leaves a ticket that already points at the in-flight branch.
* **A plan is 2–5 steps**, or more where the work needs them and the plan says why, each stating how it will be validated — the analytic result, brute-force computation or enumeration it is checked against, not "tests pass". It ends with an `Open Questions` section carrying every question on the desired behaviour, so a reviewer finds them in one place; a plan with none says so under that heading rather than omitting it.
* **PR Template:** Every PR starts from `.github/pull_request_template.md`. It carries the Definition-of-Done checklist, a benchmark-numbers table, a Documentation Sync line, and a Follow-up / Deferred Work section for anything left to a tracking issue. A second table in the Benchmark section takes the realized value of every scientific or tolerance test the PR touches — test, reference, tolerance, realized value — or the text "N/A" and no table. The template reminds; it is not a CI gate.
* **Agentic Approach:** Disjoint tickets run as parallel git worktrees and parallel pull requests; coupled changes run as a single sequential chain, each stacked on the last. Stated the same way in `ROADMAP.md` §0.2.
* **Every pull request names its base branch in its title, and targets it.** The title starts with the base branch in brackets — `[main]` for a root, `[claude/phylo-249-document-split]` for a link in a chain — and the base is set to that branch (issue #292). The base branch rather than the parent's PR number, because it is what the pull request's `base` field holds, so the `pr-title` job checks the two agree, and it survives the parent being renumbered or closed. A stacked pull request targets its *parent branch* rather than `main`, so the diff under review is the change itself and not everything beneath it — measured on the #190 chain, #225 was 64 changed files against `main` and 11 against its parent. GitHub retargets a child to `main` on its own when the parent merges, so the title prefix is the only part to update by hand, and the check fails until it is.
* **A chain merges bottom-up, with a merge commit.** Squash and rebase-merge both rewrite the parent's commits into new SHAs, after which the child no longer contains them: its diff duplicates the parent's content and every pull request below it conflicts. A merge commit preserves the ancestry, so each child's diff narrows to its own change the moment its parent lands. Bring a chain up to date the same way — cascade `main` into the root, then each parent into its child — and never rebase or force-push a branch, which invalidates any checkout of it and leaves the stale heads issue #123 records. Where two subtrees share a root, take the longer one first: whichever goes second is reconciled per branch, so the shorter chain is the cheaper one to leave until last.

### Dependency Management

1. **Request:** Explicitly request permission before adding dependencies/tools.
2. **Validate:** Must use OSI-approved licenses. Flag items with $<1000$ GitHub stars.
3. **Lock:** Run `uv lock` or update `Cargo.lock` and commit in the same PR.
4. **Justify:** Explain the inclusion in the PR description.

### Experiments

A measured comparison lives in `docs/experiments/` as one file per experiment,
written from `TEMPLATE.md` (issue #314): YAML front matter with the commit,
branch and pull request, the tickets it tests and files, the problem, fixture
and size tier, the methods compared, the budget and its unit, the shared
seeds, the hardware and a status; then fixed sections — feature under test,
setup, results, figures, finding, conclusion and actions, what is not
claimed. `infra/experiments.py` validates every file and generates the index
`README.md`; `--check` fails on an invalid file or a stale index, and
`tests/regression/test_experiments.py` runs it per pull request.

* **A number stated against a baseline lives in an experiment file.** A pull
  request that measures one method against another adds or updates the
  experiment it belongs to, and `STATUS.md` cites the file rather than
  restating its table. The matrix the ledger fills is problems × size tiers ×
  method families, every cell run through `opt.budget.compare` at one budget
  over shared seeds; the index shows each cell's status and finding.
* **Status is a claim about the record.** `open` while the comparison runs,
  `confirmed` once the finding is stated against its commit, `retracted` when
  a later measurement contradicts it (the file stays, with the contradiction
  in its finding), `superseded` when a later experiment replaces it.
* **A run store, when one exists, generates the Results section.** Until the
  Aim ledger of #75 lands, results are typed from the measurement with the
  script that produced them named; after it, `qa/experiment.py` renders them
  from the store and the Markdown stays the reviewed artefact.

### Release

A release is cut from a Release-template issue (`.github/ISSUE_TEMPLATE/release.yml`):
it drives the repository-consolidation audit (roadmap progress, doc/code
consistency, duplicated machinery, suggested follow-up tickets) and gates on
`infra/release.sh` passing before a maintainer adds the `release` label.

1. **Run the gate.** `infra/release.sh` runs every per-PR CI check
   (`ruff check`, `ruff format --check`, `mypy --strict`, `cargo clippy -D
   warnings`, `cargo fmt --check`, `cargo test --locked`) plus what CI skips
   per PR: the full `pytest` suite including `@pytest.mark.release` tests
   (see "Release-Gated" above), `sphinx-build -W`, and
   `infra/build_technical_doc.sh`. It runs every check regardless of earlier
   failures and prints a pass/fail summary at the end; a non-zero exit means
   at least one check failed.
2. **Bump the version.** Edit `[package].version` in `Cargo.toml` — the
   single version source (CLAUDE.md) — then run `cargo build` so
   `Cargo.lock`'s `oxi_snakes_and_ladders` entry picks up the new version, and commit both.
   `maturin` reads the Python package version from the same field
   (`dynamic = ["version"]` in `pyproject.toml`), so nothing else needs
   editing.
3. **Build the changelog.** Run `uv run towncrier build --version
   <version>` from the repository root: it consumes every fragment in
   `changelog.d/`, deletes them, and inserts a dated `## [<version>]` section
   into `CHANGELOG.md` (see `changelog.d/README.md`). Commit the result.
4. **Tag and publish.** Open a PR with the version bump and changelog
   commit; once merged, tag the merge commit (`git tag v<version> && git
   push origin v<version>`) and publish a GitHub release from that tag,
   with the new `CHANGELOG.md` section as its body.

**A version whose changelog section exists is already spent.** `0.1.0`'s
section was built into `CHANGELOG.md` before the repository was tagged, and
fragments accumulated after it. Running `towncrier build` at that same version
writes a second section rather than extending the first, so step 2 bumps to
the next version whenever the top section of `CHANGELOG.md` already carries
the one in `Cargo.toml`. `infra/release.sh` does not check this; issue #146
records the gap.

---

## Application Standards

`CLAUDE.md` states these and this file does not restate them: **Performance**
for when a hot path earns a GPU port and why the NumPy reference stays,
**Testing & Quality Assurance** for what an assertion must establish, and
`docs/CLAUDE.md` for how the documents are built and kept true. They
were duplicated here until issue #146; a rule with two homes acquires two
meanings.
