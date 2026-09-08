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
| `PROBLEMS.md`, `CHECKS.md` | The problem catalogue, hand-written and resolved by a test; the checks ledger, written by `infra/checks_ledger.py --write` from the `oracle` and `simulated_truth` markers and compared by CI. `infra/problems_tables.py --write` reads both into the textbook's applicability tables (`docs/tex/generated/`), which a guard holds to a regeneration. |
| `python/snakes_and_ladders/` | Python package: re-exports, typed extension stubs, stub CLI. |
| `python/snakes_and_ladders/sim/` | Data generation and ground-truth retention. |
| `python/snakes_and_ladders/likelihood/` | Felsenstein pruning; CPU dispatch landed (NumPy, PyTorch, Rust), CUDA and Metal dispatch not yet implemented. Also the phylogenetic `Objective` (`objective.py`), which adapts the recursion to `opt/`'s fitting interface — it is here because `opt/` may import no application module. |
| `python/snakes_and_ladders/opt/` | Model-agnostic continuous parameter fitting via autodiff (PyTorch): the `Objective` interface, shared constraint maps, the initializers, the Hamiltonian sampler, the temperature schedules, the budget utility, and the Potts, HMM, mixture and test-function reference instances. Imports nothing from `sim/`, `likelihood/` or `search/`, asserted by test. |
| `python/snakes_and_ladders/learn/` | Model-agnostic reinforcement learning: the `Environment` interface, the policies, REINFORCE, the critic, actor–critic, PPO and the planner, an exact trajectory-enumeration oracle, the relaxations and the learned surrogates, with the Potts-landscape and hidden-path reference instances. Imports nothing from `sim/`, `likelihood/` or `search/`, asserted by test. |
| `python/snakes_and_ladders/search/` | Move sets, the hill-climbing and large-parsimony searches (`infer.py`) that join them to `opt/`, the samplers, annealers and tempered ensembles, the exact ground states, and the coupled model's block ascent. The phylogenetic RL environment (`rl.py`) lives here too, for the reason the phylogenetic `Objective` lives in `likelihood/`: `learn/` may import no application module. |
| `python/snakes_and_ladders/qa/` | QA figures/tables for the documents; renders, doesn't recompute. |
| `python/snakes_and_ladders/sandbox/` | The oracle home: a hand-rolled implementation a framework replaced on a hot path, kept to referee it. Imported by `tests/` and `qa/` only, asserted by test; empty until an adoption is measured (issue #322). |
| `src/lib.rs` | Rust extension (`oxi_snakes_and_ladders`), exposed through PyO3. |
| `docs/tex/` | LaTeX source for the paper and the textbook, with the notation and preamble both share. |
| `infra/build_documents.sh` | Regenerates QA figures, then builds `docs/paper.pdf` and `docs/textbook.pdf` (both committed, rebuilt only by a "Rebuild the documents" pull request; the `.aux`, `.bbl`, `.log` and other files `latexmk` leaves beside them are ignored, never committed). |

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
| `tests/regression/fixtures/` | Declarative test data, one directory per problem (e.g. `tree_jc/stress.yaml`). Data, not Python. |
| `tests/` (top level) | Whole-package and binding smoke tests, which belong to no single kind or submodule — `test_run_snakes_and_ladders.py`, `test_oxiphylo_bindings.py`. |

* **Every benchmark pairs with a regression module.** `benchmarks/test_<name>_bench.py` accompanies `regression/test_<name>.py`. A benchmark without a counterpart asserts nothing about correctness, which `CLAUDE.md`'s "No Coverage Theatre" rule forbids. `profile_hotpaths.py` is not a benchmark in this sense — it ranks self time for a Rust-port audit, asserts nothing, and pairs with no regression module — so the rule does not apply to it.
* **Split by submodule only when a kind outgrows one flat directory** — `tests/regression/likelihood/`, not a top-level `tests/likelihood/`. Kind stays the outer axis; a subject-first split would fight the two directories already there. `tests/regression/` reached 39 flat modules and was split under issue #154; `tests/benchmarks/` is 39 and stays flat (re-counted 2026-09-08 at the 0.5.0 audit, where `tests/regression/` is 23 flat modules beside its seven per-module directories).
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

  The saving is uneven by design, and the reason is the import graph rather than the machinery. Attributed by `--durations=0` over one run: `qa` is 33.6% of the suite's time, benchmarks 29.6%, `search` 20.9%, and the rest under 6% each. `snakes_and_ladders.qa` imports four of the other five modules, so most changes reach the most expensive component; a change to `opt`, which everything depends on, saves almost nothing. `snakes_and_ladders.learn` was imported by nothing when this was measured, so a change there saved 90%; since `snakes_and_ladders.search.gym` adapted its protocol (issue #322) a `learn` change selects `search`, and through `likelihood.features` → `search.topology` also `likelihood` and `qa`, so it costs what the `opt/` row does less `opt` itself.

* **Coverage is measured against what was selected.** Where the whole suite runs, that is the package, as before. Where a subset runs, the claim narrows to *every module this pull request touched is at least 90% covered by that module's own tests* — stricter in one direction, since a module stops counting coverage it gets only incidentally from another module's tests, and weaker in another, since an untouched module is not re-checked. Measured on `main`: sim 100%, likelihood 100%, learn 100%, opt 99%, qa 99%, search 98%, so the gate holds without a new test. The package-wide gate still runs on every push to `main` and in `infra/release.sh`, which is what stops an unselected module rotting.
* **The `frameworks` extra is optional for the suite and required for its pins.** It installs `gymnasium`, `rustworkx`, `torchrl` and `torch_geometric` (issue #322), and enables `snakes_and_ladders.search.gym` (the Gymnasium adapter over `learn.Environment`), `PottsGraph.to_rustworkx` / `from_rustworkx`, and six test modules that `importorskip` one package each and pin our implementation against it: `tests/regression/search/test_search_gym.py` (Farama's `check_env`, and an episode round-tripped through both interfaces), `tests/regression/sim/test_graph_rustworkx.py` (the generators against `rustworkx.generators`, the cut against `networkx`), `tests/regression/search/test_search_topology_rustworkx.py` (topology equality against `rustworkx.is_isomorphic`, issue #376), `tests/regression/learn/test_learn_ppo_torchrl.py` (GAE and the clipped loss against TorchRL's), `tests/regression/learn/test_learn_reinforce_torchrl.py` (the score-function loss against `ReinforceLoss`, #376) and `tests/regression/learn/test_learn_surrogate_pyg.py` (the graph surrogate against `GINConv`). `python-tests` syncs the extra so they run per pull request; without it they skip, and nothing else in the suite changes. Three further modules compare against **`scipy`**, a core dependency since issue #376 (their `importorskip` is kept so a trimmed environment skips rather than errors): `test_opt_fit_scipy.py`, `test_likelihood_hadamard_scipy.py` and `test_search_neighbor_joining_scipy.py` (#376). They skip everywhere until scipy is declared, which is the open question that ticket leaves standing; the repository otherwise writes out the few constants it would need from it (`opt.fit`, `search.statistics`). A framework that *replaces* an implementation on a hot path is a different step, gated on a measurement, and the replaced implementation then moves to `python/snakes_and_ladders/sandbox/` as its referee (`sandbox/CLAUDE.md`).
* **Fixtures follow their blast radius.** Used by one module: keep it in that module, or in a local `conftest.py`. Shared across modules: a top-level underscore-prefixed module such as `tests/_example_hotpath.py`, which is imported rather than collected.

---

## Infrastructure & Tooling

### Build System

`maturin` builds the Rust extension natively during `pip install .`.

* **Requirement:** A Rust toolchain is required for consumers.
* **Known Gap:** The typed stub `python/snakes_and_ladders/oxi_snakes_and_ladders.pyi` is hand-written. Run `python -m mypy.stubtest snakes_and_ladders.oxi_snakes_and_ladders` periodically to prevent drift.

### Documents

Pull requests change `docs/tex/` and never `docs/paper.pdf` or `docs/textbook.pdf`. The two PDFs are committed for readers and are rebuilt, from `main`, by the pull request a **Rebuild the documents** ticket asks for (`.github/ISSUE_TEMPLATE/documents.yml`), whose title carries that phrase; CI refuses a PDF change in any other pull request. Between rebuilds the PDFs lag the sources by design. The trade (issue #369): comparing the committed PDFs on every pull request made every document pull request conflict with every merge and cost one rebuild per merge per open pull request, eleven builds in one afternoon; a lag that one ticket closes costs nothing. Every pull request still builds both documents in CI and fails on an unresolved or duplicated reference or citation, so the sources are always buildable. `infra/build_documents.sh` runs against the synced environment (`uv run --no-sync`) and no longer rebuilds the Rust extension per build.

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
| `documents` | Regenerate the QA figures the documents cite (`infra/build_documents.sh`), then LaTeX build. Fails on an undefined reference or citation, a multiply-defined label in either log, or a pull request that changes a committed PDF without being the rebuild (see Documents) |
| `notebooks` | Re-execute every notebook under `docs/nb/` (`infra/check_notebooks.py`) and fail on a re-executed output that differs from the committed one. Text is compared; a figure is checked only for still being produced. Regenerate with `--write` on the same script |
| `audit` | `pip-audit`, `cargo audit` (skips on cache hit if lockfiles are unchanged) |
| `pr-title` | The title starts with `[<base branch>]`, the branch the pull request targets (issue #292). One shell line; runs only where the workflow does, so on a stacked pull request it is the reviewer's until #273 lands |

Branch protection names each required check by the job's `name:`, not by its id, so renaming one is two changes. Issue #377 renamed the LaTeX job to **Documents (paper and textbook)**, and merged as #379 on 2026-09-08; the maintainer replaces its former name — the one entry in the branch protection no job reports under any more — with that string. Until both say it, the renamed job reports and the retired entry sits pending, blocking every merge. Confirming that replacement is a precondition of the 0.5.0 release, and is where the 0.4.0 audit left it.

`lint`, `python-tests`, `docs`, and `notebooks` restore a `~/.cache/uv` cache keyed on `uv.lock`'s hash before installing `uv`. `rust-lint`, `rust-tests`, `build`, and those same four jobs restore a shared `~/.cargo/registry`, `~/.cargo/git`, and `target/` cache keyed on `Cargo.lock`'s hash, so `oxi_snakes_and_ladders` (built via `maturin`/`pyo3` on every `uv sync` or `pip install .`) compiles from scratch only when a lockfile changes or no job has populated the cache yet. `audit`'s per-week marker cache (above) is unrelated and unaffected.

### CI & Performance Budget

* **Size Caps:** Restrict topological move tests to $n \le 10$ (exhaustive enumeration oracle).
* **No CI Profiling:** Do not rank performance on GitHub runners due to hardware variance. Benchmark on fixed hardware.
* **Three tiers, two budgets.** A test's tier is decided by *what its size is for*, never by how slow it happens to be: a size chosen so an exact oracle stays available is a CI size even when it is slow, and a size chosen to show behaviour at scale is a stress size even when it is fast.

  | tier | marker | fixture | budget | contents |
  | --- | --- | --- | --- | --- |
  | CI | none (the default) | `<problem>/ci.yaml` | **5 minutes**, worst case | correctness at the smallest size that exercises the claim |
  | developer / stress | `stress` | `<problem>/stress.yaml` | **10 minutes** | the same claims at a size the CI budget cannot hold |
  | release | `release` | `<problem>/release.yaml` | unbounded | long-running scientific validity, run by `infra/release.sh` |

  * **The tiers are the fixture files' names.** A supported problem declares
    its instance per tier under `tests/regression/fixtures/<problem>/`, and
    `snakes_and_ladders.sim.fixtures` loads it (`PROBLEMS.md`); a problem
    declares a tier only where it has an instance for it. `tests/_scale.py`'s
    `at_fixture` parameterizes one test over every tier a problem declares,
    marking each case for its own tier, so a fixture added at a tier reaches
    every such test without one of them being edited.

  * **Use `pytest -m "not release and not stress"` while developing.** That is the CI tier, and the gate a pull request is judged against.
  * **Two gates, and the merge waits for the first.** Nine of the ten jobs finish inside 156 s; `python-tests` took 1,380 s on #403 and 2,155 s on #398, because `select_tests` answers "everything" for a change it cannot attribute and both of those touched a shared fixture. On a pull request that job now runs `infra/select_tests.py --budget`, which bounds the fallback to what it can attribute plus what always runs, and the job carries a 15-minute timeout so an overrun is a defect in the bound rather than a wait. The whole suite runs on the push to `main`, where nobody waits for it, and at the release gate. A change touching a likelihood kernel, the Rust sources or a lockfile refuses the bound and takes the unbounded selection before merging, because a regression there moves a number rather than breaking a build and every branch cut before the post-merge run would inherit it (issue #409). The trade is the same one the figure stamps make: the check moves rather than disappearing.
  * **The 5 minutes is the worst case, not the average.** `infra/select_tests.py` usually selects less, but it answers "everything" for any change it cannot attribute to one module — a lockfile, a shared fixture, `infra/` — so the full CI tier is the number that has to fit.
  * **A size that exists to show scaling is parameterized, never duplicated.** `tests/_scale.py`'s `at_scale` runs one test body at both sizes, so a change to the assertion reaches the large size by construction; two tests would let the large one drift until it asserted something the small one no longer did. `stress_only` is for a claim with no smaller size that still asserts it, and states the reason on the marker.
  * **The local validation is one command and 300 s.** `infra/validate.sh` runs, against the diff from `origin/main` (or `--base <ref>`), in the order that fails fastest and with each step's wall clock printed: `ruff` and `mypy --strict` over the changed Python files (CI keeps the whole `files` set); `pytest -m critical`; the tests `infra/select_tests.py` names for the diff, capped at 180 s, with `--durations=10`; `sphinx-build -W` only when `docs/source/` or a docstring changed, keeping its doctrees under the scratch directory so a second run is seconds; the notebook checker, which skips a notebook whose inputs are unchanged since its last committed execution; and the document build when `docs/tex/` changed, which renders only the cited figures whose inputs changed (issue #372). Three classes of change against the budget, measured on the reference host (4 cores, load under 1, one BLAS thread per process), stated in the pull request that measured them and reproduced by `infra/measure_test_budget.sh`:

    | change | what `select_tests.py` names | wall | budget |
    | --- | --- | --- | --- |
    | `docs/tex/` only | the three document guards (27 tests) | **0.4 s** | 300 s (target 90 s); LaTeX alone is the rest |
    | one module (`learn/`) | `learn`, `likelihood`, `qa`, `search` and their benchmarks (1,125 tests) | **1,323 s** | 300 s: **over** |
    | `search/`-wide | `qa`, `search` and their benchmarks | not measured; a subset of the row above | 300 s |

    Measured 2026-09-08 on the 4-core reference host, one BLAS thread per process, load under 1.5. The `learn/` class fails the budget by a factor of four, and the reason is not the selection but the tier: 18 of its tests run over 10 s, led by `test_nni_hill_climb_eight_taxa_benchmark` (180 s cold, 118 s warm), the 4,000-episode critic estimator (111 s), the enumerated-support calibration (72 s) and the lazy NNI search (53 s). `infra/validate.sh` caps the selected tests at 180 s and reports the class as over budget rather than waiting; the duration guard below names the tests. The maintainer's decision (2026-09-08): every test over the cap carries `release`. The 19 tests over 10 s in that run, 817.8 s of its 1,323 s between them, moved in the pull request that records this; the class's remaining 1,106 tests average 0.46 s, so the selection is bounded by their count rather than by any one of them, and running it under `pytest-xdist` with one thread per worker stays the next step if it grows past the budget again.

  * **A test over 10 s carries `release` or `stress`, or fails the validation; the default is `release`.** A test over the cap is re-tiered, not waited for: it moves to `release`, where `infra/release.sh` runs it with every other tier, and to `stress` only when a developer needs it inside the 10-minute local budget. A claim that moves keeps a fast small-size sibling per PR where one exists; where none does, the gap is a ticket, not a new test. `tests/conftest.py` records every test's call duration; with `SAL_DURATION_CAP` set, as `infra/validate.sh` sets it on the reference host, a test over the cap without either marker fails the session by name. CI does not set it, per **No CI Profiling**, and prints `--durations` instead. The cap is asserted where the hardware is fixed and reported where it is not, so a pull request cannot add a minute to the CI tier unnoticed and cannot fail for the runner's speed either.
  * **One process is one core.** `tests/conftest.py` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 unless the caller did, and `infra/validate.sh` exports them, so the load average counts processes and four of them fit the host. The figure renderers are exempt: `snakes_and_ladders.qa.build` strips the three from a render's environment, because a committed figure is rendered the way the manifest renders it, and a reduction split over a different thread count can move its last bit.
  * **The environment is synced once per worktree.** `uv sync --locked` runs when the lockfile changes; every script exports `UV_NO_SYNC=1`, and a shell that runs `uv run` by hand should too (`INSTALL.md`). Without it each command re-resolves the environment and, in a fresh worktree, recompiles the Rust extension.
  * **Measured on one development machine, uncontended:** the CI tier runs in 141 s over 825 tests, the stress tier in 51 s over 10. Before issue #132 the same tests were one tier at 263 s, inside the 5-minute budget by 37 s and rising. `infra/measure_test_budget.sh --full` reproduces both and reports each against its budget; without `--full` it measures the critical tier only, since the full tiers take a shared host for their duration. The previously documented figure — 138 s over 540 tests — had gone stale by a factor of 1.5 in tests and 1.9 in wall clock. Neither full tier was re-measured at the 0.4.0 audit (2026-09-08): the host was shared with a document build and a second agent throughout, and a CI-tier run started under that load was stopped at 66 CPU-minutes, so the two numbers above stand from their last uncontended measurement.
  * **The budgets are not asserted in the suite.** A wall-clock assertion would fail for the machine rather than for the change, which the "No CI Profiling" rule above forbids. What is asserted is structural: `tests/regression/test_scale_tiers.py` checks that the stress tier stays reachable and that no test carries a size marker it does not use.
  * **Plain `pytest` (no `-m` filter) is the release gate's job, not a development command.** `infra/release.sh` runs it as part of cutting a release; run it by hand only when you are cutting one, or when you have changed a release-gated test itself.
* **Two axes select tests, and they answer different questions.** `infra/select_tests.py` chooses **by module path** — what a diff could have broken. The *kind* markers choose **by what a test is checked against** — `oracle`, `simulated_truth`, `mathematical`, `edge_case`, `structural`, registered in `pyproject.toml` and required of every test outside `tests/benchmarks/` by `tests/regression/test_test_kinds.py`. They sit beside each other rather than one replacing the other: path selection carries the dependency reasoning issue #161 built, and the kinds are how you ask for a class of check independently of where the change landed. `--strict-markers` is on, so a misspelled marker fails collection instead of silently selecting nothing.
* **`critical` is the early gate, and it is a second axis rather than a kind.** It marks the tests whose failure invalidates everything after them — the import graph, the documentation index, the `CLAUDE.md` pointers, `select_tests` itself, the compiled extension, and the categorical sampler. **149 tests in 13.0 s** (pytest's clock; 15 s wall; 146 run and 3 skipped without `scipy`), re-measured on 2026-09-08 at the 0.5.0 audit on the 4-core host at a 1-minute load of 0.45 with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, against a `-m "not release"` suite of several minutes; at the 0.4.0 audit it was 88 tests in 12.6 s under a load of 2.27, and the earlier uncontended figure was 76 tests in 3.0 s. The count rose with #372's stamp and duration guards and #376's problem-statement guard, and the clock did not, which is what a guard tier is for. CI runs `pytest -m critical` unconditionally before selection, so a broken invariant reports in seconds; run it locally the same way. A test is critical *and* a kind, never instead of one.
* **Benchmarks are conditional**, and are 29.6% of the suite's wall clock (40.2 s of the 136.0 s attributed to tests). They measure code a documentation or QA change cannot have altered, and since issue #161 they are selected per module rather than all together. The job itself always runs and always reports — it is a required check, and skipping the job rather than the step would leave it pending and block the merge. Coverage is unaffected, because every line a benchmark reaches is also reached by the regression module it pairs with.
* **A figure is rendered only when its inputs changed.** Beside every committed figure is a stamp, `docs/tex/figures/<stem>.inputs`, recording a digest of what the renderer *executes*, the fixtures its arguments name, the spec itself and the drawing libraries' versions, at the render that produced it (`snakes_and_ladders.inputs`, issues #372 and #418). What it executes is a walk from the renderer's module over the definitions the AST says it reaches — a call, a constant read, a base class, a decorator, a default argument — hashed as their ASTs with docstrings and comments removed, plus the Rust sources when a reachable definition names the extension. Not the text of every module it imports: `qa/runner.py` is in all 19 cited closures, so under the old rule a docstring reworded there restamped all 19 with zero PDF bytes changed (measured on #394), and two parameter constants declared there did the same. The walk over-approximates wherever it is unsure, because the two failure directions are not symmetric — a stamp that fires too often costs a render, one that misses a change publishes a figure the code no longer produces — so the entry module and every included class are hashed whole, and a module whose own text defeats the analysis (a star import, `getattr` on a module, `importlib`, a module `__getattr__`, a decorator from inside the package) is hashed whole *and named* in `Reach.fallbacks`, which enters the digest. `tests/regression/test_inputs.py` pins both directions: prose and an unread constant in the shared module stale nothing; an edit to `figure_main` stales the 18 figure renderers and an edit to `table_main` stales only the table. `snakes_and_ladders.qa.build` renders a cited figure only when its stamp differs from the digest of the current tree, so a pull request confined to `docs/tex/` renders nothing and its build is LaTeX alone; `--all`, the release gate, ignores the stamps. A figure whose stamp is stale but whose bytes still match is reported by `--check` so the stamp can be committed, and until it is the figure is re-rendered on every build, which costs time and is never wrong. A cited figure renders in at most 30 s on the reference host (`CITED_RENDER_CAP`); the manifest carries each figure's measured time, and a guard fails a cited entry over the cap unless it is waived in `CAP_WAIVERS` with the ticket that will bring it under. `rl_tree_policy`, at 101.4 s, is waived under #372. The notebooks use the same mechanism: `docs/nb/<name>.inputs` beside each, written by `infra/check_notebooks.py --write` and read by the checker, which skips a notebook whose inputs are unchanged; `--all` executes every one.
* **A reference algorithm's answer on a fixture is committed, not recomputed per pull request.** Beside `<problem>/<tier>.yaml` is `<problem>/<tier>.baseline.json`: what a *reference* algorithm achieves on that instance — the enumerated maximum over its topologies, the rate at which NNI hill climbing reaches it from 50 seeded starts, the rate an untrained policy reaches it, the exact maximum-likelihood target of every topology a surrogate is fitted against — each with the algorithm, the seed and the budget that produced it, and one digest over the fixture file, the transitive import closure of the computing modules and the versions of `numpy`, `scipy` and `torch` (`snakes_and_ladders.inputs`, issue #401). `infra/baselines.py --write` computes and writes; `snakes_and_ladders.sim.fixtures.baseline` reads and **raises** where the digest is not the current tree's, so a changed fixture or a changed search fails on the pull request that changed it rather than serving a number the tree no longer produces. The record is a second file rather than a block in the yaml, because the fixture declares an instance and is written by hand while the record states a measurement and is written by a tool.

  The recomputation moved to the release gate, where `infra/release.sh` runs `infra/baselines.py` with no flag and fails on any drift — the same trade `snakes_and_ladders.qa.build --all --check` makes for the figures neither document cites. Per pull request nothing recomputes: three claims cost 8.3 s of uniform rollouts, 7.5 s of maximum-likelihood fits and 1.9 s of exact expected returns before measuring anything about the code under test, which is most of why they left the tier. The record's *value* is deliberately outside the digest — it is the output, and a wrong one is caught by recomputing rather than by hashing itself — while the budget beside it is inside, so a record edited to match a test is refused on the next read.

* **The documents decide which figures a pull request rebuilds, and it is the *union* of what they cite.** `snakes_and_ladders.qa.manifest` states which QA outputs exist and what renders each one; `infra/build_documents.sh` passes every document to the selection. Deriving it from one document would stop regenerating the other's figures and fail nothing, which is issue #154's defect in mirror image, so `cited_stems` refuses an empty set of documents and a test pins that leaving one out selects a smaller set. The figures neither cites are regenerated and compared at the release gate instead (`infra/release.sh` runs `snakes_and_ladders.qa.build --all --check`), so the check moves rather than disappearing. The cost is the reason the split is scoped: when the manifest held thirteen figures, rendering all of them cost 281.6 s, the two documents together cited seven, and a full build of both PDFs was **60.1 s** against **5.9 s** when one document cited two; `topology_accuracy` at **124.0 s** alone was more than twice the rest of the build, which is why it stays at the release gate. At the 0.4.0 audit the manifest holds nineteen figures and the documents cite seventeen — only `sim_problem_sizes` and `topology_accuracy` are checked at the release gate (#325) — and the four new figures render in 4.6, 3.9, 27.4 and 6.6 s each on the 4-core audit host; the full-build timings were not re-measured then, because the host was shared for the whole audit. Citing a figure the manifest cannot render fails the build rather than skipping it. `infra/measure_build.sh` reproduces these numbers on fixed hardware.
* **Tolerances on a quantity that scales with problem size are relative.** The log-likelihood is a sum over sites, so an absolute bound fixed at one site count does not transfer to another: the backends agree to ~8e-13 relative at every size, but that same agreement is 7.4e-07 absolute at 200,000 sites. Absolute bounds are correct for quantities that do not scale — a transition probability, a row sum, a Monte Carlo frequency — and are kept there.
* **Concurrency:** Superseded CI runs on the same branch are automatically cancelled.

### The Continuous Optimization Contract

Moved here from the paper (issue #249): it is a statement about
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

1. `python tests/benchmarks/profile_hotpaths.py --tier enumerable` and `--tier mid` (`--module` selects one) — a `cProfile` self-time ranking of one workload per module, the top five functions with the fraction of the run each carries. Not collected by `pytest`; run by hand and read. A loop under 10% of its run is recorded in `STATUS.md` and does not proceed (#341).
2. `pytest tests/benchmarks/test_<name>_bench.py` — the NumPy or PyTorch baseline at the size the port would run at. The 10x rule is stated against realistic sizes, not the smallest that fits CI.
3. Time the port **alone** (`cargo bench`, Criterion, in `benches/`) **and through its binding** (`tests/benchmarks/`); the difference is the FFI boundary, and the pull request reports both.
4. For anything recursive, report peak memory beside time. `snakes_and_ladders.qa.likelihood_footprint` states the pruning footprint from the arrays' shapes and its test pins it to the allocator (#232); for anything else take `tracemalloc` peaks by hand and say so.
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
* **A seam is added at three consumers, and `SEAMS.md` says which exist.** `infra/seams_survey.py` regenerates `SEAMS.md` from the package: every `Protocol` and every data contract three or more modules share, with its members, the classes that satisfy it structurally, the modules that consume it, the `PROBLEMS.md` rows it reaches, and a verdict against the rule; `tests/regression/docs/test_seams_survey.py` holds the file to the tool. A proposal for a new abstraction states the consumers that would call through it and is declined below three, as issue #400 declined one EM driver over the HMM and the mixture (two callers sharing 16 lines of loop skeleton, more lines added than removed). A seam already declared but under the rule carries its reason in the survey's table and prints it beside the verdict; one with no reason is a name and is removed or argued for. The problem classes have no `Problem` protocol: the fixture registry (`snakes_and_ladders.sim.fixtures`, issue #382) is their contract, since every consumer that would call a generic operation today calls the class's own function, and the survey shows no operation with three such callers.
* **Agents run in parallel; the host runs one *measurement* at a time.** Up to eight implementation agents work at once, each in its own worktree from `main` with `OMP_NUM_THREADS=1`. The cap is set by review bandwidth, not by the reference host's 4 cores: an agent's wall clock is mostly model latency, so the host is idle while it thinks. Measured on 2026-09-08 with eight agents alive, load average was 0.03 / 0.07 / 0.68 on four cores.

  Two locks under the scratch directory, taken through `infra/locks.sh`'s `with_lock <kind> -- <command>` and waited for in the foreground:

  | lock | slots | what takes it |
  | --- | --- | --- |
  | `measure` | 1 | anything whose wall time, throughput or memory reaches a pull request, a ticket, `STATUS.md` or a docstring |
  | `validate` | 3 | correctness runs nobody times: `ruff`, `mypy`, the critical tier, the changed-file run, `review_gates.sh`, a notebook check |

  A measurement takes both — `measure`, then every `validate` slot — so it still runs alone and the budgets above stay comparable between runs. **Exclusivity is bought for comparability, so it is spent only where a number is reported**; a correctness run's wall time appears nowhere, so serializing it bought nothing. Three validation slots and not four leaves one core for the orchestrating session, every job being single-threaded already. What this fixed, measured: under one exclusive lock an agent occupied a quarter of the machine while the others waited, and issue #399's Rust agent measured 40 minutes of starvation inside a 104-minute run at five agents — serialization rather than saturation, given the load average above. `tests/regression/test_host_locks.py` pins both halves: three validations overlap, and a measurement never runs beside anything.

* **Merges to `main` land in a batch, not as each pull request goes green.** Every merge obliges each live branch to fetch, merge, re-validate and push, so *n* merges across *b* live branches cost `n * b` cycles where one batch costs `b`. Six merges across eight branches in the hour before this was written is 48 cycles where 8 would have done. A pull request that unblocks a queued ticket is the exception and merges at once, the stall being worse than the saving.

* **Batch by plan shape; split by independence.** Tickets whose plans have the same shape — the same seam behind several callers, the same profile-first exit against a framework — land as one pull request, one agent, one CI run, one review (the 2026-09-08 queue: #390, #389 and #388 as one; #387 and #386 as one). A plan whose steps are independent runs as two agents on two worktrees and integrates into one pull request before it opens (#399: the emission and mixture beside the Rust kernels).
* **Validate locally what fails fast; the pull request's CI is the final validation.** Before a push an agent runs the changed-file `ruff` and `mypy --strict` and `pytest -m critical` (15 s). The full `infra/validate.sh` runs once, before the pull request leaves draft, or when the change touches a shared fixture, `src/` or a lockfile. The pull request is then watched to green and fixed on each red check; a local run that duplicates CI is time the queue pays twice.
* **Review starts from `infra/review_gates.sh`.** It runs the Definition-of-Done gates CI does not assert or asserts late — the environment imports this checkout, the head carries the base, the two PDFs are the base's unless the pull request is a rebuild, a changelog fragment exists, every cited figure's stamp matches the tree, the critical tier passes under `SAL_DURATION_CAP`, every test the branch adds or rewrites carries a kind marker, a `Protocol` it adds is in `infra/seams_survey.py`'s catalogue with the consumers the seam rule wants or a stated reason, and `CHECKS.md`, `SEAMS.md` and the problem tables are regenerations of the tree — and prints a pass/fail table, each row with its own wall clock. **Nine rows in 28 s**, measured 2026-09-08 on the 4-core host at a 1-minute load of 4.2, of which the critical tier is 22 s and the six older rows 23 s between them; the three added under #418 cost 0.3 s (the kind markers, read from the source), 0.1 s (the seam, which returns before the survey runs when the diff adds no protocol) and 5 s (the three ledgers' own `--check`). The budget is 30 s, above which a check belongs in CI rather than in something a reviewer runs per branch — which is why the duration cap on the *changed* tests is not a row: running them cost 20 s of the 30 on the branch that added these gates, and `infra/validate.sh` has already run them under the cap on this host. Reading then covers what a script cannot: the diff against the plan on the ticket, and whether each test pins what it claims to. Those two rows are the point of a review; automating them is not on the roadmap and a gate that claimed to would be worse than none.
* **Every pull request names its base branch in its title, and targets it.** The title starts with the base branch in brackets — `[main]` for a root, `[claude/phylo-249-document-split]` for a link in a chain — and the base is set to that branch (issue #292). The base branch rather than the parent's PR number, because it is what the pull request's `base` field holds, so the `pr-title` job checks the two agree, and it survives the parent being renumbered or closed. A stacked pull request targets its *parent branch* rather than `main`, so the diff under review is the change itself and not everything beneath it — measured on the #190 chain, #225 was 64 changed files against `main` and 11 against its parent. GitHub retargets a child to `main` on its own when the parent merges, so the title prefix is the only part to update by hand, and the check fails until it is.
* **A chain merges bottom-up, with a merge commit.** Squash and rebase-merge both rewrite the parent's commits into new SHAs, after which the child no longer contains them: its diff duplicates the parent's content and every pull request below it conflicts. A merge commit preserves the ancestry, so each child's diff narrows to its own change the moment its parent lands. Bring a chain up to date the same way — cascade `main` into the root, then each parent into its child — and never rebase or force-push a branch, which invalidates any checkout of it and leaves the stale heads issue #123 records. Where two subtrees share a root, take the longer one first: whichever goes second is reconciled per branch, so the shorter chain is the cheaper one to leave until last.

### Dependency Management

1. **Request:** Explicitly request permission before adding dependencies/tools.
2. **Validate:** Must use OSI-approved licenses. Flag items with $<1000$ GitHub stars.
   `docs/external_tools.md` surveys the external phylogenetic tools against both,
   so a proposal to adopt one starts from the licence and the metric rather than
   from a search.
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

0. **Check the ledgers name open work.** Every `TICKETS.md` bullet names the
   issue that carries it, and a release audit checks each is still open: a
   bullet whose carrier has closed either describes work that landed, and is
   removed, or work that remains with nothing behind it, and is re-pointed to
   a ticket filed then. The same reading covers the notebooks' Further Work
   sections, which name an issue per line. Twelve such bullets and nine such
   lines were found at the 0.5.0 audit (issue #400); the check is a reading
   rather than a script, because it needs the issue tracker and a judgement on
   whether the work landed.
1. **Run the gate.** `infra/release.sh` runs every per-PR CI check
   (`ruff check`, `ruff format --check`, `mypy --strict`, `cargo clippy -D
   warnings`, `cargo fmt --check`, `cargo test --locked`) plus what CI skips
   per PR: the full `pytest` suite including `@pytest.mark.release` tests
   (see "Release-Gated" above), `sphinx-build -W`, and
   `infra/build_documents.sh`. It runs every check regardless of earlier
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
