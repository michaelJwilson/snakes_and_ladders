# Developing snakes_and_ladders

Repository structure, CI enforcement, and contribution rules. For setup, see [INSTALL.md](INSTALL.md); for the project's trajectory and for the development loop these rules sit inside — ticket, plan, pull request, validation, record — see [ROADMAP.md](ROADMAP.md) §0. **`CLAUDE.md` is the authoritative source for conventions; in any conflict, `CLAUDE.md` prevails.**

This file is worked in, so it carries the mechanics in full. Per `CLAUDE.md`'s altitude rule, detail may repeat between a planning document and a worked-in one but may not disagree; `ROADMAP.md` §0 is the loop's intent, and anything here that contradicts it is a defect here.

## Repository Layout

Infrastructure paths first, application paths after; `CLAUDE.md` states why
keeping the two liftable matters.

| Path | Contents |
| --- | --- |
| `benches/`, `tests/` | Criterion benchmarks (Rust), pytest suite, and integration tests. |
| `docs/source/` | Sphinx API documentation. |
| `PROBLEMS.md` | The problem catalogue, hand-written and resolved by a test. Its `Statement` column keys each row to the textbook section stating that model, and `infra/problem_join.py` joins the two, failing an orphan either way (issue #495). `infra/problems_tables.py` reads it, the suite's markers and `docs/tex/method_notes.yaml` into the textbook's applicability tables (`docs/tex/generated/`). |
| `python/snakes_and_ladders/` | Python package: re-exports, typed extension stubs, stub CLI. |
| `python/snakes_and_ladders/sim/` | Data generation and ground-truth retention. |
| `python/snakes_and_ladders/likelihood/` | Felsenstein pruning; CPU dispatch landed (NumPy, PyTorch, Rust), CUDA and Metal not implemented. Also the phylogenetic `Objective` (`objective.py`), which adapts the recursion to `opt/`'s fitting interface — here because `opt/` may import no application module. |
| `python/snakes_and_ladders/opt/` | Model-agnostic continuous parameter fitting via autodiff (PyTorch): the `Objective` interface, shared constraint maps, the initializers, the Hamiltonian sampler, the temperature schedules, the budget utility, and the Potts, HMM, mixture and test-function reference instances. Imports nothing from `sim/`, `likelihood/` or `search/`, asserted by test. |
| `python/snakes_and_ladders/learn/` | Model-agnostic reinforcement learning: the `Environment` interface, the policies, REINFORCE, the critic, actor–critic, PPO and the planner, an exact trajectory-enumeration oracle, the relaxations and the learned surrogates, with the Potts-landscape and hidden-path reference instances. Imports nothing from `sim/`, `likelihood/` or `search/`, asserted by test. |
| `python/snakes_and_ladders/search/` | Move sets, the hill-climbing and large-parsimony searches (`infer.py`) that join them to `opt/`, the samplers, annealers and tempered ensembles, the exact ground states, and the coupled model's block ascent. The phylogenetic RL environment (`rl.py`) lives here too, since `learn/` may import no application module. |
| `python/snakes_and_ladders/qa/` | QA figures/tables for the documents; renders, doesn't recompute. |
| `python/snakes_and_ladders/sandbox/` | The conserved home: an implementation a framework replaced on a hot path, or one a measurement declined, kept with the tests that referee it. Imported by `tests/` and `qa/` only, asserted by test. Carries `tropical.py` (issue #408) and `pruning_burn.py`, `burn`'s taped gradient (issue #449), whose Rust half is behind the `sandbox` Cargo feature. |
| `src/lib.rs` | Rust extension (`oxi_snakes_and_ladders`), exposed through PyO3. |
| `docs/tex/` | LaTeX source for the paper and the textbook, with the notation and preamble both share. |
| `infra/build_documents.sh` | Regenerates QA figures, checks citation integrity (`infra/check_citations.py`), then builds `docs/paper.pdf` and `docs/textbook.pdf` (both committed, rebuilt only by a "Rebuild the documents" pull request; the `.aux`, `.bbl`, `.log` and other files `latexmk` leaves beside them are ignored). `--no-figures` drops the regeneration, for the one caller that has already rendered and compared every figure (`infra/release.sh`, issue #530). |

*Note: each directory carries a localized `CLAUDE.md`, appending to rather than overriding the root.*

New issues are filed through `.github/ISSUE_TEMPLATE/task.yml`; blank issues are disabled via `.github/ISSUE_TEMPLATE/config.yml`.

## Test Layout

`tests/` is organized by **kind** at the top level and by subject within it. Where a new test goes follows from what kind of check it is, not from what it covers.

| Path | Holds |
| --- | --- |
| `tests/regression/` | Correctness. Asserts scientific validity against an independent oracle. |
| `tests/regression/{sim,likelihood,opt,learn,search,qa,sandbox}/` | Split by submodule, the outgrown-flat-directory case below. |
| `tests/regression/` (top level) | Regression tests belonging to no submodule — `test_numerics.py`, `test_claude_md_pointers.py`, `test_pairwise_distance.py` (scaffolding), and `test_sandbox.py`, whose subject is an absence in five packages rather than one module (issue #516). |
| `tests/benchmarks/` | `pytest-benchmark` timings. Asserts shape only; correctness is pinned by the regression counterpart. `profile_hotpaths.py` is the exception — a `cProfile` self-time diagnostic, not `pytest`-collected. |
| `tests/regression/fixtures/` | Declarative test data, one directory per problem (e.g. `tree_jc/stress.yaml`). Data, not Python. |
| `tests/` (top level) | Whole-package and binding smoke tests, belonging to no single kind or submodule — `test_run_snakes_and_ladders.py`, `test_oxiphylo_bindings.py`. |

* **Every benchmark pairs with a regression module.** `benchmarks/test_<name>_bench.py` accompanies `regression/test_<name>.py`. A benchmark without a counterpart asserts nothing about correctness, which `CLAUDE.md`'s "No Coverage Theatre" rule forbids. `profile_hotpaths.py` is the exception — it ranks self time for a Rust-port audit, asserts nothing, and pairs with no regression module.
* **Split by submodule only when a kind outgrows one flat directory** — `tests/regression/likelihood/`, not a top-level `tests/likelihood/`. Kind stays the outer axis. `tests/regression/` reached 39 flat modules and was split under issue #154; `tests/benchmarks/` is 39 and stays flat (re-counted 2026-09-08 at the 0.5.0 audit, where `tests/regression/` is 23 flat modules beside its seven per-module directories; `sandbox/` made an eighth under issue #516, at two modules, because the rule above is what a test referees and not how many there are).
* **A pull request runs the tests its change can affect, not all of them.** `infra/select_tests.py` turns the changed files into the test paths to run and the modules to measure coverage over; `python-tests` calls it (issue #161). Three rules make it safe: a module's dependents are derived from the import graph and run too; a change it cannot attribute to one module — a lockfile, a shared fixture, `src/`, the workflow itself — selects everything; and a diff that changed no code selects nothing. Benchmarks are selected the same way, by the regression module each pairs with.

  Measured on one development machine, `pytest -m "not release"` with the coverage gate, before and after:

  | Diff | Before | After | Selected |
  | --- | --- | --- | --- |
  | documentation only | 174.0 s | **0.0 s** | nothing |
  | `learn/` | 174.0 s | **17.1 s** | learn |
  | `qa/` | 174.0 s | **59.1 s** | qa |
  | `search/` | 174.0 s | **115.0 s** | qa, search |
  | `opt/` | 174.0 s | **163.7 s** | learn, likelihood, opt, qa, search |
  | lockfile, shared fixture, `src/` | 174.0 s | 174.0 s | everything |

  The saving is uneven because of the import graph, not the machinery. Attributed by `--durations=0` over one run: `qa` is 33.6% of the suite's time, benchmarks 29.6%, `search` 20.9%, and the rest under 6% each. `snakes_and_ladders.qa` imports four of the other five modules, so most changes reach the most expensive component; a change to `opt`, which everything depends on, saves almost nothing. `snakes_and_ladders.learn` was imported by nothing when this was measured, so a change there saved 90%; since `snakes_and_ladders.search.gym` adapted its protocol (issue #322) a `learn` change selects `search`, and through `likelihood.features` → `search.topology` also `likelihood` and `qa`.

* **Coverage is measured against what was selected.** Where the whole suite runs, that is the package. Where a subset runs, the claim narrows to *every module this pull request touched is at least 90% covered by that module's own tests* — stricter in one direction, since a module stops counting coverage it gets only incidentally, and weaker in another, since an untouched module is not re-checked. Measured on `main`: sim 100%, likelihood 100%, learn 100%, opt 99%, qa 99%, search 98%. The package-wide gate still runs on every push to `main` and in `infra/release.sh`, which is what stops an unselected module rotting.
* **The `frameworks` extra is optional for the suite and required for its pins.** It installs `gymnasium`, `rustworkx`, `torchrl` and `torch_geometric` (issue #322), and enables `snakes_and_ladders.search.gym` (the Gymnasium adapter over `learn.Environment`), `PottsGraph.to_rustworkx` / `from_rustworkx`, and six test modules that `importorskip` one package each and pin our implementation against it: `test_search_gym.py` (Farama's `check_env`, and an episode round-tripped through both interfaces), `sim/test_graph_rustworkx.py` (the generators against `rustworkx.generators`, the cut against `networkx`), `search/test_search_topology_rustworkx.py` (topology equality against `rustworkx.is_isomorphic`, issue #376), `learn/test_learn_ppo_torchrl.py` (GAE and the clipped loss against TorchRL's), `learn/test_learn_reinforce_torchrl.py` (the score-function loss against `ReinforceLoss`, #376) and `learn/test_learn_surrogate_pyg.py` (the graph surrogate against `GINConv`). `python-tests` syncs the extra so they run per pull request; without it they skip. Three further modules compare against **`scipy`**, a core dependency since issue #376 (their `importorskip` is kept so a trimmed environment skips rather than errors): `test_opt_fit_scipy.py`, `test_likelihood_hadamard_scipy.py` and `test_search_neighbor_joining_scipy.py` (#376). They skip everywhere until scipy is declared, the open question that ticket leaves standing; the repository otherwise writes out the few constants it would need (`opt.fit`, `search.statistics`). A framework that *replaces* an implementation on a hot path is a different step, gated on a measurement, and the replaced implementation moves to `python/snakes_and_ladders/sandbox/` as its referee (`sandbox/CLAUDE.md`).
* **Fixtures follow their blast radius.** Used by one module: keep it there, or in a local `conftest.py`. Shared across modules: a top-level underscore-prefixed module such as `tests/_example_hotpath.py`, imported rather than collected.

---

## Infrastructure & Tooling

### Build System

`maturin` builds the Rust extension natively during `pip install .`.

* **Requirement:** A Rust toolchain, for consumers too.
* **Known Gap:** The typed stub `python/snakes_and_ladders/oxi_snakes_and_ladders.pyi` is hand-written. Run `python -m mypy.stubtest snakes_and_ladders.oxi_snakes_and_ladders` periodically to prevent drift.
* **The `sandbox` Cargo feature is off by default and compiled at the release gate.** It turns on `src/pruning_burn.rs` and the `burn-ndarray` / `burn-autodiff` / `burn-tensor` dependencies behind it — the declined route `python/snakes_and_ladders/sandbox/` conserves (issue #449, `sandbox/CLAUDE.md`). The build cost is why it is a feature: a clean `maturin develop --release` compiles 34 crates in 37 s and links no `burn`, the same build `--features sandbox` 102 in 86 s. Neither CI nor `pip install .` passes the flag, so the wheel and every per-pull-request job pay nothing. `infra/release.sh` is the only thing that compiles it, because a `#[cfg(feature)]` route nothing builds rots unnoticed: the conserved file needed three `clippy` fixes and a `cargo fmt` to build under the current toolchain when it came back. It runs two passes with the feature: `cargo test` does not build a `[[bench]]` target and `clippy` runs no test, so the gated criterion bench is compiled by one and the gated unit tests run by the other (40 against the default 34). From a clean checkout that adds 33 s of `clippy --all-targets` and 34 s of `cargo test` to the gate's 34 s and 38 s without the feature. The Python half imports without the feature and refuses to run, so `tests/regression/sandbox/test_pruning_burn.py` skips (11 tests) against a default build and passes against `maturin develop --release --features sandbox`.

### Documents

Pull requests change `docs/tex/` and never `docs/paper.pdf` or `docs/textbook.pdf`. The two PDFs are committed for readers and are rebuilt, from `main`, by the pull request a **Rebuild the documents** ticket asks for (`.github/ISSUE_TEMPLATE/documents.yml`), whose title carries that phrase; CI refuses a PDF change in any other pull request. Between rebuilds the PDFs lag the sources by design (issue #369): comparing the committed PDFs on every pull request made every document pull request conflict with every merge and cost one rebuild per merge per open pull request, eleven builds in one afternoon. `infra/build_documents.sh` runs against the synced environment (`uv run --no-sync`) and no longer rebuilds the Rust extension per build.

**The documents build on the push to `main`, and that is not licence to commit what the build produced.** A normal pull request still must not change either PDF, and the `lint` job refuses one that does; only a **Rebuild the documents** pull request may. A merge builds the documents to find out whether the sources still typeset --- the PDFs it writes are byproducts of a runner and are thrown away with it, exactly as a local build's are reverted (below).

Building on merge rather than per pull request is issue #488: at 15.8 s of LaTeX plus 431.8 s of renders (issue #433, and every manifest entry since #490 and #492) the build is the wrong cost on the path a merge waits for, and a document that only built at a release would accumulate a release's worth of breakage. On merge the blame window is one commit wide, so a red `documents` on `main` is a defect to fix and never a known-bad background state.

**Citation integrity** --- a cited figure exists, a cited label is defined in the document that cites it, a `\cite` has an entry in `docs/tex/references.bib`, and every bibliography entry closes its braces --- is `infra/check_citations.py`, called by `infra/build_documents.sh` before `latexmk`, so the `documents` job and a contributor's `infra/validate.sh` take one path. It is text and existence, no render, and costs **55 ms over both documents** (timed 2026-09-09 on the 4-core host at load 9.4, so an upper bound) against the build's 15.8 s. `latexmk` reports the same three failures in its log and the job still greps for it, but only after both documents are typeset and without saying which document defined the label the other could not find (issue #249). The bibliography's brace walk is the fourth claim: a merge dropped an entry's closing brace during the 0.5.0 union and it was repaired by hand, and an entry that runs into the next one stops resolving while the file still reads as text.

**A figure exists because a document asked for it, and the path from the asking to the committed figure is five steps.** Issue #492. `infra/check_citations.py` fails a citation with no figure; `tests/regression/qa/test_qa_build.py::test_every_manifest_figure_is_cited_by_a_document` fails a figure with no citation. Between those ends the pipeline was undefined, and `sim_problem_sizes` and `topology_accuracy` were the symptom --- rendered on every release, cited by nothing, checked by nobody for eight releases.

1. **The request names what the figure must show**, in four parts: the claim it settles, the problem class it is drawn from, the sizes it is drawn at, and the document and section that will cite it. The missing part is almost always the fourth, which decides whether the figure is wanted at all.
2. **It becomes a ticket**, from `.github/ISSUE_TEMPLATE/task.yml`. The four parts above are its **Desired outcome**; its **How it will be validated** is step 5.
3. **The renderer joins the manifest** (`snakes_and_ladders.qa.manifest`) with a declared `seconds`, measured alone on the reference host with `infra/measure_build.sh`. Over `CITED_RENDER_CAP` it is refused unless `CAP_WAIVERS` carries it with the ticket that owns cutting it.
4. **The render is committed** --- `<stem>.pdf` or `<stem>.tex`, and `<stem>_caption.txt` --- **and the document cites it**, from the section the request named. The caption names the seed, the sizes and the model; the body says why the figure is there without restating it (`docs/CLAUDE.md`).
5. **A test pins every number the caption states**, because a figure states no result the suite does not also pin (`CLAUDE.md`). Two ways out of one number, and only two: pin it, or take it out of the caption. Taking it out is right where the quantity is discontinuous in its inputs --- a threshold crossing, a rank, a count of successes at the margin --- since `docs/CLAUDE.md` forbids publishing one of those whether or not a test would agree with it today. `topology_accuracy` lost two such numbers on the pull request that cited it and kept the comparison between the ends of its sweep, which its release-tier test asserts.

Retiring one runs the same path backwards: the citation goes, then the manifest entry and the renderer and the committed files together. A figure whose document has no room for it is deleted, never left rendering.

**Branch protection needs the second half of the change, and does not block on it.** A job the workflow skips reports the conclusion `skipped`, which GitHub counts as a pass for a required check, so **Documents (paper and textbook)** staying in the required set holds up no merge. It should still be removed: a required check that passes by not running is a check in name only — the opposite failure to issue #377's rename, where the retired name reported nothing and every merge waited.

The build writes these paths and no others, taken by hashing the tree before and after each of eight builds (issue #429):

| path | the build | committed |
| --- | --- | --- |
| `docs/tex/figures/<cited stem>.pdf`, `docs/tex/figures/<cited stem>.tex` | re-renders every cited stem; each has one or the other | yes, by the pull request that changed it |
| `docs/tex/figures/<cited stem>_caption.txt` | the same render writes it | yes, likewise |
| `docs/tex/generated/problems_tables.tex` | regenerates the applicability tables the textbook inputs (`infra/problems_tables.py --write`, issue #425) | no, ignored |
| `docs/paper.pdf`, `docs/textbook.pdf` | rewrites both on every run, whether or not a figure changed | yes, and never by an ordinary pull request |
| `docs/<document>.aux`, `docs/<document>.bbl`, `docs/<document>.blg`, `docs/<document>.fdb_latexmk`, `docs/<document>.fls`, `docs/<document>.log`, `docs/<document>.out`, `docs/<document>.toc` | `latexmk` leaves them beside the PDFs | no, ignored |

Everything else it touches it only reads: `docs/tex/*.tex`, `docs/tex/references.bib`, `PROBLEMS.md` and `docs/tex/method_notes.yaml` and the test tree, which the applicability tables are written from, the fixtures under `tests/regression/fixtures/` a renderer's arguments name, and the package sources the renderers execute. It does not write `CHECKS.md` or `SEAMS.md`: `infra/ledgers.sh` writes all three together at the review and release gates. Since issue #492 every manifest entry is cited by one of the two documents, so `<cited stem>` above ranges over the whole manifest. Nothing keeps a branch off the expensive ones any more: the stamps that narrowed the set are deleted (issue #490), and the build runs on the push to `main` rather than on the pull request (issue #488).

One row of that table needs a rule, the row both rewritten every run and tracked: **revert `docs/paper.pdf` and `docs/textbook.pdf` after any local build**. A byproduct cannot be committed by accident and a figure should be; a PDF is the only output where the right action is neither. `infra/review_gates.sh` and the `lint` job both refuse the change, so the cost of forgetting is a failed gate rather than a wrong `main`. `tests/regression/docs/test_document_build_outputs.py` holds the table to the manifest, the script and `git`.

### Continuous Integration

Ten checks run via GitHub Actions (`.github/workflows/ci.yml`). Nine run on a pull request against `main` and are required; `documents` runs on the push to `main` instead (see Documents), where its failure is `main` red:

| Job | Execution |
| --- | --- |
| `lint` | `ruff check`, `ruff format --check`, strict `mypy`, `towncrier check`, and the rule that a pull request changes a committed PDF only to rebuild it (see Documents) --- here rather than in `documents` because this is the required job a pull request runs, and it already checks out the history the diff needs |
| `rust-lint` | `cargo clippy --locked --all-targets -- -D warnings`, `cargo fmt --check` |
| `rust-tests` | `cargo test --locked`, `cargo bench` (informational) |
| `build` | `pip install .` (no lockfile, mimics fresh consumer), smoke import |
| `python-tests` | `pytest -m "not release"` in two invocations over one selection — the correctness tiers under `-n 3` with `tests/benchmarks` ignored and the coverage gate, then the selected benchmarks serially, because `pytest-benchmark` disables itself under `pytest-xdist` (issue #405, CI & Performance Budget); benchmarks are still skipped unless computational code changed. Also `infra/ledgers.sh --check`, which writes `CHECKS.md`, `SEAMS.md`, the applicability tables, and the two blocks generated from `infra/gates.py` (`pyproject.toml`'s marker list and this file's tier table), and fails if regenerating rewrote a tracked file — here rather than in `lint` because the seams survey imports the package with the `frameworks` extra. The two Markdown ledgers are written to the run summary, where a reader browses them |
| `docs` | Sphinx build (warnings as errors) |
| `documents` | **On the push to `main`, not on a pull request** (issue #488). Regenerate the QA figures the documents cite and the applicability tables the textbook inputs, check citation integrity, then LaTeX build --- all of it `infra/build_documents.sh`. Fails on a citation `infra/check_citations.py` cannot resolve, or on an undefined reference or citation or a multiply-defined label in either `latexmk` log |
| `notebooks` | Re-execute every notebook under `docs/nb/` (`infra/check_notebooks.py`) and fail on a re-executed output that differs from the committed one. The set is fixed at all six and stated in the run's log, not chosen by a digest (issue #480). The six cost **108 s** in the job (run 34385633284, all six executed) and **116 s** on the 4-core reference host, two runs at a 1-minute load of 0.94 and 1.03 with nothing else on it (2026-09-10, issue #525; 137 s at a load of 1.7 the day before, split hmm 39 s, phylo_tree 47 s, turbo 18 s, potts_chain 15 s, ldpc 9 s, spatio_sequential 8 s, one kernel start each) --- against the 110 s the job took while it was still skipping some, so the gate was not buying time. A repeat under a load of 5--8 took 468 s, almost all of the difference on `hmm.ipynb`, which is what a budget would have to be set against. Text is compared; a figure is checked only for still being produced. Regenerate with `--write` on the same script |
| `audit` | `pip-audit`, `cargo audit` (skips on cache hit if lockfiles are unchanged) |
| `pr-title` | The title starts with `[<base branch>]`, the branch the pull request targets (issue #292). One shell line; runs only where the workflow does, so on a stacked pull request it is the reviewer's until #273 lands |

Branch protection names each required check by the job's `name:`, not by its id, so renaming one is two changes and only a maintainer can make the second. Issue #377 renamed the LaTeX job to **Documents (paper and textbook)** and merged as #379 on 2026-09-08; #468 confirmed the replacement three ways — `main` took three merges on 2026-09-09, #463 reported ten check runs with `mergeable_state: clean`, and the same job later reported `skipped` without holding a pull request up. **The two failure modes are opposites.** A *retired* name no job reports under sits pending and blocks every merge. A required check that reports `skipped` blocks nothing, because GitHub counts `skipped` as a pass — which is what `documents` does now that it runs on the push to `main`, so it is a required check in name only and should leave the set. `RELEASE.md` carries the precondition and its current verdict.

`lint`, `python-tests`, `docs`, and `notebooks` restore a `~/.cache/uv` cache keyed on `uv.lock`'s hash before installing `uv`. `rust-lint`, `rust-tests`, `build`, and those same four jobs restore a shared `~/.cargo/registry`, `~/.cargo/git`, and `target/` cache keyed on `Cargo.lock`'s hash, so `oxi_snakes_and_ladders` (built via `maturin`/`pyo3` on every `uv sync` or `pip install .`) compiles from scratch only when a lockfile changes or no job has populated the cache yet. `audit`'s per-week marker cache (above) is unrelated and unaffected.

### CI & Performance Budget

* **Size Caps:** Restrict topological move tests to $n \le 10$ (exhaustive enumeration oracle).
* **No CI Profiling:** Do not rank performance on GitHub runners; hardware varies. Benchmark on fixed hardware.
* **Four tiers, three budgets.** A test's tier is decided by *what its size is for*, never by how slow it happens to be: a size chosen so an exact oracle stays available is a CI size even when it is slow, and a size chosen to show behaviour at scale is a stress size even when it is fast.

  <!-- BEGIN GENERATED tier table (infra/gates.py, via infra/ledgers.sh) -->

  | tier | marker | fixture | budget | contents |
  | --- | --- | --- | --- | --- |
  | CI | none (the default) | `<problem>/ci.yaml` | **5 minutes**, worst case | correctness at the smallest size that exercises the claim |
  | key | `key` | the instance a file marks `key`, reached as `fixture(problem, "key")` | **120 s per test** (`SAL_KEY_DURATION_CAP`) | one problem's declared instance run end to end — simulate, fit, assert. The one that exists: the coupled model at 5,041 vertices, bin factor 5, measured at **62 s** against 59 s at factor 10 and `release` at factor 1 |
  | developer / stress | `stress` | `<problem>/stress.yaml` | **10 minutes** | the same claims at a size the CI budget cannot hold |
  | release | `release` | `<problem>/release.yaml` | unbounded | long-running scientific validity, run by `infra/release.sh` |

  <!-- END GENERATED tier table -->

  * **The key tier is a named instance, not a fourth size.** A problem whose
    largest useful instance takes two minutes needs a name a study, a figure
    and a notebook can all read, or each picks its own and the three stop being
    comparable (issue #399). The fixture file marks one of its declared
    instances `key`, `snakes_and_ladders.sim.fixtures.fixture` takes `key`
    beside a tier, and the marker of the same name runs that instance's full
    test: exempt from `SAL_DURATION_CAP` and held to `SAL_KEY_DURATION_CAP`
    instead. CI's full tier runs it; the local validation runs it only when
    `infra/select_tests.py`'s `KEY_TRIGGERS` match the diff — the coupled
    model, the emissions, the fixtures or the Rust crate — so at most one key
    test runs per pull request and only where it can fail.

  * **The tiers are the fixture files' names.** A supported problem declares
    its instance per tier under `tests/regression/fixtures/<problem>/`, and
    `snakes_and_ladders.sim.fixtures` loads it (`PROBLEMS.md`); a problem
    declares a tier only where it has an instance for it. `tests/_scale.py`'s
    `at_fixture` parameterizes one test over every tier a problem declares,
    marking each case for its own tier.

  * **Use `pytest -m "not release and not stress and not key"` while developing.** That is the CI tier, and the gate a pull request is judged against. `infra/validate.sh` builds the expression from `infra/select_tests.py`'s `deselect`, which drops `not key` for a diff that could move a key fixture's result.
  * **Two gates, and the merge waits for the first.** Nine of the ten jobs finish inside 156 s; `python-tests` took 1,380 s on #403 and 2,155 s on #398, `select_tests` answering "everything" for a change it cannot attribute and both touching a shared fixture. That job runs `infra/select_tests.py` on a pull request and the whole suite on the push to `main` and at the release gate. **The selection is not bounded, and the step carries no cap of its own** (issue #425): #409's `--budget` bounded the fallback and `UNBOUNDABLE` exempted the changes a bound was least safe for, and the 15-minute cap that paired with it fired on exactly those — every `likelihood/` pull request, cancelled at 15 minutes and indistinguishable from a failure (#423, measured on #420: cancelled at 15:15 with the suite 57% run). A bound and a fixed timeout cannot coexist, so both went; the job's own 90 minutes is the only cap.
  * **The 5 minutes is the worst case, not the average.** `infra/select_tests.py` usually selects less, but it answers "everything" for any change it cannot attribute to one module — a lockfile, a shared fixture, `infra/` — so the full CI tier is the number that has to fit.
  * **A size that exists to show scaling is parameterized, never duplicated.** `tests/_scale.py`'s `at_scale` runs one test body at both sizes, so a change to the assertion reaches the large size by construction. `stress_only` is for a claim with no smaller size that still asserts it, and states the reason on the marker.
  * **The local validation is one command and 300 s.** `infra/validate.sh` runs, against the diff from `origin/main` (or `--base <ref>`), in the order that fails fastest and with each step's wall clock printed: `ruff` and `mypy --strict` over the changed Python files (CI keeps the whole `files` set); `pytest -m critical`; the tests `infra/select_tests.py` names for the diff, capped at 180 s, with `--durations=10`; `sphinx-build -W` only when `docs/source/` or a docstring changed, keeping its doctrees under the scratch directory; the notebook checker, which executes all six notebooks under `docs/nb/` in 116 s (issue #480); and the document build when `docs/tex/` changed, which renders every figure in the manifest since issue #490 deleted the stamps. Three classes of change against the budget, measured on the reference host (4 cores, load under 1, one BLAS thread per process) and reproduced by `infra/measure_test_budget.sh`:

    | change | what `select_tests.py` names | wall | budget |
    | --- | --- | --- | --- |
    | `docs/tex/` only | the three document guards (27 tests) | **0.4 s** | 300 s (target 90 s); the document build is not on the pull request (issue #488) |
    | one module (`learn/`) | `learn`, `likelihood`, `qa`, `search` and their benchmarks (1,125 tests) | **1,323 s** | 300 s: **over** |
    | `search/`-wide | `qa`, `search` and their benchmarks | not measured; a subset of the row above | 300 s |

    Measured 2026-09-08 on the 4-core reference host, one BLAS thread per process, load under 1.5. The `learn/` class fails the budget by a factor of four, and the cause is the tier rather than the selection: 18 of its tests run over 10 s, led by `test_nni_hill_climb_eight_taxa_benchmark` (180 s cold, 118 s warm), the 4,000-episode critic estimator (111 s), the enumerated-support calibration (72 s) and the lazy NNI search (53 s). `infra/validate.sh` caps the selected tests at 180 s and reports the class as over budget rather than waiting. The maintainer's decision (2026-09-08): every test over the cap carries `release`. The 19 tests over 10 s in that run, 817.8 s of its 1,323 s between them, moved in the pull request that records this; the class's remaining 1,106 tests average 0.46 s, so `pytest-xdist` with one thread per worker stays the next step if it grows past the budget again.

  * **The document build: every cited figure rendered, 431.8 s of renders plus 15.8 s of LaTeX.** The LaTeX reading was measured 2026-09-09 on the 4-core reference host by hashing the tree before and after each run (issue #429). The renders were 0 s or 305.8 s before, depending on a stamp; issue #490 deleted the stamps and issue #492 left every manifest entry cited, so the figure pass is now the manifest's declared sum over **all 23** entries and does not vary with the diff. The job runs on the push to `main`, not on the pull request (issue #488).

    | build | what it does | wall |
    | --- | --- | --- |
    | every cited figure rendered | 23 renders, then the applicability tables and `latexmk` over both documents | **431.8 s + 15.8 s** |
    | LaTeX alone, no render | the tables, then `latexmk` over both documents | **15.8 s** |
    | the same, repeated from the restored tree | the same work again | 15.6 s |
    | incremental, nothing removed | `latexmk` reports nothing to do | **3.7 s** |

    The figure pass is arithmetic over `snakes_and_ladders.qa.manifest`'s declared `seconds`, not a timed pass, and the manifest is the source rather than this restatement of it; `tests/regression/test_release_gate.py` holds `RELEASE.md`'s rounded bound above that sum. The stamped build's two clean readings measured the same tree three commits apart: at `1da89a3` nine stamps were stale and cost 216 s of a 230.7 s build, and every one of those nine renders produced a **byte-identical** figure and caption.

    **Both PDFs are reproducible.** Two clean builds from the same tree wrote 19 paths, of which 17 are byte-identical — `docs/paper.pdf`, `docs/textbook.pdf` and `docs/tex/generated/problems_tables.tex` among them; the run at `1da89a3` gave the same answer over 27 paths, 25 identical. The two that differ either time are `docs/paper.fdb_latexmk` and `docs/textbook.fdb_latexmk`, `latexmk`'s rebuild database, whose lines record each source's modification time and agree on every size and MD5. Both are ignored, so no committed output differs. No clock reaches a PDF because the build pins `SOURCE_DATE_EPOCH` and exports `FORCE_SOURCE_DATE`.
  * **A test over 10 s carries `release`, `stress` or `key`, or fails the validation; the default is `release`.** A test over the cap is re-tiered, not waited for: it moves to `release`, where `infra/release.sh` runs it with every other tier, and to `stress` only when a developer needs it inside the 10-minute local budget. A claim that moves keeps a fast small-size sibling per PR where one exists; where none does, the gap is a ticket. `tests/conftest.py` records every test's call duration; with `SAL_DURATION_CAP` set, as `infra/validate.sh` sets it on the reference host, a test over the cap carrying none of the three markers fails the session by name, and with `SAL_KEY_DURATION_CAP` set — 120 s, also `infra/validate.sh`'s — so does a `key` test over *that*, or `key` becomes the marker any slow test acquires. CI does not set it, per **No CI Profiling**, and prints `--durations` instead.  * **One process is one core.** `tests/conftest.py` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 unless the caller did, and `infra/validate.sh` exports them, so the load average counts processes and four of them fit the host. The figure renderers are exempt: `snakes_and_ladders.qa.build` strips the three from a render's environment, because a reduction split over a different thread count can move a committed figure's last bit.
  * **The environment is synced once per worktree.** `uv sync --locked` runs when the lockfile changes; every script exports `UV_NO_SYNC=1`, and a shell running `uv run` by hand should too (`INSTALL.md`). Without it each command re-resolves the environment and, in a fresh worktree, recompiles the Rust extension.
  * **Measured on one development machine, uncontended:** the CI tier runs in **1,226 s over 2,315 tests** (2,302 passed, 13 skipped, 89 deselected), measured 2026-09-10 on the 4-core host, one BLAS thread per process, `pytest -m "not release and not stress"` over the whole `tests/` tree, two readings 8 s apart on a host `ps` showed carrying nothing else (issue #525). It was 1,098 s over 2,121 tests on 2026-09-09. That is **4.1x the 5-minute budget**, which is what keeps `infra/select_tests.py` in the tree (issue #425). The stress tier holds 10 tests in 51 s, from its last uncontended measurement. Before issue #132 the same tests were one tier at 263 s. `infra/measure_test_budget.sh --full` reproduces both and reports each against its budget, over `tests/regression` alone rather than the whole tree, so its CI-tier number is smaller than the one above by the benchmarks; without `--full` it measures the critical tier only.
  * **The correctness tiers are distributed; the benchmarks are not.** Issue #405. CI runs the selection twice: `pytest -n 3 ... --ignore=tests/benchmarks` with the coverage gate, then the benchmark paths of the same selection in a serial invocation with no `-n`. `pytest-xdist` is MIT, 1.9k stars, and in the `test` extra.

    The split is not tidiness. **`pytest-benchmark` disables itself whenever a run is distributed**, without failing and without a line anyone reads, so a suite under `-n` reports its 212 benchmark tests as passed having timed none of them — and reports a larger speed-up for it, which is why a raw figure taken that way is not quoted here. `tests/conftest.py` fails a benchmark that reaches a distributed run, by name, and the same file refuses a run that sets `SAL_DURATION_CAP` or `SAL_KEY_DURATION_CAP` under `-n`: the durations are recorded in the workers and the cap is read on the controller, so every test would pass every cap. `infra/validate.sh` sets both caps and therefore runs serially; `infra/measure_test_budget.sh` measures the budgets and must.

    **The numbers, measured under issue #525 and not re-run here.** Like-for-like on the CI tier, benchmarks off on both sides: **993.7 s at `n = 1` against 361.9 s at `n = 3`, 2.75x**. Per-tier concurrency of one to four gave 1.00x, 1.89x, **2.30x** and 2.25x, so three is the peak and four is past the knee. Every cell returned the same **2,302 passed and 13 skipped**, which is what says distribution disturbs no ordering or seeding the suite depends on.

  * **The budgets are not asserted in the suite.** A wall-clock assertion would fail for the machine rather than for the change, which "No CI Profiling" forbids. What is asserted is structural: `tests/regression/test_scale_tiers.py` checks that the stress tier stays reachable and that no test carries a size marker it does not use.
  * **Plain `pytest` (no `-m` filter) is the release gate's job, not a development command.** Run it by hand only when cutting a release, or when you have changed a release-gated test itself.
* **Two axes select tests, and they answer different questions.** `infra/select_tests.py` chooses **by module path** — what a diff could have broken. The *kind* markers choose **by what a test is checked against** — `oracle`, `simulated_truth`, `mathematical`, `edge_case`, `structural`, registered in `pyproject.toml` and required of every test outside `tests/benchmarks/` by `tests/regression/test_test_kinds.py`. Path selection carries the dependency reasoning issue #161 built; the kinds ask for a class of check independently of where the change landed. `--strict-markers` is on, so a misspelled marker fails collection instead of silently selecting nothing.
* **`critical` is the early gate, and it is a second axis rather than a kind.** It marks the tests whose failure invalidates everything after them — the import graph, the documentation index, the `CLAUDE.md` pointers, `select_tests` itself, the compiled extension, and the categorical sampler. **168 tests in 15.9 s** (pytest's clock; 18 s wall), two consecutive runs on 2026-09-10 on the 4-core host at a 1-minute load of 0.54 and 0.57 with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, against a `-m "not release and not stress"` tier of twenty minutes. The host carried no other job (issue #521). It was 177 tests in 16.6 s the day before (issue #524), and the nine that went with the host lock and the figure stamps went with them (issues #521, #490). At the 0.5.0 audit it was 149 tests in 13.0 s at a load of 0.45, at the 0.4.0 audit 88 tests in 12.6 s under a load of 2.27, and the earliest uncontended figure 76 tests in 3.0 s. The count rose with #372's stamp and duration guards and #376's problem-statement guard, and the clock did not, which is what a guard tier is for. CI runs `pytest -m critical` unconditionally before selection; run it locally the same way. A test is critical *and* a kind, never instead of one.
* **Benchmarks are conditional**, and are 19.0% of the CI tier's clock: the tier is 1,226.2 s and 993.7 s with `--benchmark-disable`, a difference of 232.6 s over 212 benchmark tests (2026-09-10, issue #525; the 29.6% this replaces was 40.2 s of a 136.0 s suite and long out of date). They measure code a documentation or QA change cannot have altered, and since issue #161 they are selected per module rather than all together. The job itself always runs and always reports — it is a required check, and skipping the job rather than the step would leave it pending and block the merge. Coverage is unaffected, because every line a benchmark reaches is also reached by the regression module it pairs with.
* **Every cited figure is rendered on every documents build; nothing predicts which changed.** Beside each committed figure was a stamp, `docs/tex/figures/<stem>.inputs`, recording a digest of the renderer's import closure, the fixtures its arguments name, the spec itself and the drawing libraries' versions (`snakes_and_ladders.inputs`, issue #372). Issue #490 deleted the stamps, the digest and the notebooks' copies of both, on the measurements below. The build now renders what the documents cite and the release gate renders the manifest (`--all --check`, issue #484), and neither consults a hash.

  **What the stamps were measured to be worth.** Over 476 stale calls in two windows, **every one was a false positive**: no figure's bytes moved on any of them, at a cost of ~48 minutes in one day. Two causes are recorded rather than fixed, because deletion resolves both: the package's `__init__` re-exports `double`, so every renderer's closure named the compiled extension and any Rust change staled all 22 figures; and on the one subset-reaching change #445 measured, the closure hash staled **19 of 19** cited figures where issue #422's reachability walk staled **1** — and **0 bytes moved** under either. **That corrects the claim this file and `changelog.d/425.removed.md` carried before**, that the walk "saved none": true over the twelve branches those measured, where no change reached a subset, and false as a general statement. Both schemes were predicting renders that were unnecessary, so narrowing the prediction was never the fix.

  **What replaces each stamp.** A stale cited figure cannot reach a release because `infra/release.sh` renders **every** figure, cited and uncited, and compares bytes without overwriting (issue #484); a notebook's output cannot drift because the `notebooks` job executes **every** notebook under `docs/nb/` on each pull request and each push (issue #480). Both run the check instead of predicting whether it is needed. A cited figure renders in at most 30 s on the reference host (`CITED_RENDER_CAP`); the manifest carries each figure's measured time, and a guard fails a cited entry over the cap unless it is waived in `CAP_WAIVERS` with the ticket that will bring it under. `rl_tree_policy`, at 101.4 s, and `search_trajectory`, at 39.6 s, are waived under #372; issue #498 measured both against a five-taxon render and could pay neither debt, for the reasons the manifest records beside each entry. `topology_accuracy`, at 124.0 s, is the third, waived under #509 since #492 cited it: its cost is attributed and unreduced, being no topology sweep at all but 91.5% L-BFGS branch-length fitting over 48 inferences (#506), so the term to cut is the fit and not the taxon count.

* **A reference algorithm's answer on a fixture is committed, and recomputed on the pull requests that could have moved it.** Beside `<problem>/<tier>.yaml` is `<problem>/<tier>.baseline.json`: what a *reference* algorithm achieves on that instance — the enumerated maximum over its topologies, the rate at which NNI hill climbing reaches it from 50 seeded starts, the rate an untrained policy reaches it, the exact maximum-likelihood target of every topology a surrogate is fitted against — each with the algorithm, the seed and the budget that produced it, and the `numpy`, `scipy` and `torch` versions it was computed against (issue #401). `infra/baselines.py --write` computes and writes. The record is a second file rather than a block in the yaml: the fixture declares an instance and is written by hand, the record states a measurement and is written by a tool.

  **A record carried a digest of the tree it was written from until issue #460, and no longer does.** The digest covered the fixture bytes, the transitive import closure of the computing modules and the three library versions, and `snakes_and_ladders.sim.fixtures.baseline` raised where it was not the current tree's. Measured over the whole history of the five records, it moved **54** times and carried a moved value **0** times: every re-key was an edit somewhere in a closure spanning **66 of the 145** tracked source files, and the regeneration that followed reproduced the values byte for byte. It also made **6 of the 34** conflicts across the eight open branches at `origin/main` `31bccb0`, and on every one the three merge sides differed in the digest and in nothing else. The count is live rather than fixed: forty minutes earlier the same measurement gave 8 of 38, `main` having moved twice in between.

  `infra/baselines.py --changed` takes the pull request's changed paths and recomputes the records whose fixture, own file or module closure they touch, answering whether the *number* moved rather than whether the tree did. On the reference host that is 3 s for the cheapest record and 31 s for all five, against the 8.3 s of uniform rollouts, 7.5 s of maximum-likelihood fits and 1.9 s of exact expected returns a *test* would pay per session, which is why the numbers are committed at all. `infra/release.sh` still runs `infra/baselines.py` with no flag over every record. What stays on the cheap read is the check a recomputation cannot make: `baseline` raises `StaleBaselineError` where the installed `numpy`, `scipy` or `torch` is not the recorded one, a fact about the machine and not about the tree. A record edited to match a test is caught by the recomputation, which compares the budget, the seed and the algorithm beside the value.

* **The documents decide which figures a pull request rebuilds, and it is the *union* of what they cite.** `snakes_and_ladders.qa.manifest` states which QA outputs exist and what renders each one; `infra/build_documents.sh` passes every document to the selection. Deriving it from one document would stop regenerating the other's figures and fail nothing, so `cited_stems` refuses an empty set of documents and a test pins that leaving one out selects a smaller set. **Every** figure is regenerated and compared at the release gate (`infra/release.sh` runs `snakes_and_ladders.qa.build --all --check`, issue #484): the per-pull-request selection decides what a *branch* pays for, and the release gate checks all of them regardless. That comparison is also the gate's whole figure pass — it hands `--no-figures` to the build after it, because a comparison that passed has proved the committed bytes are a render's (issue #530). The cost is why the split is scoped: when the manifest held thirteen figures, rendering all of them cost 281.6 s, the two documents together cited seven, and a full build of both PDFs was **60.1 s** against **5.9 s** when one document cited two; `topology_accuracy` at **124.0 s** alone was more than twice the rest of the build. At the 0.4.0 audit the manifest held nineteen figures and the documents cited seventeen — `sim_problem_sizes` and `topology_accuracy` were the two a pull request never rebuilt (#325) — and the four new figures render in 4.6, 3.9, 27.4 and 6.6 s each on the 4-core audit host; the full-build timings were not re-measured then, the host being shared for the whole audit. **Issue #492 closed that gap in the other direction**: both are now cited, the manifest's twenty-three entries and the documents' citations are the same set, and a guard fails a figure no document cites. No stamp narrows either selection: the stamps are deleted on a 100% false-positive rate over 476 decisions (issues #476, #490). The split is kept because it is the mechanism, not the current numbers: a document that stops citing a figure narrows the per-branch selection the moment it does. Citing a figure the manifest cannot render fails the build rather than skipping it. `infra/measure_build.sh` reproduces these numbers on fixed hardware.
* **Tolerances on a quantity that scales with problem size are relative.** The log-likelihood is a sum over sites, so an absolute bound fixed at one site count does not transfer to another: the backends agree to ~8e-13 relative at every size, but that same agreement is 7.4e-07 absolute at 200,000 sites. Absolute bounds are correct for quantities that do not scale — a transition probability, a row sum, a Monte Carlo frequency — and are kept there.
* **Concurrency:** Superseded CI runs on the same branch are automatically cancelled.

### The Continuous Optimization Contract

Moved here from the paper (issue #249): a statement about the code's
architecture rather than about a model. The *mathematics* of the constraint map
is the textbook's; what follows is what the implementation guarantees.

* **One optimizer, three model classes.** An objective is a differentiable
  scalar over an unconstrained vector, with a map back to the parameters the
  model is stated in. Nothing in `snakes_and_ladders.opt` may import
  `snakes_and_ladders.sim`, `.likelihood` or `.search`, and a test asserts it;
  neither `ruff` nor `mypy` would notice a convenience import.
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
  slow step is located by subtracting neighbours.
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

1. `python tests/benchmarks/profile_hotpaths.py --tier enumerable` and `--tier mid` (`--module` selects one) — a `cProfile` self-time ranking of one workload per module, the top five functions with the fraction of the run each carries. Not `pytest`-collected; run by hand. A loop under 10% of its run is recorded in `STATUS.md` and does not proceed (#341).
2. `pytest tests/benchmarks/test_<name>_bench.py` — the NumPy or PyTorch baseline at the size the port would run at. The 10x rule is against realistic sizes, not the smallest that fits CI.
3. Time the port **alone** (`cargo bench`, Criterion, in `benches/`) **and through its binding** (`tests/benchmarks/`); the difference is the FFI boundary, and the pull request reports both.
4. For anything recursive, report peak memory beside time. `snakes_and_ladders.qa.likelihood_footprint` states the pruning footprint from the arrays' shapes and its test pins it to the allocator (#232); otherwise take `tracemalloc` peaks by hand and say so.
5. Pin the port against the NumPy oracle within its tolerance before reporting the speedup.

`cProfile` cannot see inside a NumPy call or a Rust kernel; `pytest-benchmark` reports wall clock and nothing about cache, branches or vector width; Criterion times a kernel with its inputs already in Rust. Each ranks or times, none explains.

### Mitigating an FFI Cost

Root `CLAUDE.md` used to say the marshalling had been the dominant term at
this boundary. It has not been, on anything measured: issue #457 put the
marshalling at 1.1–4.9% of a through-binding call against a kernel at
90–100%, and the whole per-crossing tax at 1.34% of a fit at 8 taxa and 0.79%
at 20; issue #452 cut the bytes crossing per pass by 179x and 1,881x and
moved wall time between −7.9% and +2.3%; issue #453 measured `burn`'s tape
*larger* than the PyTorch tape it was to replace, so the cost there was the
tape and not the crossing. `STATUS.md` carries the tables.

So the four mitigations below are answers to a cost this repository has not
yet observed, and **each needs a profile at the size in question before it
lands**, not a general argument that boundaries are expensive. At the sizes
measured there is under 1.4% of a fit to win, which is under the 10% rule
step 1 already applies.

* **One crossing per call, with contiguous unaliased buffers.** The standing
  rule, and the only one adopted without a further profile, because it costs
  nothing to write that way and a NumPy view that is not C-contiguous is
  refused by `as_slice` rather than copied silently. Every binding here
  already holds it.
* **A handle holding state across calls.** Licensed by a profile showing the
  same buffer crossing on consecutive calls, *and* a caller making them.
  Issue #452 built one, cut the bytes by the three orders of magnitude above,
  moved the wall time by less than a tenth in either direction, and found no
  caller on the binding to spend it on. It did not land, and the tree carries
  no handle today.
* **Releasing the GIL around a kernel.** Licensed by a profile showing Python
  work that could run concurrently with the kernel — otherwise the release
  buys an unlocked interpreter nothing is waiting for. Nothing here overlaps
  a crossing today, since `snakes_and_ladders.parallel` parallelizes over
  whole tasks and not inside one.
* **Batching many small calls into one.** Licensed by a profile showing the
  per-call fixed cost, not the per-element cost, at the top of the ranking.
  A batched signature also fixes the shape a caller may pass, so it is the
  most expensive of the four to retract.

**Where the cost could still be, and is unmeasured.** Nothing has been
profiled past 20 taxa or 10,000 sites. The 200-taxon claim of issue #436
lives only in `tests/benchmarks/test_pruning_rust_bench.py` and is a
benchmark, not a boundary measurement. If the boundary is ever the term it
will be there, and a measurement at that size is the precondition for any of
the above landing.

### Running With Workers

`snakes_and_ladders.parallel.map_tasks` is the one seam for CPU parallelism over
independent tasks (issue #344); no module keeps a pool of its own. Three sites
go through it, each taking `workers=` explicitly — `opt.fit.fit_from` (starts),
`opt.budget.compare` (cells of method × instance × seed) and
`search.support.bootstrap_support` (replicates).

* **`workers` is an argument, never a default or an environment variable.**
  `workers=1` is the serial loop; the QA runner and a test pass `1`; a run on
  a bigger machine passes its core count. A default read from the machine
  would change a run nobody edited.
* **A parallel run is bitwise the serial run.** Randomness is one generator per
  task, spawned in item order from the caller's with
  `numpy.random.Generator.spawn`, so task `i` draws the same stream at every
  worker count; results return in input order; and every worker runs at the
  intra-op thread count the site names, so the same kernels reduce in the same
  order. Each site pins `workers=1` against `workers=4` with `==` or
  `torch.equal`, never `allclose`.
* **The thread rule.** `torch` and BLAS multithread inside a kernel, so a pool
  of workers each at the default thread count oversubscribes the cores; where a
  pool pays, pin `intra_op_threads` to cores divided by workers. `map_tasks`
  sets it per worker from that argument (`None` leaves the process's setting
  alone) and restores the caller's afterwards. The three
  sites pass `None`, by measurement: pinning one thread slowed the serial
  multi-start fit 2.9× (1.40 s against 0.49 s at 8 taxa × 1000 sites),
  because torch's intra-op parallelism over the sites is the parallelism that
  pays there, and no pool reached 2×. Serial and workers then run at the
  same count on one machine, which keeps the two bitwise equal. A
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
  and left serial — all three sites are, on the 4-core host. Time only on an
  uncontended machine: a shared host at load above its core count reports the
  contention, not the code.
* **Not yet through the seam** (`TICKETS.md`, #344): the candidate fits of
  `search.infer`, `learn.rollout` batches, tempering replicas, `qa.build` and
  `infra/check_notebooks.py`. `pytest-xdist` was on this list and is adopted (issue #405): the CI & Performance Budget above states how the suite is run under it.

### Core Development Standards

* **Reproducibility:** `--locked` for CI installs, pinned runner images (`ubuntu-24.04`), and every generator seeded through `np.random.default_rng(seed)`.
* **Versioning:** Lives in `Cargo.toml` (`[package].version`), and nowhere else.
* **Definition of Done:** Follow `CLAUDE.md`'s checklist.
* **A PR implements a plan already approved.** The ticket carries a plan comment before any code exists, the issue is labelled `planned`, and a maintainer applies `approved`; only then may the pull request open. A plan that turns out to be flawed gets a revised plan posted to the thread, not a silent correction in the diff.
* **Record the branch before the first commit.** Once a branch is created for an approved plan, the first thing posted is a single issue comment naming the branch (and, once opened, the PR number) — before any further commit is pushed, so an interrupted or deferred session leaves a ticket that already points at the in-flight branch.
* **A ticket body and a pull-request body carry at most 40 content lines.** Blank lines and headings are not charged, so the cap is what a reader reads (issue #521). It is counted by `infra/experiments.py`'s `body_lines`, the same function the ten-line experiment cap uses; `infra/check_pr_body.py` charges a pull request against it in the `pr-title` job, and the issue templates state it for an author. A body that will not fit is not a body to widen the cap for: the detail belongs where it is refereed — a measurement in `STATUS.md`, a sweep in an experiment file — the ruling issue #458 made for experiments. Tickets are enforced by the template's text alone; a workflow on `issues` would add a job to every open, edit and reopen across the repository to catch what an author can see in the template.
* **A plan is 2–5 steps**, or more where the work needs them and the plan says why, each stating how it will be validated — the analytic result, brute-force computation or enumeration it is checked against, not "tests pass". It ends with an `Open Questions` section carrying every question on the desired behaviour, so a reviewer finds them in one place; a plan with none says so under that heading rather than omitting it.
* **PR Template:** Every PR starts from `.github/pull_request_template.md`: the Definition-of-Done checklist, a benchmark-numbers table, a Documentation Sync line, and a Follow-up / Deferred Work section. A second table in the Benchmark section takes the realized value of every scientific or tolerance test the PR touches — test, reference, tolerance, realized value — or the text "N/A" and no table. The template reminds; it is not a CI gate.
* **Agentic Approach:** Disjoint tickets run one at a time, each in its own git worktree and its own pull request; coupled changes run as a single sequential chain, each stacked on the last. Stated the same way in `ROADMAP.md` §0.2.
* **A seam is added at three consumers, and `SEAMS.md` says which exist.** `infra/seams_survey.py` writes `SEAMS.md` from the package: every `Protocol` and every data contract three or more modules share, with its members, the classes that satisfy it structurally, the modules that consume it, the `PROBLEMS.md` rows it reaches, and a verdict against the rule; `tests/regression/docs/test_seams_survey.py` holds the survey to what it claims. The file is not committed — write it with `infra/ledgers.sh`, or read it from a CI run's summary. A proposal for a new abstraction states the consumers that would call through it and is declined below three, as issue #400 declined one EM driver over the HMM and the mixture (two callers sharing 16 lines of loop skeleton, more lines added than removed). A seam already declared but under the rule carries its reason in the survey's table; one with no reason is removed or argued for. The problem classes have no `Problem` protocol: the fixture registry (`snakes_and_ladders.sim.fixtures`, issue #382) is their contract, and the survey shows no operation with three such callers.
* **One agent at a time, and it has the whole host.** An agent works in its own worktree from `main` (`infra/new_worktree.sh`) with `OMP_NUM_THREADS=1`, and nothing else runs beside it, so no job waits on another. The readers-writer lock that arbitrated a shared host is gone (issue #521). It was built to make measurements comparable under parallel agents and, measured, did not exclude: during an exclusive run 2.68 cores were busy and 1.75 of them foreign; `with_lock measure` gave up after its full 1,800 s wait while shared slots ran ahead of it, issue #399's Rust agent measured 40 minutes of starvation inside a 104-minute run at five agents, and three validation loops starved a measurement for an entire 18 s stream where the pre-#427 lock admitted it in 0.06 s ([#431](https://github.com/michaelJwilson/snakes_and_ladders/issues/431), closed as moot); and it taxed jobs that never needed it, `infra/baselines.py --write` costing 30 minutes of queueing for a job that runs in 28 s (#427). A job now runs when it is started, and a measurement states the load average it was taken at.

  **A quiet host is not a wide one, and measurement says it should not become one.** `tests/conftest.py` still pins one BLAS thread per process, and that stays. Issue #525 reopened it on a quiet host and found the CI tier at four threads takes **1,227.4 s against 1,226.2 s at one** — a 0.1% difference inside the spread of the one-thread readings themselves. Widening buys nothing on this suite and would stale 221 recorded timings and 59 committed baseline floats, so it is declined on its clock rather than only on its cost. A measurement of something that is itself parallel states its thread count wherever it is quoted, and sets the thread variables itself; the figure renderers are exempt, `qa.build` stripping the variables so a render matches its manifest.

* **Merges to `main` land in a batch, not as each pull request goes green.** Every merge obliges each live branch to fetch, merge, re-validate and push, so *n* merges across *b* live branches cost `n * b` cycles where one batch costs `b`. Six merges across eight branches in the hour before this was written is 48 cycles where 8 would have done. A pull request that unblocks a queued ticket is the exception and merges at once.

* **Batch by plan shape; split by independence.** Tickets whose plans have the same shape — the same seam behind several callers, the same profile-first exit against a framework — land as one pull request, one agent, one CI run, one review (the 2026-09-08 queue: #390, #389 and #388 as one; #387 and #386 as one). A plan whose steps are independent still lands as one pull request; the steps run in sequence, since one agent works at a time.
* **Validate locally what fails fast; the pull request's CI is the final validation.** Before a push an agent runs the changed-file `ruff` and `mypy --strict` and `pytest -m critical` (16 s). The full `infra/validate.sh` runs once, before the pull request leaves draft, or when the change touches a shared fixture, `src/` or a lockfile. The pull request is then watched to green and fixed on each red check.
* **Review starts from `infra/review_gates.sh`.** It runs the Definition-of-Done gates CI does not assert or asserts late — the environment imports this checkout, the head carries the base, the two PDFs are the base's unless the pull request is a rebuild, a changelog fragment exists, the critical tier passes under `SAL_DURATION_CAP`, every test the branch adds or rewrites carries a kind marker, a `Protocol` it adds is in `infra/seams_survey.py`'s catalogue with the consumers the seam rule wants or a stated reason, and regenerating `CHECKS.md`, `SEAMS.md`, the problem tables and the two blocks written from `infra/gates.py` rewrites no tracked file — and prints a pass/fail table, each row with its own wall clock. **Eight rows in 26 to 27 s**, two consecutive runs on 2026-09-09 on the 4-core host at a 1-minute load of 1.19 and 1.30, carrying no job but this one, of which the critical tier is 20 s (177 tests then, 168 since issue #526) and the ledgers 6 to 7 s; the other six rows report under 1 s between them. The two readings it replaces, 39 to 43 s at a load of 9.2 and 8.0, were upper bounds taken while a shared host starved the lock that was supposed to quiet it (issue #521). The ledger row writes the three files rather than comparing them against committed copies (issue #425), 1 s more than the `--check` calls it replaced. Writing `pyproject.toml`'s marker list and the tier table above beside them costs that row **0.07 s**, the same to two decimals over three runs on the 4-core host at a 1-minute load of 3.8 to 5.9 (issue #470). `infra/gates.py` names the eight rows and this budget in one place, and `tests/regression/test_gates.py` holds this sentence to the script (issue #469). The budget is 30 s, above which a check belongs in CI rather than in something a reviewer runs per branch — which is why the duration cap on the *changed* tests is not a row: running them cost 20 s of the 30 on the branch that added these gates, and `infra/validate.sh` has already run them under the cap on this host. Reading then covers what a script cannot: the diff against the plan on the ticket, and whether each test pins what it claims to.
* **Every pull request names its base branch in its title, and targets it.** The title starts with the base branch in brackets — `[main]` for a root, `[claude/phylo-249-document-split]` for a link in a chain — and the base is set to that branch (issue #292). The base branch rather than the parent's PR number, because it is what the pull request's `base` field holds, so the `pr-title` job checks the two agree and it survives the parent being renumbered or closed. A stacked pull request targets its *parent branch* rather than `main`, so the diff under review is the change itself — measured on the #190 chain, #225 was 64 changed files against `main` and 11 against its parent. GitHub retargets a child to `main` when the parent merges, so the title prefix is the only part to update by hand.
* **A chain merges bottom-up, with a merge commit.** Squash and rebase-merge both rewrite the parent's commits into new SHAs, after which the child no longer contains them: its diff duplicates the parent's content and every pull request below it conflicts. A merge commit preserves the ancestry, so each child's diff narrows to its own change the moment its parent lands. Bring a chain up to date the same way — cascade `main` into the root, then each parent into its child — and never rebase or force-push a branch, which invalidates any checkout of it and leaves the stale heads issue #123 records. Where two subtrees share a root, take the longer one first: whichever goes second is reconciled per branch.

### Dependency Management

1. **Request:** Explicitly request permission before adding dependencies/tools.
2. **Validate:** OSI-approved licences only. Flag items with $<1000$ GitHub
   stars. `docs/external_tools.md` surveys the external phylogenetic tools
   against both.
3. **Lock:** Run `uv lock` or update `Cargo.lock` and commit in the same PR.
4. **Justify:** Explain the inclusion in the PR description.

### Experiments

A measured comparison lives in `docs/experiments/`, one file per experiment,
written from `TEMPLATE.md` (issue #314): YAML front matter with the commit,
branch and pull request, the tickets it tests and files, the problem, fixture
and size tier, the methods compared, the budget and its unit, the shared
seeds, the hardware and a status; then three sections — question, numbers,
finding. **The body is at most ten non-blank content lines** after the front
matter's closing `---` (issue #458). Neither the title nor a section heading
is charged, and the front matter is not counted at all, being the
reproducibility record. What survives is chosen in this order: key metrics,
motivation, reproducibility. A number the cap displaces moves to `STATUS.md`
where it is evidence for a milestone, or to the pull-request body where it is
the argument for a change. `infra/experiments.py` validates every file and
generates the index `README.md`; `--check` fails on an invalid file, a body
over the cap, or a stale index, and `tests/regression/test_experiments.py`
runs it per pull request.

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
  script that produced them named.

### Release

A release is cut from a Release-template issue
(`.github/ISSUE_TEMPLATE/release.yml`): it drives the repository-consolidation
audit — roadmap progress, document/code consistency, duplicated machinery,
follow-up tickets — and gates on `infra/release.sh` before a maintainer adds
the `release` label.

[RELEASE.md](RELEASE.md) carries the procedure in full: what the cut is taken
against, the preconditions with their current verdicts, the steps in order, who
runs each, and what a release costs in hours. What a *change* meets belongs
here: `infra/release.sh` runs
every per-pull-request check above, plus what CI skips per pull request — both
`--features sandbox` `cargo` passes (see "Build System"), the `release`, `stress`
and `key` tiers of `pytest`, the full `sphinx-build -W`, every figure in
`snakes_and_ladders.qa.manifest` compared against its committed bytes, and
`infra/build_documents.sh`. It runs every step regardless of earlier failures
and exits non-zero if any failed, so a branch that lands clean per pull request
can still owe work at the gate.

**Versioning lives in `Cargo.toml`'s `[package].version` and nowhere else**
(`CLAUDE.md`). A version whose `CHANGELOG.md` section already exists is spent,
because `towncrier build` writes a second section rather than extending the
first; `infra/release.sh` does not check that, and issue #146 records the gap.

---

## Application Standards

`CLAUDE.md` states these and this file does not restate them: **Performance**
for when a hot path earns a GPU port and why the NumPy reference stays,
**Testing & Quality Assurance** for what an assertion must establish, and
`docs/CLAUDE.md` for how the documents are built and kept true. They were
duplicated here until issue #146.
