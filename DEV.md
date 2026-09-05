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
| `python/snakes_and_ladders/` | Python package: re-exports, typed extension stubs, stub CLI. |
| `python/snakes_and_ladders/sim/` | Data generation and ground-truth retention. |
| `python/snakes_and_ladders/likelihood/` | Felsenstein pruning; CPU dispatch landed (NumPy, PyTorch, Rust), CUDA and Metal dispatch not yet implemented. Also the phylogenetic `Objective` (`objective.py`), which adapts the recursion to `opt/`'s fitting interface — it is here because `opt/` may import no application module. |
| `python/snakes_and_ladders/opt/` | Model-agnostic continuous parameter fitting via autodiff (PyTorch): the `Objective` interface, shared constraint maps, and the Potts and HMM reference instances. Imports nothing from `sim/`, `likelihood/` or `search/`, asserted by test. |
| `python/snakes_and_ladders/learn/` | Model-agnostic reinforcement learning: the `Environment` interface, the policy, REINFORCE, an exact trajectory-enumeration oracle, and a Potts-landscape reference instance. Imports nothing from `sim/`, `likelihood/` or `search/`, asserted by test. |
| `python/snakes_and_ladders/search/` | Move sets, temperature schedules, and the hill-climbing search (`infer.py`) that joins them to `opt/`. The phylogenetic RL environment (`rl.py`) lives here too, for the reason the phylogenetic `Objective` lives in `likelihood/`: `learn/` may import no application module. |
| `python/snakes_and_ladders/qa/` | QA figures/tables for the documents; renders, doesn't recompute. |
| `src/lib.rs` | Rust extension (`oxi_snakes_and_ladders`), exposed through PyO3. |
| `docs/tex/` | LaTeX source for the paper and the textbook, with the notation and preamble both share. |
| `infra/build_technical_doc.sh` | Regenerates QA figures, then builds `docs/paper.pdf` and `docs/textbook.pdf` (both committed). |

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
| `tests/` (top level) | Whole-package and binding smoke tests, which belong to no single kind or submodule — `test_run_phylo.py`, `test_oxiphylo_bindings.py`. |

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

Nine required checks run via GitHub Actions (`.github/workflows/ci.yml`) on PRs against `main`:

| Job | Execution |
| --- | --- |
| `lint` | `ruff check`, `ruff format --check`, strict `mypy`, `towncrier check` |
| `rust-lint` | `cargo clippy -D warnings`, `cargo fmt --check` |
| `rust-tests` | `cargo test --locked`, `cargo bench` (informational) |
| `build` | `pip install .` (no lockfile, mimics fresh consumer), smoke import |
| `python-tests` | `pytest -m "not release"`, gated on minimum coverage; benchmarks skipped unless computational code changed |
| `docs` | Sphinx build (warnings as errors) |
| `technical-doc` | Regenerate the QA figures the documents cite (`infra/build_technical_doc.sh`), then LaTeX build. Fails on an undefined reference or citation, a multiply-defined label in either log, or a rebuilt PDF that differs from its committed copy |
| `notebooks` | Re-execute every notebook under `docs/nb/` (`infra/check_notebooks.py`) and fail on a re-executed output that differs from the committed one. Text is compared; a figure is checked only for still being produced. Regenerate with `--write` on the same script |
| `audit` | `pip-audit`, `cargo audit` (skips on cache hit if lockfiles are unchanged) |

`lint`, `python-tests`, `docs`, and `notebooks` restore a `~/.cache/uv` cache keyed on `uv.lock`'s hash before installing `uv`. `rust-lint`, `rust-tests`, `build`, and those same four jobs restore a shared `~/.cargo/registry`, `~/.cargo/git`, and `target/` cache keyed on `Cargo.lock`'s hash, so `oxi_snakes_and_ladders` (built via `maturin`/`pyo3` on every `uv sync` or `pip install .`) compiles from scratch only when a lockfile changes or no job has populated the cache yet. `audit`'s per-week marker cache (above) is unrelated and unaffected.

### CI & Performance Budget

* **Size Caps:** Restrict topological move tests to $n \le 10$ (exhaustive enumeration oracle).
* **No CI Profiling:** Do not rank performance on GitHub runners due to hardware variance. Benchmark on fixed hardware.
* **Release-Gated:** Long-running scientific validity tests run on release, not per PR. Mark them `@pytest.mark.release` (registered in `pyproject.toml`).
  * **Use `pytest -m "not release"` while developing.** That is what CI's `python-tests` job runs, so it is the gate a PR is actually judged against. Do not run the full suite to check ordinary work.
  * **The full suite is expensive and its cost is not obvious from the test count.** Measured on one development machine on the same checkout: `pytest -m "not release"` took 138 s over 540 tests; plain `pytest` took 989 s over 550 — ten extra tests, roughly 7x the wall clock. Exhaustive topological tests dominate, and they grow combinatorially with taxon count. `infra/measure_build.sh` reproduces the first of these; both are stated here so a change that moves them is visible, and both were four times stale before issue #154 measured them again.
  * **Plain `pytest` (no `-m` filter) is the release gate's job, not a development command.** `infra/release.sh` runs it as part of cutting a release; run it by hand only when you are cutting one, or when you have changed a release-gated test itself.
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

### Core Development Standards

* **Reproducibility:** Pin the environment. Use `--locked` for CI installs, pin runner images (`ubuntu-24.04`), and seed every generator through `np.random.default_rng(seed)`.
* **Versioning:** Lives in `Cargo.toml` (`[package].version`), and nowhere else.
* **Definition of Done:** Follow `CLAUDE.md`'s checklist.
* **A PR implements a plan already approved.** The ticket carries a plan comment before any code exists, the issue is labelled `planned`, and a maintainer applies `approved`; only then may the pull request open, and it implements that plan. A plan that turns out to be flawed gets a revised plan posted to the thread, not a silent correction in the diff.
* **Record the branch before the first commit.** Once a branch is created for an approved plan, the first thing posted is a single issue comment naming the branch (and, once opened, the PR number) — before any further commit is pushed, so an interrupted or deferred session leaves a ticket that already points at the in-flight branch.
* **A plan is 2–5 steps**, or more where the work needs them and the plan says why, each stating how it will be validated — the analytic result, brute-force computation or enumeration it is checked against, not "tests pass". It ends with an `Open Questions` section carrying every question on the desired behaviour, so a reviewer finds them in one place; a plan with none says so under that heading rather than omitting it.
* **PR Template:** Every PR starts from `.github/pull_request_template.md`. It carries the Definition-of-Done checklist, a benchmark-numbers table, a Documentation Sync line, and a Follow-up / Deferred Work section for anything left to a tracking issue. A second table in the Benchmark section takes the realized value of every scientific or tolerance test the PR touches — test, reference, tolerance, realized value — or the text "N/A" and no table. The template reminds; it is not a CI gate.
* **Agentic Approach:** Disjoint tickets run as parallel git worktrees and parallel pull requests; coupled changes run as a single sequential chain, each stacked on the last. Stated the same way in `ROADMAP.md` §0.2.
* **A stacked pull request names its base in its title, and targets it.** `[on #212]` for a link in a chain, `[on main]` for a root. The base branch is set to the *parent branch* rather than `main`, so the diff under review is the change itself and not everything beneath it — measured on the #190 chain, #225 was 64 changed files against `main` and 11 against its parent. GitHub retargets a child to `main` on its own when the parent merges, so the title prefix is the only part to update by hand.
* **A chain merges bottom-up, with a merge commit.** Squash and rebase-merge both rewrite the parent's commits into new SHAs, after which the child no longer contains them: its diff duplicates the parent's content and every pull request below it conflicts. A merge commit preserves the ancestry, so each child's diff narrows to its own change the moment its parent lands. Bring a chain up to date the same way — cascade `main` into the root, then each parent into its child — and never rebase or force-push a branch, which invalidates any checkout of it and leaves the stale heads issue #123 records. Where two subtrees share a root, take the longer one first: whichever goes second is reconciled per branch, so the shorter chain is the cheaper one to leave until last.

### Dependency Management

1. **Request:** Explicitly request permission before adding dependencies/tools.
2. **Validate:** Must use OSI-approved licenses. Flag items with $<1000$ GitHub stars.
3. **Lock:** Run `uv lock` or update `Cargo.lock` and commit in the same PR.
4. **Justify:** Explain the inclusion in the PR description.

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
