# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Writing Style
1.  **(Reviewer) Time is money and context windows are short:** Be concise and direct, use active voice, and limit to the most important facts, in priority.  Limit tickets and normal PRs to 40 lines, limit PRs with extended discussion to 60 lines, e.g. on release.
2.  **Be precise:** Use exact facts and numbers ("40% faster") instead of vague intensifiers ("much faster"), except where this lacks meaning, e.g. byte reproduction of figures.
3.  **Stay neutral and objective:** Avoid hype, subjective opinions, weak qualifiers, and delivering points in both positive and negative. Use nouns and verbs; avoid adjectives and adverbs.
4.  **Provide evidence:** Back every claim in PRs/commits with benchmark numbers, test validated outputs, or reproductions.
5.  **Maintain formatting:** Apply naming, terminology, notation and syntax consistently.
6.  **CLAUDE.md edits are rare** Do not add technical details to CLAUDE.md files, but principles.  These edits are rare, as (lack of) principles become apparent.
7.  **Maintain tone** maintain tone of this document throughout the repository and associated work.
 
These rules are paramount for this repository: e.g. every document, `CLAUDE.md`, docstring, comment, commit message, PR, and plan or comment posted to thread. They are stated here, rarely repeated elsewhere.

## Project
`snakes_and_ladders` is a high-performance scientific repository for inference on graphical models with  discrete/continuous optimization.  Correctness and reproducibility of numerical and scientific results are required, despite inconvenienece.

Two primary concerns are held distinct:

*   **Infrastructure:** the builds, the tests, the benchmarks, the release, the agentic loop (ticket/plan/PR).  None of it is application specific.
*   **Applications:** GMMs, HMMs, phylogenetics, ND Potts models and classical/quantum error correction.

Structure enforces the separation, e.g. `README.md` and `DEV.md` detail infrastructure before application, where distinct submodules exist.

## Repository Map
This file is authoritative. Each of the remainder has a defined task:

| Document | Job |
| --- | --- |
| `README.md` | A high level project overview, referencing the details elsewhere |
| `ROADMAP.md` | The goals and the planned path to them |
| `STATUS.md` | What has landed against each roadmap milestone, the evidence, and the PR carrying it |
| `TICKETS.md` | Titles of planned tickets remaining on the roadmap |
| `CHECKS.md`, `SEAMS.md` | The evidence the repository is founded on, and the abstractions/apis that ensure usability and extensibility|
| `CHANGELOG.md` | What has landed, per dated release section; built from `changelog.d/` fragments by `towncrier` |
| `INSTALL.md` | Installing, building and running locally |
| `DEV.md` | Layout, the CI jobs, repository settings, the CI budget, how a change is reviewed |
| `RELEASE.md` | Cutting a release: what it is cut against, the preconditions, the steps, who runs each, and what it costs |
| `REFERENCES.md` | The texts and papers each part of the work is cited against |
| `docs/tex/` | The academic paper reporting results, and the textbook of problem statements, algorithms, and properties that define them |
| `docs/nb/` | One worked notebook per problem, from the fixture (of given size) to likelihood, optimization and learned policy |

Submodules include `infra/`, `sim/`, `likelihood/`, `opt/`, `search/`, learn/`, `qa/`, `sandbox/`, and `docs/` with each carrying their own `CLAUDE.md` of specific details.  All are subject to the same **writing style, and rules** as defined in this doc.  `ROADMAP.md`, `STATUS.md` and `TICKETS.md` plan and trackat a level at a higher level. `DEV.md` and `INSTALL.md` are detailed, e.g. step by step.  There may be some light repetion betwen the two then.

## Environment & Tooling
*   **Python (3.12):** Manage via `uv`. Run `uv sync --locked --all-extras`. Regenerate locks with `uv lock` and commit `uv.lock` in the same PR.
*   **Rust:** Compiler pinned via `rust-toolchain.toml`. Lockfile is `Cargo.lock`. Update with `cargo update` and commit.
*   **Lint/Format (Python):** `ruff check .` and `ruff format --check .`
*   **Type Check (Python):** `mypy --strict`, over the paths in `pyproject.toml`'s `files` (`python/`, `tests/`, `infra/`).
*   **Lint/Format (Rust):** `cargo clippy --locked --all-targets -- -D warnings` and `cargo fmt --check`.
*   **Audit:** `pip-audit` (Python) and `cargo audit` (Rust).
*   **Docs:** Build with `sphinx-build -W` in `docs/source/`.

## High Performance frameworks
*   **GPU (PyTorch, Triton, JAX):** Target if the hot path is data-parallel and earns $\ge 10\times$ speedup over vectorized NumPy at realistic problem sizes.
*   **Autodiff:** **PyTorch**, decided. Its MPS backend is the path on Apple Silicon, which `ROADMAP.md` targets alongside CUDA.
*   **Rust Backend (`oxi_snakes_and_ladders`):** Target for CPU-bound hot paths (control flow, tree traversal, irregular memory access, small sizes).
*   **Measurement:** Benchmark candidates against the NumPy reference before committing to a port. Report both numbers in the PR.
*   **The Oracle:** Every accelerated kernel keeps its pure Python/NumPy implementation as an oracle. Regression tests must pin the accelerated output against it, and recover known values on sims.

## High Performance coding
*   **Profile first.** `cProfile` decides what is worth teststing and whether alternatives are superior.
*   **L1/2/3 caches.** Design code to fully utilize simd and the cache.
*   **Memory layout.** Contiguous, row-major arrays walked in stride order; neighbour lists as offsets, not lists of lists.
*   **Vectorization and SIMD.** Inner loops contiguous, unaliased, without early exit or data-dependent reduction, so NumPy and the Rust compiler vectorize.
*   **Branch misprediction.** minimize branch misses, queries etc.
*   **Inlining.** No Python-level call per site or per node; hoist it or vectorize it. In Rust, `#[inline]` the small hot helpers; use smallvec on the stack.
*   **Allocation.** Preallocate and reuse buffers across sweeps.
*   **The FFI boundary.** Cross it once per call with contiguous arrays.
*   **Parallel over independent tasks.** A loop of independent bodies — starts, seeds, replicates — should utilize `snakes_and_ladders.parallel`.
*   **The GIL.** A compiled kernel that touches no Python object releases it, or the thread backend cannot use it. Held, four Python threads took 3.82x the wall of one on the Potts sweep — serialization exactly; released, 1.08x, for a throughput of 3.70x beside a NumPy control's 3.19x (#604).
*   **Compiled backends.** `njit` for ease, Rust carries the load.

## Testing & Quality Assurance
*   **Simulate Component-Wise:** Build fixtures across a set of sizes; simulate from a known generative model with n explicit a seeded generator. Test components individually and in combination.
*   **Fixtures Carry Their Own Truth; oracles, and recover of known parameters/configurations will determine validate and are required.
*   **Pin to Independent Sources:** Validate expected values against analytic properties, brute-force computations, or secondary implementations with stated tolerances.
*   **Check known Math properties** limits, invariants, etc.
*   **Cross-Device Agreement Is a Tolerance:** given `float32` and `float64`, tolerance is the higher precision, etc.
*   **No Coverage Theatre:** Tests asserting only output shapes or successful execution without exceptions are forbidden. Leave gaps documented where necessary and track with tickets.x
*   **Every Test Says What It Is Checked Against:** a test carries marker tags for meaning and selection, e.g. naming its referee.  `pyproject.toml` registers the names.
*   **Notebooks:** `docs/nb/` carries one notebook per problem (of various sizes), demonstrating functionality is supports, and evidence by oracles.  A **Further Work** specifies gaps/relevant roadmap.
*   **Scientific Outputs:** The suite must emit plots, tables and the  LaTeX documents.
*   **Time is money:** test and build frameworks should be justified wrt time and computational budget, e.g. cached; a high priority is to quickly standup a minimal implementation against the ROADMAP.md with  test-driven development.  
*   **The per-PR tier is the fast gate; the release gate runs everything.** A test over the per-PR duration cap, or whose claim is not needed to gate a merge, carries the `release` marker and runs for a release.

## Definition of Done
1.  **Regression Test:** Asserts scientific validity against known simulations, simpler/alternative algorithms, and pins expected output.
2.  **Benchmark:** New/changed hot functions include a `pytest-benchmark` (Python) or `criterion` bench (Rust). Baselines reported in PR.
3.  **Coverage:** `--cov-fail-under` gate is maintained or raised. Never lower it to pass a PR.
4.  **Docs & Tooling:** CI covers the new code. `ruff`, `mypy`, and `cargo` checks pass locally. Documentation Sync above is satisfied.
5.  **Dependency Hygiene:** Follows OSI-license and external tools rules.

## Dependencies & External Tools
*   Ask for explicit permission before adding new tools/dependencies.
*   Must be open source (OSI-approved license).
*   Flag any proposed dependency with $<1,000$ GitHub stars (or equivalent ecosystem metric).

## Conventions
*   **Documentation Sync:** Any change affecting behavior, CI, dev setup, or math models must update, in the same PR, whichever of the documents it makes inaccurate: e.g. `README.md`, `CLAUDE.md` (including modules), `DEV.md`, `INSTALL.md`, `docs/tex/`, `docs/nb/`, etc. If the change is user-visible, add a fragment under `changelog.d/`.
*   **Code Standards:** Use type hints where possible. Do not introduce silent behavior changes (e.g., default parameters). Keep dependencies minimal and justify additions.
*   **Define abstractions/apis where they align and simplify  multiple use cases**
*   **Dev Standards:** One PR per ticket to minimize review and tests.
*   **A tickets report as they go:** Open a new branch and draft a PR immediately when startign work.
*   **Throughput:** One agent works at a time, in its own worktree, and it has the whole host.
*   **Model Routing:** Claude Fable carries the judgement: it plans tickets, writes plans and PRs, reviews work by lesser models, and handles review tickets. Delegate to Opus otherwise.