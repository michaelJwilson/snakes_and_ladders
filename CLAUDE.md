# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Writing Style
1.  **(Reviewer) Time is money and context windows are finite and vital:** Be as concise/succinct as possible, use active voice, and lead with (only) the most important facts first.
2.  **Be precise:** Use exact facts and numbers ("40% faster") instead of vague intensifiers ("much faster").
3.  **Stay neutral and objective:** Avoid hype, subjective opinions, and weak qualifiers. Use nouns and verbs; avoid adjectives and adverbs.
4.  **Provide evidence:** Back every claim in PRs/commits with benchmark numbers, test validated outputs, or reproductions.
5.  **Maintain formatting:** Apply naming, terminology, and syntax consistently.
6.  **CLAUDE.md edits are rare** Do not add technical details to CLAUDE.md files, but principles.  These edits are rare, as principles become clear.

These rules govern all work for this repository: e.g. every document, each module's `CLAUDE.md` included, and every docstring, comment, commit message, PR, and plan or comment posted to a ticket thread. They are stated here once and referenced from the module files rather than copied into them, so there is one text to change and nothing to fall out of step with it.

**Expected Reader:** a well-educated developer with scientific and performance-computing background, but not an application expert, e.g. phylogenetics. Keep the documents streamlined — hyperlinks and citations over inline derivation — and push required application background (e.g. NNI, other standard algorithms) into a dedicated appendix, cited from the point of use rather than re-derived there. Treat the main text as a high-level overview of the current best-known approach (simulation, models, results) in terms of the roadmap, not an exhaustive record; link out to supporting docs, with plots, and results for dedicated studies that informed them. Adopt the style of an academic paper, supported by a textbook on domain-specific material likely new to the developer — two documents rather than one file with appendices (issue #249), so the paper can report results without carrying the formulations that support them, and the textbook can state an algorithm without naming any code that implements it.

## Project
`snakes_and_ladders` is a high-performance scientific repository. Correctness and reproducibility of numerical and scientific results take priority over convenience.

Two concerns are supported, and they must stay separable:

*   **Infrastructure:** the build, the checks, the release process, the agentic workflow. None of it names an application.
*   **Applications:** inference on graphical models, e.g. phylogenetics, likelihoods, tree search, and the standards this science requires.

An infrastructure rule that acquires an application reference has lost the separation. Structure enforces it: `README.md` and `DEV.md` put infrastructure before application, and the file layout keeps them apart (`infra/` against `python/snakes_and_ladders/*` and `docs/tex/`).

## Repository Map
This file is authoritative. Each of the remainder has a defined task:

| Document | Job |
| --- | --- |
| `README.md` | What the project is, and where everything else lives |
| `INSTALL.md` | Installing, building, running locally |
| `DEV.md` | Layout, the CI jobs, repository settings, the CI budget, how a change is reviewed |
| `ROADMAP.md` | The goals and a path to them |
| `STATUS.md` | What has landed against each roadmap milestone, the evidence, and the PR carrying it |
| `TICKETS.md` | The titles of the tickets that remain between `STATUS.md` and `ROADMAP.md` |
| `CHECKS.md`, `SEAMS.md` | The checks the roadmap's claims rest on, and the seams the package has; both generated from the tree, neither committed |
| `CHANGELOG.md` | What has landed, per dated release section; built from `changelog.d/` fragments by `towncrier` |
| `docs/tex/` | Two documents: a paper reporting results, and a textbook of the problem statements, algorithms, and the properties that referee them |
| `docs/nb/` | One worked notebook per problem class, from a fixture to a learned policy |

`python/snakes_and_ladders/sim/`, `likelihood/`, `opt/`, `learn/`, `search/`, `qa/`, `sandbox/`, `infra/`, and `docs/` each carry their own `CLAUDE.md`. Those add what applies only inside one module; they never override this file except for the vital **writing style rules, which binds everything**. A rule that binds the whole repository belongs here, not in one of them.

**Altitude, and what may repeat.** `ROADMAP.md`, `STATUS.md` and `TICKETS.md` plan and track — what the project is doing, how far it has got, what remains — at a level a reader holds in their head. `DEV.md` and `INSTALL.md` are followed step by step, so they carry their detail in full rather than as pointers: someone working through one of them should not have to assemble the answer from three. Detail may therefore repeat between them, and where it repeats it must agree — a copy that has drifted is a defect, and this file settles which reading is right.

## Environment & Tooling
*   **Python (3.12):** Manage via `uv`. Run `uv sync --locked --all-extras`. Regenerate locks with `uv lock` and commit `uv.lock` in the same PR.
*   **Rust:** Compiler pinned via `rust-toolchain.toml`. Lockfile is `Cargo.lock`. Update with `cargo update` and commit.
*   **Lint/Format (Python):** `ruff check .` and `ruff format --check .`
*   **Type Check (Python):** `mypy --strict`, over the paths in `pyproject.toml`'s `files` (`python/`, `tests/`).
*   **Lint/Format (Rust):** `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`.
*   **Audit:** `pip-audit` (Python) and `cargo audit` (Rust).
*   **Docs:** Build with `sphinx-build -W` in `docs/source/`.

## Conventions
*   **Documentation Sync:** Any change affecting behavior, CI, dev setup, or math models must update, in the same PR, whichever of the documents it makes untrue: e.g. `README.md`, `CLAUDE.md` (including the modules), `DEV.md`, `INSTALL.md`, `ROADMAP.md`, `STATUS.md`, `TICKETS.md`, `docs/tex/`, `docs/nb/`. If the change is user-visible, add a fragment under `changelog.d/` (see `changelog.d/README.md`) rather than editing `CHANGELOG.md` directly — `towncrier` merges fragments into `CHANGELOG.md` at release time, and CI's `towncrier check` enforces one exists.
*   **Single Version Source:** The package version lives exclusively in `Cargo.toml`'s `[package].version`.
*   **Package Surface:** `python/snakes_and_ladders/__init__.py` re-exports nothing beyond the package's own top-level utilities (currently `double`);
*   **Code Standards:** Use type hints on all Python functions. Do not introduce silent behavior changes (e.g., default parameters). Keep dependencies minimal and justify additions.
*   **A seam earns its place by its consumers.** A `Protocol` or shared contract is added where three or more modules call through it; below that the operation is named consistently and the seam is not written, and an existing one is kept only for a reason stated where it is declared.
*   **Dev Standards:** The number of PRs should be minimized to limit the amount of review work and test runs, particularly given tickets are typically scoped to a work item.
*   **A tickets report as they go:** PRs open as a draft with its branch exists when the work begins.
*   **Throughput:** One agent works at a time, in its own worktree, and it has the whole host; there is no contention to arbitrate and nothing to serialize.
*   **Model Routing:** Claude Fable carries the judgement: it plans tickets, implements the plans, writes the pull requests, reviews work before it is posted, and takes review tickets. Everything else is delegated to Claude Opus subagents.

## Performance
*   **GPU (PyTorch, Triton, JAX):** Target if the hot path is data-parallel and earns $\ge 10\times$ speedup over vectorized NumPy at realistic problem sizes.
*   **Rust Backend (`oxi_snakes_and_ladders`):** Target for CPU-bound hot paths (control flow, tree traversal, irregular memory access, small sizes).
*   **Autodiff:** **PyTorch**, decided. Its MPS backend is the path on Apple Silicon, which `ROADMAP.md` targets alongside CUDA.
*   **Measurement:** Benchmark candidates against the NumPy reference before committing to a port. Report both numbers in the PR.
*   **The Oracle:** Every accelerated kernel keeps its pure Python/NumPy implementation as an oracle. Regression tests must pin the accelerated output against it, and recover known values on sims.

### Runtime Optimization Opportunities
Checked in this order when a hot path is proposed; `DEV.md` carries the procedure, `STATUS.md` the numbers.
*   **Profile first.** `cProfile` self time ranks the loop to port; a term that is a small fraction of the runtime pays for no port in any language (Gorelick & Ozsvald ch. 2).
*   **L1/L2/L3 cache.** Keep a recursion's live set — partials, messages, the current sweep — inside cache; release intermediates and report peak memory beside time (Bryant & O'Hallaron ch. 6).
*   **Memory layout.** Contiguous, row-major arrays walked in stride order; neighbour lists as offsets into one array, not lists of lists (Bryant & O'Hallaron ch. 6; Gorelick & Ozsvald ch. 6).
*   **Vectorization and SIMD.** Inner loops contiguous, unaliased, without early exit or data-dependent reduction, so NumPy and the Rust compiler vectorize; confirm by benchmark, never by asserting a width (Bryant & O'Hallaron ch. 5).
*   **Branch misprediction.** A data-dependent branch in an inner loop is free or a stall; the branchless form (mask, select, table) wins only where a measurement shows the branch does not predict (Bryant & O'Hallaron ch. 5).
*   **Inlining and call overhead.** No Python-level call per site or per node; hoist it or vectorize it. In Rust, `#[inline]` the small hot helpers (Gorelick & Ozsvald ch. 4; Bryant & O'Hallaron ch. 5).
*   **Allocation.** Preallocate and reuse buffers across sweeps; NumPy `out=` and in-place operators over temporaries (Gorelick & Ozsvald ch. 6).
*   **The FFI boundary.** Cross it once per call with contiguous arrays; minimize the perimeter.
*   **Parallel over independent tasks.** A loop of independent bodies — starts, seeds, replicates — runs through `snakes_and_ladders.parallel`.
*   **Compiled backends.** Two, each for a reason: Rust carries the load, so it stays opt-in; `numba`'s `njit` carries the ease.

## Testing & Quality Assurance
*   **Simulate Component-Wise:** Build fixtures by simulating from a known generative model under an explicitly seeded generator. Test components individually and in combination.
*   **Fixtures Carry Their Own Truth; an External Solver Is Surveyed Before It Is Adopted:** every claim here is refereed by an oracle or by the parameters that generated the data, and an empirical alignment supplies neither — it has no true tree, so a result on it can be reported but not checked. The repository therefore works on simulated fixtures today and takes no real data. External phylogenetic software is surveyed and considered, not adopted: nothing is installed now, and admitting one is a decision taken on its own evidence when there is a use for it. Three things would have to hold — a role it fills that this repository's own oracles cannot, an OSI-approved licence, and the **Dependencies & External Tools** rules satisfied. `docs/external_tools.md` carries the survey, so that decision starts from facts rather than from a search.
*   **Pin to Independent Sources:** Validate expected values against analytic results, brute-force computations, or secondary implementations with stated tolerances. Where no oracle is affordable at a size, recovering the simulated truth — the generating parameters or structure of a seeded fixture, to a stated tolerance or coverage — is sufficient, and the oracle remains desired and is ticketed.
*   **Check Math Invariants:** Rows of a transition matrix sum to 1, a reversible model satisfies detailed balance, gradients match finite differences, and a fit's likelihood increases monotonically.
*   **Cross-Device Agreement Is a Tolerance:** `float32` and `float64` behave differently across CPU, CUDA, and Metal, and deep recursions accumulate that. Agreement is checked against the tolerance stated in `likelihood/CLAUDE.md`, with the measurements it is derived from, and implemented in `snakes_and_ladders.likelihood.device`, never bitwise. A discrepancy inside it is not a bug and must not be "fixed". Two rules that fall out of it: the tolerance is **relative**, because the log-likelihood is a sum over sites and an absolute bound fixed at one problem size does not transfer to another; and it is keyed on the **lowest precision** in the comparison, because Metal cannot do `float64` and one bound loose enough for `float32` would let a broken `float64` backend pass.
*   **No Coverage Theatre:** Tests asserting only output shapes or successful execution without exceptions are forbidden. Leave gaps unwritten and track them as GitHub issues rather than writing meaningless tests.
*   **Every Test Says What It Is Checked Against:** a test carries a marker naming the kind of thing that referees it, and a test that fits no kind is either a missing category or a test with nothing to assert. This is what makes the rule above checkable rather than a matter of a reviewer noticing. The kind is a *tag* and not a partition: one test may be refereed two ways, and splitting it to fit a taxonomy would make it worse. Whether a test gates early is a separate axis from what it checks, because a test is critical *and* refereed somehow, never instead. `pyproject.toml` registers the names, `DEV.md` says which selection runs when, and a guard in `tests/regression/` enforces that every test carries one.
*   **Worked Notebooks:** `docs/nb/` carries one notebook per problem class, each running the application end to end against the oracles the regression suite already establishes, and each ending with a **Further Work** section that names, with its issue number, what it could not demonstrate because the feature is not built. A notebook states no result the suite does not also pin. A change that alters a number a notebook prints re-runs it in the same pull request, as it would regenerate a figure. A CI job re-executes every notebook and fails a pull request whose output disagrees with the committed one, so a notebook is held to the standard a `docs/tex/` figure is; `docs/nb/README.md` states what that comparison covers, and what it deliberately does not.
*   **Scientific Outputs:** The suite must emit plots and tables for the LaTeX documents. Update the LaTeX captions concurrently. Every figure is rendered from the code it reports on, ships with a caption naming the seed, sizes and model that produced it, and is committed under `docs/tex/figures/` so a changed plot is visible in review rather than only after a document build.
*   **Time is money:** test and build frameworks should be justified, time/computationally, e.g. cached; a high priority is to standup a minimal implementation against the ROADMAP.md with corresponding ablation studies with a fast test-driven development cycle.  Rely on the tests run on a PR as final validation where appropriate (late in development), rather than duplicating the effort - you will monitor the PR and fix issues before merging. The pull request's gate is bounded and the whole suite runs after the merge, so what a merge waits for and what the repository checks are two different sets; `DEV.md` states which is which and what refuses the bound.
*   **The per-PR tier is the fast gate; the release runs everything.** A test over the per-PR duration cap, or whose claim is not needed to gate a merge, carries the `release` marker and runs in the release process, which runs every tier; the duration guard in `tests/` enforces the cap, so a slow test is re-tiered rather than waited for. A claim moved to release keeps a fast small-size sibling per PR where one exists, and where none does the gap is a ticket, not a test. `DEV.md` states the cap and the tiers.

## Definition of Done
1.  **Regression Test:** Asserts scientific validity (not just shape/execution/coverage theatre) and pins expected output.
2.  **Benchmark:** New/changed hot functions include a `pytest-benchmark` (Python) or `criterion` bench (Rust). Baseline numbers reported in PR.
3.  **Coverage:** `--cov-fail-under` gate is maintained or raised. Never lower it to pass a PR.
4.  **Docs & Tooling:** CI covers the new code. `ruff`, `mypy`, and `cargo` checks pass locally. Documentation Sync above is satisfied.
5.  **Dependency Hygiene:** Follows OSI-license and external tools rules.

## Dependencies & External Tools
*   Must be open source (OSI-approved license).
*   Ask for explicit permission before adding new tools/dependencies.
*   Flag any proposed dependency with $<1,000$ GitHub stars (or equivalent ecosystem metric) for explicit review.
