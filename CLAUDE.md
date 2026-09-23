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
8.  **Open with a TL;DR:** every ticket, pull request, plan and review opens with the result — the number, the decision, or what broke — in O(1) lines before any context, so the opening does not grow with the body.
9.  **A document does not certify itself:** it states what it shows and names where each number is checked; it claims nothing about its own completeness.
10. **A notebook teaches:** every call in it carries a comment naming what the step does and why it is there.
 
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
| `CHECKS.md` | The evidence the repository is founded on, read from the tree and not committed. The abstractions are not listed beside it: a `Protocol` is declared in the package and found at import, so the declaration is the record (issue #586) |
| `CHANGELOG.md` | What has landed, per dated release section; built from `changelog.d/` fragments by `towncrier` |
| `INSTALL.md` | Installing, building and running locally |
| `DEV.md` | Layout, the CI jobs, repository settings, the CI budget, how a change is reviewed |
| `RELEASE.md` | Cutting a release: what it is cut against, the preconditions, the steps, who runs each, and what it costs |
| `REFERENCES.md` | The texts and papers each part of the work is cited against |
| `docs/tex/` | The academic paper reporting results, the textbook of problem statements, algorithms, and properties that define them, the API map --- the package's own surface, generated from its docstrings and edited by no one --- and the mind map of the package, which is the one document that names code |
| `docs/nb/` | One worked notebook per problem, from the fixture (of given size) to likelihood, optimization and learned policy |
| `docs/reviews/` | Dated reviews of the whole between releases: what is established, where the documents and the code disagree, what is missing, and what is worth doing next; each fixes what is stale and tickets the rest |

Submodules include `infra/`, `sim/`, `likelihood/`, `opt/`, `search/`, learn/`, `qa/`, `sandbox/`, `validation/`, and `docs/` with each carrying their own `CLAUDE.md` of specific details.  All are subject to the same **writing style, and rules** as defined in this doc.  `ROADMAP.md` and `STATUS.md` plan and track at a higher level; open work is in the issue tracker, not a file (#804). `DEV.md` and `INSTALL.md` are detailed, e.g. step by step.  There may be some light repetion betwen the two then.

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
*   **Rust Backend (`oxi_snakes_and_ladders`):** Target for CPU-bound hot paths (control flow, tree traversal, irregular memory access, small sizes), and keep it only where it earns $\ge 2\times$ over the vectorized NumPy reference at realistic problem sizes. The bar is the GPU rule's, set lower because the cost it buys off is lower: a second language in the build, a second implementation to keep in step with its oracle, and a kernel a reader must cross a language boundary to follow. Below it the simpler code wins and the port is reverted rather than kept — a backend that exists and is never faster is a maintenance cost with no counterpart.
*   **Measurement:** Benchmark candidates against the NumPy reference before committing to a port. Report both numbers in the PR. **The two concerns do not share a size policy.** Validity is established at every tier — at gate sizes so a merge has something to gate on, and at stress sizes because a claim that holds only where the problem is small is not the claim the roadmap makes. A *speedup* is established at stress sizes alone: that is where the work is done and what a user pays. So a ratio read at a gate size decides nothing in either direction — a regression there is not a defect to fix, a win there is not a result to report — and an optimization whose only evidence is a gate-sized benchmark has not been measured.
*   **The Oracle:** Every accelerated kernel keeps its pure Python/NumPy implementation as an oracle. Regression tests must pin the accelerated output against it, and recover known values on sims.

## High Performance coding
*   **Profile first.** `cProfile` decides what is worth teststing and whether alternatives are superior.
*   **A ratio is half the case; the effect size is the other half.** The thresholds above are ratios --- $\ge 10\times$ for the GPU, a CPU-bound hot path for Rust --- and a ratio alone admits a port significant in neither direction: a $10\times$ on a term worth 40 us buys 36 us, and the rule as written says take it. A port states both, and states the absolute saving *against something*: the call that encloses it, or the budget `DEV.md` declares. **Profile first** ranks what to port; this says when a ranked candidate is still not worth porting. The precedent is the judgement made before the rule existed --- a hoisted multiply predicted to dominate, measured at 1.2 ms against 97.8 ms of `digamma` in a ~270 ms solve, under 0.5%, and left alone with nothing to cite. This is not a threshold to tune: where ratio and effect size disagree, say which and why, and the saying is the point.
*   **Do less work before doing the same work faster.** An algorithmic cut outranks a mechanical one and the profile says which is available; #289 bought 6.3x by scoring fewer candidates, where the layout work on the same tree bought 3.7x.
*   **Warm starts before cold ones.** A search recomputing per step what it could update incrementally pays the full cost per step, and asking for that comes before reaching for a faster language.
*   **Recompute or store is a decision, and unmade it defaults to recompute.** A derived quantity rebuilt inside a loop is a store nobody has chosen yet: `compressed_adjacency` was 2.5 ms of a 3.5 ms cluster move, rebuilt per call to touch one cluster.
*   **Cost depends on the data, not only on its size.** A route chosen on problem size alone is chosen on the wrong variable, and a default taken from one fixture is a default taken from one dataset; at one size, replacing a fixture's branch lengths moved taped against analytic from 2.1x to 0.6x (#443).
*   **L1/2/3 caches.** Design code to fully utilize simd and the cache.
*   **Memory layout.** Contiguous, row-major arrays walked in stride order; neighbour lists as offsets, not lists of lists. **The rule stops at the Python boundary:** it is about a NumPy or compiled consumer, and for a pure-Python inner loop it inverts — over 16,384 rows of degree six a row walk is 2.0 ms as lists, 3.9 ms flat with offsets and 25.6 ms as a NumPy slice.
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
*   **Bitwise Is the Target, the Declared Tolerance Is the Floor:** an equality is the strongest thing a referee can assert, so it is what a comparison strives for and what a change is expected to preserve. It is not a veto. Where a change is justified on its own terms --- a temporary removed, a reduction reordered, a backend swapped --- and the only cost is bitwise agreement, it may back off to the tolerance declared for that comparison, with the measured difference stated beside it and the guard restated as that tolerance rather than deleted. Backing off is a step down a scale, never off it: a difference outside the declared tolerance is a defect and the ticket stays open. What this forbids is loosening a bound *to admit a result* --- the tolerance is where a comparison lands, never what it is fitted to.
*   **Cross-Device Agreement Is a Tolerance:** given `float32` and `float64`, tolerance is the higher precision, etc.
*   **No Coverage Theatre:** coverage counts a test only when it judges the science against something outside the implementation: `end2end` drives `sal` through a real path and judges the output against the truth that generated the data, `oracle` judges it against an exact independent answer. Strive for both over the complete functionality of every problem, component by component and as one script that runs the whole path. `analytic`, `patch`, `backend`, `bug`, `warning`, `snapshot`, `smoke` and `experiment` say what a test is and do not count; `infra` is used sparingly. A test asserting only a shape or the absence of an exception is `smoke`; one pinning which of two runs of the package reached further is `experiment`. Leave gaps documented where necessary and track with tickets.
*   **Every Test Says What It Is Checked Against:** a test carries marker tags for meaning and selection, e.g. naming its referee. `pyproject.toml` registers the ones an author writes; a marker a hook derives is registered where it is derived, so nothing is maintained by hand twice.
*   **Notebooks:** `docs/nb/` carries one notebook per problem (of various sizes), demonstrating functionality is supports, and evidence by oracles.  A **Further Work** specifies gaps/relevant roadmap.
*   **Scientific Outputs:** The suite must emit plots, tables and the  LaTeX documents.
*   **Time is money:** test and build frameworks should be justified wrt time and computational budget, e.g. cached; a high priority is to quickly standup a minimal implementation against the ROADMAP.md with  test-driven development.  
*   **The per-PR tier is the fast gate; the release gate runs everything.** A test over the per-PR duration cap, or whose claim is not needed to gate a merge, carries the `release` marker and runs for a release.
*   **Select narrowly to iterate; gate on the tier.** Selection composes over what a test *is* --- the problem it exercises and the kind that referees it --- and the tier it runs in, so reach for the narrowest selection that can still refute the change. The narrow run is for iterating and is never reported as the gate: on #601 and #595 this session the full tier caught what the `critical` gate could not, twelve failures and two stale references. `DEV.md` holds the expressions.
*   **A plan says how the work will be developed, not only how it will be checked.** A plan that names an oracle and a tolerance but no selection has said nothing about how the work will be iterated on, or about which tier decides it. `ROADMAP.md` §0.2 states what a plan must name; it is not restated here.

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