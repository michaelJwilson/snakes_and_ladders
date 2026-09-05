# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Writing Style
1.  **(Reviewer) Time is money and context windows are finite and vital:** Be as concise as possible, use active voice, and lead with (only) the most important facts first.
2.  **Be precise:** Use exact facts and numbers ("40% faster") instead of vague intensifiers ("much faster").
3.  **Stay neutral and objective:** Avoid hype, subjective opinions, and weak qualifiers. Use nouns and verbs; avoid adjectives and adverbs.
4.  **Provide evidence:** Back every claim in PRs/commits with benchmark numbers, test validated outputs, or reproductions.
5.  **Maintain formatting:** Apply naming, terminology, and syntax consistently.
6.  ***CLAUDE.md edits are rare* Do not add technical details to CLAUDE.md files, but principles.  These edits are rare, as principles become clear.

These rules govern everything written in this repository: every document, each module's `CLAUDE.md` included, and every docstring, comment, commit message, pull-request body, and plan or comment posted to a ticket thread. They are stated here once and referenced from the module files rather than copied into them, so there is one text to change and nothing to fall out of step with it.

**Expected Reader:** a well-educated developer with scientific and performance-computing background, but not an application expert, e.g. phylogenetics. Keep tech. doc. streamlined — hyperlinks and citations over inline derivation — and push required application background (e.g. NNI, other standard algorithms) into a dedicated appendix, cited from the point of use rather than re-derived there. Treat the main text as a high-level overview of the current best-known approach (simulation, models, results) in terms of the roadmap, not an exhaustive record; link out to supporting docs, with plots, and results for dedicated studies that informed the technical doc. Adopt the style of an academic paper, supported by textbook style appendices on domain-specific material likely new to the developer.

## Project
`snakes_and_ladders` is a high-performance scientific repository. Correctness and reproducibility of numerical and scientific results take priority over convenience.

Two concerns are supported, and they must stay separable:

*   **Infrastructure:** the build, the checks, the release process, the agentic workflow. None of it names an application.
*   **Application:** phylogenetic substitution models, likelihoods, tree search, and the standards this science requires.

An infrastructure rule that acquires an application reference has lost the separation. Structure enforces it: `README.md` and `DEV.md` put infrastructure before application, and the file layout keeps them apart (`infra/` against `python/phylo/*` and `docs/tex/`).

## Repository Map
This file is authoritative. Each of the remainder has a defined task:

| Document | Job |
| --- | --- |
| `README.md` | What the project is, and where everything else lives |
| `INSTALL.md` | Installing, building, running the tests locally |
| `DEV.md` | Layout, the CI jobs, repository settings, the CI budget, how a change is reviewed |
| `ROADMAP.md` | The development loop, the scientific goal, requirements, and milestones |
| `STATUS.md` | What has landed against each roadmap milestone, the evidence, and the PR carrying it |
| `TICKETS.md` | The titles of the tickets that remain between `STATUS.md` and `ROADMAP.md` |
| `CHANGELOG.md` | What has landed, per dated release section; built from `changelog.d/` fragments by `towncrier` |
| `docs/tex/` | The technical document: background, equations, algorithms |
| `docs/nb/` | One worked notebook per problem class, from a fixture to a learned policy |

`python/phylo/sim/`, `likelihood/`, `opt/`, `learn/`, `search/`, `qa/`, `infra/`, and `docs/` each carry their own `CLAUDE.md`. Those add what applies only inside one module; they never override this file except for the vital **writing style rules, which bind every one of them**. A rule that binds the whole repository belongs here, not in one of them.

**Altitude, and what may repeat.** `ROADMAP.md`, `STATUS.md` and `TICKETS.md` plan and track — what the project is doing, how far it has got, what remains — at a level a reader holds in their head. `DEV.md` and `INSTALL.md` are followed step by step, so they carry their detail in full rather than as pointers: someone working through one of them should not have to assemble the answer from three. Detail may therefore repeat between them, and where it repeats it must agree — a copy that has drifted is a defect, and this file settles which reading is right.

Every `CLAUDE.md`, this one included, is the exception, on rule 6: it carries the principle and names where the detail lives, never the detail itself. A measurement belongs to the thing that produced it — `STATUS.md` where it is evidence for a milestone, the module that defines the constant where a caller must act on it — and a `CLAUDE.md` that restates it acquires a second copy to keep true. The Writing Style above is the one text referenced rather than copied, because it binds every file at once.

## Environment & Tooling
*   **Python (3.12):** Manage via `uv`. Run `uv sync --locked --all-extras`. Regenerate locks with `uv lock` and commit `uv.lock` in the same PR.
*   **Rust:** Compiler pinned via `rust-toolchain.toml`. Lockfile is `Cargo.lock`. Update with `cargo update` and commit.
*   **Lint/Format (Python):** `ruff check .` and `ruff format --check .`
*   **Type Check (Python):** `mypy --strict`, over the paths in `pyproject.toml`'s `files` (`python/`, `tests/`).
*   **Lint/Format (Rust):** `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`.
*   **Audit:** `pip-audit` (Python) and `cargo audit` (Rust).
*   **Docs:** Build with `sphinx-build -W` in `docs/source/`.

## Conventions
*   **Documentation Sync:** Any change affecting behavior, CI, dev setup, or math models must update, in the same PR, whichever of these it makes untrue: `README.md`, `CLAUDE.md` (including a module's), `DEV.md`, `INSTALL.md`, `ROADMAP.md`, `STATUS.md`, `TICKETS.md`, `docs/tex/`, `docs/nb/`. If the change is user-visible, add a fragment under `changelog.d/` (see `changelog.d/README.md`) rather than editing `CHANGELOG.md` directly — `towncrier` merges fragments into `CHANGELOG.md` at release time, and CI's `towncrier check` enforces one exists.
*   **Single Version Source:** The package version lives exclusively in `Cargo.toml`'s `[package].version`.
*   **Package Surface:** `python/phylo/__init__.py` re-exports nothing beyond the package's own top-level utilities (currently `double`); import submodule contents explicitly (`from phylo.likelihood import ...`), not through the top-level namespace.
*   **Code Standards:** Use type hints on all Python functions. Do not introduce silent behavior changes (e.g., default parameters). Keep dependencies minimal and justify additions.
*   **Dev Standards:** The number of PRs should be minimized to limit the amount of review work and test runs, particularly given tickets are typically scoped to a work item. 

## Performance
*   **GPU (PyTorch, Triton, JAX):** Target if the hot path is data-parallel and earns $\ge 10\times$ speedup over vectorized NumPy at realistic problem sizes.
*   **Rust Backend (`oxiphylo`):** Target for CPU-bound hot paths (control flow, tree traversal, irregular memory access, small sizes).
*   **Autodiff:** **PyTorch**, decided. Its MPS backend is the path on Apple Silicon, which `ROADMAP.md` targets alongside CUDA.
*   **Measurement:** Benchmark candidates against the NumPy reference before committing to a port. Report both numbers in the PR.
*   **The Oracle:** Every accelerated kernel keeps its pure Python/NumPy implementation as an oracle. Regression tests must pin the accelerated output against it within an explicit tolerance.

## Testing & Quality Assurance
*   **Simulate Component-Wise:** Build fixtures by simulating from a known generative model under an explicitly seeded generator. Test components individually and in combination.
*   **Pin to Independent Sources:** Validate expected values against analytic results, brute-force computations, or secondary implementations with stated tolerances.
*   **Check Math Invariants:** Rows of a transition matrix sum to 1, a reversible model satisfies detailed balance, gradients match finite differences, and a fit's likelihood increases monotonically.
*   **Cross-Device Agreement Is a Tolerance:** `float32` and `float64` behave differently across CPU, CUDA, and Metal, and deep recursions accumulate that. Agreement is checked against the tolerance stated in `likelihood/CLAUDE.md`, with the measurements it is derived from, and implemented in `phylo.likelihood.device`, never bitwise. A discrepancy inside it is not a bug and must not be "fixed". Two rules that fall out of it: the tolerance is **relative**, because the log-likelihood is a sum over sites and an absolute bound fixed at one problem size does not transfer to another; and it is keyed on the **lowest precision** in the comparison, because Metal cannot do `float64` and one bound loose enough for `float32` would let a broken `float64` backend pass.
*   **No Coverage Theatre:** Tests asserting only output shapes or successful execution without exceptions are forbidden. Leave gaps unwritten and track them as GitHub issues rather than writing meaningless tests.
*   **Every Test Says What It Is Checked Against:** a test carries a marker naming the kind of thing that referees it, and a test that fits no kind is either a missing category or a test with nothing to assert. This is what makes the rule above checkable rather than a matter of a reviewer noticing. The kind is a *tag* and not a partition: one test may be refereed two ways, and splitting it to fit a taxonomy would make it worse. Whether a test gates early is a separate axis from what it checks, because a test is critical *and* refereed somehow, never instead. `pyproject.toml` registers the names, `DEV.md` says which selection runs when, and a guard in `tests/regression/` enforces that every test carries one.
*   **Worked Notebooks:** `docs/nb/` carries one notebook per problem class, each running the application end to end against the oracles the regression suite already establishes, and each ending with a **Further Work** section that names, with its issue number, what it could not demonstrate because the feature is not built. A notebook states no result the suite does not also pin. A change that alters a number a notebook prints re-runs it in the same pull request, as it would regenerate a figure. A CI job re-executes every notebook and fails a pull request whose output disagrees with the committed one, so a notebook is held to the standard a `docs/tex/` figure is; `docs/nb/README.md` states what that comparison covers, and what it deliberately does not.
*   **Scientific Outputs:** The suite must emit plots and tables for the LaTeX technical document. Update the LaTeX captions concurrently. Every figure is rendered from the code it reports on, ships with a caption naming the seed, sizes and model that produced it, and is committed under `docs/tex/figures/` so a changed plot is visible in review rather than only after a document build.
*   **Time is money:** test and build frameworks should be justified, time/computationally, e.g. cahced; a high priority is to standup a minimal implementation against the ROADMAP.md with corresponding ablation studies with a fast test-driven development cycle.  Rely on the tests run on a PR as final validation where appropriate (late in development), rather than duplicating the effort - you will monitor the PR and fix issues before merging.

## Technical Document & Reference Sources
`docs/tex/` is treated as code. Cite these texts where they carry the material, and state any deviation from their standard algorithms explicitly. The core references are a routing table, grouped by what they inform:

**Infrastructure (Build, Structure, and Speed)**
*   **Software Craft:** Martin (*Clean Code*); Blandy et al. (*Programming Rust*)
*   **Systems & Hardware:** Bryant & O'Hallaron (*Computer Systems*); Hwu et al. (*Programming Massively Parallel Processors*)

**Optimization (Discrete and Continuous)**
*   **Algorithms & Math:** Cormen et al. (*Introduction to Algorithms*); Rosen (*Discrete Mathematics and Its Applications*)
*   **Numerical Optimization:** Nocedal & Wright (*Numerical Optimization*)
*   **Probabilistic Inference:** MacKay (*Information Theory...*); Koller & Friedman (*Probabilistic Graphical Models*); Frey (*Graphical Models...*); Ortega (*Introduction to Graph Signal Processing*)
*   **Statistical Physics:** Mézard & Montanari (*Information, Physics, and Computation*); Newman & Barkema (*Monte Carlo Methods in Statistical Physics*)
*   **Information Geometry:** Amari (*Information Geometry and Its Applications*)
*   **Learning & RL:** Goodfellow et al. / Prince (*Deep Learning*); Sutton & Barto (*Reinforcement Learning*); Lapan (*Deep RL Hands-On*); Raschka (*Build a Large Language Model*)

**Application (The Science)**
*   **Phylogenetics:** Felsenstein (*Inferring Phylogenies*); Durbin et al. (*Biological Sequence Analysis*); Compeau & Pevzner (*Bioinformatics Algorithms*); Pachter & Sturmfels (*Algebraic Statistics for Computational Biology*)
*   **Information/Quantum:** Blahut (*Algebraic Codes for Data Transmission*); Nielsen & Chuang (*Quantum Computation and Quantum Information* — background only)

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
