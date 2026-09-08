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

**Expected Reader:** a well-educated developer with scientific and performance-computing background, but not an application expert, e.g. phylogenetics. Keep the documents streamlined — hyperlinks and citations over inline derivation — and push required application background (e.g. NNI, other standard algorithms) into a dedicated appendix, cited from the point of use rather than re-derived there. Treat the main text as a high-level overview of the current best-known approach (simulation, models, results) in terms of the roadmap, not an exhaustive record; link out to supporting docs, with plots, and results for dedicated studies that informed them. Adopt the style of an academic paper, supported by a textbook on domain-specific material likely new to the developer — two documents rather than one file with appendices (issue #249), so the paper can report results without carrying the formulations that support them, and the textbook can state an algorithm without naming any code that implements it.

## Project
`snakes_and_ladders` is a high-performance scientific repository. Correctness and reproducibility of numerical and scientific results take priority over convenience.

Two concerns are supported, and they must stay separable:

*   **Infrastructure:** the build, the checks, the release process, the agentic workflow. None of it names an application.
*   **Application:** phylogenetic substitution models, likelihoods, tree search, and the standards this science requires.

An infrastructure rule that acquires an application reference has lost the separation. Structure enforces it: `README.md` and `DEV.md` put infrastructure before application, and the file layout keeps them apart (`infra/` against `python/snakes_and_ladders/*` and `docs/tex/`).

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
| `PROBLEMS.md` | The supported problems, each with the code that simulates, evaluates, fits, searches and learns on it |
| `CHECKS.md` | The checks the roadmap's claims rest on, generated from the tests' own markers |
| `CHANGELOG.md` | What has landed, per dated release section; built from `changelog.d/` fragments by `towncrier` |
| `docs/tex/` | Two documents: a paper reporting results, and a textbook of the problem statements, algorithms, and the properties that referee them |
| `docs/nb/` | One worked notebook per problem class, from a fixture to a learned policy |

`python/snakes_and_ladders/sim/`, `likelihood/`, `opt/`, `learn/`, `search/`, `qa/`, `sandbox/`, `infra/`, and `docs/` each carry their own `CLAUDE.md`. Those add what applies only inside one module; they never override this file except for the vital **writing style rules, which bind every one of them**. A rule that binds the whole repository belongs here, not in one of them.

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
*   **Package Surface:** `python/snakes_and_ladders/__init__.py` re-exports nothing beyond the package's own top-level utilities (currently `double`); import submodule contents explicitly (`from snakes_and_ladders.likelihood import ...`), not through the top-level namespace.
*   **Code Standards:** Use type hints on all Python functions. Do not introduce silent behavior changes (e.g., default parameters). Keep dependencies minimal and justify additions.
*   **Dev Standards:** The number of PRs should be minimized to limit the amount of review work and test runs, particularly given tickets are typically scoped to a work item. 
*   **Model Routing:** Claude Fable carries the judgement: it plans tickets, implements the plans, writes the pull requests, reviews work before it is posted, and takes review tickets. Everything else is delegated to Claude Opus subagents: the mechanical steps such as merges and rebuilds, measurements, searches, and re-executions, each briefed with the plan and validated by Fable before it lands.
*   **A release ticket reports as it goes:** the pull request for a Release-template ticket opens as a draft as soon as its branch exists, and its body is updated every 10 minutes with the work still to be completed and the estimated time to the finished pull request, so the maintainer reads the state of the release from the pull request rather than asking for it. The draft leaves that state when the plan's steps are all landed and validated.

## Performance
*   **GPU (PyTorch, Triton, JAX):** Target if the hot path is data-parallel and earns $\ge 10\times$ speedup over vectorized NumPy at realistic problem sizes.
*   **Rust Backend (`oxi_snakes_and_ladders`):** Target for CPU-bound hot paths (control flow, tree traversal, irregular memory access, small sizes).
*   **Autodiff:** **PyTorch**, decided. Its MPS backend is the path on Apple Silicon, which `ROADMAP.md` targets alongside CUDA.
*   **Measurement:** Benchmark candidates against the NumPy reference before committing to a port. Report both numbers in the PR.
*   **The Oracle:** Every accelerated kernel keeps its pure Python/NumPy implementation as an oracle. Regression tests must pin the accelerated output against it within an explicit tolerance.

### Runtime Optimization Opportunities
Checked in this order when a hot path is proposed; `DEV.md` carries the procedure, `STATUS.md` the numbers.
*   **Profile first.** `cProfile` self time ranks the loop to port; a term that is a small fraction of the runtime pays for no port in any language (Gorelick & Ozsvald ch. 2).
*   **L1/L2/L3 cache.** Keep a recursion's live set — partials, messages, the current sweep — inside cache; release intermediates and report peak memory beside time (Bryant & O'Hallaron ch. 6).
*   **Memory layout.** Contiguous, row-major arrays walked in stride order; neighbour lists as offsets into one array, not lists of lists (Bryant & O'Hallaron ch. 6; Gorelick & Ozsvald ch. 6).
*   **Vectorization and SIMD.** Inner loops contiguous, unaliased, without early exit or data-dependent reduction, so NumPy and the Rust compiler vectorize; confirm by benchmark, never by asserting a width (Bryant & O'Hallaron ch. 5).
*   **Branch misprediction.** A data-dependent branch in an inner loop is free or a stall; the branchless form (mask, select, table) wins only where a measurement shows the branch does not predict (Bryant & O'Hallaron ch. 5).
*   **Inlining and call overhead.** No Python-level call per site or per node; hoist it or vectorize it. In Rust, `#[inline]` the small hot helpers (Gorelick & Ozsvald ch. 4; Bryant & O'Hallaron ch. 5).
*   **Allocation.** Preallocate and reuse buffers across sweeps; NumPy `out=` and in-place operators over temporaries (Gorelick & Ozsvald ch. 6).
*   **Double buffering.** Reading and writing one array in a sweep is a *different Markov chain* from reading the previous buffer; the docstring says which and the oracle pins it before either is timed.
*   **The FFI boundary.** Cross it once per call with contiguous arrays; time a kernel alone *and* through its binding, since the marshalling has been the dominant term here (Gorelick & Ozsvald ch. 7; Antão).
*   **Parallel over independent tasks.** A loop of independent bodies — starts, seeds, replicates — runs through `snakes_and_ladders.parallel`, one seam and no pool of its own; `workers` is explicit and `1` is serial. A parallel run is bitwise the serial run or it is not a parallel run: one generator per task, spawned in item order, and one intra-op thread per worker, since a pool of multithreaded kernels oversubscribes the machine. A site under 2× at 4 workers stays serial and `STATUS.md` records why.
*   **Compiled backends.** Two, each for a reason: Rust carries the sampling sweep, whose agreement with its oracle is distributional, so it stays opt-in; `numba`'s `njit` carries deterministic kernels whose pin against the oracle is bitwise, so it may be the default. Every backend is one more implementation held to the NumPy oracle, and a third joins only against a measurement on an existing hot path (Gorelick & Ozsvald ch. 7).

## Testing & Quality Assurance
*   **Simulate Component-Wise:** Build fixtures by simulating from a known generative model under an explicitly seeded generator. Test components individually and in combination.
*   **Pin to Independent Sources:** Validate expected values against analytic results, brute-force computations, or secondary implementations with stated tolerances. Where no oracle is affordable at a size, recovering the simulated truth — the generating parameters or structure of a seeded fixture, to a stated tolerance or coverage — is sufficient, and the oracle remains desired and is ticketed.
*   **Check Math Invariants:** Rows of a transition matrix sum to 1, a reversible model satisfies detailed balance, gradients match finite differences, and a fit's likelihood increases monotonically.
*   **Cross-Device Agreement Is a Tolerance:** `float32` and `float64` behave differently across CPU, CUDA, and Metal, and deep recursions accumulate that. Agreement is checked against the tolerance stated in `likelihood/CLAUDE.md`, with the measurements it is derived from, and implemented in `snakes_and_ladders.likelihood.device`, never bitwise. A discrepancy inside it is not a bug and must not be "fixed". Two rules that fall out of it: the tolerance is **relative**, because the log-likelihood is a sum over sites and an absolute bound fixed at one problem size does not transfer to another; and it is keyed on the **lowest precision** in the comparison, because Metal cannot do `float64` and one bound loose enough for `float32` would let a broken `float64` backend pass.
*   **No Coverage Theatre:** Tests asserting only output shapes or successful execution without exceptions are forbidden. Leave gaps unwritten and track them as GitHub issues rather than writing meaningless tests.
*   **Every Test Says What It Is Checked Against:** a test carries a marker naming the kind of thing that referees it, and a test that fits no kind is either a missing category or a test with nothing to assert. This is what makes the rule above checkable rather than a matter of a reviewer noticing. The kind is a *tag* and not a partition: one test may be refereed two ways, and splitting it to fit a taxonomy would make it worse. Whether a test gates early is a separate axis from what it checks, because a test is critical *and* refereed somehow, never instead. `pyproject.toml` registers the names, `DEV.md` says which selection runs when, and a guard in `tests/regression/` enforces that every test carries one.
*   **Worked Notebooks:** `docs/nb/` carries one notebook per problem class, each running the application end to end against the oracles the regression suite already establishes, and each ending with a **Further Work** section that names, with its issue number, what it could not demonstrate because the feature is not built. A notebook states no result the suite does not also pin. A change that alters a number a notebook prints re-runs it in the same pull request, as it would regenerate a figure. A CI job re-executes every notebook and fails a pull request whose output disagrees with the committed one, so a notebook is held to the standard a `docs/tex/` figure is; `docs/nb/README.md` states what that comparison covers, and what it deliberately does not.
*   **Scientific Outputs:** The suite must emit plots and tables for the LaTeX documents. Update the LaTeX captions concurrently. Every figure is rendered from the code it reports on, ships with a caption naming the seed, sizes and model that produced it, and is committed under `docs/tex/figures/` so a changed plot is visible in review rather than only after a document build.
*   **Time is money:** test and build frameworks should be justified, time/computationally, e.g. cahced; a high priority is to standup a minimal implementation against the ROADMAP.md with corresponding ablation studies with a fast test-driven development cycle.  Rely on the tests run on a PR as final validation where appropriate (late in development), rather than duplicating the effort - you will monitor the PR and fix issues before merging.

## Documents & Reference Sources
`docs/tex/` is treated as code. Cite these texts where they carry the material, and state any deviation from their standard algorithms explicitly. The core references are a routing table, grouped by what they inform:

**Infrastructure (Build, Structure, and Speed)**
*   **Software Craft:** Martin (*Clean Code*); Blandy et al. (*Programming Rust*); Ramalho (*Fluent Python*)
*   **Systems & Hardware:** Bryant & O'Hallaron (*Computer Systems*); Hwu et al. (*Programming Massively Parallel Processors*)
*   **Python Performance:** Gorelick & Ozsvald (*High Performance Python*); Antão (*Fast Python*)

**Optimization (Discrete and Continuous)**
*   **Algorithms & Math:** Cormen et al. (*Introduction to Algorithms*); Rosen (*Discrete Mathematics and Its Applications*); Papadimitriou & Steiglitz (*Combinatorial Optimization*, the exact baselines and their complexity). Papers: Kolmogorov & Zabih 2004 (which energies a cut minimizes, the minimum-cut and alpha-expansion solvers); Boykov & Kolmogorov 2004 (the max-flow those solvers run)
*   **Numerical Optimization:** Nocedal & Wright (*Numerical Optimization*); Boyd & Vandenberghe (*Convex Optimization*, the Max-Cut SDP certificate). Papers: Hansen & Ostermeier 2001 (CMA-ES, the population baseline)
*   **Probabilistic Inference:** MacKay (*Information Theory...*); Koller & Friedman (*Probabilistic Graphical Models*); Frey (*Graphical Models...*); Ortega (*Introduction to Graph Signal Processing*); Bishop (*Pattern Recognition and Machine Learning*, EM and variational inference); Murphy (*Probabilistic Machine Learning: Advanced Topics*). Review: Wainwright & Jordan 2008 (the exponential-family frame the bounds sit in). Papers: Wainwright, Jaakkola & Willsky 2005 (tree-reweighted bounds on `log Z` and on the MAP); Hsu, Kakade & Zhang 2012 (spectral HMM initialization)
*   **Statistical Physics:** Mézard & Montanari (*Information, Physics, and Computation*); Newman & Barkema (*Monte Carlo Methods in Statistical Physics*); Krauth (*Statistical Mechanics: Algorithms and Computations*); Landau & Binder (*A Guide to Monte Carlo Simulations in Statistical Physics*). Reviews: Betancourt 2017 (Hamiltonian Monte Carlo); Schollwöck 2011 (matrix-product states, the contraction oracle). Papers: Machta 2010 (population annealing)
*   **Information Theory & Geometry:** Amari (*Information Geometry and Its Applications*); Cover & Thomas (*Elements of Information Theory*)
*   **Learning & RL:** Goodfellow et al. / Prince (*Deep Learning*); Sutton & Barto (*Reinforcement Learning*); Lapan (*Deep RL Hands-On*); Raschka (*Build a Large Language Model*). Papers: Sutton, Precup & Singh 1999 (options, the compound-moves item); Bacon, Harb & Precup 2017 (option-critic); Schrittwieser et al. 2020 (planning with a learned model)
*   **Learning for Combinatorial Optimization:** Reviews: Bengio, Lodi & Prouvost 2021 (the field Stage 2 sits in); Mazyavkina et al. 2021 (reinforcement learning for combinatorial optimization). Papers: Khalil et al. 2017 (a graph-network policy); Kool, van Hoof & Welling 2019 (an attention policy); Paulus et al. 2020 (stochastic softmax tricks over structured spaces); Henderson et al. 2018 (seeds and reporting); Agarwal et al. 2021 (the interval statistics §2.4's paired test needs); Schuetz, Brubaker & Katzgraber 2022 (physics-inspired GNN ground states); Angelini & Ricci-Tersenghi 2023 (its greedy critique)

**Application (The Science)**
*   **Phylogenetics:** Felsenstein (*Inferring Phylogenies*); Durbin et al. (*Biological Sequence Analysis*); Compeau & Pevzner (*Bioinformatics Algorithms*); Pachter & Sturmfels (*Algebraic Statistics for Computational Biology*); Yang (*Molecular Evolution: A Statistical Approach*, the substitution models). Papers: Whidden & Matsen 2015 (mixing over the SPR graph); Azouri et al. 2021 (learned SPR ranking); Zhang & Matsen 2018 (subsplit networks); Zhang & Matsen 2019 (variational inference over topologies, the differentiable-topology oracle); Speyer & Sturmfels 2004 (the tropical Grassmannian); Altekar et al. 2004 (Metropolis-coupled MCMC over topologies); Minh et al. 2020 (IQ-TREE 2, external baseline); Kozlov et al. 2019 (RAxML-NG, external baseline); Saitou & Nei 1987 (neighbor joining); Atteson 1999 (its radius); Steel 1994 (the log-det distance); Hendy & Penny 1993 (the Hadamard conjugation); Mossel & Roch 2006 (spectral learning of a nonsingular phylogeny, the bridge from the HMM)
*   **Information/Quantum:** Blahut (*Algebraic Codes for Data Transmission*); Richardson & Urbanke (*Modern Coding Theory*, the LDPC decoder and density evolution); Nielsen & Chuang (*Quantum Computation and Quantum Information* — background only). Papers: Gallager 1962 (low-density parity-check codes); Nachmani et al. 2018 (neural belief propagation, the learned decoder)

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
