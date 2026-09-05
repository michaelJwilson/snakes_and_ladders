# snakes_and_ladders

[![CI](https://github.com/michaelJwilson/snakes_and_ladders/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelJwilson/snakes_and_ladders/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Mixed discrete-continuous optimization over graph-structured models —
phylogenetic trees, Potts models in an external field, and hidden Markov
models. Autodiff fits the continuous half, a learned policy is intended to
propose the discrete half, and a Rust backend (`snakes_and_ladders.oxi_snakes_and_ladders`, via
[PyO3](https://pyo3.rs)/[maturin](https://www.maturin.rs)) carries the
CPU-bound recursions.

Development is agent-assisted, and the repository is built so that claim is
checkable: every result is pinned to an oracle that shares no code with what it
checks.

Two concerns stay separate. **The infrastructure** — the build, the checks, the
release process, the agentic workflow — names no application. **The
application** is the science.

## Quick start

```
uv sync --locked --all-extras
source .venv/bin/activate
pytest -m "not release"
```

[INSTALL.md](INSTALL.md) covers the full workflow: prerequisites, building the
Rust extension, running both test suites, the checks CI enforces, dependency
audits, and building the docs.

---

# Infrastructure

## Development is agent-assisted

An agent's output is reviewed on the same terms as a human's. The claim is not
that an agent wrote the code; it is that the process establishes whether the
code is right. The summary below names the stages;
[ROADMAP.md](ROADMAP.md) §0 states each one as a deliverable and the gate that
holds it.

1. **A ticket is filed** through [`.github/ISSUE_TEMPLATE/task.yml`](.github/ISSUE_TEMPLATE/task.yml),
   which asks for the outcome, the non-goals, and — the field that does the
   work — *how it will be validated*. Blank issues are disabled: a task that
   cannot say what would falsify it does not get filed.
2. **A plan is posted to the thread** and the issue is labelled `planned`.
   Review happens before any code exists, the cheapest point to reject an
   approach.
3. **A maintainer applies `approved`.** Only then may a pull request open, and
   it must implement the plan already in the thread. A plan that turns out to
   be flawed gets a revised plan posted, not a silent correction.
4. **The pull request answers [a fixed checklist](.github/pull_request_template.md)**:
   the Definition of Done, benchmark numbers, the realized value of any
   tolerance-based test beside the tolerance it was checked against, and which
   documents the change made untrue.
5. **A release is itself a ticket**, gated on `infra/release.sh` passing before
   a version is tagged.

## The contract an agent reads first

`CLAUDE.md` is authoritative: where it and any other document disagree, it
wins. Each module carries its own, adding the rules local to it and never
overriding the root. Read the root file, then the one for the directory you are
working in.

| Contract | Governs |
| --- | --- |
| [`CLAUDE.md`](CLAUDE.md) | The repository-wide rules: environment, conventions, performance, testing, the Definition of Done, the reference routing table |
| [`python/snakes_and_ladders/sim/CLAUDE.md`](python/snakes_and_ladders/sim/CLAUDE.md) | Data generation and ground-truth retention |
| [`python/snakes_and_ladders/likelihood/CLAUDE.md`](python/snakes_and_ladders/likelihood/CLAUDE.md) | Pruning, the backends, and the cross-device tolerance they are held to |
| [`python/snakes_and_ladders/opt/CLAUDE.md`](python/snakes_and_ladders/opt/CLAUDE.md) | Continuous fitting, and why no application may be imported here |
| [`python/snakes_and_ladders/learn/CLAUDE.md`](python/snakes_and_ladders/learn/CLAUDE.md) | The RL interface, its oracles, and the rules a reward and a baseline obey |
| [`python/snakes_and_ladders/search/CLAUDE.md`](python/snakes_and_ladders/search/CLAUDE.md) | Move sets, search budgets, and the phylogenetic environment |
| [`python/snakes_and_ladders/qa/CLAUDE.md`](python/snakes_and_ladders/qa/CLAUDE.md) | QA figures: rendering, never recomputing |
| [`infra/CLAUDE.md`](infra/CLAUDE.md) | CI/CD, the agentic workflow, experiment tracking |
| [`docs/CLAUDE.md`](docs/CLAUDE.md) | How the documents are built and kept true |

[`.github/labels.yml`](.github/labels.yml) defines the labels and a workflow
applies them, so the taxonomy cannot drift from the documentation.

## What enforces the claims

Nine required checks run on every pull request: `ruff` and `mypy --strict`,
`clippy` and `cargo fmt`, the Rust and Python suites, the Sphinx build with
warnings as errors, the technical-document build, the re-execution of every
committed notebook, and dependency audits.
Three further rules constrain what the suite may contain:

- **No coverage theatre.** A test asserting only shapes, or only that nothing
  raised, is forbidden. Gaps are left unwritten and tracked as issues.
- **Every accelerated path keeps its reference implementation.** The
  vectorized NumPy version stays as the oracle the Rust, PyTorch and future
  GPU backends are pinned against. Deleting the slow path removes the only
  thing that says the fast path is right.
- **Correctness comes from an independent source**, not from a second
  backend: analytic results, brute-force computation, or exhaustive
  enumeration.

| Document | Contents |
| --- | --- |
| [INSTALL.md](INSTALL.md) | Installing, building, running the tests, working locally |
| [DEV.md](DEV.md) | Repository layout, test layout, CI jobs, the CI budget, the release procedure |
| [CHANGELOG.md](CHANGELOG.md) | What has landed, per dated release |
| [`docs/source/`](docs/source) | Sphinx API documentation, built from the docstrings |
| [`docs/nb/`](docs/nb) | Worked notebooks, one per problem class, fixture to learned policy |

---

# Application

## One abstraction, three problem classes

The scientific problem is a search over discrete structure where scoring any
one candidate requires a continuous fit. In phylogenetics it is a search over
tree topologies — the large parsimony problem — scoring each candidate by its
likelihood under a model of character substitution. The search is **discrete**
over topologies, but scoring one requires a **continuous** fit of that tree's
branch lengths, rate matrix and root distribution, and neither half separates
from the other: a better topology scored with badly fitted parameters looks
worse than a poor one scored well.

That shape is not unique to phylogenies. Felsenstein pruning, the HMM forward
algorithm, and the Potts transfer matrix are the same sum-product recursion on
different graphs — a tree, a chain, a lattice — so one discrete/continuous
interface serves all three. The project treats that as a design constraint
rather than a coincidence, and enforces it structurally: `snakes_and_ladders.opt` and
`snakes_and_ladders.learn` may import no application module, asserted by test.

[ROADMAP.md](ROADMAP.md) states the goal, the accuracy and hardware
requirements, and the milestones.

## Features

**A model-agnostic optimization interface.** An `Objective` is an unconstrained
parameter vector, a differentiable scalar, and a map back to named constrained
parameters. Four instances run against it unchanged — a Potts chain, a discrete
HMM, branch lengths on a fixed topology, and the GTR substitution model — and
none required a change to `snakes_and_ladders.opt`.

**Fitting with intervals, not just convergence.** L-BFGS under a strong-Wolfe
line search, with confidence intervals from the observed Fisher information
pushed through the constraint map by the delta method, and convergence judged
on the gradient relative to the objective's own magnitude. Validation is
parameter recovery against known truth, not a falling loss curve.

**Three pruning backends against one oracle.** Vectorized NumPy is the
reference; differentiable PyTorch keeps branch lengths in the autograd graph;
Rust carries the CPU-bound recursion. All three are pinned against independent
brute-force marginalization at a worst relative deviation of 4.0e-14.

**Device dispatch with a declared tolerance.** Selection prefers CUDA, then
Metal/MPS, then CPU. Agreement is a *relative* bound keyed on the lowest
precision in the comparison — 1e-11 in `float64`, 1e-6 where either side is
`float32` — derived from measured agreement, never bitwise.

**Discrete move sets with closed-form checks.** NNI and SPR neighbourhoods
behind one interface, verified exhaustively against `2(n-3)` and
`2(n-3)(2n-7)` at `n = 5..8`, plus exhaustive enumeration of unrooted
topologies as the oracle that makes "did the search find the best tree" a
question with an answer.

**Reinforcement learning pinned to a closed form.** An `Environment`
interface, a softmax-over-scored-actions policy, REINFORCE with a baseline, and
an exact trajectory-enumeration oracle for the expected return and its
gradient. Claims rest on that oracle rather than on a training curve.

**A QA pipeline that is the evidence.** Every figure and table in the technical
document is rendered by `snakes_and_ladders.qa` from the code it reports on, and CI rebuilds
and compares them, so a plot cannot drift from what produced it.

## What exists, measured

Simulation under `k`-state Jukes–Cantor and GTR models, truth retained
alongside the data and validated against the closed-form transition
probabilities. Felsenstein pruning agreeing with brute-force marginalization to
4.0e-14 relative across three backends. Parameters recovered inside intervals
whose 95% coverage is measured at the nominal rate over 60 replicates. Hill
climbing reaching the exhaustively enumerated maximum from all 12 starting
points on a 6-taxon fixture. Normalized Robinson–Foulds distance meeting the
0.05 requirement from 125 sites upward. On a Potts landscape, a learned policy
reaching the optimum from 86.6% of starts against greedy's 80.2%, in 8 of 8
seeds. A gradient update costing 203 ms at `n = 100`, `L = 1000`.

Not claimed: that a learned policy beats hill climbing on trees, any comparison
against IQ-TREE 2 or RAxML-NG, GPU dispatch, or rate variation across sites.

[STATUS.md](STATUS.md) records what has landed against each milestone, the
oracle that established it, and the pull request that carries it;
[TICKETS.md](TICKETS.md) records, as titles, what has not.

## Reading the science

| Document | Contents |
| --- | --- |
| [ROADMAP.md](ROADMAP.md) | The development loop, the scientific goal, the requirements, and the milestones |
| [STATUS.md](STATUS.md) | What has landed against each milestone, with its evidence and pull request |
| [TICKETS.md](TICKETS.md) | The titles of the tickets remaining to complete the roadmap |
| [`docs/tex/paper.tex`](docs/tex/paper.tex) | The paper: abstract, the MDP formulation, results, conclusions |
| [`docs/tex/textbook.tex`](docs/tex/textbook.tex) | The textbook: problem statements, algorithms, and the properties each is validated against |
| [`docs/paper.pdf`](docs/paper.pdf), [`docs/textbook.pdf`](docs/textbook.pdf) | The rendered documents, committed and regenerated by `infra/build_technical_doc.sh` |
| [`docs/tex/figures/`](docs/tex/figures) | The QA figures the documents rest on, committed so a changed plot is visible in review |
| [`docs/nb/`](docs/nb) | One worked notebook per problem class: simulate, fit, search, learn, and what is not built yet |

Every figure in the document is rendered from the code it reports on and ships
with a caption naming the seed, the sizes, and the model that produced it. A
figure that cannot say what generated it is not evidence.

## References

The literature this work is built against is a routing table rather than a
bibliography: each group informs one concern, and `CLAUDE.md` holds the full
list with the rule that a deviation from a standard algorithm is stated
explicitly where it is taken. `docs/tex/textbook.tex` cites them at the point of use, and
its Reference Taxonomy appendix groups them the same way.

| Concern | Anchor texts |
| --- | --- |
| Software craft, systems and hardware | Martin, *Clean Code*; Blandy et al., *Programming Rust*; Bryant & O'Hallaron, *Computer Systems*; Hwu et al., *Programming Massively Parallel Processors* |
| Algorithms, discrete mathematics and continuous optimization | Cormen et al., *Introduction to Algorithms*; Rosen, *Discrete Mathematics*; Nocedal & Wright, *Numerical Optimization* |
| Probabilistic inference and graphical models | MacKay, *Information Theory, Inference, and Learning Algorithms*; Koller & Friedman, *Probabilistic Graphical Models*; Frey, *Graphical Models*; Ortega, *Graph Signal Processing* |
| Statistical physics and information geometry | Mézard & Montanari, *Information, Physics, and Computation*; Newman & Barkema, *Monte Carlo Methods in Statistical Physics*; Amari, *Information Geometry* |
| Learning and reinforcement learning | Goodfellow et al. and Prince, *Deep Learning*; Sutton & Barto, *Reinforcement Learning*; Lapan, *Deep RL Hands-On*; Raschka, *Build a Large Language Model* |
| Phylogenetics and sequence analysis | Felsenstein, *Inferring Phylogenies*; Durbin et al., *Biological Sequence Analysis*; Compeau & Pevzner, *Bioinformatics Algorithms*; Pachter & Sturmfels, *Algebraic Statistics for Computational Biology* |

Full entries are in [`docs/tex/references.bib`](docs/tex/references.bib).

## License

MIT — see [LICENSE](LICENSE).
