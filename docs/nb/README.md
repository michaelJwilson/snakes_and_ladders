# docs/nb/

One notebook per problem class, each a single pass from a seeded fixture to a
learned search policy (the coupled model's ends at fitted labels, its policy
being #290's later work; the code's at a decoded word, its optimization
framing being #340's), checking every claim against an oracle that shares no
code with what it checks.

| Notebook | Problem | Oracles it rests on |
| --- | --- | --- |
| [`potts_chain.ipynb`](potts_chain.ipynb) | Potts chain in an external field, and the Potts lattice as a Markov random field | Exhaustive enumeration of the partition function on the chain and on the 3x3 lattice (19,683 configurations); enumeration of all 81 configurations; the closed-form transition `J_c = ln(1 + sqrt(3))` |
| [`phylo_tree.ipynb`](phylo_tree.ipynb) | Phylogenetic trees, 6 taxa | Brute-force marginalization over ancestral states; all 105 unrooted topologies enumerated |
| [`hmm.ipynb`](hmm.ipynb) | Discrete hidden Markov model | Enumeration over all `3**8` hidden paths; the retained hidden path; Baum-Welch as an independent algorithm |
| [`ldpc.ipynb`](ldpc.ipynb) | Low-density parity-check code, Gallager's (3,6) ensemble | The general sum-product on the parity-check factor graph; enumeration of all 32,768 codewords of a cycle-free code; the density-evolution threshold 0.4294 |
| [`turbo.ipynb`](turbo.ipynb) | Turbo code, two memory-2 (7,5) recursive systematic encoders through a seeded interleaver | Enumeration of all 4,096 messages of the `K = 12` instance; the tree schedule of the general sum-product on the trellis factor graph; a parity-check matrix built by GF(2) nullspace; the uncoded closed form `Q(sqrt(2 E_b / N_0))` |
| [`spatio_sequential.ipynb`](spatio_sequential.ipynb) | Coupled spatio-sequential model: a Potts prior over class labels gating one hidden chain per class | Enumeration over all 65,536 joint states of the CI fixture; the per-class forward recursion as a second route to the evidence; planted labels on a 10x10 lattice |

Each ends with a **Further Work** section naming what it could not demonstrate
and the issue that carries it. Those sections are the point as much as the
results are: a notebook that quietly skipped the unbuilt half would misreport
the state of the repository. `infra/check_notebooks.py` checks the shape —
the last cell is that section, and every line in it names an issue or a
`TICKETS.md` section — because re-execution compares outputs and a markdown
cell has none; all three notebooks said "no job re-runs it" for months after
one did (issue #278).

## Running them

The notebooks import `snakes_and_ladders` and take their instance from the
fixture registry --- `snakes_and_ladders.sim.fixtures.fixture(problem, tier)`,
one file per problem and tier under `tests/regression/fixtures/` --- which
resolves the repository root from wherever they are opened. A notebook builds
no instance of its own, and a guard refuses one that does (`PROBLEMS.md`,
issue #382): a notebook demonstrating a different instance from the suite's
would be demonstrating a different problem. Install the package first
(`INSTALL.md`), then open them with any Jupyter front end.

Continuous integration re-executes every notebook here and fails a pull
request whose re-executed output disagrees with the committed one
(`infra/check_notebooks.py`, issue #203), so these numbers are held to the
standard the figures in `docs/tex/` are — CI regenerates those and
byte-compares the rebuilt PDF.

**Every notebook here is re-executed, on every run.** Which ones a run checks
is its arguments and nothing else — the notebooks named, or all six when none
is — and the run states the set before executing any of it. A digest decided
it until issue #480: `<name>.inputs` beside each notebook recorded a hash of
its code cells, the modules they import and the fixtures they name (#372),
and the checker skipped a notebook whose stamp still matched. `turbo.ipynb`
was skipped under that rule for as long as no diff reached its import
closure, so the disagreement it had been carrying (#507) surfaced only when
an unrelated merge moved the hash and the check ran. Whether an input changed
and whether a correctness check runs are separate questions, and the second
is not the first's to answer; a set too expensive to run whole would be cut
by a budget `DEV.md` states, never by a hash. The stamps are still written by
`--write` and read by nothing, until issue #490 removes them.

**Text is compared; images are not.** Every number a notebook prints is
determined by its seeds, so a re-executed stream output must match exactly.
Rendered figures embed metadata that is not stable across matplotlib builds,
and comparing them would reproduce the `SOURCE_DATE_EPOCH` problem
`docs/CLAUDE.md` records for `docs/tex/` — for a weaker payoff, since the
printed numbers are what the notebooks assert with. What is checked for a
figure is that the cell still produced one.

This makes a notebook's printed numbers subject to the rule `docs/CLAUDE.md`
states for a generated caption: **only quantities continuous in their inputs**.
A near-zero residual is not one. The check's first run rejected two notebooks
that printed a converged optimizer's gradient norm, which moved by two orders
of magnitude between machines while every parameter it reported agreed to four
decimals. They print the tolerance it cleared instead. Three more did the same
thing where nothing had been running to catch it: `turbo.ipynb` printed two
BCJR agreements at 1e-13 and 1e-14, one of which read 1.07e-14 on one machine
and 1.15e-14 on another (#507); `phylo_tree.ipynb` a 1.33e-16 relative
deviation, a residual gradient norm, and two log-likelihoods to a twelfth
decimal that is the order 200 site terms were summed in; `potts_chain.ipynb` a
1.27e-16 deviation, below float64's epsilon. Each now prints the verdict
against the tolerance the regression suite pins that same comparison at, so
the notebook states the claim the suite pins rather than the sample that
carried it (#480).

Install the kernel with `uv sync --extra notebooks`; a normal `pip install .`
does not need it.

## Keeping them true

A change to `snakes_and_ladders.sim`, `snakes_and_ladders.opt`, `snakes_and_ladders.likelihood`, `snakes_and_ladders.search` or
`snakes_and_ladders.learn` that alters a number these notebooks print must re-run them in
the same pull request, exactly as it must regenerate a `docs/tex/` figure —
and CI now enforces that rather than trusting it.

Regenerate with the same tool that checks them:

```
uv run python infra/check_notebooks.py --write
```

Checking and regenerating live in one tool because they must execute a
notebook identically; a regenerator that differed in working directory,
timeout or kernel would write a notebook the checker then rejects. Running it
when nothing has moved rewrites nothing — the wall-clock timestamps nbclient
records per cell are stripped, so a regeneration diff shows the change and not
the time of day.

Nothing here may state a result the regression suite does not also pin.
