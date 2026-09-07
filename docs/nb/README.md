# docs/nb/

One notebook per problem class, each a single pass from a seeded fixture to a
learned search policy, checking every claim against an oracle that shares no
code with what it checks.

| Notebook | Problem | Oracles it rests on |
| --- | --- | --- |
| [`potts_chain.ipynb`](potts_chain.ipynb) | Potts chain in an external field | Exhaustive enumeration of the partition function; enumeration of all 81 configurations |
| [`phylo_tree.ipynb`](phylo_tree.ipynb) | Phylogenetic trees, 6 taxa | Brute-force marginalization over ancestral states; all 105 unrooted topologies enumerated |
| [`hmm.ipynb`](hmm.ipynb) | Discrete hidden Markov model | Enumeration over all `3**8` hidden paths; the retained hidden path; Baum-Welch as an independent algorithm |

Each ends with a **Further Work** section naming what it could not demonstrate
and the issue that carries it. Those sections are the point as much as the
results are: a notebook that quietly skipped the unbuilt half would misreport
the state of the repository.

## Running them

The notebooks import `snakes_and_ladders` and read fixtures from
`tests/regression/fixtures/`, resolving the repository root from wherever they
are opened. Install the package first (`INSTALL.md`), then open them with any
Jupyter front end.

Continuous integration re-executes every notebook here and fails a pull
request whose re-executed output disagrees with the committed one
(`infra/check_notebooks.py`, issue #203), so these numbers are held to the
standard the figures in `docs/tex/` are — CI regenerates those and
byte-compares the rebuilt PDF.

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
decimals. They print the tolerance it cleared instead.

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
