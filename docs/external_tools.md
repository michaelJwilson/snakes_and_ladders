# External phylogenetic software

Root `CLAUDE.md` states the stance: this repository works on simulated fixtures
with known truth, and adopts no external solver today. This file is the survey
that rule points at, and it is a list and a recommendation and nothing else.
Nothing here is installed; nothing here appears in `pyproject.toml`, `uv.lock`
or `Cargo.toml`. Adopting one of these is a separate decision, taken on its own
evidence, under `DEV.md`'s Dependency Management steps.

Three roles a candidate could fill (issue #408):

- **Fixture definition** — writing or reading the alignment, tree and model
  that define a test case.
- **Validation** — an independent implementation of a quantity this repository
  already computes, so the two can be compared.
- **Benchmarking** — a search this repository's search is scored against under
  an equal budget of objective evaluations.

## Candidates

Every licence below was read on 2026-09-08 from the project's own `LICENSE`
file in its repository, not from memory; the star counts were read from the
same repositories on that date. `CLAUDE.md` flags any candidate under 1,000 stars for explicit
review, so the column records the count rather than a verdict.

| Tool | What it is | Licence | OSI-approved | Stars | Role it could fill |
| --- | --- | --- | --- | --- | --- |
| [IQ-TREE 2](https://github.com/iqtree/iqtree2) | Maximum-likelihood tree inference: model selection, NNI/SPR search, ultrafast bootstrap (Minh et al. 2020) | GPL-2.0 | Yes | 336 | Benchmarking; validation of a log-likelihood on a fixed tree and model |
| [RAxML-NG](https://github.com/amkozlov/raxml-ng) | Maximum-likelihood tree inference, the successor to RAxML (Kozlov et al. 2019) | AGPL-3.0 | Yes | 477 | Benchmarking; a second independent likelihood |
| [Biopython](https://github.com/biopython/biopython) | General bioinformatics library; `Bio.Phylo` reads and writes Newick, Nexus and PhyloXML | Biopython License Agreement, some files dual-licensed BSD-3-Clause | BSD-3-Clause yes; the Biopython License Agreement is a variant of the OSI-approved HPND and is not itself on the OSI list | 5,200 | Fixture definition (tree and alignment I/O) |
| [DendroPy](https://github.com/jeetsukumaran/DendroPy) | Phylogenetic computing library: tree and character-matrix objects, simulation, tree distances | BSD-3-Clause | Yes | 235 | Fixture definition; validation of tree distances (Robinson–Foulds) |
| [ETE](https://github.com/etetoolkit/ete) | Tree manipulation, annotation and visualization toolkit | GPL-3.0 | Yes | 884 | Fixture definition; figure rendering |
| [scikit-bio](https://github.com/scikit-bio/scikit-bio) | Bioinformatics data structures and algorithms on NumPy, including `TreeNode` and distance matrices | BSD-3-Clause | Yes | 1,200 | Fixture definition; validation of tree distances |
| [TreeSwift](https://github.com/niemasd/TreeSwift) | Pure-Python tree library optimized for traversal and I/O at large leaf counts | GPL-3.0 | Yes | 88 | Fixture definition at sizes where object-per-node libraries dominate the runtime |

The four OSI licences here (GPL-2.0, GPL-3.0, AGPL-3.0, BSD-3-Clause) are on
the [OSI list](https://opensource.org/licenses). The distinction that matters
if one is ever adopted is not the approval but the reach: a copyleft library
imported into this package carries its terms into this package, while a
copyleft binary invoked as a subprocess does not. The two inference tools are
binaries; the five Python libraries are imports.

## Recommendation

**None is worth adopting today, and the survey's point is which would be first
if the stance changed.** The three roles are not equally open. Fixture
definition is filled: `snakes_and_ladders.sim.fixtures` generates every fixture
together with the parameters that generated it, which no external reader can
supply, and the tree I/O these libraries offer replaces code this repository
already has. Validation is partly filled: an oracle referees every accelerated
path, and what it cannot referee is the NumPy implementation itself, since it
is compared only against its own derivatives. Benchmarking is the role our own
referees genuinely cannot fill — enumeration stops at eight taxa, and simulated
truth gives a Robinson–Foulds distance but no competitive likelihood to beat at
50 to 200 taxa, which is what `ROADMAP.md` §2.3 and the external-baseline
candidate in `docs/blue_sky.md` ask for.

So the first adoption, if there is one, is **IQ-TREE 2, for benchmarking**, and
for the fixed-tree likelihood that would cross-check the NumPy oracle from
outside. It is GPL-2.0 and invoked as a subprocess, so its terms do not reach
this package; RAxML-NG is the same shape under AGPL-3.0 and is the second
opinion rather than the first. Its 336 stars are below the 1,000 the rule sets
and would need the explicit review the rule requires — for a compiled research
tool the star count understates use, and the metric to argue from is the
citation record of Minh et al. 2020, not the repository page.

What would trigger it: a measurement this repository needs, cannot referee with
an oracle or with simulation truth, and has a use for — a budget-matched
comparison at 50 to 200 taxa is the standing example. Absent that, the survey
stands and nothing is installed.
