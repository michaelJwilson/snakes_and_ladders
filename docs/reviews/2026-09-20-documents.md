# The documents against each other and the tree, 2026-09-20

**TL;DR:** ten findings, read by hand across `README.md`, `DEV.md`, `INSTALL.md`,
`STATUS.md`, `ROADMAP.md`, `PROBLEMS.md`, `REFERENCES.md`, root and module
`CLAUDE.md` and `docs/`. Seven are fixed here; three are ticketed (#847, #844,
#839). The largest: `docs/textbook.pdf`, which three documents say is committed,
has not been in the tree since f16a26ab's rebuild deleted it.

| # | Where | Finding | Disposition |
| --- | --- | --- | --- |
| 1 | `README.md`, `DEV.md` x2, `docs/CLAUDE.md` | Four PDFs committed; the tree carries three, the textbook's deleted by f16a26ab (462,655 to 0 bytes) | #847; the three documents now say so |
| 2 | `DEV.md` layout | `opt/` listed the samplers, schedules and initializers #777 and #830 moved; `search/` listed the samplers and `rl.py`, moved by #777 and #779; no `sample/` row | fixed |
| 3 | `DEV.md`, `README.md` | `learn/` "imports nothing from `sim/`, `likelihood/` or `search/`" and "may import no application module"; `learn/CLAUDE.md` names four exceptions and the guard admits them | fixed: the exceptions are named |
| 4 | `DEV.md` CI table | `documents` "on the push to `main`, not on a pull request"; the job has typeset on a pull request touching an input since #625 | fixed |
| 5 | `README.md` contract table | Ten module contracts listed; `sample/CLAUDE.md` absent | fixed |
| 6 | `STATUS.md` summary | Row 4.1 "the Aim run store not started (#75)" beside its own paragraph "Aim is the store #75 asked for" | fixed |
| 7 | `STATUS.md` 2.1 | The fitted reward's 113.7 ms read as a cold-start cost a warm start would cut; measured at #821: 1.22x with 9 of 60 optima moved, 0.98x with the trap closed | fixed by #840 |
| 8 | `docs/tex/textbook.tex` | One PDF; `PROBLEMS.md` and every notebook point into one section | #844, #846 |
| 9 | `search.infer` | The search's warm start inherits collapsed branches and flags convergence at them | #839 |
| 10 | `README.md` root exports | Seven names today; `Simulator` (#835), `Params` and `load_params` (#838) arrive with their pull requests, which carry the line | to those pull requests |

What agreed: the ten CI jobs `README.md` counts are the ten `ci.yml` declares;
`PROBLEMS.md`'s 18 rows are the 18 `README.md` cites; the six extras
`INSTALL.md` names are `pyproject.toml`'s six; `Cargo.toml` at `0.3.0` is the
version `STATUS.md` and the last `CHANGELOG.md` section state; every module
`CLAUDE.md` is within the 120-line budget and named from the root, which the
guards already assert.

Blue-sky candidates proposed this day are in `docs/blue_sky.md` §3.
