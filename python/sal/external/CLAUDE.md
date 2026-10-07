# external/

The package's one path to external solvers: gco, PyMaxflow, HiGHS, hmmlearn
and BlackJAX, each posed in sal's types (#1282). `validation/` drives the same
frameworks as referees for the suite and imports its runner and registry from
here.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and every docstring, comment and commit message in this
module. Its **API Conventions** bind every call added here: match the sibling,
one result type per concept, NumPy at the boundary. They are referenced here,
never restated. What follows is local.

## Local rules

- **Every framework runs in a subprocess.** `runner.run` is the one spawner in
  the package; `tests/regression/test_duplication_guards.py` counts a second
  one. No file here imports a framework: the script under
  `validation/scripts/` is the only one that does, so GPL and research-only
  terms, a native library and its threads stay out of the package process.
- **A refusal comes before the subprocess.** A call states the `Capability`
  set it needs; `require` refuses what the `Solver` does not declare, naming
  it, and an absent framework raises `ExternalUnavailable` naming its extra.
  Both are read in this process, before `runner.run`. An absent framework is
  never a fallback to the package's own solver.
- **One solver, one framework, one declaration.** A `Solver` member is one
  framework method, with an entry in `solvers.DECLARED` naming its
  `FRAMEWORKS` key and capabilities. A framework arrives through
  `validation/`'s rule (an extra, an adapter, a script and a test), and a
  solver over it is declared here.
- **Licence travels with the answer.** Every result carries the
  `Provenance` its framework's registry entry states, read from the installed
  distribution's metadata without importing it.
- **The import runs one way.** `validation/` imports from here; nothing here
  imports `validation/`. No hot-path package (`sim`, `likelihood`, `opt`,
  `search`, `learn`) imports from here.
