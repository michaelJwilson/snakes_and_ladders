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

- **Every framework runs in a subprocess.** `runner` is the one spawner in
  the package, of a one-shot script (`run`) and of a session's worker
  (`worker`); `tests/regression/test_duplication_guards.py` counts a second
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
- **A session is the one-shot call, served by one worker.** The worker runs
  the script a one-shot call runs, so a call's outputs are bitwise those of
  `invoke` under every `Transport`; `tests/regression/test_external_session.py`
  pins each. The worker is shut down when the `with` block exits, normally or
  by an exception, and a worker that dies is a `ScriptError` carrying its
  standard error; the session is closed from then on.
- **The parent owns every block.** The parent creates and unlinks each
  memory-mapped file, inputs and outputs alike; the
  worker attaches and never unlinks, so a killed worker leaves no orphan, a
  count the kill tests hold at 0.
- **A session sends C-contiguous `float64`, `int64` or `bool` arrays.**
  Anything else is refused before the call, under every transport, so the
  inputs do not depend on the transport chosen.
- **A problem family is a submodule; the root is infrastructure.** The
  root exports `Solver`, `Capability`, `Provenance`, `session`, `Transport`
  and the runner, and no call that solves a problem; the Potts calls are
  `potts`, and a later family takes its own module (`hmm`, `hmc`).
  `tests/regression/test_external.py` holds the root to it.
- **A call takes its sibling's signature and returns its result type.**
  `potts.ground_state` is `search.ground_state.ground_state`'s arguments with a
  `Solver` for the method, and returns its `MethodRun` as `ExternalRun`,
  which adds the `Provenance`; the guard is
  `tests/regression/test_external_ground_state.py`. `potts.lower_bound` is
  `search.trws.trws`'s `graph` and `field` with a `Solver` after them, and
  returns its `BoundedLabelling` as `ExternalBound`
  (`tests/regression/test_external_lower_bound.py`). A problem's bytes are
  posed once, in `potts_inputs.py`, for this call and the adapters alike, so
  the answers are bitwise (`tests/validation/test_gco.py`,
  `test_pymaxflow.py`, `test_highs.py`).
- **A transport is kept on a measurement.** `MMAP` is the default: it beat
  `NPZ` in 13 of 15 cells of #1288's grid and trailed it in none by more than
  the host's run-to-run spread (0.4 ms at 0.2 MB, 4 ms of 198 at 10 MB, N = 1).
  `SHARED` (`multiprocessing.shared_memory`) was dropped, `MMAP` beating it in
  10 of 12 cells at 10 and 128 MB. A transport that stops earning its place
  goes.
