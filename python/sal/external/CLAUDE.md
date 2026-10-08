# external/

The package's one path to external solvers: gco, PyMaxflow, HiGHS, hmmlearn,
BlackJAX and OpenGM, each posed in sal's types (#1282, #1279). `validation/` drives the same
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
  it, and an absent framework raises `ExternalUnavailable` naming its extra,
  or for a source build the script that builds it (OpenGM's
  `infra/build_opengm.sh`, #1279).
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
  `potts`, the HMM's `hmm`, and the Hamiltonian chain's `hmc`.
  `tests/regression/test_external.py` holds the root to it.
- **A call takes its sibling's signature and returns its result type.**
  `potts.ground_state` is `search.ground_state.ground_state`'s arguments with a
  `Solver` for the method, and returns its `MethodRun` as `ExternalRun`,
  which adds the `Provenance`; the guard is
  `tests/regression/test_external_ground_state.py`. `potts.lower_bound` is
  `search.trws.trws`'s `graph` and `field` with a `Solver` after them, then
  its `max_iterations` and `tolerance`, which OpenGM reads and HiGHS
  refuses, and returns its `BoundedLabelling` as `ExternalBound`
  (`tests/regression/test_external_lower_bound.py`). A problem's bytes are
  posed once, in `potts_inputs.py`, for this call and the adapters alike, so
  the answers are bitwise (`tests/validation/test_gco.py`,
  `test_pymaxflow.py`, `test_highs.py`). `hmm.fit`, `hmm.viterbi` and
  `hmm.forward_log_likelihood` take `opt.hmm.baum_welch_family`'s,
  `likelihood.hmm.viterbi`'s and `opt.hmm.forward_log_likelihood`'s leading
  arguments with a `Solver` after them, and return `EmFit` as `ExternalFit`,
  `ExternalPaths` and `ExternalLogLikelihood`
  (`tests/regression/test_external_hmm.py`); their bytes are posed in
  `hmm_inputs.py`, which `validation.hmmlearn.baum_welch` sends too
  (`tests/validation/test_hmmlearn.py`). A family hmmlearn has no model for
  needs a `Capability` it does not declare, and is refused. `hmc.sample`
  takes `sample.hmc.sample`'s three positional arguments with a `Solver`
  after them, and returns `HmcChain` as `ExternalChain`
  (`tests/regression/test_external_hmc.py`); its target is the kernel the
  objective declares (`supported_gradient`), rebuilt in JAX by the script,
  since a closure cannot cross the process boundary, and its bytes are posed
  in `hmc_inputs.py`, which `validation.blackjax` sends too
  (`tests/validation/test_blackjax.py`). A warm-up the framework cannot
  honour as asked is refused, never approximated in silence.
- **An algorithm has one home: its explicit call (#1304).**
  `potts.ground_state`, `potts.lower_bound` and `hmm.fit` are callable
  namespaces: called with a `Solver`, each is a `match` onto one explicit
  call, `ground_state.expansion` or `lower_bound.lp`, which holds the
  algorithm and its keywords. The `match` maps a `Solver` to a call and its
  `by` and holds no other logic, so the two cannot drift; its `case _`
  refuses a solver without the task by name. A keyword the explicit call
  does not take is refused before any subprocess starts. `by: Framework`
  sits only where several frameworks run one algorithm. A name is not
  repeated across levels: `lower_bound.lp`, not `lower_bound.lp_bound`.
  `tests/regression/test_external_explicit.py` holds every `Solver` member
  to one case and each pair of calls to equal results.
- **A transport is kept on a measurement.** The default is the transport
  that won #1288's grid, and a dropped one is named with its measurement in
  `transport`'s docstring. A transport that stops earning its place goes.
