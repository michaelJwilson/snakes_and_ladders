# validation/

The home of external frameworks the package is checked or timed against.
`sandbox/` conserves the package's own replaced implementations; this drives
other people's.

Root `CLAUDE.md` holds the repository-wide rules, and its **Writing Style**
section binds this file and every docstring, comment and commit message in this
module. It is referenced here, never restated. What follows is local.

## Local rules

- **Every framework runs in a subprocess.** An adapter writes an `.npz`, runs
  its script under `sys.executable` through `runner.run`, and reads an `.npz`
  back. The script is the only file that imports the framework, whatever its
  licence: GPL and research-only terms stay out of the package process, and
  so do a framework's native library, threads and second autodiff stack.
  The one exception is a framework that is also a core dependency: JAX is
  the HMM objectives' gradient (#1000), so the package imports it; its
  validation pair still runs in a subprocess.
- **A script imports its framework inside `main`.** The docs build imports
  every module under `docs/source/index.rst` without the extras, so a
  module-level import of the framework would fail it.
- **A script times and weighs the framework's own call and nothing else.** It
  wraps the call in `protocol.timed` inside `protocol.peaked`, so interpreter
  start-up, the imports and the file round trip are charged to neither
  figure. The package's side of a pair is read the same way, in a fresh
  interpreter, by `scripts/package.py`.
- **One framework, one extra, one module, one test file.** A framework arrives
  as a `validation-<name>` extra in `pyproject.toml`, a `Framework` entry in
  `FRAMEWORKS`, an adapter `<name>.py` here, a script `scripts/<name>.py` and
  `tests/validation/test_<name>.py`. The extra is added by the framework's
  own ticket, with the permission root `CLAUDE.md` requires.
- **A framework's runtime sets a goal.** Where a benchmark pair times the
  framework on a declared fixture, its runtime there is hardcoded as a `Goal` in
  `tests/validation/test_goals.py`, with when and where it was measured. The
  goal test times the package alone, so it needs no framework, and fails
  until the package meets the figure. It runs in a non-blocking step of the
  `validation` job.
- **Only `tests/` imports from here.**
- **A test here skips where its framework is absent** and carries the
  `validation` marker beside its kind. CI's `validation` job installs every
  `validation-*` extra and runs `-m validation`.
- **Nothing here is on a hot path.** A framework that would replace an
  implementation is a different step, gated on a measurement, and the replaced
  implementation moves to `sandbox/`.
