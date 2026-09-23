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
- **A script imports its framework inside `main`.** The docs build imports
  every module under `docs/source/index.rst` without the extras, so a
  module-level import of the framework would fail it.
- **A script times the framework's own call and nothing else.** It wraps the
  call in `protocol.timed`, so interpreter start-up and the file round trip
  are not charged to the framework in a benchmark pair.
- **One framework, one extra, one module, one test file.** A framework arrives
  as a `validation-<name>` extra in `pyproject.toml`, a `Framework` entry in
  `FRAMEWORKS`, an adapter `<name>.py` here, a script `scripts/<name>.py` and
  `tests/validation/test_<name>.py`. The extra is added by the framework's
  own ticket, with the permission root `CLAUDE.md` requires.
- **A framework that is faster sets a goal.** Where a benchmark pair shows
  the framework ahead on a declared fixture, a `goal` test in
  `tests/validation/` times both in one run and fails until the package meets
  the framework's runtime (`tests/validation/_goals.py`). It runs in a
  non-blocking step of the `validation` job.
- **Only `tests/` imports from here.**
- **A test here skips where its framework is absent** and carries the
  `validation` marker beside its kind. CI's `validation` job installs every
  `validation-*` extra and runs `-m validation`.
- **Nothing here is on a hot path.** A framework that would replace an
  implementation is a different step, gated on a measurement, and the replaced
  implementation moves to `sandbox/`.
