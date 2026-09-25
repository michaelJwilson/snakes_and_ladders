# Installing and running locally

Everything needed for a working checkout: the environment, the build, the test
suites, and the checks pre-commit runs. For the repository layout, the CI jobs,
and how a change is reviewed, see [DEV.md](DEV.md).

## Prerequisites

| Tool | Version | Notes |
| --- | --- | --- |
| Python | >= 3.12.2 | `requires-python` in `pyproject.toml` |
| Rust | 1.94.1 | pinned by `rust-toolchain.toml`; `rustup` installs it automatically |
| [uv](https://docs.astral.sh/uv/) | 0.8.17 | the version CI pins |

The package compiles a Rust extension on install, so a Rust toolchain is
required even for a Python-only workflow.

## Environment

```
uv sync --locked --all-extras
source .venv/bin/activate
```

After changing a dependency, run `uv lock` and commit the updated lockfile in
the same PR.

Sync once per worktree, then set `UV_NO_SYNC=1` in the shell: every `uv run`
otherwise re-resolves the environment and, in a fresh worktree, recompiles the
Rust extension — one to two minutes of a core per command (issues #369, #372).
The repository's scripts export it themselves.

Where several worktrees share one environment through a `.venv` symlink, put
the guard ahead of `uv` as well — `infra/new_worktree.sh` emits both lines:

```
export PATH=/path/to/main/checkout/infra/bin:$PATH
```

`infra/bin/uv` refuses `sync`, `add`, `remove` and a syncing `run` when `.venv`
is a symlink, and passes everything else through untouched. A sync narrower
than the extras installed uninstalls the rest — 15 of 24 declared requirements,
twice (issue #615) — and one with `--all-extras` repoints the shared editable
install (issue #556); `infra/repair_environment.py` is what adds back what is
missing. A real `.venv` directory has one owner and is never refused, which is
why CI, and the first sync above, are unaffected. `UV_NO_SYNC=1` does not
substitute: it is a `uv run` option, and `uv sync` ignores it.

Before pushing, run `infra/validate.sh`: lint, types, the critical gate, the
tests the diff selects, and whichever of the Sphinx, notebook and document
checks the diff calls for, each timed, under the 300 s budget `DEV.md`
states.

The extras are `dev` (ruff, mypy, pre-commit, pip-audit), `test` (pytest and
plugins, NumPy), `docs` (Sphinx), `notebooks` (a kernel, for re-executing
`docs/nb/`), and `track` (Aim, the run store), beside the `validation-*` extras
below. Sync a single one with `uv sync --locked --extra test`; `--all-extras`
installs all of them. `notebooks` carries Jupyter Notebook itself, so a
notebook opens where it is edited:
`uv run --extra notebooks jupyter notebook docs/nb/`. Nothing in the core
install needs an external framework, and no package module imports one.

The `validation-<framework>` extras (issue #972) each install one external
framework the package is checked or timed against, and nothing imports one
into the package process: its script under
`python/sal/validation/scripts/` runs in a subprocess. Sync one
with `uv sync --locked --extra test --extra validation-<name>`, or every one
with `uv sync --locked --extra test $(python3 infra/validation_extras.py)`,
then run `uv run pytest -m validation tests/validation`. Without its extra a
test there skips.

`track` is the one extra with an advisory against it, and the one to sync
deliberately. It installs `aim`, the optional store behind
`sal.track.Run` (issue #778). Nothing in the package imports
it: `Run` is a Protocol written with `aim.Run`'s own signatures, so an Aim
run is passed in and nothing is adapted, and the default run records nothing
and needs nothing. PYSEC-2026-1087 and PYSEC-2026-1088 stand unfixed against
3.29.1, its current release; both are in the server `aim up` runs. No CI job
installs the extra --- the audit job syncs `dev` --- so `uv run pip-audit`
is clean on this tree and reports those two after `uv sync --locked
--extra track` or `--all-extras`, which is the trade a reader who wants the
UI is making.

`scipy` is a core dependency since the 0.5.0 audit (issue #376): three
regression modules referee our neighbor joining, Hadamard transform and fit
against it, and `search.maxflow`'s replacement (#388) will front its maximum
flow.

## Building

`maturin` is the PEP 517 build backend, so a normal install compiles the Rust
extension:

```
pip install .
```

This makes `sal.oxisal` importable from Python: `double`, an example
binding; `pruning_log_likelihood`, the Rust CPU Felsenstein pruning backend
behind `sal.likelihood.pruning.rust`; `sample_rows`, the categorical
sampler behind `sal.numerics_rust`, which `sal.sim` and `sal.opt` draw
their fixtures through; and `max_flow` with `ising_ground_state`, the
minimum-cut kernels behind `sal.search.maxflow.rust`. Reinstall after
editing anything under `src/`; the compiled module does not rebuild itself.

One binding is not in that build. `pruning_gradient`, the `burn` taped
gradient issue #449 measured and declined, sits behind the `sandbox` Cargo
feature so the default install compiles no `burn` (34 crates in 37 s against
102 in 86 s; `DEV.md`, Build System). `sal.sandbox.pruning_burn`
imports either way and refuses to run without it, and its tests skip. To run
them, build with the feature:

```
maturin develop --release --features sandbox
pytest tests/regression/sandbox/test_pruning_burn.py
```

## Running the tests

```
pytest      # Python: regression tests (tests/regression), a pytest-benchmark
            # suite (tests/benchmarks), and an integration test that the
            # Rust extension imports correctly
            # (tests/test_oxisal_bindings.py)
cargo test  # Rust: unit tests for the PyO3 bindings (src/lib.rs)
cargo bench # Rust: Criterion benchmarks (benches/)
```

Add `--features sandbox` to either `cargo` command for the conserved route's
own tests and bench; `infra/release.sh` runs both that way, and nothing else
does.

`pytest` reads its configuration from `pyproject.toml`. To reproduce the CI
gate, including coverage:

```
pytest --cov=sal --cov-report=term-missing --cov-fail-under=90
```

## Checks CI will run

Run these before pushing; all of them are required checks.

```
ruff check .
ruff format --check .
mypy                                       # strict, over python/, tests/ and infra/
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --check
```

Add a fragment under `changelog.d/` for the change as well, if it is
user-visible (see `changelog.d/README.md`); `towncrier` merges fragments into
`CHANGELOG.md` at release time.

`pre-commit install` runs these same checks on every `git commit`, but not the
dependency audits below; CI runs those when a lockfile changes.

## Dependency audits

```
pip-audit    # Python, from the dev extra
cargo audit  # Rust; install once with `cargo install cargo-audit --locked`
```

Both run in CI's `audit` job when `uv.lock` or `Cargo.lock` changed, and
weekly on `main` regardless, so a newly disclosed advisory against a pinned
dependency fails the build. [DEV.md](DEV.md) describes the caching.

## Building the documentation

API documentation, from the NumPy-style docstrings:

```
sphinx-build -b html docs/source docs/_build/html -W
```

Open `docs/_build/html/index.html`. The `-W` flag turns warnings into errors,
matching CI, so a broken docstring or cross-reference fails locally.

The documents — the paper and the textbook — are LaTeX under `docs/tex/`. The
`sal.qa` scripts render the figures and tables they include
(`sal.qa.manifest` lists them), so building them regenerates the
cited ones before running `latexmk`; the applicability tables the textbook
inputs are not committed and are written by `infra/problems_tables.py --write`,
which `infra/build_documents.sh` runs for you:

```
sudo apt-get install -y --no-install-recommends latexmk texlive-latex-base \
  texlive-latex-recommended texlive-fonts-recommended texlive-science \
  texlive-pictures
uv sync --locked --extra test
infra/build_documents.sh
```

Open `docs/paper.pdf` and `docs/textbook.pdf`. CI runs the same script and
fails on an undefined or multiply-defined reference, an undefined citation, or
a pull request that changes either committed PDF without being the rebuild the
"Rebuild the documents" ticket asks for. The PDFs are *not* compared against
the rebuild: they lag `docs/tex/` between rebuilds by design (`DEV.md`,
Documents).

**Revert `docs/paper.pdf` and `docs/textbook.pdf` after the build.** It rewrites
both on every run, they are tracked, and only a "Rebuild the documents" pull
request may carry the change:

```
git checkout -- docs/paper.pdf docs/textbook.pdf
```

The rest needs no such care: the `.aux`, `.log`, `.fdb_latexmk` and other
files `latexmk` leaves in `docs/`, and the applicability tables under
`docs/tex/generated/`, are ignored and cannot be committed, while a re-rendered
figure under `docs/tex/figures/` — its `.pdf` or `.tex` and its `_caption.txt`
— is committed by the pull request that changed it. `DEV.md` lists every path,
and gives the wall clock on the 4-core reference host: **15.8 s** for the
LaTeX, **3.7 s** for a second run that finds nothing to do, and **441.3 s** for
the figures, all 23 in the manifest on every build since issue #490 deleted the
stamps that used to select them and issue #492 left every entry cited.

## Benchmarking locally

CI runs the benchmarks but asserts nothing against their timings: GitHub-hosted
runner hardware varies between runs. Compare against a local baseline instead:

```
pytest tests/benchmarks --benchmark-autosave            # establish a baseline
pytest tests/benchmarks --benchmark-compare=0001 \
                        --benchmark-compare-fail=mean:5%

cargo bench -- --save-baseline main                      # Criterion equivalent
cargo bench -- --baseline main
```
