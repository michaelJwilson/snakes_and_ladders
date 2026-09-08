# Installing and running locally

Everything needed for a working checkout: the environment, the build, the test
suites, and the checks pre-commit runs against your branch. For the repository
layout, the CI jobs, and how a change is reviewed, see [DEV.md](DEV.md).

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
otherwise re-resolves the environment, and in a fresh worktree recompiles the
Rust extension, one to two minutes of a core per command (issues #369, #372).
The repository's scripts export it themselves.

Before pushing, run `infra/validate.sh`: lint, types, the critical gate, the
tests the diff selects, and whichever of the Sphinx, notebook and document
checks the diff calls for, each timed, under the 300 s budget `DEV.md`
states.

The extras are `dev` (ruff, mypy, pre-commit, pip-audit), `test` (pytest and
plugins, NumPy), `docs` (Sphinx), `notebooks` (a kernel, for re-executing
`docs/nb/`), and `frameworks` (Gymnasium, rustworkx, TorchRL and PyTorch
Geometric: the external implementations the suite pins its own against, and
the Gymnasium adapter's one import). `--all-extras` installs all five; sync a
single one with `uv sync --locked --extra test`. Nothing in the core install
needs `frameworks`: every test that uses one of its packages skips without
it, and `snakes_and_ladders.search.gym` is the only module that imports one
at module level.

`scipy` is a core dependency since the 0.5.0 audit (issue #376): three
regression modules referee our neighbor joining, our Hadamard transform and
our fit against it, and `search.maxflow`'s replacement (#388) will front its
maximum flow.

## Building

`maturin` is the PEP 517 build backend, so a normal install compiles the Rust
extension:

```
pip install .
```

This makes `snakes_and_ladders.oxi_snakes_and_ladders` importable from Python: `double`, an example
binding; `pruning_log_likelihood`, the Rust CPU Felsenstein pruning backend
behind `snakes_and_ladders.likelihood.pruning_rust`; `sample_rows`, the categorical
sampler behind `snakes_and_ladders.numerics_rust`, which `snakes_and_ladders.sim` and `snakes_and_ladders.opt` draw
their fixtures through; and `max_flow` with `ising_ground_state`, the
minimum-cut kernels behind `snakes_and_ladders.search.maxflow_rust`. Reinstall after
editing anything under `src/`; the compiled module does not rebuild itself.

## Running the tests

```
pytest      # Python: regression tests (tests/regression), a pytest-benchmark
            # suite (tests/benchmarks), and an integration test that the
            # Rust extension imports correctly
            # (tests/test_oxiphylo_bindings.py)
cargo test  # Rust: unit tests for the PyO3 bindings (src/lib.rs)
cargo bench # Rust: Criterion benchmarks (benches/)
```

`pytest` reads its configuration from `pyproject.toml`. To reproduce the CI
gate, including coverage:

```
pytest --cov=snakes_and_ladders --cov-report=term-missing --cov-fail-under=90
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

`pre-commit install` runs these same checks on every `git commit`. It does not
run the dependency audits below; CI runs those when a lockfile changes.

## Dependency audits

```
pip-audit    # Python, from the dev extra
cargo audit  # Rust; install once with `cargo install cargo-audit --locked`
```

Both run in CI's `audit` job when `uv.lock` or `Cargo.lock` changed, and
weekly on `main` regardless, so a newly disclosed advisory against a pinned
dependency fails the build without every run auditing an unchanged graph.
[DEV.md](DEV.md) describes the caching.

## Building the documentation

API documentation, from the NumPy-style docstrings:

```
sphinx-build -b html docs/source docs/_build/html -W
```

Open `docs/_build/html/index.html`. The `-W` flag turns warnings into errors,
matching CI, so a broken docstring or cross-reference fails locally rather
than in review.

The documents — the paper and the textbook, carrying the scientific background,
equations, and algorithms — are LaTeX under `docs/tex/`. The
`snakes_and_ladders.qa` scripts render the figures and tables they include
(`snakes_and_ladders.qa.manifest` lists them), so building them regenerates the
cited ones first rather than only running `latexmk`; the applicability tables
the textbook inputs are written by `infra/problems_tables.py --write`:

```
sudo apt-get install -y --no-install-recommends latexmk texlive-latex-base \
  texlive-latex-recommended texlive-fonts-recommended texlive-science \
  texlive-pictures
uv sync --locked --extra test
infra/build_documents.sh
```

Open `docs/paper.pdf` and `docs/textbook.pdf`. CI runs the same script on
every PR, and fails on an undefined or multiply-defined reference, an undefined
citation, or a committed PDF that differs from the rebuild.

## Benchmarking locally

CI runs the benchmarks but asserts nothing against their timings, because
GitHub-hosted runner hardware varies between runs. Compare against a local
baseline instead:

```
pytest tests/benchmarks --benchmark-autosave            # establish a baseline
pytest tests/benchmarks --benchmark-compare=0001 \
                        --benchmark-compare-fail=mean:5%

cargo bench -- --save-baseline main                      # Criterion equivalent
cargo bench -- --baseline main
```
