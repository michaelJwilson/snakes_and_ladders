#!/usr/bin/env bash
# Build `oxisal` for THIS worktree and put it where the
# package will import it.
#
# Issue #630. The shared `.venv` holds one editable install, pointing at the
# primary worktree; a worktree is reached with `PYTHONPATH=<worktree>/python`,
# which shadows that install and its extension. So a worktree needs its own,
# and `maturin develop` cannot supply it: it writes the environment, which is
# shared, and `infra/bin/uv` refuses that for the reason issue #615 gives.
#
# `cargo build` plus a copy is what is left, and it is what this is.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# The suffix CPython will import under, asked of **the interpreter that will
# import it** rather than of whatever `python3` resolves to: the environment
# is 3.12 and a login shell here answered 3.11, which names a file the suite
# then does not see.
interpreter=".venv/bin/python"
if [ ! -x "$interpreter" ]; then
  interpreter="$(command -v python3)"
fi
suffix="$("$interpreter" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
target="python/sal/oxisal${suffix}"

# `--iter` builds `[profile.iter]` (issue #1224): no LTO, 16 codegen units,
# incremental, so one edit in `src/` rebuilds in about 5 s against the release
# profile's 19. The release profile's fat LTO is what a benchmark quotes, so
# rebuild without the flag before reading one. `CARGO_TARGET_DIR`, when set,
# is where the library is found, so one cached target serves every worktree.
profile="release"
if [ "${1:-}" = "--iter" ]; then
  profile="iter"
  shift
fi

# A shared `CARGO_TARGET_DIR` holds one fingerprint for the crate whatever the
# worktree, and cargo reads a worktree whose sources are older than the last
# build as fresh: it would copy another worktree's library (issue #1268).
# Touching one source marks the crate dirty, so it recompiles from these
# sources; the dependencies stay cached.
if [ -n "${CARGO_TARGET_DIR:-}" ]; then
  touch src/lib.rs
fi

cargo build --profile "$profile" --locked "$@"
cp "${CARGO_TARGET_DIR:-target}/$profile/liboxisal.so" "$target"
echo "built $target ($profile)"
