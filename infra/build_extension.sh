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
target="python/snakes_and_ladders/oxisal${suffix}"

cargo build --release --locked "$@"
cp "target/release/liboxisal.so" "$target"
echo "built $target"
