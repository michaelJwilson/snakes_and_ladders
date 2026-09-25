#!/usr/bin/env bash
# Create an agent worktree, complete: the branch, the shared environment, the
# gitignored extension, and the PYTHONPATH that resolves to it.
#
# The five-step convention this replaces was repeated in every agent brief and
# failed four times on 2026-09-08: a sibling worktree's code imported and
# judged (issue #401), 164 collection errors from a real `.venv` directory
# rather than the symlink (issue #404), and two worktrees without the compiled
# extension. Each step below is one of those failures.
#
# The symlink this makes is also what a `uv sync` narrower than the extras
# installed would strip for all of them at once, so the last line emitted puts
# `infra/bin` --- the guard that refuses that --- ahead of `uv` on `PATH`
# (issue #615).
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: ${0##*/} <path> <branch> [base; default origin/main]" >&2
  exit 2
fi
path=$1
branch=$2
base=${3:-origin/main}

# The checkout holding `.git`, whose `.venv` every worktree shares.
main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")

git -C "$main" fetch --quiet --no-tags origin
git -C "$main" worktree add -b "$branch" "$path" "$base"
ln -s "$main/.venv" "$path/.venv"

# Copy an extension only if it is newer than every `src/*.rs`; otherwise build
# one. The old loop took the *first* `.so` it found anywhere, and a stale build
# imports fine and fails only when something calls the changed signature --
# which is why the check below is on mtime and not on import (issue #556). At
# filing, 121 of 154 extensions across the worktrees predated the newest
# `src/*.rs`.
newest_source=$(ls -t "$main"/src/*.rs | head -1)
extension=""
for tree in "$main" $(git -C "$main" worktree list --porcelain | sed -n 's/^worktree //p'); do
  for candidate in "$tree"/python/sal/*.so; do
    if [ -f "$candidate" ] && [ "$candidate" -nt "$newest_source" ]; then
      extension=$candidate; break 2
    fi
  done
done
if [ -z "$extension" ]; then
  echo "no extension newer than $newest_source; building one (about 13 s incremental)" >&2
  (cd "$path" && cargo build --release) || exit 1
  built=$(ls "$path"/target/release/liboxisal.* 2>/dev/null | head -1)
  [ -n "$built" ] || { echo "cargo build produced no library" >&2; exit 1; }
  cp "$built" "$path/python/sal/oxisal.cpython-312-x86_64-linux-gnu.so"
else
  cp "$extension" "$path/python/sal/"
fi

# The import must resolve here and not in a sibling, which is issue #401.
resolved=$(cd "$path" && PYTHONPATH="$path/python" UV_NO_SYNC=1 \
  uv run python -c 'import sal as s; print(s.__file__)')
case "$resolved" in
  "$path"/*) ;;
  *) echo "import resolved to $resolved, outside $path" >&2; exit 1 ;;
esac

echo "$branch at $path, from $base, importing $resolved"
echo "export PATH=$main/infra/bin:\$PATH"
echo "export PYTHONPATH=$path/python"
