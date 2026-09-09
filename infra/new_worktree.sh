#!/usr/bin/env bash
# Create an agent worktree, complete: the branch, the shared environment, the
# gitignored extension, and the PYTHONPATH that resolves to it.
#
# The five-step convention this replaces was repeated in every agent brief and
# failed four times on 2026-09-08: a sibling worktree's code imported and
# judged (issue #401), 164 collection errors from a real `.venv` directory
# rather than the symlink (issue #404), and two worktrees without the compiled
# extension. Each step below is one of those failures.
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

extension=""
for tree in "$main" $(git -C "$main" worktree list --porcelain | sed -n 's/^worktree //p'); do
  for candidate in "$tree"/python/snakes_and_ladders/*.so; do
    if [ -f "$candidate" ]; then extension=$candidate; break 2; fi
  done
done
if [ -z "$extension" ]; then
  echo "no compiled extension in any worktree; build one with maturin" >&2
  exit 1
fi
cp "$extension" "$path/python/snakes_and_ladders/"

# The import must resolve here and not in a sibling, which is issue #401.
resolved=$(cd "$path" && PYTHONPATH="$path/python" UV_NO_SYNC=1 \
  uv run python -c 'import snakes_and_ladders as s; print(s.__file__)')
case "$resolved" in
  "$path"/*) ;;
  *) echo "import resolved to $resolved, outside $path" >&2; exit 1 ;;
esac

echo "$branch at $path, from $base, importing $resolved"
echo "export PYTHONPATH=$path/python"
