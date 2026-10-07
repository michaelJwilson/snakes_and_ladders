#!/usr/bin/env bash
# Build OpenGM's validation library, `libsal_opengm.so`, into a cache (issue #1279).
#
# OpenGM is not on PyPI (#962), and its inference is header-only C++
# templates. This fetches the pinned commit's headers, compiles
# `infra/opengm/sal_opengm.cxx` against them with the host's g++, and writes
# the library, the commit and the licence terms into one directory, which
# `sal.external` finds and the validation tests skip without. No Boost, no
# HDF5, no Python binding and no downloaded external solver is built: every
# algorithm in scope is OpenGM's own MIT code (#1279's spike), and the
# min s-t cut alpha expansion and the swap need is the file's own.
#
#   infra/build_opengm.sh            # build, or report the cached build current
#   SAL_OPENGM_HOME=/some/dir infra/build_opengm.sh
#
# Idempotent: a build whose stamp (commit, driver source, flags) matches is
# kept, and anything else is rebuilt. The directory is
# `$SAL_OPENGM_HOME`, else `${XDG_CACHE_HOME:-$HOME/.cache}/sal/opengm`, the
# rule `sal.external.frameworks.built_home` reads.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

#: The commit built: OpenGM's last, 2019-07-26; the project is unmaintained.
commit="decdacf4caad223b0ab5478d38a855f8767a394f"
repository="https://github.com/opengm/opengm"
home="${SAL_OPENGM_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/sal/opengm}"
source_file="$root/infra/opengm/sal_opengm.cxx"
flags="-std=c++11 -O2 -shared -fPIC -w"
compiler="${CXX:-g++}"

stamp="$commit $(sha256sum "$source_file" | cut -d' ' -f1) $compiler $flags"
if [ -f "$home/STAMP" ] && [ "$(cat "$home/STAMP")" = "$stamp" ] && [ -f "$home/libsal_opengm.so" ]; then
  echo "opengm: current at $home"
  exit 0
fi

mkdir -p "$home"
if [ ! -d "$home/src/include/opengm" ] || [ "$(git -C "$home/src" rev-parse HEAD 2>/dev/null)" != "$commit" ]; then
  rm -rf "$home/src"
  git init -q "$home/src"
  git -C "$home/src" fetch -q --depth 1 "$repository" "$commit"
  git -C "$home/src" checkout -q FETCH_HEAD
fi

start=$(date +%s)
# shellcheck disable=SC2086 # the flags are words
"$compiler" $flags -I"$home/src/include" "$source_file" -o "$home/libsal_opengm.so.tmp"
mv "$home/libsal_opengm.so.tmp" "$home/libsal_opengm.so"
echo "$commit" > "$home/COMMIT"
cat > "$home/LICENCES" <<TERMS
OpenGM $commit ($repository): MIT, Licence-OpenGM.txt.
Built: OpenGM's header-only ICM, loopy BP, A*, TRWSi (Savchynskyy's TRW-S),
dual decomposition (subgradient, dynamic-programming subproblems), alpha
expansion and alpha-beta swap; the min s-t cut is infra/opengm/sal_opengm.cxx's.
Not fetched or built: the externals OpenGM's CMake downloads (Kolmogorov's
TRW-S v1.3 and MRF-LIB, maxflow v3.02, QPBO v1.3: research-only or GPL;
gco-v3.0: research-only; IBFS, AD3, ConicBundle, and the rest).
TERMS
echo "$stamp" > "$home/STAMP"
echo "opengm: built $home/libsal_opengm.so in $(( $(date +%s) - start )) s"
