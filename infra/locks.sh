#!/usr/bin/env bash
# The two host locks, and the wrapper every agent takes them through.
#
# The host runs one *measurement* at a time, not one heavy step at a time.
# Those are different rules, and the difference is where the throughput went:
# exclusivity is bought for comparability, so it is spent only where a number
# is reported (CLAUDE.md, Throughput).
#
#   measure  one slot.   Anything whose wall time, throughput or memory reaches
#                        a pull request, a ticket, STATUS.md or a docstring:
#                        a pytest-benchmark run, a criterion bench, a cProfile
#                        ranking, a timing quoted anywhere.
#   validate three slots. Correctness runs nobody times: ruff, mypy, the
#                        critical tier, the changed-file run, review_gates.sh,
#                        a notebook check, a document build whose output is
#                        compared rather than timed.
#
# A measurement takes BOTH -- `measure` first, then every slot of `validate` --
# so it still runs alone and the budgets in DEV.md stay comparable across runs.
# `with_lock measure` does that for you; do not hand-roll it.
#
# Three validate slots and not four: every job is single-threaded already
# (OMP_NUM_THREADS=1), the reference host has four cores, and one is left for
# the orchestrating session. The measured problem this solves: one agent under
# the old exclusive lock occupied a quarter of the machine while seven waited,
# and #399B measured 40 minutes of starvation inside a 104-minute run at five
# agents while load average sat at 0.68 -- serialization, not saturation.
#
# Usage:
#   . infra/locks.sh
#   with_lock validate -- uv run pytest -m critical
#   with_lock measure  -- uv run pytest tests/benchmarks -q
#
# Waits in the foreground, as the agent rules require: no sleep, no polling.
# SAL_LOCK_WAIT (default 1800) bounds the wait; exceeding it fails rather than
# proceeding unlocked, because a measurement taken beside another job is worse
# than no measurement.

set -uo pipefail

: "${SAL_LOCK_WAIT:=1800}"
: "${SAL_SCRATCH:=${TMPDIR:-/tmp}}"

#: Slots on the validation lock. See the note above before changing it.
SAL_VALIDATE_SLOTS=3

_lock_dir() {
  mkdir -p "$SAL_SCRATCH/locks"
  printf '%s/locks' "$SAL_SCRATCH"
}

with_lock() {
  # with_lock <measure|validate> -- <command...>
  local kind="${1:?with_lock needs a kind: measure or validate}"
  shift
  [ "${1:-}" = "--" ] && shift
  [ $# -gt 0 ] || { echo "with_lock: no command given" >&2; return 2; }

  local dir
  dir="$(_lock_dir)"

  case "$kind" in
    validate) _with_validate_slot "$dir" "$@" ;;
    measure)
      # Exclusive against other measurements, then against every validation:
      # the measurement runs alone, which is the only reason this lock exists.
      flock -w "$SAL_LOCK_WAIT" "$dir/measure.lock" \
        "${BASH_SOURCE[0]}" --hold-all-validate-slots "$dir" "$@"
      ;;
    *) echo "with_lock: unknown kind '$kind'" >&2; return 2 ;;
  esac
}

_with_validate_slot() {
  # Takes whichever of the numbered slots frees first. `flock` releases a slot
  # when its file descriptor closes, so a killed job never leaks one.
  local dir="$1"
  shift
  local slot
  for slot in $(seq 1 "$SAL_VALIDATE_SLOTS"); do
    if flock -n "$dir/validate.$slot.lock" true 2>/dev/null; then
      flock -w "$SAL_LOCK_WAIT" "$dir/validate.$slot.lock" -c "$(_quote "$@")"
      return $?
    fi
  done
  # Every slot busy: block on the first rather than spinning over all three.
  flock -w "$SAL_LOCK_WAIT" "$dir/validate.1.lock" -c "$(_quote "$@")"
}

_quote() {
  # `flock -c` takes one shell string, so the command is requoted rather than
  # interpolated: a path with a space would otherwise split into two arguments.
  local out=""
  local arg
  for arg in "$@"; do
    out+="$(printf '%q ' "$arg")"
  done
  printf '%s' "$out"
}

_hold_all_validate_slots() {
  # Recurses through this file so each slot is held by a nested `flock`, then
  # runs the command with all three held.
  local dir="$1"
  shift
  local slot="${SAL_HELD_SLOT:-1}"
  if [ "$slot" -gt "$SAL_VALIDATE_SLOTS" ]; then
    "$@"
    return $?
  fi
  SAL_HELD_SLOT=$((slot + 1)) flock -w "$SAL_LOCK_WAIT" "$dir/validate.$slot.lock" \
    "${BASH_SOURCE[0]}" --hold-all-validate-slots "$dir" "$@"
}

if [ "${1:-}" = "--hold-all-validate-slots" ]; then
  shift
  _hold_all_validate_slots "$@"
  exit $?
fi
