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
#                        a notebook check.
#
# A FIGURE OR DOCUMENT RENDER TAKES `measure`, not `validate`, even though
# nobody times it. `snakes_and_ladders.qa.build` deliberately strips the three
# thread variables from a render's environment (DEV.md: a committed figure is
# rendered the way the manifest renders it, and a reduction split over a
# different thread count can move its last bit), so a render is the one
# validation that is NOT single-threaded. Measured at 22:54 on the 4-core
# host: two concurrent `qa.opt_coverage` renders at 149% CPU each, driving
# load average to 10.65 -- 2.7x oversubscribed -- while three agents worked.
# Three slots sized for single-threaded jobs do not bound a job that takes
# 1.5 cores, so a render takes the exclusive lock. The reason differs from a
# measurement's -- footprint rather than comparability -- and the mechanism
# is the same.
#
# A measurement takes BOTH -- `measure` first, then every slot of `validate` --
# so it still runs alone and the budgets in DEV.md stay comparable across runs.
# `with_lock measure` does that for you; do not hand-roll it.
#
# THREADS. Holding the machine is not the same as using it. A measurement runs
# at one thread by default even though it owns all four cores, and that is
# deliberate: every baseline in STATUS.md and DEV.md was taken at one thread,
# so a number taken at four is not comparable with any of them. Exclusivity
# buys a quiet machine, not a wide one.
#
# The exception is measuring something that is itself parallel -- rayon in the
# Rust backend, a torch intra-op reduction, the CPU parallelism of #344 --
# where one thread measures the wrong thing. `--wide` raises the three thread
# variables to the core count for that command, and is legal only under
# `measure`, which is the only kind that owns the machine. A number taken
# under `--wide` MUST state its thread count wherever it is quoted, or it is
# indistinguishable from a one-thread number and silently wrong.
#
# Three validate slots, by measurement rather than by arithmetic. Running
# `pytest -m critical` at 1, 2, 3 and 4 concurrent on the idle 4-core
# reference host, one BLAS thread each:
#
#   concurrent   total     throughput   vs one   per-job latency
#   1            22.0 s    0.045/s      1.00x    --
#   2            23.4 s    0.086/s      1.89x    +6%
#   3            28.7 s    0.104/s      2.30x    +30%
#   4            39.2 s    0.102/s      2.25x    +78%
#
# Three is the peak; four is past the knee and buys nothing for a 78% latency
# cost. Note the trade the third slot makes: throughput 2.30x, but each
# individual validation takes 30% longer than it would alone. That is the
# right trade for correctness runs nobody times, and the wrong one for a
# measurement, which is why measurements do not share.
#
# The problem this solves: one agent under the old exclusive lock occupied a
# quarter of the machine while seven waited, and #399B measured 40 minutes of
# starvation inside a 104-minute run at five agents while load average sat at
# 0.68 -- serialization, not saturation.
#
# Usage:
#   . infra/locks.sh
#   with_lock validate -- uv run pytest -m critical
#   with_lock measure  -- infra/build_technical_doc.sh          # a render
#   with_lock measure  -- uv run pytest tests/benchmarks -q
#   with_lock measure --wide -- cargo bench --bench parallel_sweep
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
  # with_lock <measure|validate> [--wide] -- <command...>
  local kind="${1:?with_lock needs a kind: measure or validate}"
  shift
  local wide=0
  if [ "${1:-}" = "--wide" ]; then
    wide=1
    shift
  fi
  [ "${1:-}" = "--" ] && shift
  [ $# -gt 0 ] || { echo "with_lock: no command given" >&2; return 2; }

  if [ "$wide" = 1 ]; then
    if [ "$kind" != "measure" ]; then
      # A validation shares the host with two others, so it may not take the
      # cores they are using. Refusing beats quietly oversubscribing.
      echo "with_lock: --wide needs 'measure'; '$kind' does not own the host" >&2
      return 2
    fi
    local cores
    cores="$(nproc)"
    set -- env "OMP_NUM_THREADS=$cores" "OPENBLAS_NUM_THREADS=$cores" \
      "MKL_NUM_THREADS=$cores" "RAYON_NUM_THREADS=$cores" "SAL_MEASURE_THREADS=$cores" "$@"
  fi

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
  # Takes whichever slot frees first. The acquisition must be atomic: an
  # earlier version probed a slot with `flock -n ... true`, which releases it
  # again before the command starts, so three jobs launched together all saw
  # slot 1 free and then queued on it -- the split bought nothing, and the
  # overlap test caught it. Holding a file descriptor across the `flock` makes
  # taking the slot and keeping it one step. The descriptor closes when this
  # shell exits, so a killed job never leaks a slot.
  local dir="$1"
  shift
  local slot fd status
  for slot in $(seq 1 "$SAL_VALIDATE_SLOTS"); do
    exec {fd}>"$dir/validate.$slot.lock"
    if flock -n "$fd"; then
      "$@"
      status=$?
      exec {fd}>&-
      return $status
    fi
    exec {fd}>&-
  done
  # Every slot busy: block on the first rather than spinning over all three.
  exec {fd}>"$dir/validate.1.lock"
  if ! flock -w "$SAL_LOCK_WAIT" "$fd"; then
    exec {fd}>&-
    echo "with_lock: timed out waiting for a validate slot" >&2
    return 1
  fi
  "$@"
  status=$?
  exec {fd}>&-
  return $status
}

_hold_all_validate_slots() {
  # A measurement runs alone, so it holds every validation slot as well as the
  # measure lock. Each slot gets its own descriptor, taken in order; all are
  # released when this shell exits.
  local dir="$1"
  shift
  local slot fd status
  local -a held=()
  for slot in $(seq 1 "$SAL_VALIDATE_SLOTS"); do
    exec {fd}>"$dir/validate.$slot.lock"
    if ! flock -w "$SAL_LOCK_WAIT" "$fd"; then
      echo "with_lock: timed out taking validate slot $slot" >&2
      return 1
    fi
    held+=("$fd")
  done
  "$@"
  status=$?
  for fd in "${held[@]}"; do
    exec {fd}>&-
  done
  return $status
}

if [ "${1:-}" = "--hold-all-validate-slots" ]; then
  shift
  _hold_all_validate_slots "$@"
  exit $?
fi
