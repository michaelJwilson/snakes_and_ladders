#!/usr/bin/env bash
# One readers-writer lock on the host, and the wrapper every agent takes it
# through. `DEV.md` carries the measurements that set the shape; this file
# carries the rule for deciding which kind a job is.
#
#   measure  exclusive. Runs alone.
#   validate shared, capped at three concurrent. Runs beside two siblings.
#
# WHICH KIND IS THIS JOB? Two questions, in order; either "yes" is `measure`.
#
#   1. Would a busy host change a number this run reports? A pytest-benchmark
#      run, a criterion bench, a cProfile ranking, any wall time, throughput or
#      memory figure quoted in a pull request, a ticket, STATUS.md or a
#      docstring: yes. Exclusivity is bought for comparability, so it is spent
#      exactly where the host is part of the answer.
#   2. Does the job use more than one core? Three slots are sized for
#      single-threaded jobs and do not bound a job that takes 1.5 of them. A
#      figure or document render is the case that exists: `qa.build` strips the
#      thread variables so a render matches its manifest (DEV.md), so a render
#      takes the exclusive lock though nobody times it.
#
# Otherwise `validate`, however long the job takes and however much it writes.
# Cost is not the test and neither is recording a number: `infra/baselines.py
# --write` spends minutes and commits its results, but every one of them is a
# deterministic function of the fixture, the code and a seed -- identical on an
# idle host and a loaded one -- so it is a correctness run. Taking `measure`
# for it once cost 30 minutes of queueing for a job that runs in 28 seconds
# (issue #427).
#
# LOCK ORDER. A measurement takes the exclusive lock and nothing else, so it
# holds nothing until it holds everything. Acquiring the exclusive lock first
# and then blocking for the shared capacity is the inversion issue #427 fixed:
# a measurement waiting that way holds slots a validation could have run in,
# and 3.3 s of a queued validation's wait was spent behind a measurement that
# had not started.
#
# THREADS. Holding the machine is not using it. A measurement runs at whatever
# thread count the caller set -- one, per `tests/conftest.py` -- because every
# baseline in `STATUS.md` and `DEV.md` was taken at one thread and a four-thread
# number is comparable with none of them.
#
# Usage:
#   . infra/locks.sh
#   with_lock validate -- uv run pytest -m critical
#   with_lock measure  -- infra/build_documents.sh          # a render
#   with_lock measure  -- uv run pytest tests/benchmarks -q
#
# Waits in the foreground, as the agent rules require: no sleep, no polling.
# SAL_LOCK_WAIT (default 1800) bounds the wait; exceeding it fails rather than
# proceeding unlocked, because a measurement taken beside another job is worse
# than no measurement.

set -uo pipefail

: "${SAL_LOCK_WAIT:=1800}"
: "${SAL_SCRATCH:=${TMPDIR:-/tmp}}"

#: Concurrent validations. Three is the measured peak; see `DEV.md`.
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

  local mode
  case "$kind" in
    measure) mode=-x ;;
    validate) mode=-s ;;
    *) echo "with_lock: unknown kind '$kind'" >&2; return 2 ;;
  esac

  local dir host slot fd status
  dir="$(_lock_dir)"

  # The one wait a measurement does. Each descriptor closes when this shell
  # exits, so a killed job leaks nothing.
  exec {host}>"$dir/host.lock"
  if ! flock -w "$SAL_LOCK_WAIT" "$mode" "$host"; then
    exec {host}>&-
    echo "with_lock: timed out waiting for the host lock" >&2
    return 1
  fi

  if [ "$kind" = validate ]; then
    # The shared lock says a measurement is not running; a slot caps how many
    # validations share the host. Taking a slot and keeping it must be one
    # step: an earlier version probed with `flock -n ... true`, which releases
    # the slot again before the command starts, so three jobs launched together
    # all saw slot 1 free and then queued on it -- the overlap test caught it.
    fd=
    for slot in $(seq 1 "$SAL_VALIDATE_SLOTS"); do
      exec {fd}>"$dir/validate.$slot.lock"
      flock -n "$fd" && break
      exec {fd}>&-
      fd=
    done
    if [ -z "$fd" ]; then
      # Every slot busy: block on the first rather than spinning over all three.
      exec {fd}>"$dir/validate.1.lock"
      if ! flock -w "$SAL_LOCK_WAIT" "$fd"; then
        exec {fd}>&-
        exec {host}>&-
        echo "with_lock: timed out waiting for a validate slot" >&2
        return 1
      fi
    fi
  fi

  "$@"
  status=$?
  [ -n "${fd:-}" ] && exec {fd}>&-
  exec {host}>&-
  return $status
}
