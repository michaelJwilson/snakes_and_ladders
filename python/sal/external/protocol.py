"""The file protocol between an adapter and its script (issues #972, #1282).

A script is run as ``python -m sal.validation.scripts.<name>
<inputs.npz> <outputs.npz>``. It reads its inputs with :func:`load`, times the
framework's own call with :func:`timed`, and writes its outputs with
:func:`dump`, which records the time under :data:`SECONDS`. NumPy is the only
dependency, so the protocol costs a script nothing it would not import anyway.

:func:`peaked` measures the resident memory a call adds at its peak (issue
#987): it resets the kernel's high-water mark through
``/proc/self/clear_refs``, reads ``VmRSS``, runs the call and reads
``VmHWM``. The kernel counts every page the process touches, NumPy's,
Rust's and a framework's alike, where ``tracemalloc`` sees Python's heap
alone; the reading is Linux-only, which is where the reference host and CI
run. :func:`dump` records it under :data:`PEAK_BYTES` when given.

:func:`received` and :func:`dump` read and write through a :class:`Codec`,
the ``.npz`` pair by default. A session's worker (#1282, step 2b) serves a
script under :func:`through`, so the script hands its arrays to the worker's
transport and is the same script a one-shot call runs.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

#: The output key that carries the seconds the script measured.
SECONDS = "__seconds__"

#: The output key that carries the peak resident bytes the call added.
PEAK_BYTES = "__peak_bytes__"

T = TypeVar("T")


def load(path: str | Path) -> dict[str, np.ndarray]:
    """Every array in an ``.npz``, read eagerly so the file can be removed."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write ``arrays`` to an uncompressed ``.npz``, one entry per name."""
    # `numpy.savez`'s stub types every keyword as its own `allow_pickle`
    # flag, so a mapping of arrays is handed over as `Any`.
    named: dict[str, Any] = dict(arrays)
    np.savez(path, **named)


def dump(
    path: str | Path,
    outputs: Mapping[str, np.ndarray],
    seconds: float,
    peak_bytes: int | None = None,
) -> None:
    """Write ``outputs``, the measured ``seconds`` and any ``peak_bytes`` to an ``.npz``."""
    if SECONDS in outputs or PEAK_BYTES in outputs:
        message = f"{SECONDS!r} and {PEAK_BYTES!r} are reserved for the measures"
        raise ValueError(message)
    measures = {SECONDS: np.asarray(seconds, dtype=np.float64)}
    if peak_bytes is not None:
        measures[PEAK_BYTES] = np.asarray(peak_bytes, dtype=np.int64)
    _codec.write(path, {**outputs, **measures})


def timed(call: Callable[[], T]) -> tuple[T, float]:
    """``call()`` and the wall seconds it took, on ``time.perf_counter``."""
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def _status(field: str) -> int:
    """A ``/proc/self/status`` field in bytes; the kernel reports kB."""
    with Path("/proc/self/status").open() as status:
        for line in status:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    message = f"/proc/self/status has no {field}"
    raise OSError(message)


def peaked(call: Callable[[], T]) -> tuple[T, int]:
    """``call()`` and the resident bytes it added at its peak.

    The high-water mark is reset first, so what the process held before the
    call is subtracted and only what the call touched on top of it remains.
    """
    Path("/proc/self/clear_refs").write_text("5")
    before = _status("VmRSS")
    result = call()
    return result, max(_status("VmHWM") - before, 0)


def measured(call: Callable[[], T]) -> tuple[T, float, int]:
    """``call()``, the wall seconds it took and the resident bytes it added at its peak (issue #1010).

    :func:`timed` inside :func:`peaked`, the pairing six scripts wrote out.
    """
    (result, seconds), peak_bytes = peaked(lambda: timed(call))
    return result, seconds, peak_bytes


def received(argv: list[str] | None = None) -> tuple[dict[str, np.ndarray], Path]:
    """A script's inputs, loaded, and the path its outputs go to (issue #1010)."""
    given, returned = paths(argv)
    return _codec.read(given), returned


@dataclass(frozen=True)
class Codec:
    """How :func:`received` reads a script's inputs and :func:`dump` writes its outputs."""

    #: The inputs, from the path the script was given.
    read: Callable[[str | Path], dict[str, np.ndarray]]
    #: The outputs and measures, to the path the script was given.
    write: Callable[[str | Path, Mapping[str, np.ndarray]], None]


#: The ``.npz`` file pair, what a one-shot call and a session's ``NPZ`` transport use.
NPZ = Codec(read=load, write=save)

_codec = NPZ


@contextmanager
def through(codec: Codec) -> Iterator[None]:
    """Serve :func:`received` and :func:`dump` through ``codec`` for the block's duration."""
    global _codec  # noqa: PLW0603 -- the worker's one switch, restored on exit
    previous, _codec = _codec, codec
    try:
        yield
    finally:
        _codec = previous


def paths(argv: list[str] | None = None) -> tuple[Path, Path]:
    """The inputs and outputs paths a script is called with."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        usage = "usage: <script> <inputs.npz> <outputs.npz>"
        raise SystemExit(usage)
    return Path(arguments[0]), Path(arguments[1])
