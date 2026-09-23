"""The file protocol between an adapter and its script (issue #972).

A script is run as ``python -m snakes_and_ladders.validation.scripts.<name>
<inputs.npz> <outputs.npz>``. It reads its inputs with :func:`load`, times the
framework's own call with :func:`timed`, and writes its outputs with
:func:`dump`, which records the time under :data:`SECONDS`. NumPy is the only
dependency, so the protocol costs a script nothing it would not import anyway.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

#: The output key that carries the seconds the script measured.
SECONDS = "__seconds__"

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


def dump(path: str | Path, outputs: Mapping[str, np.ndarray], seconds: float) -> None:
    """Write ``outputs`` and the measured ``seconds`` to an ``.npz``."""
    if SECONDS in outputs:
        message = f"{SECONDS!r} is reserved for the measured time"
        raise ValueError(message)
    save(path, {**outputs, SECONDS: np.asarray(seconds, dtype=np.float64)})


def timed(call: Callable[[], T]) -> tuple[T, float]:
    """``call()`` and the wall seconds it took, on ``time.perf_counter``."""
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def paths(argv: list[str] | None = None) -> tuple[Path, Path]:
    """The inputs and outputs paths a script is called with."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        usage = "usage: <script> <inputs.npz> <outputs.npz>"
        raise SystemExit(usage)
    return Path(arguments[0]), Path(arguments[1])
