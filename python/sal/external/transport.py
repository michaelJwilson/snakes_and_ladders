"""How a session's arrays reach its worker and come back (issue #1282, step 2b).

:class:`Transport` names the three ways. ``NPZ`` is the one-shot protocol: an
``.npz`` file per call each way. ``MMAP``, the default, places each array in
one file mapped by ``np.memmap`` in the session's temporary directory; the
worker maps it as ``np.ndarray(buffer=...)`` and copies nothing on the way in.
A ``multiprocessing.shared_memory`` transport was measured and dropped: ``MMAP``
beat it in 10 of 12 cells at 10 and 128 MB (#1288).

A :class:`Block` is one array's bytes. :func:`create` is the parent's side:
the parent creates every block, inputs and outputs alike, and
:meth:`Block.release` closes and unlinks it. :func:`attach` is the worker's
side: it maps a block the parent named and never unlinks one, so a worker
that dies leaves nothing behind it. Each block's ``spec`` is what crosses the
pipe: the transport, the block's name, its dtype and its shape.

:func:`admit` states what a session sends: C-contiguous ``float64``,
``int64`` or ``bool`` arrays, each read by its declared dtype and shape. It
refuses anything else before a call starts, under every transport, so a
call's inputs do not depend on the transport chosen.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class Transport(StrEnum):
    """How a session moves arrays between the parent and its worker."""

    #: An ``.npz`` file each way, the one-shot protocol.
    NPZ = "npz"
    #: One ``np.memmap`` file per array, in the session's temporary directory;
    #: the default, faster than ``NPZ`` in 13 of 15 measured cells and behind
    #: it in none by more than the host's run-to-run spread (#1288).
    MMAP = "mmap"


#: The dtypes a session sends; each is read back by its declared name.
ADMITTED = frozenset({np.dtype(np.float64), np.dtype(np.int64), np.dtype(np.bool_)})


def admit(inputs: Mapping[str, Any]) -> None:
    """Refuse any input that is not a C-contiguous ``float64``, ``int64`` or ``bool`` array.

    Raises :class:`TypeError` naming every refused input and what it is.
    """
    refused = []
    for name, array in inputs.items():
        if not isinstance(array, np.ndarray):
            refused.append(f"{name} is a {type(array).__name__}, not an ndarray")
        elif array.dtype not in ADMITTED:
            refused.append(f"{name} is {array.dtype}")
        elif not array.flags.c_contiguous:
            refused.append(f"{name} is not C-contiguous")
    if refused:
        allowed = "C-contiguous float64, int64 or bool"
        message = f"a session sends {allowed} arrays only: " + "; ".join(refused)
        raise TypeError(message)


class Block:
    """One array's bytes in a mapped file, and the view over them."""

    def __init__(
        self,
        spec: dict[str, Any],
        array: np.ndarray,
        medium: np.memmap | None,
        *,
        owner: bool,
    ) -> None:
        #: What crosses the pipe: ``transport``, ``handle``, ``dtype``, ``shape``.
        self.spec = spec
        #: The array, a view over the medium; ``None`` once released.
        self.array: np.ndarray | None = array
        self._medium = medium
        self._owner = owner

    def release(self) -> None:
        """Drop the view and unmap the medium; the parent also unlinks it."""
        self.array = None
        medium, self._medium = self._medium, None
        if medium is not None:
            del medium
            if self._owner:
                Path(self.spec["handle"]).unlink(missing_ok=True)


def _view(
    medium: np.memmap | None, dtype: np.dtype, shape: tuple[int, ...]
) -> np.ndarray:
    """``dtype`` and ``shape`` over the medium's bytes, copying none."""
    if medium is None:
        return np.empty(shape, dtype=dtype)
    return np.ndarray(shape, dtype=dtype, buffer=medium)


def create(
    transport: Transport,
    handle: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> Block:
    """Create, as the parent and owner, a block for one array.

    ``handle`` is the file path. An empty array needs no medium and crosses as
    its spec.
    """
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    medium: np.memmap | None = None
    if nbytes and transport is Transport.MMAP:
        medium = np.memmap(handle, dtype=np.uint8, mode="w+", shape=(nbytes,))
    elif nbytes:
        message = f"{transport} moves no arrays through blocks"
        raise ValueError(message)
    spec = {
        "transport": str(transport),
        "handle": handle if nbytes else None,
        "dtype": dtype.str,
        "shape": list(shape),
    }
    return Block(spec, _view(medium, dtype, shape), medium, owner=True)


def attach(spec: Mapping[str, Any]) -> Block:
    """Map, as the worker, the block ``spec`` names; it is never unlinked from here."""
    dtype = np.dtype(spec["dtype"])
    shape = tuple(int(n) for n in spec["shape"])
    handle = spec["handle"]
    medium: np.memmap | None = None
    if handle is not None:
        size = Path(handle).stat().st_size
        medium = np.memmap(handle, dtype=np.uint8, mode="r+", shape=(size,))
    return Block(dict(spec), _view(medium, dtype, shape), medium, owner=False)
