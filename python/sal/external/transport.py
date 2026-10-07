"""How a session's arrays reach its worker and come back (issue #1282, step 2b).

:class:`Transport` names the three ways. ``NPZ`` is the one-shot protocol: an
``.npz`` file per call each way. ``SHARED`` places each array in one named
``multiprocessing.shared_memory`` block, and ``MMAP`` in one file mapped by
``np.memmap`` in the session's temporary directory; the worker maps either as
``np.ndarray(buffer=...)`` and copies nothing on the way in.

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

import sys
from collections.abc import Mapping
from contextlib import suppress
from enum import StrEnum
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np


class Transport(StrEnum):
    """How a session moves arrays between the parent and its worker."""

    #: An ``.npz`` file each way, the one-shot protocol; the default.
    NPZ = "npz"
    #: One ``multiprocessing.shared_memory`` block per array.
    SHARED = "shared"
    #: One ``np.memmap`` file per array, in the session's temporary directory.
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
    """One array's bytes in a shared medium, and the view over them."""

    def __init__(
        self,
        spec: dict[str, Any],
        array: np.ndarray,
        medium: SharedMemory | np.memmap | None,
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
        if isinstance(medium, SharedMemory):
            # A script that kept a view keeps the mapping until it exits.
            with suppress(BufferError):
                medium.close()
            if self._owner:
                medium.unlink()
        elif medium is not None:
            del medium
            if self._owner:
                Path(self.spec["handle"]).unlink(missing_ok=True)


def _view(
    medium: SharedMemory | np.memmap | None, dtype: np.dtype, shape: tuple[int, ...]
) -> np.ndarray:
    """``dtype`` and ``shape`` over the medium's bytes, copying none."""
    if medium is None:
        return np.empty(shape, dtype=dtype)
    buffer = medium.buf if isinstance(medium, SharedMemory) else medium
    return np.ndarray(shape, dtype=dtype, buffer=buffer)


def create(
    transport: Transport,
    handle: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> Block:
    """Create, as the parent and owner, a block for one array.

    ``handle`` is the shared-memory name under ``SHARED`` and the file path
    under ``MMAP``. An empty array needs no medium and crosses as its spec.
    """
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    medium: SharedMemory | np.memmap | None = None
    if nbytes and transport is Transport.SHARED:
        medium = SharedMemory(name=handle, create=True, size=nbytes)
    elif nbytes and transport is Transport.MMAP:
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


def _attached(name: str) -> SharedMemory:
    """Map an existing block without registering it for removal at exit.

    Python 3.12 registers an attached block with the attaching process's
    resource tracker, which unlinks it when that process exits; the parent
    owns the block, so the worker's registration is suppressed (3.13 names
    this ``track=False``).
    """
    if sys.version_info >= (3, 13):
        return SharedMemory(name=name, track=False)  # type: ignore[call-arg,unused-ignore]
    register = resource_tracker.register
    resource_tracker.register = lambda *_: None
    try:
        return SharedMemory(name=name)
    finally:
        resource_tracker.register = register


def attach(spec: Mapping[str, Any]) -> Block:
    """Map, as the worker, the block ``spec`` names; it is never unlinked from here."""
    dtype = np.dtype(spec["dtype"])
    shape = tuple(int(n) for n in spec["shape"])
    handle = spec["handle"]
    medium: SharedMemory | np.memmap | None = None
    if handle is not None and spec["transport"] == Transport.SHARED:
        medium = _attached(handle)
    elif handle is not None:
        size = Path(handle).stat().st_size
        medium = np.memmap(handle, dtype=np.uint8, mode="r+", shape=(size,))
    return Block(dict(spec), _view(medium, dtype, shape), medium, owner=False)
