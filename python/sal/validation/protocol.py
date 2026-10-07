"""The file protocol's old import path, kept for one release (issue #1282).

:mod:`sal.external.protocol` holds the protocol; this module re-exports it,
as the packages #1010 split re-export what they held, so
``from sal.validation.protocol import dump`` still resolves. Import from
:mod:`sal.external.protocol`; this module goes with the next release.
"""

from __future__ import annotations

from sal.external.protocol import (
    PEAK_BYTES,
    SECONDS,
    dump,
    load,
    measured,
    paths,
    peaked,
    received,
    save,
    timed,
)

__all__ = [
    "PEAK_BYTES",
    "SECONDS",
    "dump",
    "load",
    "measured",
    "paths",
    "peaked",
    "received",
    "save",
    "timed",
]
