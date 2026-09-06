"""Reading a declared fixture, once, for every model that has one.

Four loaders opened with the same four lines (issue #132) --- a frozen set of
required fields, ``yaml.safe_load``, a set difference, and an identical error
naming the file and the missing keys --- and differed only in the per-field
validation after it. That preamble lives here now; each model keeps its own
fields and its own checks, which are the parts that are genuinely different.

The module names no model, so `phylo.opt` may import it without acquiring an
application reference, on the same terms as :mod:`phylo.numerics` and
:mod:`phylo.enumeration`.

**Scale is part of a fixture's identity, not of its caller.** A size chosen
so an exact oracle stays available and a size chosen to show behaviour at
scale are different fixtures with different jobs, and the difference decides
which time budget a test using one falls under (`DEV.md`, CI & Performance
Budget). :class:`Scale` names that, so a test asks for the scale it needs and
the sizes live in one place rather than in each test's literals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class Scale(StrEnum):
    """Which time budget a fixture is sized for.

    ``CI`` is the size at which an exact oracle is still available and the
    per-pull-request suite stays inside its budget; ``STRESS`` is the size
    that demonstrates behaviour where the CI size cannot, and is run under
    the ``stress`` marker rather than on every pull request.

    The distinction is *what the size is for*, not how slow it happens to
    be: a slow test at an oracle size is still ``CI``, and a fast test whose
    size exists to show scaling is still ``STRESS``.
    """

    CI = "ci"
    STRESS = "stress"


def load_declared(path: Path, required: Iterable[str]) -> Mapping[str, Any]:
    """Read a fixture yaml and check that every required field is present.

    Parameters
    ----------
    path : Path
        The yaml file. Reported in every error, because a missing field is
        usually a mis-named file rather than a mis-written one.
    required : Iterable[str]
        Fields the caller cannot proceed without. Per-field validation stays
        with the caller: only presence is general.

    Returns
    -------
    Mapping[str, Any]
        The parsed mapping, unmodified.

    Raises
    ------
    ValueError
        If the file does not parse to a mapping, or a required field is
        absent. The mapping check is here rather than in each caller because
        an empty or list-valued yaml otherwise fails later as an
        ``AttributeError`` from inside a loader.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = pathlib.Path(directory) / "params.yaml"
    ...     _ = path.write_text("seed: 1\\nsites: 10\\n")
    ...     sorted(load_declared(path, ("seed", "sites")))
    ['seed', 'sites']
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        msg = f"{path}: expected a mapping of fields, got {type(raw).__name__}"
        raise ValueError(msg)

    missing = set(required) - raw.keys()
    if missing:
        msg = f"{path}: missing required field(s) {sorted(missing)}"
        raise ValueError(msg)

    return raw
