"""Reading a declared fixture, once, for every model that has one.

Four loaders opened with the same four lines (issue #132) --- a frozen set of
required fields, ``yaml.safe_load``, a set difference, and an identical error
naming the file and the missing keys --- and differed only in the per-field
validation after it. That preamble lives here now; each model keeps its own
fields and its own checks, which are the parts that are genuinely different.

The module names no model, so `snakes_and_ladders.opt` may import it without acquiring an
application reference, on the same terms as :mod:`snakes_and_ladders.numerics` and
:mod:`snakes_and_ladders.enumeration`.

**Scale is part of a fixture's identity, not of its caller.** A size chosen
so an exact oracle stays available and a size chosen to show behaviour at
scale are different fixtures with different jobs, and the difference decides
which time budget a test using one falls under (`DEV.md`, CI & Performance
Budget). :class:`Scale` names that, so a test asks for the scale it needs and
the sizes live in one place rather than in each test's literals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self, TypeVar

import yaml


class Scale(StrEnum):
    """Which time budget a fixture is sized for.

    ``CI`` is the size at which an exact oracle is still available and the
    per-pull-request suite stays inside its budget; ``STRESS`` is the size
    that demonstrates behaviour where the CI size cannot, and is run under
    the ``stress`` marker rather than on every pull request; ``RELEASE`` is
    the size only the release gate pays for. The three are `DEV.md`'s tiers,
    and are the tier a fixture file is named for
    (:mod:`snakes_and_ladders.sim.fixtures`).

    The distinction is *what the size is for*, not how slow it happens to
    be: a slow test at an oracle size is still ``CI``, and a fast test whose
    size exists to show scaling is still ``STRESS``.
    """

    CI = "ci"
    STRESS = "stress"
    RELEASE = "release"


@dataclass(frozen=True)
class BinInstance:
    """One coarser instance a fixture declares: a factor, what it counts, and its tier.

    The declaration is the same yaml block for every model that carries one
    --- ``bin: [{factor: 5, marker: key}]`` --- and it was read into two
    classes with two validations (issue #862). What differed is the only
    thing that varies with the model, so it is a field: **what one bin
    holds**. A spatio-sequential fixture sums ``factor`` consecutive
    positions, because the positions are coupled and a sum is the coarser
    observation; a mixture's observations are independent, so there is
    nothing to sum and the coarse instance is ``n_samples // factor`` draws
    from the same parameters, distributed exactly as a subsample of the fine
    draw.

    Parameters
    ----------
    factor : int
        Units of the fine instance per unit of this one, ``>= 1``. ``1`` is
        the fine instance the file declares.
    unit : str
        What one bin holds --- ``draw`` or ``position`` --- naming the
        reduction and reported when a factor is refused.
    marker : str
        The tier the full test at this factor runs in, measured rather than
        assumed (``DEV.md``, CI & Performance Budget). Which markers a model
        admits is that model's own check, since a fixture with one instance
        per tier and one naming a key instance declare different sets.

    Raises
    ------
    ValueError
        If the factor is below one.
    """

    factor: int
    unit: str
    marker: str

    def __post_init__(self) -> None:
        if self.factor < 1:
            msg = f"a bin holds at least one {self.unit}, got {self.factor}"
            raise ValueError(msg)


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


class Params(Protocol):
    """A fixture's declared truth, built from the mapping its yaml declares.

    Sixteen loaders shared one signature, one preamble and sixteen bodies
    whose only difference was the per-field validation (issue #832). The
    validation is each dataclass's own now, in :meth:`from_declared`, and the
    preamble is :func:`load_params`, once. The seam earns its place by the
    rule in the calling review: sixteen implementers, three consumers.
    """

    #: The fields :func:`load_params` checks are present before it calls
    #: :meth:`from_declared`; the per-field validation stays with the class.
    required_fields: ClassVar[frozenset[str]]

    @classmethod
    def from_declared(cls, declared: Mapping[str, Any], path: Path, /) -> Self:
        """Validate ``declared`` field by field and build the truth.

        ``path`` is the file ``declared`` was read from, named in every error
        so a bad value is found in the file and not in the loader.
        """
        ...


P = TypeVar("P", bound=Params)


def load_params(path: Path, kind: type[P]) -> P:
    """Read a fixture yaml and build its declared truth as ``kind``.

    Parameters
    ----------
    path : Path
        The yaml file, reported in every error.
    kind : type[Params]
        Which truth the file declares: the dataclass whose
        :attr:`~Params.required_fields` are checked and whose
        :meth:`~Params.from_declared` validates the rest.

    Returns
    -------
    P
        The parsed, validated truth, equal record for record to what the
        loader it replaced returned (issue #832).

    Raises
    ------
    ValueError
        From :func:`load_declared` if the file does not parse to a mapping or
        a required field is absent, and from ``kind.from_declared`` on its own
        terms.
    """
    return kind.from_declared(load_declared(path, kind.required_fields), path)
