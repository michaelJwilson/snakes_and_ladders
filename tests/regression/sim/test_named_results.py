"""The `sim/` package's named results iterate in their declared order (issue #865).

Six entry points return a frozen dataclass where they returned a tuple, and
callers still unpack them, so ``__iter__`` must yield fields in declared
order; ``_Pruned`` and ``_RowEchelon`` are unpacked positionally too.
Sentinel values: this checks order; each module's tests check values.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from snakes_and_ladders.sim.convolutional import Encoded
from snakes_and_ladders.sim.count_pairs import ChannelCovariates
from snakes_and_ladders.sim.css import XError
from snakes_and_ladders.sim.graph import CompressedAdjacency, Endpoints
from snakes_and_ladders.sim.ldpc import _RowEchelon
from snakes_and_ladders.sim.topology import _Pruned

#: Every result type this package's part of the ticket names.
NAMED = [
    ChannelCovariates,
    CompressedAdjacency,
    Encoded,
    Endpoints,
    XError,
    _Pruned,
    _RowEchelon,
]


@pytest.mark.smoke
@pytest.mark.parametrize("result", NAMED, ids=lambda cls: cls.__name__)
def test_a_named_result_unpacks_to_its_fields_in_order(result: type) -> None:
    names = [field.name for field in fields(result)]
    instance = result(*names)

    assert list(instance) == names
    assert [getattr(instance, name) for name in names] == names
