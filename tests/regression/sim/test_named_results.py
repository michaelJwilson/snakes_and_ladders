"""The `sim/` package's named results iterate in their declared order (issue #865).

The six entry points the ticket names return a frozen dataclass where they
returned a bare tuple, and every caller that unpacks one is left unpacking.
What makes that safe is a single property: ``__iter__`` yields the fields in
the order they are declared, which is the order the tuple had. It is asserted
here over the types themselves, so a field inserted in the middle without the
matching line in ``__iter__`` fails here rather than at whichever call site
unpacks next.

The two private types are held to the same property: ``_Pruned`` is unpacked
by a test and ``_RowEchelon`` is what its callers reach for positionally
elsewhere, so their order is as load-bearing as a public one's.

The values are sentinels: this checks the order, and the arithmetic that
fills the fields is checked by the module's own tests.
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
