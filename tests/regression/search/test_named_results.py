"""`search/`'s named results iterate in their declared field order (issue #865).

Nine entry points return a frozen dataclass where they returned a tuple, and
callers still unpack them, so ``__iter__`` must yield fields in declared
order. Sentinel values: this checks order; each module's tests check values.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from sal.search.alpha_expansion import Labelling
from sal.search.ground_state import Bracket, RungField
from sal.search.max_cut import MaxCut
from sal.search.maxflow import GroundState, PairedArcs

#: Every result type this package's share of the ticket names. `Labelling`
#: carries three entry points and `GroundState` two, so the count here is
#: six against nine functions: a type is reused wherever two of them report
#: the same fields with the same meaning.
NAMED = [Bracket, GroundState, Labelling, MaxCut, PairedArcs, RungField]


@pytest.mark.smoke
@pytest.mark.parametrize("result", NAMED, ids=lambda cls: cls.__name__)
def test_a_named_result_unpacks_to_its_fields_in_order(result: type) -> None:
    names = [field.name for field in fields(result)]
    instance = result(*names)

    assert list(instance) == names
    assert [getattr(instance, name) for name in names] == names
