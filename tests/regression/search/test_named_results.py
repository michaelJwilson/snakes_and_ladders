"""`search/`'s named results iterate in their declared field order (issue #865).

Nine entry points in this package returned a bare tuple and now return a
frozen dataclass, and every caller that unpacks one is left unpacking. What
makes that safe is a single property: ``__iter__`` yields the fields in the
order they are declared, which is the order the tuple had. It is asserted
here over the types themselves, so a field inserted in the middle without
the matching line in ``__iter__`` fails here rather than at whichever call
site unpacks next.

The values are sentinels: this checks the order, and the arithmetic that
fills the fields is checked by the module's own tests.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from snakes_and_ladders.search.alpha_expansion import Labelling
from snakes_and_ladders.search.ground_state import Bracket, RungField
from snakes_and_ladders.search.max_cut import MaxCut
from snakes_and_ladders.search.maxflow import GroundState, PairedArcs

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
