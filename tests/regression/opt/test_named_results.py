"""`opt/`'s named results iterate in their declared field order (issue #865).

Three entry points return a frozen dataclass where they returned a tuple, and
callers still unpack them, so ``__iter__`` must yield fields in declared
order. Sentinel values: this checks order; each module's tests check values.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from sal.opt.hmm import CategoricalFit
from sal.opt.potts import GraphStatistics
from sal.opt.testfunctions import NearestMinimum

#: Every result type this package's share of the ticket names, one per
#: entry point: no two of the three report the same fields.
NAMED = [CategoricalFit, GraphStatistics, NearestMinimum]


@pytest.mark.smoke
@pytest.mark.parametrize("result", NAMED, ids=lambda cls: cls.__name__)
def test_a_named_result_unpacks_to_its_fields_in_order(result: type) -> None:
    names = [field.name for field in fields(result)]
    instance = result(*names)

    assert list(instance) == names
    assert [getattr(instance, name) for name in names] == names
