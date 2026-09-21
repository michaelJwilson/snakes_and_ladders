"""The package's named results iterate in their declared field order (issue #865).

Each of the thirteen entry points listed on the ticket returns a frozen
dataclass where it returned a bare tuple, and every caller that unpacks one
is left unpacking. What makes that safe is a single property: ``__iter__``
yields the fields in the order they are declared, which is the order the
tuple had. It is asserted here over the types themselves, so a field
inserted in the middle without the matching line in ``__iter__`` fails here
rather than at whichever call site unpacks next.

The values are sentinels: this checks the order, and the arithmetic that
fills the fields is checked by the module's own tests.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from snakes_and_ladders.likelihood.blocks import Extremes
from snakes_and_ladders.likelihood.distance import DistanceMatrix, TreeDistances
from snakes_and_ladders.likelihood.features import Adjacency
from snakes_and_ladders.likelihood.forward_backward import StepKernels
from snakes_and_ladders.likelihood.hadamard import Spectrum
from snakes_and_ladders.likelihood.message_passing import _Run
from snakes_and_ladders.likelihood.message_passing_rust import TreeMessages
from snakes_and_ladders.likelihood.ragged_rust import Posteriors
from snakes_and_ladders.likelihood.schedule import UpwardMessageSchedule
from snakes_and_ladders.likelihood.spatio_sequential_rust import (
    EmissionRows,
    EmissionTables,
)
from snakes_and_ladders.likelihood.surrogate import EnergyBounds

#: Every result type the ticket names, including the private one `_run`
#: returns: it is unpacked by its two callers and by a test, so its order is
#: as load-bearing as a public one's.
NAMED = [
    Adjacency,
    DistanceMatrix,
    EmissionRows,
    EmissionTables,
    EnergyBounds,
    Extremes,
    Posteriors,
    Spectrum,
    StepKernels,
    TreeDistances,
    TreeMessages,
    _Run,
]


@pytest.mark.smoke
@pytest.mark.parametrize("result", NAMED, ids=lambda cls: cls.__name__)
def test_a_named_result_unpacks_to_its_fields_in_order(result: type) -> None:
    names = [field.name for field in fields(result)]
    instance = result(*names)

    assert list(instance) == names
    assert [getattr(instance, name) for name in names] == names


@pytest.mark.smoke
def test_the_upward_schedule_refuses_a_scale_it_was_not_given() -> None:
    # `_run` reports no scale on the compiled route, which accumulates none.
    # The upward schedule is the one that reads it, is not compiled, and so
    # is never handed a `None`; it says so rather than adding to it.
    with pytest.raises(ValueError, match="does not accumulate"):
        # The layout and the messages are never reached: the refusal is first.
        UpwardMessageSchedule().log_partition(None, None, 0.0, None)  # type: ignore[arg-type]
