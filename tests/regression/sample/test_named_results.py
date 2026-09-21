"""The package's named results iterate in their declared field order (issue #865).

The `sample/` half of the ticket: each entry point that returned a bare
tuple returns a frozen dataclass, and every caller that unpacks one is left
unpacking. What makes that safe is a single property --- ``__iter__`` yields
the fields in the order they are declared, which is the order the tuple had
--- asserted here over the types themselves, so a field inserted in the
middle without the matching line in ``__iter__`` fails here rather than at
whichever call site unpacks next.

The values are sentinels: this checks the order, and the arithmetic that
fills the fields is checked by the module's own tests.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from snakes_and_ladders.sample.balanced import Change
from snakes_and_ladders.sample.gibbs import TopologyStep
from snakes_and_ladders.sample.hmc import Chain, PhaseSpace, Transition
from snakes_and_ladders.sample.potts_mcmc import Clusters, PottsPair, TemperedModel

#: Every result type this package names. :class:`Transition` covers both
#: kernels: `langevin._LangevinKernel` returns the one `hmc.Kernel` declares
#: rather than a second type of its own.
NAMED = [
    Chain,
    Change,
    Clusters,
    PhaseSpace,
    PottsPair,
    TemperedModel,
    TopologyStep,
    Transition,
]


@pytest.mark.smoke
@pytest.mark.parametrize("result", NAMED, ids=lambda cls: cls.__name__)
def test_a_named_result_unpacks_to_its_fields_in_order(result: type) -> None:
    names = [field.name for field in fields(result)]
    instance = result(*names)

    assert list(instance) == names
    assert [getattr(instance, name) for name in names] == names
