"""The package's named results iterate in their declared field order (issue #865).

The `sample/` half: entry points return a frozen dataclass where they returned
a tuple, and callers still unpack them, so ``__iter__`` must yield fields in
declared order. Sentinel values: this checks order; each module's tests check values.
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
