"""The package's named results iterate in their declared field order (issue #865).

Thirteen entry points return a frozen dataclass where they returned a tuple,
and callers still unpack them, so ``__iter__`` must yield fields in declared
order. Sentinel values: this checks order; each module's tests check values.
``Posteriors`` is defined in ``opt`` since #1166 and is checked there; this
file checks the re-export is the same class.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
from sal.likelihood.blocks import Extremes
from sal.likelihood.distance import DistanceMatrix, TreeDistances
from sal.likelihood.features import Adjacency
from sal.likelihood.forward_backward import StepKernels
from sal.likelihood.hadamard import Spectrum
from sal.likelihood.message_passing import _Run
from sal.likelihood.message_passing.rust import TreeMessages
from sal.likelihood.ragged import Paths, Posteriors
from sal.likelihood.schedule import UpwardMessageSchedule
from sal.likelihood.spatio_sequential.rust import (
    EmissionRows,
    EmissionTables,
)
from sal.likelihood.surrogate import EnergyBounds
from sal.opt.hmm import forward

#: Every result type the ticket names that ``likelihood`` defines, including
#: the private one `_run` returns: it is unpacked by its two callers and by a test, so its order is
#: as load-bearing as a public one's.
NAMED = [
    Adjacency,
    DistanceMatrix,
    EmissionRows,
    EmissionTables,
    EnergyBounds,
    Extremes,
    Paths,
    Spectrum,
    StepKernels,
    TreeDistances,
    TreeMessages,
    _Run,
]


@pytest.mark.smoke
def test_a_named_result_unpacks_to_its_fields_in_order() -> None:
    for result in NAMED:
        names = [field.name for field in fields(result)]
        instance = result(*names)

        assert list(instance) == names, result.__name__
        assert [getattr(instance, name) for name in names] == names, result.__name__


@pytest.mark.smoke
def test_the_ragged_posteriors_are_the_type_opt_defines() -> None:
    # One class under two import paths (#1166): a second definition would
    # pass every field check and fail an `isinstance` across the seam. The
    # module names a problem, so the kind is `smoke`, not `infra` (#729).
    assert Posteriors is forward.Posteriors
    assert Posteriors.__module__ == "sal.opt.hmm.forward"


@pytest.mark.smoke
def test_the_upward_schedule_refuses_a_scale_it_was_not_given() -> None:
    # `_run` reports no scale on the compiled route, which accumulates none.
    # The upward schedule is the one that reads it, is not compiled, and so
    # is never handed a `None`; it says so rather than adding to it.
    with pytest.raises(ValueError, match="does not accumulate"):
        # The layout and the messages are never reached: the refusal is first.
        UpwardMessageSchedule().log_partition(None, None, 0.0, None)  # type: ignore[arg-type]
