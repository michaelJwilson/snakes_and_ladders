"""Deprecated: moved to :mod:`snakes_and_ladders.sample.hmc` (issue #777).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name from the new one; the objects are the same objects, which
``tests/regression/sample/test_sample_deprecated_paths.py`` asserts with ``is``.
The shim is removed in the release after 0.3.0, whose number issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.sample.hmc import (
    Adaptation,
    Adapted,
    Annealed,
    HmcChain,
    Integrator,
    Kernel,
    TempSchedule,
    Tempered,
    WithGaussianPrior,
    anneal,
    effective_sample_size,
    hamiltonian,
    leapfrog,
    parallel_tempering,
    sample,
    yoshida,
)

warnings.warn(
    "snakes_and_ladders.opt.hmc moved to snakes_and_ladders.sample.hmc "
    "(issue #777); this shim is removed in the release after 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Adaptation",
    "Adapted",
    "Annealed",
    "HmcChain",
    "Integrator",
    "Kernel",
    "TempSchedule",
    "Tempered",
    "WithGaussianPrior",
    "anneal",
    "effective_sample_size",
    "hamiltonian",
    "leapfrog",
    "parallel_tempering",
    "sample",
    "yoshida",
]
