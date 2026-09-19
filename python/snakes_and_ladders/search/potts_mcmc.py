"""Deprecated: moved to :mod:`snakes_and_ladders.sample.potts_mcmc` (issue #777).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name the new one defines; the objects are the same objects,
which ``tests/regression/sample/test_sample_deprecated_paths.py`` asserts
with ``is``. The shim is removed in the release after 0.3.0, whose number
issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.sample.potts_mcmc import (
    AnnealedPotts,
    ClusterCounter,
    MoveKind,
    PottsChain,
    PottsMove,
    Recolour,
    TemperedChains,
    adapt_ladder_potts,
    anneal_potts,
    autodiff_log_ratios,
    energies,
    houdayer_cluster,
    niedermayer_threshold,
    parallel_tempering,
    sample_potts,
    sample_potts_pair,
    taylor_log_ratios,
    tempered,
)

warnings.warn(
    "snakes_and_ladders.search.potts_mcmc moved to snakes_and_ladders.sample.potts_mcmc "
    "(issue #777); this shim is removed in the release after 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AnnealedPotts",
    "ClusterCounter",
    "MoveKind",
    "PottsChain",
    "PottsMove",
    "Recolour",
    "TemperedChains",
    "adapt_ladder_potts",
    "anneal_potts",
    "autodiff_log_ratios",
    "energies",
    "houdayer_cluster",
    "niedermayer_threshold",
    "parallel_tempering",
    "sample_potts",
    "sample_potts_pair",
    "taylor_log_ratios",
    "tempered",
]
