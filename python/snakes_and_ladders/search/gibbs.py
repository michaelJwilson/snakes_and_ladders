"""Deprecated: moved to :mod:`snakes_and_ladders.sample.gibbs` (issue #777).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name from the new one; the objects are the same objects, which
``tests/regression/sample/test_sample_deprecated_paths.py`` asserts with ``is``.
The shim is removed in the release after 0.3.0, whose number issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.sample.gibbs import (
    Annealed,
    AnnealedTopology,
    GibbsChain,
    GibbsMove,
    TempSchedule,
    anneal_factor_graph,
    anneal_topology,
    balanced_sweep,
    cached_topology_score,
    chain_block_sweep,
    draw_change,
    factor_autodiff_log_ratios,
    factor_taylor_log_ratios,
    gibbs_sweep,
    log_balanced_weights,
    log_metropolis_ratio,
    log_normalizer,
    log_ratios,
    sample_factor_graph,
    topology_step,
)

warnings.warn(
    "snakes_and_ladders.search.gibbs moved to snakes_and_ladders.sample.gibbs "
    "(issue #777); this shim is removed in the release after 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Annealed",
    "AnnealedTopology",
    "GibbsChain",
    "GibbsMove",
    "TempSchedule",
    "anneal_factor_graph",
    "anneal_topology",
    "balanced_sweep",
    "cached_topology_score",
    "chain_block_sweep",
    "draw_change",
    "factor_autodiff_log_ratios",
    "factor_taylor_log_ratios",
    "gibbs_sweep",
    "log_balanced_weights",
    "log_metropolis_ratio",
    "log_normalizer",
    "log_ratios",
    "sample_factor_graph",
    "topology_step",
]
