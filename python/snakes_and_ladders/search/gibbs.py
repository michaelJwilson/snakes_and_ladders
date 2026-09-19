"""Deprecated: moved to :mod:`snakes_and_ladders.sample.gibbs` (issue #777).

Importing this module emits one :class:`DeprecationWarning` and re-exports
every public name the new one defines; the objects are the same objects,
which ``tests/regression/sample/test_sample_deprecated_paths.py`` asserts
with ``is``. The shim is removed in the release after 0.3.0, whose number
issue #522 names.
"""

from __future__ import annotations

import warnings

from snakes_and_ladders.sample.gibbs import (
    Annealed,
    AnnealedTopology,
    GibbsChain,
    GibbsMove,
    anneal_factor_graph,
    anneal_topology,
    balanced_sweep,
    cached_topology_score,
    chain_block_sweep,
    factor_autodiff_log_ratios,
    factor_taylor_log_ratios,
    gibbs_sweep,
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
    "anneal_factor_graph",
    "anneal_topology",
    "balanced_sweep",
    "cached_topology_score",
    "chain_block_sweep",
    "factor_autodiff_log_ratios",
    "factor_taylor_log_ratios",
    "gibbs_sweep",
    "sample_factor_graph",
    "topology_step",
]
