"""A factor graph's log-density at one assignment, compiled, and reproduced bitwise (issues #563, #1059).

:meth:`sal.sim.factor_graph.FactorGraph.log_density` is the oracle and
:mod:`sal.sample.gibbs` the caller; this is the ``numba`` twin, moved out of
``sample/kernels.py`` by #1059. It takes no exponential, so summing the same
terms in the same order is bitwise reproduction outright, and the pin needs
no bound. What it does need is that order -- floating-point addition is not
associative, so the kernel sums factors left to right in graph order as the
oracle does, and a vectorized sum would be a different number.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def factor_graph_log_density(
    state: np.ndarray,
    tables: np.ndarray,
    factor_start: np.ndarray,
    factor_offsets: np.ndarray,
    factor_column: np.ndarray,
    factor_stride: np.ndarray,
) -> float:
    """``sum_f log psi_f`` at one full assignment, over the edge layout.

    The density :meth:`sal.sim.factor_graph.FactorGraph.log_density`
    defines, read from the ``tables`` of
    :class:`sal.sample.gibbs._EdgeLayout` rather than from a table
    and a tuple key per factor: each factor's element is the offset its axes
    fix, and the sum runs left to right over factors in graph order.

    That order is the whole of the pin. Floating-point addition is not
    associative, so a pairwise or vectorized sum may move the last place of a
    recorded log-density; this accumulates the same terms in the same sequence
    as the oracle and reproduces it **bitwise**. No exponential is taken, so
    nothing here depends on which ``exp`` :func:`gibbs_sweep_sites` had to
    bound.

    **The pin is not a cost, which #651 established by measuring it.** Gathering
    the offsets and calling ``np.sum`` is *slower* at every size from a thousand
    factors to a million --- 0.60x to 0.77x, the extra array and its second pass
    costing more than fusing the accumulate saves --- so the ordered form is
    also the fast one and nothing is traded for the reproduction.
    ``tests/benchmarks/test_factor_density_sum_bench.py`` holds both.
    """
    total = 0.0
    for factor in range(factor_start.shape[0]):
        offset = factor_start[factor]
        for axis in range(factor_offsets[factor], factor_offsets[factor + 1]):
            offset += state[factor_column[axis]] * factor_stride[axis]
        total += tables[offset]
    return total
