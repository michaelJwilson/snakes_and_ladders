"""The factor-graph heat-bath sweep, compiled, and reproduced bitwise by its oracle (issues #561, #1059).

:func:`sal.sample.gibbs.gibbs_sweep` is the gateway and the NumPy oracle;
this is its ``numba`` twin, moved out of ``sample/kernels.py`` by #1059.

That exactness is why this is ``numba`` rather than Rust. Root
``CLAUDE.md``'s backend rule admits one compiled path per measurement; the
Rust extension already carries the Potts *sampling* sweep
(:func:`sal.sample.potts_mcmc.sample_potts` on the extension), and a second copy of it
is what the rule exists to refuse. A kernel whose pin is exact carries no
such cost.

A heat-bath draw exponentiates, and ``libm``'s ``exp`` and NumPy's disagree
in the last place on 4.6% of ``float64`` inputs, so the arithmetic alone
would make it a distributional port. What keeps the pin exact is that
:func:`gibbs_sweep_sites` decides a site only where the draw clears every
cumulative boundary by more than the two exponentials can move it, and hands
the rest back; a bound, not an assumption about rounding.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def gibbs_sweep_sites(
    state: np.ndarray,
    draws: np.ndarray,
    cardinality: np.ndarray,
    tables: np.ndarray,
    entry_offsets: np.ndarray,
    entry_start: np.ndarray,
    entry_stride: np.ndarray,
    term_offsets: np.ndarray,
    term_column: np.ndarray,
    term_stride: np.ndarray,
    local: np.ndarray,
    beta: float,
    guard: float,
    first: int,
) -> int:
    """Heat-bath updates from ``first`` on, in place; returns where it stopped.

    The update :func:`sal.sample.gibbs.gibbs_sweep` states, over the
    edge layout :class:`sal.sample.gibbs._EdgeLayout` builds: the
    variable's conditional is gathered from ``tables`` by the entries that
    touch it, scaled by ``beta``, shifted by its maximum, exponentiated into a
    running cumulative sum, and searched for ``draws[position]`` times that
    sum's last entry.

    Every step but one is the arithmetic NumPy performs, operation for
    operation, so it is reproduced bitwise. The exception is ``exp``, which
    NumPy computes by its own SIMD polynomial and this kernel by ``libm``:
    the two disagree in the last place on 4.6% of ``float64`` inputs
    (measured, 2e6 draws), and one draw across a moved threshold sends two
    chains apart. So the kernel decides a site only where the decision cannot
    depend on that last place -- where the draw clears every cumulative
    boundary by ``slack``, which bounds the difference the two exponentials
    can make to the boundary and to the draw together. Where it does not,
    the kernel returns that position without touching ``state``, and the
    NumPy path decides that one site. That is what makes the compiled sweep
    the *same chain* rather than a chain of the same law, and it is why this
    kernel is the default. Issue #599 has since carried the same
    construction into the Rust sampling sweep, which is the default too
    (:mod:`sal.backend`).

    ``guard`` sets that boundary in units of the last place, per state:
    :func:`sal.sample.gibbs.gibbs_sweep` passes the bound it derives,
    and a test raises it to drive every site onto the NumPy path and pin the
    handing back itself, which no realistic draw reaches.

    Returns
    -------
    int
        ``state.shape[0]`` if every remaining site was decided, otherwise the
        position of the first site it declined to decide.
    """
    n_variables = int(state.shape[0])
    for position in range(first, n_variables):
        n_states = cardinality[position]
        for label in range(n_states):
            local[label] = 0.0
        for entry in range(entry_offsets[position], entry_offsets[position + 1]):
            base = entry_start[entry]
            for term in range(term_offsets[entry], term_offsets[entry + 1]):
                base += state[term_column[term]] * term_stride[term]
            stride = entry_stride[entry]
            for label in range(n_states):
                local[label] += tables[base + label * stride]
        top = local[0] * beta
        for label in range(n_states):
            local[label] = local[label] * beta
            top = max(top, local[label])
        running = 0.0
        for label in range(n_states):
            running += np.exp(local[label] - top)
            local[label] = running
        target = draws[position] * running
        # 2**-52, the largest relative gap between neighbouring float64 values.
        slack = guard * n_states * 2.220446049250313e-16 * running
        chosen = -1
        for label in range(n_states):
            gap = local[label] - target
            if not (gap > slack or -gap > slack):
                return position
            if chosen < 0 and gap >= 0.0:
                chosen = label
        if chosen < 0:
            return position
        state[position] = chosen
    return n_variables
