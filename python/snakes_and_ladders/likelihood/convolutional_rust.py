"""Rust log-MAP BCJR (``snakes_and_ladders.oxi_snakes_and_ladders.bcjr_forward_backward``),
pinned against :func:`snakes_and_ladders.likelihood.convolutional.bcjr`, the
NumPy oracle (``likelihood/CLAUDE.md``, "The reference implementation is the
oracle and it stays").

**Why this one is a port.** Root ``CLAUDE.md`` reserves the Rust backend for
CPU-bound hot paths, and the trellis recursion is sequential in the step:
the only axis NumPy can vectorize is the state, which the declared ``(7, 5)``
register makes four wide, so a pass is ``2 (K + m)`` NumPy calls on
four-element arrays. Issue #754's stress profile put ``bcjr`` at 95.8% of one
pass and 95.5% of an eight-iteration turbo decode, the highest fraction the
survey ranked; ``docs/experiments/020`` carries the ranking and ``024`` the
measurement that admitted the port.

**The boundary is crossed once and copies nothing.**
:class:`~snakes_and_ladders.sim.convolutional.Trellis` already holds
``next_state`` and ``parity`` as C-contiguous ``(n_states, 2)`` arrays, so
the flat form the kernel reads is a view and ``ascontiguousarray`` is free.
Nothing is cached on ``Trellis``: the marshalling is four ``reshape`` views
per call against ``O(K n_states)`` of arithmetic inside, which is the
recompute side of root ``CLAUDE.md``'s recompute-or-store decision, made
rather than defaulted.

**Bitwise on every register this repository declares, and the departure
above it is NumPy's reduction, not this one's arithmetic.** ``logaddexp`` in
``src/bcjr.rs`` is NumPy's ``npy_logaddexp`` branch for branch, the
log-sum-exp is ``numerics.logsumexp``'s shift by the row maximum, and the
branch metric is formed in the same order --- so agreement is **bitwise** at
``memory`` 1 and 2, the four-state ``(7, 5)`` register every fixture and
both benchmark lengths use. Past that the state reduction parts: NumPy's
pairwise sum switches to eight accumulators at eight terms and this one
stays left to right, which is a reassociation of a floating sum and nothing
else. Realized at ``memory`` 3 and 4 over ``K`` up to 1,024: **2.3e-13**
absolute and **2.2e-12** relative in the posterior and extrinsic ratios, the
log evidence still bitwise. That is inside
:data:`~snakes_and_ladders.likelihood.device.CROSS_DEVICE_RTOL_FLOAT64`,
which is the bound it is held to, and
``tests/regression/likelihood/test_convolutional_rust.py`` asserts the
equality and the bound separately rather than the looser one everywhere.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders import oxi_snakes_and_ladders
from snakes_and_ladders.likelihood.convolutional import TrellisDecoding
from snakes_and_ladders.sim.convolutional import Trellis


def bcjr(
    trellis: Trellis,
    systematic_llr: np.ndarray,
    parity_llr: np.ndarray,
    apriori_llr: np.ndarray,
    *,
    terminated: bool = True,
) -> TrellisDecoding:
    """One forward--backward pass over the trellis, computed in Rust.

    :func:`snakes_and_ladders.likelihood.convolutional.bcjr`'s arguments and
    its return, with ``apriori_llr`` required rather than optional: the
    caller there has already replaced ``None`` with zeros, and a second
    default would be a second place for it to be wrong.

    Parameters
    ----------
    trellis : Trellis
    systematic_llr, parity_llr, apriori_llr : np.ndarray
        Shape ``(T,)`` each, ``float64``, in ``eq:ldpc-llr``'s convention.
    terminated : bool
        Whether the register was driven back to the zero state.

    Returns
    -------
    TrellisDecoding
        The posterior and extrinsic ratios and the log evidence. Bitwise
        equal to the NumPy oracle's at ``memory`` 1 and 2 --- every declared
        register, the ``ci`` turbo fixture included --- and within
        **2.3e-13** absolute, **2.2e-12** relative at ``memory`` 3 and 4
        over ``K`` up to 1,024, the log evidence bitwise throughout. The
        module docstring says which reduction the last place is in.

    Raises
    ------
    ValueError
        If the ratio vectors disagree in length, or ``next_state`` names a
        state outside the trellis or is not a permutation in either input.
    """
    # `reshape(-1)` on a C-contiguous array is a view, so the trellis crosses
    # without a copy; `ascontiguousarray` is the guard that keeps `as_slice`
    # on the Rust side from failing on a strided caller.
    posterior, extrinsic, log_evidence = oxi_snakes_and_ladders.bcjr_forward_backward(
        np.ascontiguousarray(trellis.next_state, dtype=np.int64).reshape(-1),
        np.ascontiguousarray(trellis.parity, dtype=np.uint8).reshape(-1),
        np.ascontiguousarray(systematic_llr, dtype=np.float64),
        np.ascontiguousarray(parity_llr, dtype=np.float64),
        np.ascontiguousarray(apriori_llr, dtype=np.float64),
        terminated,
    )
    return TrellisDecoding(
        posterior_llr=np.asarray(posterior),
        extrinsic_llr=np.asarray(extrinsic),
        log_evidence=float(log_evidence),
    )
