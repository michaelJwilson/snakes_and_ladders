"""Which implementation runs a kernel, chosen by the caller and never silently.

Root ``CLAUDE.md`` keeps every accelerated kernel beside its pure Python or
NumPy oracle, and its backend rule admits a compiled path only against a
measurement on an existing hot path. The enum is the seam that makes the
choice explicit at the call site: a test that pins a kernel against its
oracle names both, and a caller that wants the oracle can say so.

Two compiled backends exist here for two different reasons, and the split is
deliberate rather than a stack. **Rust** carries the sampling sweep, where
the loop is over an adjacency structure and there is no array arithmetic for
NumPy to vectorize. **Numba** carries the kernels whose arithmetic is
NumPy's operation for operation -- descent, energy evaluation, the Gibbs
sweep.

Both are defaults, and on the same terms. ``f64::exp`` and NumPy's differ in
the last place, and ``searchsorted`` is a threshold, so a sampling port is
distributional unless something bounds that: each compiled sweep decides a
site only where the draw clears every cumulative boundary by more than the
two exponentials can move it, and hands the rest back. The pin is then exact
and the compiled path is the default without a committed number moving
(issues #561, #599).
"""

from __future__ import annotations

from enum import StrEnum


class Backend(StrEnum):
    """Where a kernel runs."""

    PYTHON = "python"
    """The pure Python or NumPy oracle."""

    NUMBA = "numba"
    """A ``numba.njit`` kernel over the compressed-row adjacency."""

    RUST = "rust"
    """The ``oxi_snakes_and_ladders`` extension."""
