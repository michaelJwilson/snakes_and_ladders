"""Which implementation runs a kernel, chosen by the caller and never silently.

Root ``CLAUDE.md`` keeps every accelerated kernel beside its pure Python or
NumPy oracle, and its backend rule admits a compiled path only against a
measurement on an existing hot path. The enum is the seam that makes the
choice explicit at the call site: a test that pins a kernel against its
oracle names both, and a caller that wants the oracle can say so.

Two compiled backends exist here for two different reasons, and the split is
deliberate rather than a stack. **Rust** carries the sampling sweep, because
a sampler's agreement with its oracle is distributional -- ``f64::exp`` and
NumPy's differ in the last place and one draw across a moved threshold sends
two chains apart -- so that port lives behind an opt-in and is refereed by
the distribution it converges to. **Numba** carries the deterministic
kernels -- descent, energy evaluation -- whose output is an integer labelling
or a sum that the oracle reproduces *bitwise*, so the pin is exact and the
compiled path can be the default without moving a single committed number.
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
