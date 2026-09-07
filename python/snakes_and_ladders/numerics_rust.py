"""Rust inverse-CDF categorical sampling (`snakes_and_ladders.oxi_snakes_and_ladders.sample_rows`) --
pinned against :func:`snakes_and_ladders.numerics.sample_rows`, the NumPy oracle.

Issue #181's audit measured `sample_rows` at 94-96% of
`simulate_alignment`'s self time, at both a CI-sized and a larger fixture,
and found nothing else in `sim`, `search` or `learn` both that hot
end-to-end and free of an autodiff dependency. This is the port (issue
#187).

The oracle stays, and not only as a courtesy to root ``CLAUDE.md``'s rule
that deleting the slow path removes the only thing saying the fast path is
right. Because the uniforms are drawn here and handed to Rust, the two
implementations differ in *arithmetic alone* -- so the comparison between
them is exact equality rather than a tolerance, and a regression that
changed a single index would be visible.

**The generator does not cross the boundary.** ``snakes_and_ladders.sim``'s
reproducibility contract is that a seeded ``numpy.random.Generator``
determines the alignment; a second stream inside Rust would break that
without failing anything. So this module draws ``rng.random`` exactly as the
oracle does, in the same order and the same count, and Rust performs only
the lookup.
"""

from __future__ import annotations

import numpy as np

from snakes_and_ladders import oxi_snakes_and_ladders


def sample_rows(
    rng: np.random.Generator, distributions: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Draw one categorical index per entry of ``rows``, from the row it selects.

    Signature-compatible with :func:`snakes_and_ladders.numerics.sample_rows`, and
    bit-identical to it for the same generator state.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded generator. One value is drawn per entry of ``rows``, in order,
        matching the oracle's consumption exactly -- a caller may swap the
        two implementations without its stream diverging.
    distributions : np.ndarray
        Row-stochastic, shape ``(n_rows, n_categories)``. Rows need only sum
        to 1 within rounding; the last cumulative column is clamped to 1, as
        in the oracle.
    rows : np.ndarray
        Which row each draw comes from, shape ``(n_draws,)``, entries in
        ``[0, n_rows)``.

    Returns
    -------
    np.ndarray
        Sampled category per entry, shape ``(n_draws,)``, dtype ``int64``.

    Raises
    ------
    ValueError
        If ``distributions`` is not 2-D, matching the oracle, or if Rust
        rejects a row index outside the distribution count.
    """
    if distributions.ndim != 2:
        msg = (
            f"expected distributions of shape (n_rows, n_categories), got "
            f"{distributions.shape}"
        )
        raise ValueError(msg)

    draws = rng.random(size=(int(rows.shape[0]),))
    # Every array is borrowed rather than copied, and the result is written
    # into `sampled` rather than returned. The boundary, not the kernel,
    # decided whether this port was worth having: passing lists cost 64 ms
    # marshalling in and 125 ms back at two million draws against a NumPy
    # oracle that finishes in 86 ms, and copying through the buffer protocol
    # still left the caller at 1.3x where the kernel runs at 3.6x. Borrowing
    # (issue #202) puts it at 2.6x, and stops the ratio shrinking as the
    # array grows.
    #
    # `ascontiguousarray` is what makes the borrow safe: Rust's `as_slice`
    # accepts only a C-contiguous array, and this is free when the input
    # already is one.
    sampled = np.empty(int(rows.shape[0]), dtype=np.int64)
    oxi_snakes_and_ladders.sample_rows(
        np.ascontiguousarray(distributions, dtype=np.float64).reshape(-1),
        int(distributions.shape[1]),
        np.ascontiguousarray(rows, dtype=np.int64),
        draws,
        sampled,
    )
    return sampled
