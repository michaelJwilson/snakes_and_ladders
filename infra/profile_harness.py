"""A generic self-time profiling harness, with no application reference.

`select_tests.py` decides *what* CI runs; this decides *where a candidate
Rust port should look first* -- both read code to answer a question no
scientific model bears on, which is why both live in `infra/` rather than
beside `snakes_and_ladders.sim`, `snakes_and_ladders.search` or `snakes_and_ladders.learn`. Imported the same way
`select_tests.py` is: insert `infra/` onto ``sys.path`` and import it by its
bare module name (see ``tests/benchmarks/profile_hotpaths.py``, the caller).

``cProfile`` over ``py-spy``: the latter is not a repository dependency
(root `CLAUDE.md` requires explicit permission before adding one), and
``cProfile``'s deterministic instrumentation is sufficient to rank
functions by self time, which is all a Rust-port audit needs.
"""

from __future__ import annotations

import cProfile
import io
import pstats
from collections.abc import Callable


def self_time_ranking(
    fn: Callable[[], object], *, repeats: int = 1, top_n: int = 15
) -> str:
    """Profile ``fn`` and return its top ``top_n`` functions by self time.

    Parameters
    ----------
    fn : Callable[[], object]
        Zero-argument callable to profile. Its return value is discarded.
    repeats : int
        Number of times to call ``fn`` inside one profiling session, so a
        cheap call accumulates enough self time to rank reliably.
    top_n : int
        Number of rows to keep, ordered by descending self (``tottime``)
        time -- the statistic that ranks a function's own cost, excluding
        callees, which is what a port replaces.

    Returns
    -------
    str
        ``pstats``' formatted table, unmodified beyond the row limit.

    Raises
    ------
    ValueError
        If ``repeats`` is not positive.
    """
    if repeats < 1:
        msg = f"repeats must be positive, got {repeats}"
        raise ValueError(msg)

    profile = cProfile.Profile()
    profile.enable()
    for _ in range(repeats):
        fn()
    profile.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).sort_stats("tottime")
    stats.print_stats(top_n)
    return stream.getvalue()
