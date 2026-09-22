"""Regression tests for snakes_and_ladders.qa.starts_table (issue #891).

The layout is pinned, not the rendering: one row per start in the order
given, the gap and the seconds formatted as `sim_problem_sizes` formats a
number, and a caption the LaTeX check accepts.
"""

from __future__ import annotations

import pytest
from snakes_and_ladders.qa.figure import check_latex_safe
from snakes_and_ladders.qa.starts_table import build_table
from snakes_and_ladders.search.mixture_starts import SolverComparison


def _row(
    start: str, gaps: tuple[float, ...], *, deterministic: bool
) -> SolverComparison:
    return SolverComparison(
        start=start,
        deterministic=deterministic,
        seeded=tuple(-1.0 for _ in gaps),
        reached=tuple(-1.0 for _ in gaps),
        gap=gaps,
        passes=0.0,
        state_bytes=320,
        seconds=tuple(2.0 + gap for gap in gaps),
        seeding_seconds=tuple(0.1 for _ in gaps),
        recovery=tuple(0.5 for _ in gaps),
        mean_error=tuple(0.1 for _ in gaps),
    )


@pytest.mark.smoke
def test_the_table_has_one_row_per_start_in_order_and_a_safe_caption() -> None:
    rows = [
        _row("emission++", (1.25,), deterministic=False),
        _row("burn-in", (1.0, 3.0), deterministic=False),
        _row("quantile", (-2.0,), deterministic=True),
    ]

    body, caption = build_table(rows, instance="emission mixture stress", passes=6)

    lines = body.splitlines()
    assert lines[0] == r"\begin{tabular}{lrrr}"
    assert lines[4:7] == [
        r"  \texttt{emission++} & 1 & 1.2 & 3.2 \\",
        r"  \texttt{burn-in} & 2 & 2.0 [1.0, 3.0] & 4.0 [3.0, 5.0] \\",
        r"  \texttt{quantile} & det. & -2.0 & 0.0 \\",
    ]
    assert lines[-1] == r"\end{tabular}"
    check_latex_safe(caption)
    assert "6 passes" in caption
