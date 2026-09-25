"""Regression tests for sal.qa.starts_table (issues #891, #898).

The layout is pinned, not the rendering: one row per start in the order
given, five columns --- the initializer, the gap and the seconds at the
handover, the gap and the seconds where the polish stopped --- each a mean to
one decimal with the sample standard deviation beside it over more than one
seed, and a caption the LaTeX check accepts. The ``array`` form carries the
same cells.
"""

from __future__ import annotations

import pytest
from sal.qa.figure import check_latex_safe
from sal.qa.starts_table import build_array, build_table
from sal.search.mixture_starts import StartRow


def _row(start: str, gaps: tuple[float, ...], *, deterministic: bool) -> StartRow:
    return StartRow(
        start=start,
        deterministic=deterministic,
        seeded=tuple(-1.0 for _ in gaps),
        seeded_gap=tuple(100.0 + gap for gap in gaps),
        reached=tuple(-1.0 for _ in gaps),
        gap=gaps,
        passes=0.0,
        state_bytes=320,
        seconds=tuple(2.0 + gap for gap in gaps),
        seeding_seconds=tuple(0.1 for _ in gaps),
        recovery=tuple(0.5 for _ in gaps),
        mean_error=tuple(0.1 for _ in gaps),
        converged=tuple(True for _ in gaps),
        emptied=tuple(False for _ in gaps),
        iterations=tuple(10 for _ in gaps),
    )


#: One seed, two seeds (mean 2.0, sample standard deviation 1.4) and a
#: deterministic start.
ROWS = [
    _row("emission++", (1.25,), deterministic=False),
    _row("burn-in", (1.0, 3.0), deterministic=False),
    _row("quantile", (-2.0,), deterministic=True),
]


@pytest.mark.smoke
def test_the_table_has_five_columns_rows_by_final_gap_and_a_safe_caption() -> None:
    body, caption = build_table(
        ROWS, instance="emission mixture stress", seconds=120, seeds=2
    )

    lines = body.splitlines()
    assert lines[0] == r"\begin{tabular}{lrrrr}"
    assert lines[2] == (
        r"  initializer & init time [s] & init. gap [nats] & final time [s] "
        r"& final gap [nats] \\"
    )
    # Each time before its gap, and the rows from the lowest mean final gap.
    assert lines[4:7] == [
        r"  \texttt{quantile} & 0.1 & 98.0 & 0.0 & -2.0 \\",
        r"  \texttt{emission++} & 0.1 & 101.2 & 3.2 & 1.2 \\",
        r"  \texttt{burn-in} & 0.1 $\pm$ 0.0 & 102.0 $\pm$ 1.4 & 4.0 $\pm$ 1.4 "
        r"& 2.0 $\pm$ 1.4 \\",
    ]
    assert lines[-1] == r"\end{tabular}"
    check_latex_safe(caption)
    assert "120 s" in caption
    assert "2 seeds" in caption
    assert "host" in caption
    # At one seed there is no spread to state, and the caption says so.
    _, alone = build_table(
        ROWS, instance="emission mixture stress", seconds=120, seeds=1
    )
    check_latex_safe(alone)
    assert "one seed's" in alone
    assert "standard deviation" not in alone


@pytest.mark.smoke
def test_the_array_sets_the_tabular_s_cells_in_the_form_mathjax_renders() -> None:
    # The same three rows: every cell the tabular sets, in the same place,
    # with the booktabs rules as \hline, the sign as \pm and the header as
    # \text (issue #898).
    tabular, _ = build_table(
        ROWS, instance="emission mixture stress", seconds=120, seeds=2
    )

    lines = build_array(ROWS).splitlines()
    assert lines[0] == r"\begin{array}{lrrrr}"
    assert lines[2] == (
        r"  \text{initializer} & \text{init time [s]} & \text{init. gap [nats]} "
        r"& \text{final time [s]} & \text{final gap [nats]} \\"
    )
    assert lines[4:7] == [
        line.replace(r"$\pm$", r"\pm") for line in tabular.splitlines()[4:7]
    ]
    assert [lines[1], lines[3], lines[7]] == [r"  \hline"] * 3
    assert lines[-1] == r"\end{array}"
