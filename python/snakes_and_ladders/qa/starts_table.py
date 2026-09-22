"""QA table: each initializer's gap and wall clock at the handover and at the end (issue #891).

One row per start, in the order given, in five columns (issue #898): the
initializer, the gap below the generating parameters' log-likelihood at the
handover and the seconds to it, and the gap where the polish stopped and the
seconds of the start and its polish under their one budget. Built from the
:class:`~snakes_and_ladders.search.mixture_starts.StartRow` rows the starts
notebook reads, in the ``tabular`` shape
:func:`snakes_and_ladders.qa.figure.write_qa_table` writes, so a document can
``\\input`` it, and as the ``array`` a notebook's MathJax renders. The
builders return the body and the caption and write nothing (``qa/CLAUDE.md``).

The seconds belong to the host, so a caller prints this table in a cell the
notebook checker does not compare, and the caption says so.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from snakes_and_ladders.qa.figure import check_latex_safe, latex_escape
from snakes_and_ladders.search.mixture_starts import StartRow

#: The header of both forms, one entry per column.
HEADER = (
    "initializer",
    "init. gap [nats]",
    "init time [s]",
    "final gap [nats]",
    "final time [s]",
)


def _cell(values: Sequence[float], plus_minus: str) -> str:
    """The mean to one decimal, and over more than one seed the sample standard deviation beside it."""
    mean = f"{np.mean(values):.1f}"
    if len(values) == 1:
        return mean
    return f"{mean} {plus_minus} {np.std(values, ddof=1):.1f}"


def _body(rows: Sequence[StartRow], plus_minus: str) -> list[list[str]]:
    """Each row's five cells, with ``plus_minus`` as the form sets the sign."""
    return [
        [
            f"\\texttt{{{latex_escape(row.start)}}}",
            _cell(row.seeded_gap, plus_minus),
            _cell(row.seeding_seconds, plus_minus),
            _cell(row.gap, plus_minus),
            _cell(row.seconds, plus_minus),
        ]
        for row in rows
    ]


def build_table(
    rows: Sequence[StartRow], *, instance: str, seconds: int, seeds: int
) -> tuple[str, str]:
    """The ``tabular`` of gap and seconds per initializer, at the handover and at the end, and its caption.

    Parameters
    ----------
    rows : Sequence[StartRow]
        One per start, in the order the table takes them.
    instance : str
        The fixture the starts ran on, as the caption names it.
    seconds : int
        The one budget each (start, seed) cell ran under, start and polish.
    seeds : int
        The seeds each start that reads a generator ran at.

    Returns
    -------
    tuple[str, str]
        A complete ``tabular`` environment, and a caption that
        :func:`~snakes_and_ladders.qa.figure.check_latex_safe` accepts.
    """
    table = "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"  \toprule",
            "  " + " & ".join(HEADER) + r" \\",
            r"  \midrule",
            *("  " + " & ".join(cells) + r" \\" for cells in _body(rows, r"$\pm$")),
            r"  \bottomrule",
            r"\end{tabular}",
        ]
    )
    caption = (
        f"Each initializer of the count-pair mixture of {instance}, polished by "
        f"expectation-maximization, the start and its polish under one budget "
        f"of {seconds} s: the gap below the log-likelihood the generating "
        f"parameters reach, in nats, and the wall clock, at the handover and "
        f"where the polish stopped. A negative gap is a fit that passed the "
        f"generating parameters. Values are means over {seeds} seeds with the "
        f"sample standard deviation beside them; a start that reads no "
        f"generator ran once and shows its one value. The times are those of "
        f"the host the table was built on."
    )
    check_latex_safe(caption)
    return table, caption


def build_array(rows: Sequence[StartRow]) -> str:
    """The cells of :func:`build_table`'s ``tabular``, as an ``array`` MathJax renders.

    A notebook displays LaTeX through MathJax, which sets math and not a
    ``tabular``: the environment is ``array``, the rules are ``\\hline``
    (``booktabs`` is not MathJax's), the sign is ``\\pm`` rather than
    ``$\\pm$``, and the header is ``\\text``, since in math mode a word is
    set as a product of italic symbols.

    Returns
    -------
    str
    """
    return "\n".join(
        [
            r"\begin{array}{lrrrr}",
            r"  \hline",
            "  " + " & ".join(rf"\text{{{name}}}" for name in HEADER) + r" \\",
            r"  \hline",
            *("  " + " & ".join(cells) + r" \\" for cells in _body(rows, r"\pm")),
            r"  \hline",
            r"\end{array}",
        ]
    )
