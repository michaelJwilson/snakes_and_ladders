"""QA table: the gap and the wall clock of each start of a mixture fit (issue #891).

One row per start, in the order given: the start's name, its trials, the gap
below the generating parameters' log-likelihood in nats, and the seconds of
the start and its polish. Built from the
:class:`~snakes_and_ladders.search.mixture_starts.StartRow` rows the
starts notebook reads, in the ``tabular`` shape
:func:`snakes_and_ladders.qa.figure.write_qa_table` writes, so a document can
``\\input`` it. The builder returns the body and the caption and writes
nothing (``qa/CLAUDE.md``).

The seconds belong to the host, so a caller prints this table in a cell the
notebook checker does not compare, and the caption says so.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from snakes_and_ladders.qa.figure import check_latex_safe, latex_escape
from snakes_and_ladders.search.mixture_starts import StartRow


def _cell(values: Sequence[float]) -> str:
    """The mean to one decimal, and at more than one trial the range beside it."""
    mean = f"{np.mean(values):.1f}"
    if len(values) == 1:
        return mean
    return f"{mean} [{min(values):.1f}, {max(values):.1f}]"


def build_table(
    rows: Sequence[StartRow], *, instance: str, passes: int
) -> tuple[str, str]:
    """The ``tabular`` of gap and seconds per start, and its caption.

    Parameters
    ----------
    rows : Sequence[StartRow]
        One per start, in the order the table takes them.
    instance : str
        The fixture the starts ran on, as the caption names it.
    passes : int
        The polish's budget in passes.

    Returns
    -------
    tuple[str, str]
        A complete ``tabular`` environment, and a caption that
        :func:`~snakes_and_ladders.qa.figure.check_latex_safe` accepts.
    """
    body = [
        " & ".join(
            [
                f"\\texttt{{{latex_escape(row.start)}}}",
                "det." if row.deterministic else str(row.trials),
                _cell(row.gap),
                _cell(row.seconds),
            ]
        )
        + r" \\"
        for row in rows
    ]
    table = "\n".join(
        [
            r"\begin{tabular}{lrrr}",
            r"  \toprule",
            r"  Start & Trials & Gap (nats) & Seconds \\",
            r"  \midrule",
            *(f"  {line}" for line in body),
            r"  \bottomrule",
            r"\end{tabular}",
        ]
    )
    caption = (
        f"Each start of the count-pair mixture of {instance}, polished by "
        f"expectation-maximization for {passes} passes: the gap below the "
        f"log-likelihood the generating parameters reach, in nats, and the wall "
        f"clock of the start and its polish, on the host the table was built on. "
        f"A negative gap is a fit that passed the generating parameters. Values "
        f"are means over the trials, with the range beside them where there is "
        f"more than one; det. marks a start that reads no generator."
    )
    check_latex_safe(caption)
    return table, caption
