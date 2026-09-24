"""One test over a table of rows, failing once with every row that failed.

A seed or scalar grid parametrised case by case collects one test per row,
each asserting the same claim against the same referee (issue #982, R22).
Folded into a loop, a plain ``assert`` would stop at the first failing row
and hide the rest; `every_row` runs them all and names each failure, so the
message says as much as the parametrised cases did.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest


def every_row(rows: Iterable[tuple[object, ...]], check: Callable[..., object]) -> None:
    """Call ``check(*row)`` on every row; fail once, listing each row that raised.

    An exception of any type counts as that row failing, as it would have
    failed its own parametrised case; the loop does not stop at it.
    """
    failures = []
    count = 0
    for row in rows:
        count += 1
        try:
            check(*row)
        except (Exception, pytest.fail.Exception) as error:
            failures.append(f"{row!r}: {type(error).__name__}: {error}")
    if failures:
        pytest.fail(
            f"{len(failures)} of {count} rows fail:\n" + "\n".join(failures),
            pytrace=False,
        )


def every_value(values: Iterable[object], check: Callable[[Any], object]) -> None:
    """`every_row` over one axis: ``check(value)`` for every value."""
    every_row(((value,) for value in values), check)
