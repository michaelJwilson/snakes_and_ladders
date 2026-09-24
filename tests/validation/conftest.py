"""A goal runs only where it is asked for (issue #972).

A `goal` test fails until the package meets an external framework's figure,
so the per-PR tier and the push to `main`, whose `-m` expressions do not
name it, skip it. CI's `validation` job asks for it by name, `-m goal`, in a
step that reports and does not block.
"""

from __future__ import annotations

import pytest

#: Why an unrequested goal skips.
REASON = "a goal runs only under an -m expression naming it (issue #972)"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if "goal" in (config.option.markexpr or ""):
        return
    for item in items:
        if item.get_closest_marker("goal") is not None:
            item.add_marker(pytest.mark.skip(reason=REASON))
