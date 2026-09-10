"""What the figure manifest must state before a build may trust its selection.

The manifest declares each figure's renderer and its measured render time,
and two claims rest on that time: a cited figure fits the cap `DEV.md` gives
the documents build, and every entry states a time at all --- an entry with
none would leave the release pass's total (`tests/regression/test_release_gate.py`)
short by an unmeasured figure. Both were asserted beside the input cache
until issue #490 removed it; neither is about staleness, so both moved here.
"""

from __future__ import annotations

import pytest
from snakes_and_ladders.qa import build
from snakes_and_ladders.qa.manifest import (
    CAP_WAIVERS,
    CITED_RENDER_CAP,
    FIGURES,
    cited_stems,
)


@pytest.mark.critical
@pytest.mark.structural
def test_a_cited_figure_renders_inside_the_cap_or_carries_a_waiver() -> None:
    # The documents build is the sum of its cited figures, every one of them
    # since #490 removed the stamps that skipped some, so a cited figure over
    # the cap is a third of the whole validation budget on its own. One over
    # it carries a waiver naming the ticket that owns cutting it.
    cited = cited_stems(*build.DEFAULT_DOCUMENTS)
    over = {
        spec.stem: spec.seconds
        for spec in FIGURES
        if spec.stem in cited and spec.seconds > CITED_RENDER_CAP
    }

    assert set(over) <= set(CAP_WAIVERS), (
        f"cited figures over the {CITED_RENDER_CAP:.0f} s cap without a waiver: {over}"
    )
    assert all(ticket.startswith("#") for ticket in CAP_WAIVERS.values())
    assert set(CAP_WAIVERS) <= {spec.stem for spec in FIGURES}


@pytest.mark.critical
@pytest.mark.structural
def test_every_manifest_entry_states_a_measured_time() -> None:
    assert all(spec.seconds > 0 for spec in FIGURES)
