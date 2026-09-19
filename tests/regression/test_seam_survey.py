"""The seam survey and the seam gate answer to one definition of a consumer.

Issue #813, step 1. `infra/appraise_seams.py` reports every seam the package
declares; `infra/gate_new_seams.py` refuses a branch that adds one without
the consuming modules root `CLAUDE.md` admits an abstraction for. Two tools
reading the same tree for the same word is how the pair in issue #244 drifted,
so the survey imports the gate's `consumers` rather than writing a second one,
and this holds it to that: a survey that agreed with the gate by coincidence
would stop agreeing the first time either changed.

What this does **not** assert is that every seam under the rule states a
reason. One does not --- `sample.hmc.Kernel`, which #790 carries --- and a
guard with a hand-maintained exemption for it is the shape issue #557 exists
to refuse. The audit records it; the ticket removes it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "infra"))

import appraise_seams  # noqa: E402
import gate_new_seams  # noqa: E402


@pytest.mark.critical
@pytest.mark.infra
def test_the_survey_reads_the_gates_consumers_for_every_seam() -> None:
    sources = gate_new_seams.source_modules()
    for seam in appraise_seams.seams():
        assert seam.consumers == gate_new_seams.consumers(
            seam.module, seam.name, sources
        ), f"{seam.module}.{seam.name}"


@pytest.mark.critical
@pytest.mark.infra
def test_the_survey_finds_the_seams_the_package_declares() -> None:
    # Not a count, which would be a pin on a growing tree: the two seams the
    # repository argues about most, one `Protocol` and the one `ABC`, and the
    # rule read from the gate rather than repeated.
    found = {f"{seam.module}.{seam.name}": seam for seam in appraise_seams.seams()}

    assert "opt.objective.Objective" in found
    assert found["likelihood.schedule.MessageSchedule"].kind == "ABC"
    assert appraise_seams.CONSUMER_RULE == gate_new_seams.CONSUMER_RULE


@pytest.mark.infra
def test_a_generic_protocol_is_seen_as_a_seam() -> None:
    # The shape the gate's own reader misses: `class M(Protocol[T])` is a
    # subscript, not a name, and issue #803 found one it could not see.
    declared = appraise_seams._declared(
        "from typing import Protocol\n"
        "class Plain(Protocol):\n    ...\n"
        "class Generic(Protocol[int]):\n    ...\n"
    )

    assert declared == [("Plain", "Protocol"), ("Generic", "Protocol")]
