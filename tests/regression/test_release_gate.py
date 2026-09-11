"""What `infra/release.sh` must rebuild in full, and in what order.

Two of the gate's steps exist because prediction failed: the figure stamps
were wrong on 476 of 476 decisions (issue #476), so the release gate renders
every figure (issue #484), and an incremental Sphinx build reports on what a
saved environment says changed, so the release gate reads all 134 modules
whatever ``docs/_build/`` holds (issue #485). Both are
edits away from being cheap again --- a `--only`, a dropped `-E` --- and
neither loss would fail anything at the time it was made, which is what these
tests are for.

The ordering is checked as well as the flags. `infra/build_documents.sh`
renders a stale cited figure *into* `docs/tex/figures/`, so a comparison run
after it compares a rebuild against bytes that build had just written.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
from snakes_and_ladders.qa.manifest import FIGURES

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_GATE = REPO_ROOT / "infra" / "release.sh"

#: `RELEASE.md`'s rounded bound on the whole-manifest figure pass, in minutes.
_FIGURE_PASS_BOUND = re.compile(
    r"`qa\.build --all --check`[^|]*\|\s*\*\*Around (\d+) minutes\*\*"
)

#: One `run_check "<name>" <command>`, with the line continuations joined.
_RUN_CHECK = re.compile(r"^run_check\s+(.*)$", re.MULTILINE)


def _steps() -> list[tuple[str, list[str]]]:
    """The gate's steps, in the order it runs them.

    Returns
    -------
    list[tuple[str, list[str]]]
        Each step's name and the argument vector it runs.
    """
    text = RELEASE_GATE.read_text().replace("\\\n", " ")
    steps = []
    for match in _RUN_CHECK.finditer(text):
        words = shlex.split(match.group(1))
        steps.append((words[0], words[1:]))
    return steps


def _step_named(fragment: str) -> tuple[int, list[str]]:
    """The one step whose name contains ``fragment``, and its position."""
    found = [
        (index, command)
        for index, (name, command) in enumerate(_steps())
        if fragment in name
    ]
    assert len(found) == 1, (
        f"infra/release.sh has {len(found)} steps named like {fragment!r}; "
        "the gate's shape has changed"
    )
    return found[0]


@pytest.mark.structural
@pytest.mark.critical
def test_the_gate_renders_every_figure_without_consulting_a_stamp() -> None:
    """`--all --check`: no digest, no skip, and no overwrite.

    `--all` is what ignores the stamps whose false-positive rate is 100%, and
    `--check` is what makes a mismatch a failure rather than a refresh. A
    `--only` or a `--document` here would narrow the gate back to a
    prediction.
    """
    _, command = _step_named("QA figures")
    assert command[:3] == ["uv", "run", "python"], command
    assert "snakes_and_ladders.qa.build" in command
    assert "--all" in command, (
        "the release gate's figure step does not pass --all, so the stamps "
        "decide what it renders (issue #484)"
    )
    assert "--check" in command, (
        "the release gate's figure step does not pass --check, so a stale "
        "figure is silently refreshed instead of failing (issue #484)"
    )
    for narrowing in ("--only", "--document"):
        assert narrowing not in command, (
            f"the release gate's figure step passes {narrowing}, which "
            "renders a subset (issue #484)"
        )


@pytest.mark.structural
@pytest.mark.critical
def test_the_figures_are_compared_before_the_document_build_rewrites_them() -> None:
    """The comparison must read the committed bytes, not a fresh render.

    `infra/build_documents.sh` renders every stale cited figure into
    `docs/tex/figures/`. Run first, it supplies the very bytes the comparison
    is against, and every cited figure passes by construction.
    """
    figures, _ = _step_named("QA figures")
    documents, command = _step_named("documents")
    assert "infra/build_documents.sh" in command
    assert figures < documents, (
        "infra/release.sh compares the figures after infra/build_documents.sh "
        "has rendered them into docs/tex/figures/, so the comparison is "
        "against its own output (issue #484)"
    )


@pytest.mark.structural
@pytest.mark.critical
def test_the_gate_builds_the_documentation_in_full() -> None:
    """`-E -a -W`: every module re-read, every output written, warnings fatal.

    Without `-E` the build reuses whatever ``docs/_build/`` holds from an
    earlier branch or an interrupted run and states only what changed --- 8.2 s
    against 32.2 s cold (issue #485) --- while a release claims all 134 modules
    are clean.
    """
    _, command = _step_named("sphinx-build")
    assert "sphinx-build" in command
    assert "docs/source" in command
    for flag, why in (
        ("-E", "a saved environment decides which modules the run re-reads"),
        ("-a", "unchanged outputs are not rewritten"),
        ("-W", "a warning does not fail the build"),
    ):
        assert flag in command, (
            f"the release gate's Sphinx step does not pass {flag}: {why} (issue #485)"
        )


@pytest.mark.structural
def test_release_md_bounds_the_figure_pass_above_the_manifest_s_total() -> None:
    """`RELEASE.md`'s stated bound is not less than the manifest's sum.

    The number this replaced --- ~6 min a figure --- was a whole re-stamp
    pass's total read as one render, and it stood because nothing in the tree
    disagreed with it (issue #476). `snakes_and_ladders.qa.manifest` carries a
    measured `seconds` per figure, so the cost has a source a reader can add
    up.

    An inequality rather than an equality, because `RELEASE.md`'s numbers are
    rounded upper bounds by design (issue #525) and an exact-equality guard
    would fail on the rounding rather than on the drift. This fires when the
    figure set outgrows what the document tells a reader to expect, which is
    the drift worth catching.
    """
    total = sum(spec.seconds for spec in FIGURES)
    release = (REPO_ROOT / "RELEASE.md").read_text()
    match = _FIGURE_PASS_BOUND.search(release)
    assert match is not None, (
        "RELEASE.md no longer states a bound for the `qa.build --all --check` "
        "step, so nothing couples the document to the manifest (issue #525)"
    )
    bound = int(match.group(1)) * 60
    assert bound >= total, (
        f"RELEASE.md bounds the figure pass at ~{match.group(1)} min, under "
        f"the {total:.1f} s the {len(FIGURES)} entries of "
        "snakes_and_ladders.qa.manifest declare between them: the manifest has "
        "outgrown the stated bound (issue #525)"
    )
