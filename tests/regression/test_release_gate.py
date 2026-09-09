"""What `infra/release.sh` must rebuild in full, and in what order.

Two of the gate's steps exist because prediction failed: the figure stamps
were wrong on 476 of 476 decisions (issue #476), so the release gate renders
every figure (issue #484), and an incremental Sphinx build re-reads only what
changed, so the release gate reads all 134 modules (issue #485). Both are
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

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_GATE = REPO_ROOT / "infra" / "release.sh"

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

    Without `-E` the build reuses a saved environment and re-reads only what
    changed, so a warning in an untouched docstring is one it never emits ---
    8.2 s and no finding, against 32.2 s cold (issue #485).
    """
    _, command = _step_named("sphinx-build")
    assert "sphinx-build" in command
    assert "docs/source" in command
    for flag, why in (
        ("-E", "the saved environment is reused, so unchanged modules are not re-read"),
        ("-a", "unchanged outputs are not rewritten"),
        ("-W", "a warning does not fail the build"),
    ):
        assert flag in command, (
            f"the release gate's Sphinx step does not pass {flag}: {why} (issue #485)"
        )
