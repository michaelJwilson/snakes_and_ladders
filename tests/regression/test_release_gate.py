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

The same two steps are also where the gate paid for the manifest twice
(issue #530), so what the ordering allows and what the gate spends are
pinned together: the comparison comes first, and because it did, the build
after it is told not to render.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest
from snakes_and_ladders.qa.manifest import FIGURES

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_GATE = REPO_ROOT / "infra" / "release.sh"
BUILD_DOCUMENTS = REPO_ROOT / "infra" / "build_documents.sh"

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
def test_the_gate_renders_each_figure_once() -> None:
    """The document build is told not to render what the step before it did.

    Both steps rendered the whole manifest until issue #530. The figure step
    compares every entry against the committed bytes, so those bytes *are* a
    fresh render; a second pass can only write them back. The stamps that once
    let it skip most of that are deleted (issue #490) and issue #492 left
    every entry cited, so the two selections are the same 23 entries and the
    gate paid the manifest's declared total twice.
    """
    _, command = _step_named("documents")
    assert "--no-figures" in command, (
        "the release gate's document build renders every figure the step "
        "before it has already rendered and compared (issue #530)"
    )


@pytest.mark.structural
def test_the_build_script_renders_the_figures_unless_it_is_told_not_to() -> None:
    """`--no-figures` is one caller's flag, not the script's default.

    The `documents` job on the push to `main` runs the script with no
    arguments and must still render; only the release gate, which has just
    compared every figure, may skip. An unknown argument is refused rather
    than ignored, so a typo cannot silently turn the render off.
    """
    script = BUILD_DOCUMENTS.read_text()
    assert "--no-figures) figures=0" in script
    assert 'if [ "$figures" -eq 1 ]; then' in script, (
        "the figure pass in infra/build_documents.sh is unconditional, so "
        "--no-figures does not reach it (issue #530)"
    )

    refused = subprocess.run(
        ["bash", str(BUILD_DOCUMENTS), "--render-everything-twice"],
        capture_output=True,
        check=False,
    )
    assert refused.returncode == 2, refused.stderr.decode()

    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "infra/build_documents.sh --no-figures" not in workflow, (
        "the documents job skips the render, so nothing regenerates the "
        "figures it typesets (issue #530)"
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
