"""What `infra/release.sh` must rebuild in full, and in what order.

Two steps exist because prediction failed: figure stamps were wrong on 476 of
476 decisions (#476), so every figure renders (#484); an incremental Sphinx
build reports only what a saved environment says changed, so all 134 modules
are read (#485). A `--only` or a dropped `-E` would undo either silently. The
comparison runs before `infra/build_documents.sh`, which writes stale figures
into `docs/tex/figures/`, and the build after it does not render (#530).
"""

from __future__ import annotations

import re
import shlex
import subprocess

import pytest
from sal.qa.manifest import FIGURES

from tests._paths import REPO_ROOT

RELEASE_GATE = REPO_ROOT / "infra" / "release.sh"
BUILD_DOCUMENTS = REPO_ROOT / "infra" / "build_documents.sh"

#: `RELEASE.md`'s rounded bound on the whole-manifest figure pass, in minutes.
_FIGURE_PASS_BOUND = re.compile(
    r"`qa\.build --all --check`[^|]*\|\s*\*\*Around (\d+) minutes\*\*"
)

#: One `run_check "<name>" <command>`, with the line continuations joined.
_RUN_CHECK = re.compile(r"^run_check\s+(.*)$", re.MULTILINE)


def _steps() -> list[tuple[str, list[str]]]:
    """The gate's steps in run order, as (name, argument vector)."""
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


@pytest.mark.infra
@pytest.mark.critical
def test_the_gate_renders_every_figure_without_consulting_a_stamp() -> None:
    """`--all --check`: no digest, no skip, and no overwrite.

    `--all` ignores stamps wrong on 100% of decisions; `--check` fails, not refreshes.
    """
    _, command = _step_named("QA figures")
    assert command[:3] == ["uv", "run", "python"], command
    assert "sal.qa.build" in command
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


@pytest.mark.infra
@pytest.mark.critical
def test_the_figures_are_compared_before_the_document_build_rewrites_them() -> None:
    """The comparison must read the committed bytes, not a fresh render.

    Run after the document build, it would compare figures against themselves.
    """
    figures, _ = _step_named("QA figures")
    documents, command = _step_named("documents")
    assert "infra/build_documents.sh" in command
    assert figures < documents, (
        "infra/release.sh compares the figures after infra/build_documents.sh "
        "has rendered them into docs/tex/figures/, so the comparison is "
        "against its own output (issue #484)"
    )


@pytest.mark.infra
@pytest.mark.critical
def test_the_gate_renders_each_figure_once() -> None:
    """The document build is told not to render what the step before it did.

    Both rendered the same 23 entries until #530, paying the manifest twice.
    """
    _, command = _step_named("documents")
    assert "--no-figures" in command, (
        "the release gate's document build renders every figure the step "
        "before it has already rendered and compared (issue #530)"
    )


@pytest.mark.infra
def test_the_build_script_renders_the_figures_unless_it_is_told_not_to() -> None:
    """`--no-figures` is one caller's flag, not the script's default.

    Release (#530) and PRs (#488: 431.8 s against 15.8 s) skip; `main` renders.
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
    skipping = [
        line.strip()
        for line in workflow.splitlines()
        if "build_documents.sh" in line and "--no-figures" in line
    ]
    rendering = [
        line.strip()
        for line in workflow.splitlines()
        if "build_documents.sh" in line and "--no-figures" not in line
    ]
    assert rendering, (
        "no invocation of infra/build_documents.sh renders, so nothing "
        "regenerates the figures the documents typeset (issue #530)"
    )
    # The call that skips is reachable only from the pull-request arm, and the
    # call that renders only from the other. Asserting the shape of the `if`
    # rather than the presence of a string is what lets one job carry both.
    assert len(skipping) <= 1, skipping
    if skipping:
        arm = workflow.split("infra/build_documents.sh --no-figures")[0]
        assert arm.rstrip().endswith("= 'pull_request' ]; then"), (
            "infra/build_documents.sh --no-figures is reachable outside the "
            "pull-request arm, so a merge would typeset stale figures "
            "(issues #530, #625)"
        )


@pytest.mark.infra
@pytest.mark.critical
def test_the_gate_builds_the_documentation_in_full() -> None:
    """`-E -a -W`: every module re-read, every output written, warnings fatal.

    Without `-E`: 8.2 s against 32.2 s cold, over changes only (issue #485).
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


@pytest.mark.infra
def test_release_md_bounds_the_figure_pass_above_the_manifest_s_total() -> None:
    """`RELEASE.md`'s stated bound is not less than the manifest's sum.

    Measured `seconds` per figure (#476); an inequality, as bounds round up (#525).
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
        "sal.qa.manifest declare between them: the manifest has "
        "outgrown the stated bound (issue #525)"
    )
