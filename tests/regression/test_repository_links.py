"""Every link to this project names the repository this project is in.

Issue #250. `STATUS.md` cited 82 pull requests, `README.md`'s CI badge pointed
twice, and `Cargo.toml`'s `repository` field once, all at
the repository's former path under this owner -- one this project does not
live at. Correct links in the repository: zero. The numbers were right and only the path
was wrong, which is why nothing caught it: every link resolved to a plausible
page, and a reviewer reading one saw a pull request that existed.

`Cargo.toml`'s is the one with consequences beyond a reader: it is the metadata
a published crate carries.

The check is a guard rather than a one-time fix because the wrong path is what
a copied line reproduces. It reads tracked text files, so a link added to any
document is covered without listing the documents.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The account, and the only repository under it this project may name.
OWNER = "michaelJwilson"
REPOSITORY = "snakes_and_ladders"

#: Any `github.com/<owner>/<repo>` reference, in Markdown, TOML or LaTeX alike.
_LINK = re.compile(rf"github\.com/{OWNER}/([A-Za-z0-9_.-]+)")

#: Where a link can be written. Binary artifacts and generated output are
#: excluded: `docs/draft.pdf` is a build product, and a link inside it comes
#: from `docs/tex/`, which is covered.
SUFFIXES = (".md", ".toml", ".tex", ".py", ".rst", ".yml", ".yaml", ".sh", ".bib")


def _tracked_text_files() -> list[Path]:
    """Every tracked file a link could be written in.

    Uses `git ls-files` rather than a walk, so untracked scratch files and
    build output cannot fail the check.
    """
    listing = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        REPO_ROOT / name
        for name in listing.stdout.splitlines()
        if name.endswith(SUFFIXES)
    ]


def _wrong_links(text: str) -> list[str]:
    """The repository names in ``text`` that are not this one."""
    return [name for name in _LINK.findall(text) if name != REPOSITORY]


@pytest.mark.critical
@pytest.mark.structural
def test_every_link_names_this_repository() -> None:
    """No tracked file points at another repository under this owner.

    A link under a *different* owner is somebody else's project and is not this
    check's business; one under this owner naming something other than
    `snakes_and_ladders` is this project under a name it no longer has.
    """
    offenders = {
        str(path.relative_to(REPO_ROOT)): sorted(set(wrong))
        for path in _tracked_text_files()
        if (wrong := _wrong_links(path.read_text(encoding="utf-8", errors="replace")))
    }
    assert not offenders, (
        f"{len(offenders)} file(s) link to another repository under {OWNER}: "
        f"{offenders}. This project is {OWNER}/{REPOSITORY}."
    )


@pytest.mark.critical
@pytest.mark.structural
def test_the_guard_fails_on_a_link_to_the_old_name() -> None:
    """The guard rejects exactly what it exists to reject.

    A guard that only passes on the current tree says nothing about the next
    line somebody copies -- the rule this repository settled on after the
    documentation index needed four repairs by hand before a test closed it
    (#223). The former path is the case that actually occurred, 85 times.

    The offending URL is assembled from parts rather than written out: this
    file is itself scanned, so a literal one here would fail the check above --
    which is the guard working, but on its own test.
    """
    former = "phylo"
    assert _wrong_links(f"see https://github.com/{OWNER}/{former}/pull/49") == [former]
    assert _wrong_links(f"https://github.com/{OWNER}/{REPOSITORY}/pull/49") == []
    # A different owner is a different project, not this one under an old name.
    assert _wrong_links("https://github.com/numpy/numpy") == []
