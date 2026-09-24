"""Every link to this project names the repository this project is in.

Issue #250. `STATUS.md` cited 82 pull requests, `README.md`'s CI badge twice
and `Cargo.toml`'s `repository` field once, all at the former path under this
owner; every link resolved to a plausible page. `Cargo.toml`'s is the metadata
a published crate carries. A guard, since a copied line reproduces the path;
it reads every tracked text file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests._paths import REPO_ROOT

#: The account, and the only repository under it this project may name.
OWNER = "michaelJwilson"
REPOSITORY = "snakes_and_ladders"

#: Any `github.com/<owner>/<repo>` reference, in Markdown, TOML or LaTeX alike.
_LINK = re.compile(rf"github\.com/{OWNER}/([A-Za-z0-9_.-]+)")

#: Where a link can be written. Binary artifacts and generated output are
#: excluded: `docs/paper.pdf` and `docs/textbook.pdf` are build products, and
#: a link inside them comes from `docs/tex/`, which is covered.
SUFFIXES = (".md", ".toml", ".tex", ".py", ".rst", ".yml", ".yaml", ".sh", ".bib")


def _tracked_text_files() -> list[Path]:
    """Every tracked file a link could be written in, by `git ls-files`."""
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
@pytest.mark.infra
def test_every_link_names_this_repository() -> None:
    """No tracked file points at another repository under this owner.

    Other owners are other projects; this owner's other names are former ones.
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
@pytest.mark.infra
def test_the_guard_fails_on_a_link_to_the_old_name() -> None:
    """The guard rejects exactly what it exists to reject.

    The former path (85 occurrences), assembled from parts: this file is scanned.
    """
    former = "phylo"
    assert _wrong_links(f"see https://github.com/{OWNER}/{former}/pull/49") == [former]
    assert _wrong_links(f"https://github.com/{OWNER}/{REPOSITORY}/pull/49") == []
    # A different owner is a different project, not this one under an old name.
    assert _wrong_links("https://github.com/numpy/numpy") == []
