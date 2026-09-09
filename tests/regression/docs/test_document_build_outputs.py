"""What `infra/build_documents.sh` writes is what `DEV.md` says it writes.

Issue #429. The build rewrites `docs/paper.pdf` and `docs/textbook.pdf` on every
run and only a "Rebuild the documents" pull request may carry that change; it
also rewrites a cited figure whose stamp is stale, and those *are* committed by
the pull request that changed them. A contributor who does not know which is
which commits the wrong one, so `DEV.md` carries the table --- and a table
nothing checks drifts from the script the first time either changes.

The comparison is against the build's own sources rather than against a second
copy of the list: the stems from `snakes_and_ladders.qa.manifest` and the
documents' citations, the documents from `infra/build_documents.sh`, what is
committed from `git`, and what cannot be from `.gitignore`. A figure added to
the manifest and cited, a figure the documents stop citing, or a third document
therefore fails here until `DEV.md` says so.

Not checked here: that a build *run* writes these and no others. That is a
render, which takes the host's exclusive lock (`DEV.md`, Throughput) and does
not belong in a test session. It was measured instead --- eight builds, the tree
hashed before and after each --- and `DEV.md` records the result beside the wall
times.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from snakes_and_ladders.qa.manifest import FIGURES, cited_stems

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The sentence in `DEV.md` opening the contract table. Prose around it may be
#: reworded; this is the anchor, and its loss fails the test rather than
#: silently checking nothing.
_ANCHOR = "The build writes these paths and no others"

#: A backticked token inside a table cell.
_QUOTED = re.compile(r"`([^`]+)`")

#: The documents `infra/build_documents.sh` runs `latexmk` over, read back from
#: the script so a third document cannot arrive without this list growing.
_DOCUMENT_LOOP = re.compile(r"^for document in (.+); do$", re.MULTILINE)

#: The sources the script hands the figure selection, and where it puts the
#: renders. Read from the script rather than from
#: `snakes_and_ladders.qa.build`'s defaults: those resolve against the checkout
#: the package was installed from, which on a shared environment is another
#: worktree, and this test is about *this* tree's paths.
_SELECTION_DOCUMENT = re.compile(r"^\s*--document (\S+)", re.MULTILINE)
_OUTPUT_DIR = re.compile(r"^\s*--output-dir (\S+)", re.MULTILINE)


def _script() -> str:
    """The build script's text."""
    return (REPO_ROOT / "infra" / "build_documents.sh").read_text()


def figures_directory() -> Path:
    """Where the build writes the figures, per its own `--output-dir`."""
    found: list[str] = _OUTPUT_DIR.findall(_script())
    assert len(found) == 1, (
        f"expected one --output-dir in the build script, got {found}"
    )
    return REPO_ROOT / found[0]


def selection_documents() -> list[Path]:
    """The documents whose citations drive the selection, per the script."""
    found: list[str] = _SELECTION_DOCUMENT.findall(_script())
    assert found, "the build script passes no --document to the figure selection"
    return [REPO_ROOT / name for name in found]


def documents() -> tuple[str, ...]:
    """The document stems the build runs `latexmk` over."""
    match = _DOCUMENT_LOOP.search(_script())
    assert match is not None, "infra/build_documents.sh no longer loops over documents"
    return tuple(match.group(1).split())


def _rendered(stem: str) -> Path:
    """The committed output of one figure: a `.pdf` or a `.tex`, never both."""
    found = [
        candidate
        for suffix in (".pdf", ".tex")
        if (candidate := figures_directory() / f"{stem}{suffix}").is_file()
    ]
    assert len(found) == 1, (
        f"{stem} has {len(found)} rendered outputs committed; the manifest "
        "renders exactly one"
    )
    return found[0]


def regenerated_and_committed() -> set[str]:
    """The paths a build rewrites that are also tracked.

    Returns
    -------
    set[str]
        Repository-relative: three files per cited figure, and the two PDFs.
    """
    paths = {f"docs/{document}.pdf" for document in documents()}
    for stem in cited_stems(*selection_documents()):
        paths.add(str(_rendered(stem).relative_to(REPO_ROOT)))
        for name in (f"{stem}_caption.txt", f"{stem}.inputs"):
            path = figures_directory() / name
            assert path.is_file(), f"{name} is not committed"
            paths.add(str(path.relative_to(REPO_ROOT)))
    return paths


def _rows() -> list[tuple[list[str], str]]:
    """`DEV.md`'s contract table as (paths, committed) per row."""
    lines = (REPO_ROOT / "DEV.md").read_text().splitlines()
    start = next((index for index, line in enumerate(lines) if _ANCHOR in line), None)
    assert start is not None, f"DEV.md no longer says {_ANCHOR!r}"
    rows: list[tuple[list[str], str]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        assert stripped.startswith("|"), f"DEV.md's contract table ended at {line!r}"
        if set(stripped) <= set("| -"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        assert len(cells) == 3, f"expected three columns, got {cells}"
        if cells[0] == "path":
            continue
        rows.append((_QUOTED.findall(cells[0]), cells[2]))
    assert rows, "DEV.md's contract table has no rows"
    return rows


def _expand(pattern: str) -> set[str]:
    """The concrete paths one documented pattern stands for."""
    if "<" not in pattern:
        return {pattern}
    stems = "|".join(
        re.escape(stem) for stem in sorted(cited_stems(*selection_documents()))
    )
    names = "|".join(re.escape(name) for name in sorted(documents()))
    regex = re.escape(pattern)
    regex = regex.replace(re.escape("<cited stem>"), f"(?:{stems})")
    regex = regex.replace(re.escape("<document>"), f"(?:{names})")
    candidates = {
        path
        for path in _tracked(REPO_ROOT / "docs") | _byproducts()
        if re.fullmatch(regex, path)
    }
    assert candidates, f"DEV.md names {pattern!r}, which matches nothing in the tree"
    return candidates


def _tracked(directory: Path) -> set[str]:
    """Every path `git` tracks under one directory, repository-relative."""
    listing = subprocess.run(
        ["git", "ls-files", str(directory.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(listing.stdout.split())


def _byproducts() -> set[str]:
    """The names `latexmk` can leave beside the PDFs, for pattern expansion."""
    suffixes = (
        ".aux",
        ".bbl",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".out",
        ".toc",
    )
    return {
        f"docs/{document}{suffix}" for document in documents() for suffix in suffixes
    }


@pytest.mark.structural
def test_dev_md_names_every_regenerated_path_that_is_committed() -> None:
    """The table's committed rows and the build's own sources name one set.

    This is the half a contributor acts on: a path that is rewritten by a build
    *and* tracked either belongs in their commit or is refused by a gate, and
    nothing else in the tree distinguishes the two.
    """
    documented: set[str] = set()
    for paths, committed in _rows():
        if not committed.startswith("yes"):
            continue
        for pattern in paths:
            documented |= _expand(pattern)
    written = regenerated_and_committed()
    assert documented == written, (
        "DEV.md's document-build contract and the build's sources disagree; "
        f"only DEV.md: {sorted(documented - written)}; "
        f"only the build: {sorted(written - documented)}"
    )


@pytest.mark.structural
def test_the_uncommitted_rows_are_the_ones_git_cannot_carry() -> None:
    """A row marked "no" is ignored; a row marked "yes" is tracked.

    The table's third column is a claim about `git`, so it is checked against
    `git` rather than read as prose. An entry that becomes committable, or a
    committed output that falls out of the index, fails here.
    """
    for paths, committed in _rows():
        for pattern in paths:
            for path in _expand(pattern):
                ignored = subprocess.run(
                    ["git", "check-ignore", "-q", path],
                    cwd=REPO_ROOT,
                    check=False,
                ).returncode
                if committed.startswith("yes"):
                    assert ignored != 0, f"DEV.md calls {path} committed; it is ignored"
                    assert (REPO_ROOT / path).is_file(), f"{path} is not in the tree"
                else:
                    assert ignored == 0, (
                        f"DEV.md calls {path} uncommitted, but .gitignore does not "
                        "cover it, so a pull request can carry it"
                    )


@pytest.mark.structural
def test_the_two_pdfs_are_the_paths_the_gates_name() -> None:
    """Only the PDFs need a rule, and both gates that enforce it name both.

    A byproduct cannot be committed and a re-rendered figure should be. The PDFs
    are the only outputs that are rewritten every run *and* tracked, which is
    why `infra/review_gates.sh` and the `documents` job check them by name; a
    third document would need adding to both.
    """
    pdfs = {f"docs/{document}.pdf" for document in documents()}
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    gates = (REPO_ROOT / "infra" / "review_gates.sh").read_text()
    for pdf in pdfs:
        assert pdf in workflow, f"the CI PDF gate does not name {pdf}"
        assert pdf in gates, f"infra/review_gates.sh does not name {pdf}"


@pytest.mark.structural
def test_every_tracked_figure_file_belongs_to_a_manifest_stem() -> None:
    """`docs/tex/figures/` holds three files per manifest entry and nothing else.

    The contract table above says "cited stem" because
    `infra/build_documents.sh` regenerates the cited figures alone, and the
    release gate the rest (`--all`). Since issue #492 every entry is cited, so
    that column ranges over the whole manifest and this asserts the range
    rather than the split: the selection is still the mechanism --- a document
    that drops a citation narrows it the same day, which
    `tests/regression/qa/test_qa_build.py` pins on documents it writes itself
    --- but no file here is outside the build's reach today.

    What the check is for is unchanged either way: a tracked file under the
    figures directory that no manifest entry renders is regenerated by
    nothing, and a stem missing one of its three is published without the
    caption or the stamp that referees it.
    """
    cited = set(cited_stems(*selection_documents()))
    every = {spec.stem for spec in FIGURES}
    assert cited == every, (
        f"the build writes only what a document cites, and these are cited by "
        f"nothing: {sorted(every - cited)}"
    )

    tracked = _tracked(figures_directory())
    expected = {str(_rendered(stem).relative_to(REPO_ROOT)) for stem in every} | {
        str((figures_directory() / f"{stem}{name}").relative_to(REPO_ROOT))
        for stem in every
        for name in ("_caption.txt", ".inputs")
    }
    assert tracked == expected, (
        "docs/tex/figures/ holds files no manifest entry renders, or is missing "
        f"one: extra {sorted(tracked - expected)}; missing {sorted(expected - tracked)}"
    )
