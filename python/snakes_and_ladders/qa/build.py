"""Regenerate QA figures, selected by what the technical document cites.

Per pull request, only the figures the documents under ``docs/tex/`` include need
rebuilding: those are the ones a change could make stale in the committed
document, and regenerating the rest cost most of the ``technical-doc`` job
while proving nothing about it (issue #154).

The figures the document does not cite are still checked, at the release gate,
where ``infra/release.sh`` runs ``--all --check``. That is the trade this
module implements: the check moves rather than disappearing, so a figure the
document has stopped citing cannot rot unnoticed.

``--check`` regenerates into a temporary directory and compares bytes instead
of overwriting, so a verification run cannot itself produce the state it was
meant to detect.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from snakes_and_ladders.qa.manifest import (
    FIGURES,
    FigureSpec,
    cited_stems,
    select,
    unknown_stems,
)

# The committed figures are reproducible artifacts, so the clock they were
# rendered against is part of what renders them: matplotlib embeds
# SOURCE_DATE_EPOCH as the PDF's /CreationDate, and without it two rebuilds of
# an unchanged figure differ. Pinned here rather than left to each caller,
# because a `--check` that reports every figure stale unless the caller
# remembered an environment variable is worse than no check. `2025-01-01`.
SOURCE_DATE_EPOCH = "1735689600"

# Development tooling, so a source checkout is assumed: the defaults below
# locate the document and the committed figures relative to this file. An
# installed copy has neither, and nothing in the shipped package imports this.
REPO_ROOT = Path(__file__).resolve().parents[3]
#: The documents the repository builds, in build order. Plural since issue
#: #249: the paper carries the results and the textbook the algorithms, and
#: the per-pull-request figure selection is the union of what they cite --- a
#: selection taken from one of them alone would stop regenerating the other's
#: figures without failing anything.
DEFAULT_DOCUMENTS = (
    REPO_ROOT / "docs" / "tex" / "paper.tex",
    REPO_ROOT / "docs" / "tex" / "textbook.tex",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "tex" / "figures"


class UncitedFigureError(RuntimeError):
    """Raised when the document cites a figure the manifest cannot render."""


def selected(
    documents: Sequence[Path], every: bool, only: Sequence[str] = ()
) -> tuple[FigureSpec, ...]:
    """Choose the figures to regenerate.

    Parameters
    ----------
    documents : Sequence[Path]
        The documents whose citations drive the per-PR selection, taken
        together. Every document the repository builds belongs here: one
        left out is a figure set that is silently too small.
    every : bool
        Select the whole manifest instead, as the release gate does.
    only : Sequence[str]
        Render just these stems, whatever the document cites. For timing one
        figure at a time; it overrides the other two.

    Returns
    -------
    tuple[FigureSpec, ...]
        The specs to render, in manifest order.

    Raises
    ------
    UncitedFigureError
        If a document cites a stem the manifest does not know, or ``only``
        names one. Silently skipping it would leave a figure in the document
        that no build regenerates, which is the failure this whole selection
        risks and so the one it must refuse.
    """
    if only:
        missing = unknown_stems(only)
        if missing:
            msg = f"no snakes_and_ladders.qa.manifest entry renders {sorted(missing)}"
            raise UncitedFigureError(msg)
        return select(only)
    if every:
        return FIGURES
    cited = cited_stems(*documents)
    missing = unknown_stems(cited)
    if missing:
        names = ", ".join(str(document) for document in documents)
        msg = (
            f"{names} cite {sorted(missing)}, which no "
            f"snakes_and_ladders.qa.manifest entry renders; add an entry or "
            f"correct the reference"
        )
        raise UncitedFigureError(msg)
    return select(cited)


def render(spec: FigureSpec, output_dir: Path) -> None:
    """Run one figure's script, writing into ``output_dir``.

    Each figure renders in its own process, as it did when a shell script
    invoked them, so none inherits matplotlib state from the one before it.

    Raises
    ------
    subprocess.CalledProcessError
        If the script exits non-zero.
    """
    environment = dict(os.environ)
    # An explicit setting wins, so a caller can still render against another
    # clock; absent one, this is the clock the committed figures assume.
    environment.setdefault("SOURCE_DATE_EPOCH", SOURCE_DATE_EPOCH)
    subprocess.run(
        [sys.executable, *spec.command(output_dir)],
        cwd=REPO_ROOT,
        check=True,
        env=environment,
    )


def compare(rebuilt_dir: Path, committed_dir: Path) -> list[str]:
    """Find every rebuilt file that differs from its committed counterpart.

    Parameters
    ----------
    rebuilt_dir : Path
        Freshly rendered output.
    committed_dir : Path
        The committed figures to compare against.

    Returns
    -------
    list[str]
        Filenames that differ or are missing from ``committed_dir``, sorted.
    """
    stale = []
    for rebuilt in sorted(rebuilt_dir.iterdir()):
        committed = committed_dir / rebuilt.name
        if not committed.is_file() or not filecmp.cmp(
            rebuilt, committed, shallow=False
        ):
            stale.append(rebuilt.name)
    return stale


def main(argv: list[str] | None = None) -> int:
    """Regenerate or check the selected QA figures.

    Returns
    -------
    int
        ``0`` on success; ``1`` if ``--check`` found a stale figure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document",
        type=Path,
        action="append",
        default=[],
        dest="documents",
        metavar="TEX",
        help=(
            "a LaTeX source whose citations drive the selection; repeatable, "
            "and every document the repository builds should be given"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--all",
        action="store_true",
        dest="every",
        help="render every manifest entry, not only what the document cites",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare a rebuild against the committed figures without overwriting",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="STEM",
        help="render only this figure; repeatable, overrides --all",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="print the selected stems and exit",
    )
    args = parser.parse_args(argv)

    specs = selected(args.documents or list(DEFAULT_DOCUMENTS), args.every, args.only)

    if args.list_only:
        for spec in specs:
            print(spec.stem)
        return 0

    if not args.check:
        for spec in specs:
            render(spec, args.output_dir)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        rebuilt_dir = Path(tmp)
        for spec in specs:
            render(spec, rebuilt_dir)
        stale = compare(rebuilt_dir, args.output_dir)

    if stale:
        print(
            f"stale QA figures: {', '.join(stale)} -- "
            f"run infra/build_technical_doc.sh and commit the result",
            file=sys.stderr,
        )
        return 1
    print(f"{len(specs)} QA figures match the committed output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
