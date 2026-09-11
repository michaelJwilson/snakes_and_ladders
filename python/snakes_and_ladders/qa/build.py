"""Regenerate QA figures, selected by what the documents cite.

Per pull request, only the figures the documents under ``docs/tex/`` include need
rebuilding: those are the ones a change could make stale in the committed
document, and regenerating the rest cost most of the ``documents`` job
while proving nothing about it (issue #154).

Every figure is rendered and compared at the release gate, where
``infra/release.sh`` runs ``--all --check`` (issue #484): cited and uncited
alike. The guarantee --- a figure whose rendered bytes would change cannot
reach a release claiming to be current --- is enforced by rendering every
figure rather than predicted for any of them. The selection below is a cost
decision under that gate, not a substitute for it.

``--check`` regenerates into a temporary directory and compares bytes instead
of overwriting, so a verification run cannot itself produce the state it was
meant to detect.

**Every selected figure is rendered; nothing predicts which could have
changed** (issue #490). A stamp beside each committed figure recorded a
digest of the renderer's import closure, and a figure whose stamp matched
the tree was skipped. Over 476 decisions the stamps never once skipped a
figure whose bytes would have moved, and never once let one through: every
stale call was a false positive, at ~48 minutes in one day. Rendering the
selection unconditionally costs the cited set's declared time --- since issue
#492 the whole manifest, 431.8 s over the 23 entries
`snakes_and_ladders.qa.manifest` states --- on the push to ``main`` where the
documents job runs, and answers with the bytes rather than with a hash of
their inputs.

The release gate pays that once rather than twice. ``--check`` renders the
manifest into a temporary directory and compares it, so a pass proves the
committed bytes are a render's; `infra/build_documents.sh` therefore runs
there with ``--no-figures`` and typesets what is already in the tree (issue
#530).
"""

from __future__ import annotations

import argparse
import filecmp
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from snakes_and_ladders.log import get_logger, phase
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


#: Thread-count variables a figure's process does not inherit. The suite
#: pins them to one so a process is one core (``tests/conftest.py``); a
#: figure is rendered the way the manifest renders it, with the threading
#: the committed bytes were produced under, since a reduction split across a
#: different number of threads can move a last bit (``qa/CLAUDE.md``).
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


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
    for variable in THREAD_VARIABLES:
        environment.pop(variable, None)
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
    log = get_logger(__name__, start_time=time.time())

    specs = selected(args.documents or list(DEFAULT_DOCUMENTS), args.every, args.only)
    log.info("%d figures selected", len(specs))

    if args.list_only:
        # Output, not a log line: the release script reads the stems.
        for spec in specs:
            print(spec.stem)
        return 0

    if not args.check:
        for spec in specs:
            with phase(f"render {spec.stem}"):
                render(spec, args.output_dir)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        rebuilt_dir = Path(tmp)
        for spec in specs:
            with phase(f"render {spec.stem}"):
                render(spec, rebuilt_dir)
        with phase("compare"):
            differing = compare(rebuilt_dir, args.output_dir)

    if differing:
        log.error(
            "stale QA figures: %s -- run infra/build_documents.sh and commit the result",
            ", ".join(differing),
        )
        return 1
    log.info("%d QA figures match the committed output", len(specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
