"""Re-execute the committed notebooks and compare what they print.

`docs/nb/` ships notebooks with their outputs committed, and until now
nothing re-ran them. That gap was not theoretical: `snakes_and_ladders.sim.hmm` landing in
#182 broke `hmm.ipynb`'s import outright, and #187 switched three call sites
to a Rust sampler that could have moved every simulated number. Both were
caught by hand. A `docs/tex/` figure cannot rot that way because CI
regenerates it and byte-compares the rebuilt PDF; this is the notebooks'
equivalent (issue #203).

**Text is compared; images are not.** Every number a notebook prints is
deterministic given its seeds, so a re-executed stream output must match the
committed one exactly. Rendered figures embed metadata that is not stable
across matplotlib builds, and comparing them would reproduce the
`SOURCE_DATE_EPOCH` problem `docs/CLAUDE.md` records for `docs/tex/` -- for a
weaker payoff, since the printed numbers are what the notebooks assert with.
What is checked for a figure is that the cell still produced one.

Exits 0 when every notebook agrees, 1 on the first that does not, printing a
unified diff of the cell's output.

``--write`` re-executes and saves instead of comparing, which is how a
notebook is regenerated after a change moves what it prints. Both live here
rather than in two tools because they must execute a notebook *identically* --
a regenerator that differed from the checker in working directory, timeout or
kernel would write a notebook the checker then rejects.

`nbformat` and `nbclient` are imported inside the functions that run a
notebook, not at module scope. Comparing two runs is pure dict arithmetic and
needs neither; importing them here would put the whole Jupyter stack behind
`tests/regression/test_check_notebooks.py`, which the `python-tests` job does
not install (`uv sync --extra test`) and does not need to.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "docs" / "nb"

# Generous: `potts_chain.ipynb` trains eight policies. A timeout here would
# read as a rotted notebook, which is the one failure this must not invent.
CELL_TIMEOUT = 900


def text_outputs(cell: dict[str, Any]) -> list[str]:
    """Everything a code cell printed, in order.

    Streams and text/plain execute results count; images do not, for the
    reason the module docstring gives.

    Parameters
    ----------
    cell : dict[str, Any]
        A notebook cell.

    Returns
    -------
    list[str]
        One entry per text-bearing output.
    """
    collected = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            collected.append("".join(output.get("text", [])))
            continue
        if output.get("output_type") not in {"execute_result", "display_data"}:
            continue
        data = output.get("data", {})
        if "image/png" in data:
            # A figure's `text/plain` is `<Figure size 560x340 with 1 Axes>`
            # -- a repr of the artist, not a measurement, and it moves with
            # the figure size. The figure itself is counted by `image_count`.
            continue
        plain = data.get("text/plain")
        if plain is not None:
            collected.append("".join(plain) if isinstance(plain, list) else str(plain))
    return collected


def image_count(cell: dict[str, Any]) -> int:
    """How many outputs of this cell carry an image."""
    return sum(
        1 for output in cell.get("outputs", []) if "image/png" in output.get("data", {})
    )


def differences(
    name: str,
    committed: Sequence[dict[str, Any]],
    executed: Sequence[dict[str, Any]],
) -> list[str]:
    """Report where two runs of the same notebook disagree.

    Separated from execution so it can be tested without a kernel, which is
    what `tests/regression/test_check_notebooks.py` does --- the figure-repr
    exclusion in :func:`text_outputs` was a real bug and a 92-second test
    would not have been run often enough to catch it.

    Parameters
    ----------
    name : str
        The notebook's filename, for the diff headers.
    committed, executed : Sequence[dict[str, Any]]
        The two runs' cells, in order.

    Returns
    -------
    list[str]
        Human-readable differences; empty when they agree.
    """
    problems = []
    code_cells = [
        (before, after)
        for before, after in zip(committed, executed, strict=True)
        if before["cell_type"] == "code"
    ]
    for index, (before, after) in enumerate(code_cells, start=1):
        expected, realized = text_outputs(before), text_outputs(after)
        if expected != realized:
            diff = difflib.unified_diff(
                "".join(expected).splitlines(keepends=True),
                "".join(realized).splitlines(keepends=True),
                fromfile=f"{name} cell {index}: committed",
                tofile=f"{name} cell {index}: re-executed",
            )
            problems.append("".join(diff))
        if image_count(before) != image_count(after):
            problems.append(
                f"{name} cell {index}: committed {image_count(before)} "
                f"figure(s), re-executed produced {image_count(after)}"
            )
    return problems


def execute(path: Path) -> Any:
    """Run ``path`` in place and return the executed notebook.

    Parameters
    ----------
    path : Path
        The notebook to run. It is executed in its own directory, because
        the notebooks resolve the repository root by walking up from the
        working directory.

    Returns
    -------
    Any
        The notebook with outputs replaced by what this run produced.
    """
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    nbformat.validator.normalize(notebook)
    NotebookClient(
        notebook,
        timeout=CELL_TIMEOUT,
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    for cell in notebook.cells:
        # nbclient records four wall-clock timestamps per cell. They are the
        # notebooks' version of the `\today` that made the committed PDFs fail
        # its own staleness check (`docs/CLAUDE.md`): nothing reads them, they
        # differ on every run, and left in they would make each regeneration
        # a diff of times with the real change buried inside it.
        cell.get("metadata", {}).pop("execution", None)
    return notebook


def rewrite(path: Path) -> None:
    """Re-execute ``path`` and save it, replacing the committed outputs.

    The regeneration half of this tool. A change that moves what a notebook
    prints runs this and commits the result, exactly as a change that moves a
    figure runs ``infra/build_technical_doc.sh``.
    """
    import nbformat

    nbformat.write(execute(path), path)


def compare(path: Path) -> list[str]:
    """Re-execute ``path`` and report where it disagrees with what is committed.

    Parameters
    ----------
    path : Path
        The notebook to check.

    Returns
    -------
    list[str]
        Human-readable differences; empty when the notebook still produces
        what it claims.
    """
    import nbformat
    from nbclient.exceptions import CellExecutionError

    committed = nbformat.read(path, as_version=4)
    try:
        executed = execute(path)
    except CellExecutionError as failure:
        # A notebook that no longer runs is the loudest way it can rot, and
        # reporting that as a crash of this tool rather than as a failure of
        # that notebook would bury it. `snakes_and_ladders.sim.hmm` landing in #182 broke
        # `hmm.ipynb`'s import outright, which is exactly this case.
        return [f"{path.name} did not execute:\n{failure}"]

    return differences(path.name, committed.cells, executed.cells)


def main(argv: list[str] | None = None) -> int:
    """Check every notebook, or the ones named.

    Returns
    -------
    int
        0 when all agree, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebooks to check; default is every one under docs/nb/.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Re-execute and save instead of comparing. Run this after a "
            "change moves what a notebook prints, then commit the result."
        ),
    )
    arguments = parser.parse_args(argv)
    paths = arguments.notebooks or sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not paths:
        print(f"no notebooks found under {NOTEBOOK_DIR}", file=sys.stderr)
        return 1

    if arguments.write:
        for path in paths:
            rewrite(path)
            print(f"wrote {path}")
        return 0

    failed = False
    for path in paths:
        problems = compare(path)
        if problems:
            failed = True
            print(f"FAIL {path}", file=sys.stderr)
            for problem in problems:
                print(problem, file=sys.stderr)
            print(
                "  regenerate with: uv run python infra/check_notebooks.py "
                f"--write {path}",
                file=sys.stderr,
            )
        else:
            print(f"ok   {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
