"""Re-execute the committed notebooks and compare what they print.

`docs/nb/` ships notebooks with their outputs committed, and until now nothing
re-ran them. The gap was not theoretical: `sal.sim.hmm` landing
in #182 broke `hmm.ipynb`'s import outright, and #187 switched three call sites
to a Rust sampler that could have moved every simulated number. Both were caught
by hand. A `docs/tex/` figure cannot rot that way because CI regenerates it and
byte-compares the rebuilt PDF; this is the notebooks' equivalent (issue #203).

**Text is compared; images are not.** Every number a notebook prints is
deterministic given its seeds, so a re-executed stream output must match the
committed one exactly. Rendered figures embed metadata that is not stable across
matplotlib builds, and comparing them would reproduce the `SOURCE_DATE_EPOCH`
problem `docs/CLAUDE.md` records for `docs/tex/`, for a weaker payoff. What is
checked for a figure is that the cell still produced one.

**A cell tagged ``host-dependent`` is executed and its text is not compared.**
Two kinds of reading belong to the host and not to the seeds: a wall clock,
which moves with the machine and its load, and a quantity discontinuous in
floating point, such as a recovery read through a permutation match whose
near-ties flip when a reduction is ordered differently (issue #891: every
log-likelihood agreed on the CI runner while a recovery moved from 0.048 to
0.046). A cell printing either carries the tag, still runs, and is still
checked for its figures; every other number stays under the comparison.

**The Further Work section is checked for shape, not only presence.** Root
`CLAUDE.md` makes it load-bearing: the last cell of every notebook names, with
an issue number, what the notebook could not demonstrate. All three carried a
sentence that had been false since this tool landed ("no job re-runs it") and
nothing noticed, because re-execution compares outputs and a markdown cell has
none (issue #278). So the last cell must be markdown headed
``## Further work``, and every bullet under it must name an issue (``#N``).
Whether that issue is still open is the release gate's question.

Exits 0 when every notebook agrees, 1 on the first that does not, printing a
unified diff of the cell's output.

**A notebook's execution is held to a stated budget**, :data:`NOTEBOOK_BUDGET`
seconds of wall clock or its own in :data:`NOTEBOOK_BUDGETS`, and one over it fails with its time beside the budget
(issue #891). A run too expensive to make whole is cut by that budget.

**Every notebook this is given is executed.** Which ones a run checks is a
property of its arguments alone: the notebooks named, or every one under
``docs/nb/`` when none is, listed before the first is run. A staleness digest
used to decide it instead (issue #372) and skipped ``turbo.ipynb`` until an
unrelated merge moved the hash --- at which point the check ran and found a
disagreement the notebook had carried for as long as it had been skipped
(issues #480, #507). Whether an input changed and whether a correctness check
runs are separate questions. A run too expensive to make unconditional is cut
by a stated budget, never by a hash; ``DEV.md`` carries the budget. The digest
and the ``<name>.inputs`` stamps are gone with issue #490.

**A pull request runs the notebooks it changes; the release gate runs all of
them** (issue #1087). ``--changed`` reads changed paths on stdin and checks what
:func:`changed_notebooks` selects: a changed notebook itself, every notebook when
``docs/nb/data/`` or this checker changes, and none otherwise. The full run moved
to ``infra/release.sh`` because it was 590 to 1,438 s of a local CI run. A
notebook a change breaks without touching it is then found at the next release;
the digest #490 removed skipped one for as long as nothing merged, which a
release bounds.

``--write`` re-executes and saves instead of comparing, which regenerates a
notebook after a change moves what it prints. Both live here rather than in two
tools because they must execute a notebook *identically* --- a regenerator
differing from the checker in working directory, timeout or kernel would write
a notebook the checker then rejects.

`nbformat` and `nbclient` are imported inside the functions that run a
notebook. Comparing two runs is dict arithmetic and needs neither; importing
them at module scope would put the whole Jupyter stack behind
`tests/regression/test_check_notebooks.py`, which the `python-tests` job does
not install (`uv sync --extra test`).
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT

NOTEBOOK_DIR = REPO_ROOT / "docs" / "nb"

# Generous: `potts_chain.ipynb` trains eight policies. A timeout here would
# read as a rotted notebook, which is the one failure this must not invent.
CELL_TIMEOUT = 900

#: Seconds of wall clock one notebook's execution may take, the budget
#: `DEV.md` states for the `notebooks` job. Stated, not measured: a notebook
#: over it fails the check with its time beside the budget, and is cut ---
#: fewer trials, a smaller instance --- rather than skipped (issue #891).
#: 600 s, raised to 900 s while `emission_mixture_starts` ran 620 s on the
#: runner (#909) and restored once the compiled count M step took it to 142 s
#: on the 4-core reference host with #905's best-of group (#925).
NOTEBOOK_BUDGET = 600

#: The notebooks held to a larger budget than :data:`NOTEBOOK_BUDGET`, and
#: theirs: the two starts comparisons, whose cells are a start and its polish
#: per (start, seed), at 1,000 s from #912, whose polished best-of group
#: polishes every seeding of every cell. Every other notebook keeps 600 s.
NOTEBOOK_BUDGETS: dict[str, int] = {
    "emission_mixture_starts.ipynb": 1000,
    "potts_starts.ipynb": 1000,
}


def budget_for(name: str) -> int:
    """The seconds ``name`` may spend: its own in :data:`NOTEBOOK_BUDGETS`, else :data:`NOTEBOOK_BUDGET`."""
    return NOTEBOOK_BUDGETS.get(name, NOTEBOOK_BUDGET)


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
    collected: list[str] = []
    collected_names: list[str | None] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            # The kernel splits one cell's stream into chunks at times that
            # differ from run to run, so two runs of the same cell can carry
            # the same text as a different number of outputs. Consecutive
            # chunks of the same stream are one entry, and only the text is
            # compared.
            text = "".join(output.get("text", []))
            name = output.get("name")
            if collected and collected_names[-1] == name:
                collected[-1] += text
            else:
                collected.append(text)
                collected_names.append(name)
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
            collected_names.append(None)
    return collected


#: The cell tag exempting a code cell's text from the comparison: what it
#: prints belongs to the host --- a wall clock, or a quantity discontinuous in
#: floating point --- and no rerun on another machine reproduces it (issue #891).
HOST_DEPENDENT_TAG = "host-dependent"


def is_host_dependent(cell: dict[str, Any]) -> bool:
    """Whether ``cell`` carries the :data:`HOST_DEPENDENT_TAG` tag in its metadata."""
    return HOST_DEPENDENT_TAG in cell.get("metadata", {}).get("tags", [])


def image_count(cell: dict[str, Any]) -> int:
    """How many outputs of this cell carry an image."""
    return sum(
        1 for output in cell.get("outputs", []) if "image/png" in output.get("data", {})
    )


#: The heading the last cell must carry; case-insensitive on the second word
#: because root `CLAUDE.md` writes "Further Work" and the notebooks "Further
#: work", and either is the section.
FURTHER_WORK = re.compile(r"^## Further work\s*$", re.IGNORECASE | re.MULTILINE)

#: What a Further Work bullet must name: an issue. `TICKETS.md` was the second
#: form until #804 deleted it, since a section heading names nothing that can be
#: assigned, closed or found from a pull request.
NAMES_A_TICKET = re.compile(r"#\d+")


def structure_problems(name: str, cells: Sequence[dict[str, Any]]) -> list[str]:
    """Report where a notebook's Further Work section is missing or unanchored.

    Parameters
    ----------
    name : str
        The notebook's name, for the messages.
    cells : Sequence[dict[str, Any]]
        Its cells.

    Returns
    -------
    list[str]
        Human-readable problems; empty when the last cell is a markdown cell
        headed ``## Further work`` whose every bullet names an issue, and no
        code cell's metadata sets ``scrolled: true``, which boxes a figure
        into a scroll pane (issue #891).
    """
    if not cells:
        return [f"{name}: has no cells"]
    last = cells[-1]
    source = last.get("source", "")
    text = "".join(source) if isinstance(source, list) else str(source)
    heading = FURTHER_WORK.search(text)
    if last.get("cell_type") != "markdown" or heading is None:
        return [
            f"{name}: the last cell is not a markdown cell headed '## Further work'"
        ]

    problems = [
        f"{name}: code cell {index} is set to scroll its output; a figure or a "
        f"table is shown whole (`--write` sets scrolled: false)"
        for index, cell in enumerate(
            (cell for cell in cells if cell.get("cell_type") == "code"), start=1
        )
        if cell.get("metadata", {}).get("scrolled") is True
    ]
    for bullet in re.split(r"^- ", text[heading.end() :], flags=re.MULTILINE)[1:]:
        if not NAMES_A_TICKET.search(bullet):
            first_line = bullet.strip().splitlines()[0] if bullet.strip() else ""
            problems.append(
                f"{name}: Further work bullet names no issue: {first_line!r}"
            )
    return problems


def differences(
    name: str,
    committed: Sequence[dict[str, Any]],
    executed: Sequence[dict[str, Any]],
) -> list[str]:
    """Report where two runs of the same notebook disagree.

    A code cell tagged :data:`HOST_DEPENDENT_TAG` has its text exempted and its
    figures still counted.

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
        # The committed cell declares the tag: it is the notebook's statement
        # of which outputs belong to the host, and a rerun preserves the metadata.
        if expected != realized and not is_host_dependent(before):
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


def over_budget(name: str, seconds: float) -> list[str]:
    """Report a notebook whose execution took longer than its budget, :func:`budget_for`.

    Parameters
    ----------
    name : str
        The notebook's filename, for the message.
    seconds : float
        Wall clock of its execution.

    Returns
    -------
    list[str]
        One message naming the measured time and the budget, or nothing.
    """
    budget = budget_for(name)
    if seconds <= budget:
        return []
    return [
        f"{name} executed in {seconds:.0f} s, over the {budget} s budget "
        f"a notebook may spend (DEV.md, notebooks); cut what it runs"
    ]


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
    figure runs ``infra/build_documents.sh``.
    """
    import nbformat

    notebook = execute(path)
    # Every output shown whole: a front end that honours the two keys never
    # boxes a tall figure or a long table into a scroll pane (issue #891).
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.metadata["scrolled"] = False
            cell.metadata["collapsed"] = False
    nbformat.write(notebook, path)


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
    problems = structure_problems(path.name, committed.cells)
    started = time.perf_counter()
    try:
        executed = execute(path)
    except CellExecutionError as failure:
        # A notebook that no longer runs is the loudest way it can rot, and
        # reporting that as a crash of this tool rather than as a failure of
        # that notebook would bury it. `sal.sim.hmm` landing in #182 broke
        # `hmm.ipynb`'s import outright, which is exactly this case.
        return [*problems, f"{path.name} did not execute:\n{failure}"]

    return [
        *problems,
        *over_budget(path.name, time.perf_counter() - started),
        *differences(path.name, committed.cells, executed.cells),
    ]


#: A changed path under one of these selects every notebook: the data they
#: read, and this checker itself (issue #1087).
SELECTS_ALL = ("docs/nb/data/", "infra/check_notebooks.py")


def changed_notebooks(changed: Iterable[str]) -> list[Path]:
    """The notebooks a change re-executes on a pull request (issue #1087).

    A changed notebook selects itself; a changed path under
    :data:`SELECTS_ALL` selects every notebook; any other change selects
    none. Every notebook runs at release (`infra/release.sh`) whatever this
    returns, so a notebook a change breaks without touching it is found at
    the next release, never skipped for good (issues #480, #507).
    """
    paths = [line.strip() for line in changed if line.strip()]
    if any(path.startswith(SELECTS_ALL) for path in paths):
        return sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    return sorted(
        REPO_ROOT / path
        for path in paths
        if path.startswith("docs/nb/")
        and path.endswith(".ipynb")
        and "/" not in path.removeprefix("docs/nb/")
        and (REPO_ROOT / path).exists()
    )


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
    parser.add_argument(
        "--changed",
        action="store_true",
        help=(
            "Read changed paths, one per line, from stdin and check only the "
            "notebooks they select (changed_notebooks); the release gate "
            "checks every notebook."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="With --changed: print the selected notebooks and exit.",
    )
    arguments = parser.parse_args(argv)
    if arguments.changed:
        selected = changed_notebooks(sys.stdin)
        if arguments.list:
            print(" ".join(str(path.relative_to(REPO_ROOT)) for path in selected))
            return 0
        if not selected:
            print("no notebook selected: notebooks run at release (issue #1087)")
            return 0
        arguments.notebooks = selected
    from sal.log import get_logger, phase

    log = get_logger("check_notebooks", start_time=time.time())
    paths = arguments.notebooks or sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not paths:
        log.error("no notebooks found under %s", NOTEBOOK_DIR)
        return 1
    # The set is stated before any of it runs, so a log says which claims this
    # run verified without the reader inferring it from the absences.
    log.info(
        "executing %d notebook(s): %s",
        len(paths),
        ", ".join(path.name for path in paths),
    )

    if arguments.write:
        import nbformat

        failed = False
        for path in paths:
            with phase(f"execute {path.name}"):
                rewrite(path)
            log.info("wrote %s", path)
            for problem in structure_problems(
                path.name, nbformat.read(path, as_version=4).cells
            ):
                failed = True
                log.error("%s", problem)
        return 1 if failed else 0

    failed = False
    for path in paths:
        with phase(f"compare {path.name}"):
            problems = compare(path)
        if problems:
            failed = True
            log.error("FAIL %s", path)
            for problem in problems:
                log.error("%s", problem)
            log.error(
                "regenerate with: uv run python infra/check_notebooks.py --write %s",
                path,
            )
        else:
            log.info("ok   %s", path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
