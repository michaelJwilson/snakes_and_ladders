"""QA table: problem-size parameters across the simulation fixtures.

Tabulates taxa count, site count, seed, and Monte Carlo tolerance for a set
of ``simulation_params.yaml``-format fixtures, read directly from the yaml
rather than hardcoded, so the technical document's numbers cannot drift from
what the regression suite actually runs (``tests/regression/test_jc_simulate.py``).
"""

from __future__ import annotations

from pathlib import Path

from snakes_and_ladders.qa.figure import QATable, latex_integer
from snakes_and_ladders.qa.runner import ParamsArgument, table_main
from snakes_and_ladders.sim.params import SimulationParams, load_simulation_params
from snakes_and_ladders.sim.tree import preorder


def _n_taxa(params: SimulationParams) -> int:
    return sum(1 for node in preorder(params.tau) if node.is_leaf)


def _escape(text: str) -> str:
    """Escape the LaTeX specials a fixture filename can contain."""
    return text.replace("_", "\\_")


def render_problem_sizes(
    fixture_names: list[str], params_by_fixture: dict[str, SimulationParams]
) -> str:
    """Build a LaTeX ``tabular`` of problem-size parameters, one row per fixture.

    A typeset table rather than a matplotlib image: it matches the
    surrounding type, scales with the document, and can be selected and
    searched. The numbers are read from the yaml, so they cannot drift from
    what the regression suite runs.

    Parameters
    ----------
    fixture_names : list[str]
        Fixture filenames, in the row order to display.
    params_by_fixture : dict[str, SimulationParams]
        Loaded parameters, keyed by the same filenames.

    Returns
    -------
    str
        A complete ``tabular`` environment.
    """
    rows = [
        " & ".join(
            [
                f"\\texttt{{{_escape(fixture_name)}}}",
                str(_n_taxa(params_by_fixture[fixture_name])),
                latex_integer(params_by_fixture[fixture_name].n_sites),
                # A seed is an identifier, not a magnitude: separators would
                # make 20260902 look like a quantity rather than a date.
                str(params_by_fixture[fixture_name].seed),
                f"{params_by_fixture[fixture_name].tolerance:g}",
            ]
        )
        + r" \\"
        for fixture_name in fixture_names
    ]
    return "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"  \toprule",
            r"  Fixture & Taxa & Sites & Seed & Tolerance \\",
            r"  \midrule",
            *(f"  {row}" for row in rows),
            r"  \bottomrule",
            r"\end{tabular}",
        ]
    )


def build_caption(fixture_names: list[str]) -> str:
    """Caption text for the problem-sizes table.

    Parameters
    ----------
    fixture_names : list[str]
        Fixture filenames tabulated, in row order.

    Returns
    -------
    str
        Plain-text caption, safe to ``\\input`` into LaTeX verbatim.
    """
    return (
        f"Problem-size parameters across the {len(fixture_names)} simulation "
        "regression fixtures backing the Jukes-Cantor simulation test: taxa "
        "count, site count, seed, and the Monte Carlo tolerance each "
        "validates simulated substitution frequencies within."
    )


def _load_named(path: Path) -> tuple[str, SimulationParams]:
    """Load one fixture, keeping the filename the caption reports.

    The caption names every fixture it tabulates, so the filename is part of
    what this table reports and cannot be recovered from the loaded params.

    Returns
    -------
    tuple[str, SimulationParams]
        The fixture's filename and its loaded contents.
    """
    return path.name, load_simulation_params(path)


# Repeated, because the table is one row per fixture and the row order is the
# order the flags are given in.
NAMED_PARAMS = ParamsArgument("params", _load_named, repeated=True)


def build_table(named: list[tuple[str, SimulationParams]]) -> tuple[str, str]:
    """Build the problem-sizes ``tabular`` body and its caption.

    Parameters
    ----------
    named : list[tuple[str, SimulationParams]]
        Each fixture's filename and loaded contents, in row order.

    Returns
    -------
    tuple[str, str]
        The ``tabular`` body and its caption.
    """
    fixture_names = [name for name, _ in named]
    params_by_fixture = dict(named)
    body = render_problem_sizes(fixture_names, params_by_fixture)
    return body, build_caption(fixture_names)


def main(argv: list[str] | None = None) -> QATable:
    """Render the table from the command line.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; ``None`` reads ``sys.argv``.

    Returns
    -------
    QATable
        Paths written, and the caption.
    """
    return table_main(
        stem="sim_problem_sizes",
        description=__doc__,
        params=[NAMED_PARAMS],
        build=build_table,
        argv=argv,
    )


if __name__ == "__main__":
    main()
