"""The shared QA entry point's own contract.

The thirteen figure and table scripts exercise :mod:`phylo.qa.runner`
end-to-end, so what is pinned here is what routing them through one function
is supposed to guarantee and what no individual figure's test would catch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from phylo.qa.runner import (
    Option,
    ParamsArgument,
    figure_main,
    table_main,
)


def _one_line(path: Path) -> str:
    """Load a file as its stripped contents.

    Returns
    -------
    str
        The file's text, stripped.
    """
    return path.read_text().strip()


def _write_params(tmp_path: Path, name: str, text: str) -> Path:
    """Write a stand-in parameters file.

    Returns
    -------
    Path
        The written file.
    """
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.mark.edge_case
def test_a_figure_is_closed_even_when_writing_it_fails(tmp_path: Path) -> None:
    # The reason the close lives in a `finally` in the runner rather than in
    # each builder. A caption carrying an unescaped LaTeX special is refused
    # by `check_latex_safe` inside the write, and a figure leaked on that path
    # accumulates across a build until matplotlib warns and memory grows.
    params_path = _write_params(tmp_path, "params.txt", "ignored")
    leaked: list[Figure] = []

    def build(_: str) -> tuple[Figure, str]:
        fig = plt.figure()
        leaked.append(fig)
        # `_` is exactly what check_latex_safe rejects.
        return fig, "a caption with an unescaped under_score"

    with pytest.raises(ValueError, match="unescaped LaTeX special"):
        figure_main(
            stem="unwritable",
            description=None,
            params=[ParamsArgument("params", _one_line)],
            build=build,
            argv=["--params", str(params_path), "--output-dir", str(tmp_path)],
        )

    assert len(leaked) == 1
    assert not plt.fignum_exists(leaked[0].number)


@pytest.mark.structural
def test_repeated_parameters_reach_the_builder_in_the_order_given(
    tmp_path: Path,
) -> None:
    # The problem-sizes table is one row per fixture, and the row order is the
    # order the flags were given, so the runner must not reorder or dedupe.
    first = _write_params(tmp_path, "first.txt", "alpha")
    second = _write_params(tmp_path, "second.txt", "beta")
    third = _write_params(tmp_path, "third.txt", "alpha")

    def build(loaded: list[str]) -> tuple[str, str]:
        return " ".join(loaded), "caption"

    written = table_main(
        stem="ordered",
        description=None,
        params=[ParamsArgument("params", _one_line, repeated=True)],
        build=build,
        argv=[
            "--params",
            str(second),
            "--params",
            str(first),
            "--params",
            str(third),
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert written.table_path.read_text().strip() == "beta alpha alpha"


@pytest.mark.structural
def test_parameters_reach_the_builder_in_declaration_order_not_argv_order(
    tmp_path: Path,
) -> None:
    # A builder takes its parameters positionally, so the declaration order is
    # its signature. If the runner passed them in the order the flags happened
    # to appear, two files through the same loader -- the rooted and unrooted
    # fixtures of `opt_branch_recovery` -- would silently swap.
    left = _write_params(tmp_path, "left.txt", "left-value")
    right = _write_params(tmp_path, "right.txt", "right-value")

    def build(first: str, second: str) -> tuple[str, str]:
        return f"{first}|{second}", "caption"

    written = table_main(
        stem="declaration_order",
        description=None,
        params=[
            ParamsArgument("first-params", _one_line),
            ParamsArgument("second-params", _one_line),
        ],
        build=build,
        argv=[
            "--second-params",
            str(right),
            "--first-params",
            str(left),
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert written.table_path.read_text().strip() == "left-value|right-value"


@pytest.mark.structural
def test_an_absent_option_reaches_the_builder_as_its_default(
    tmp_path: Path,
) -> None:
    # `sim_example` keeps a `--n-sites-shown` default of 10, and the build
    # script never passes the flag, so the default is the value every
    # committed rendering of that figure was produced with.
    params_path = _write_params(tmp_path, "params.txt", "ignored")

    def build(_: str, n_sites_shown: int = -1) -> tuple[str, str]:
        return str(n_sites_shown), "caption"

    written = table_main(
        stem="defaulted",
        description=None,
        params=[ParamsArgument("params", _one_line)],
        build=build,
        options=[Option("n-sites-shown", int, 10)],
        argv=["--params", str(params_path), "--output-dir", str(tmp_path)],
    )

    assert written.table_path.read_text().strip() == "10"


@pytest.mark.structural
def test_an_option_given_on_the_command_line_overrides_its_default(
    tmp_path: Path,
) -> None:
    params_path = _write_params(tmp_path, "params.txt", "ignored")

    def build(_: str, n_sites_shown: int = -1) -> tuple[str, str]:
        return str(n_sites_shown), "caption"

    written = table_main(
        stem="overridden",
        description=None,
        params=[ParamsArgument("params", _one_line)],
        build=build,
        options=[Option("n-sites-shown", int, 10)],
        argv=[
            "--params",
            str(params_path),
            "--output-dir",
            str(tmp_path),
            "--n-sites-shown",
            "3",
        ],
    )

    assert written.table_path.read_text().strip() == "3"


@pytest.mark.edge_case
def test_a_missing_parameters_file_argument_is_refused(tmp_path: Path) -> None:
    # Every parameters file is required: a figure rendered from a default
    # fixture nobody named would carry a caption claiming provenance it was
    # not given.
    def build(_: str) -> tuple[str, str]:
        return "body", "caption"

    with pytest.raises(SystemExit):
        table_main(
            stem="incomplete",
            description=None,
            params=[ParamsArgument("params", _one_line)],
            build=build,
            argv=["--output-dir", str(tmp_path)],
        )
